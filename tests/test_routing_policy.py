"""
测试 RoutingPolicyManager 核心逻辑

纯逻辑测试，无外部依赖。
"""

import pytest
from app.routing_policy import (
    RoutingPolicyManager,
    WorkerConfig,
    RetryStrategy,
    create_default_policy,
    create_conservative_policy,
)


class TestWorkerConfig:
    """测试 WorkerConfig 数据类"""

    def test_default_values(self):
        cfg = WorkerConfig(name="test")
        assert cfg.name == "test"
        assert cfg.max_attempts == 1
        assert cfg.retry_strategy == RetryStrategy.ONCE
        assert cfg.priority == 5
        assert cfg.fallback_workers == []
        assert cfg.error_keywords == []

    def test_custom_values(self):
        cfg = WorkerConfig(
            name="sqler",
            max_attempts=3,
            retry_strategy=RetryStrategy.TWICE,
            priority=9,
            fallback_workers=["vec_kg"],
            error_keywords=["error", "timeout"],
        )
        assert cfg.max_attempts == 3
        assert cfg.priority == 9
        assert "vec_kg" in cfg.fallback_workers


class TestRoutingPolicyManager:
    """测试 RoutingPolicyManager 核心方法"""

    def _make_manager(self):
        configs = [
            WorkerConfig(name="chat", max_attempts=2, priority=8, fallback_workers=["sqler"]),
            WorkerConfig(name="sqler", max_attempts=2, priority=9, fallback_workers=["vec_kg", "chat"]),
            WorkerConfig(name="vec_kg", max_attempts=1, priority=6, fallback_workers=["chat"]),
            WorkerConfig(name="graph_kg", max_attempts=1, priority=6, fallback_workers=["vec_kg"]),
        ]
        return RoutingPolicyManager(configs)

    # --- can_call_worker ---

    def test_can_call_worker_within_limit(self):
        mgr = self._make_manager()
        assert mgr.can_call_worker("chat") is True

    def test_can_call_worker_exceeds_limit(self):
        mgr = self._make_manager()
        mgr.record_attempt("vec_kg", success=True)
        assert mgr.can_call_worker("vec_kg") is False  # max_attempts=1

    def test_can_call_worker_unknown(self):
        mgr = self._make_manager()
        assert mgr.can_call_worker("nonexistent") is False

    def test_can_call_worker_after_reset(self):
        mgr = self._make_manager()
        mgr.record_attempt("vec_kg", success=True)
        assert mgr.can_call_worker("vec_kg") is False
        mgr.reset_for_new_query()
        assert mgr.can_call_worker("vec_kg") is True

    # --- record_attempt ---

    def test_record_attempt_increments_count(self):
        mgr = self._make_manager()
        mgr.record_attempt("chat", success=True)
        assert mgr.state.worker_attempts["chat"] == 1
        mgr.record_attempt("chat", success=True)
        assert mgr.state.worker_attempts["chat"] == 2

    def test_record_attempt_success_updates_rate(self):
        mgr = self._make_manager()
        initial_rate = mgr.state.worker_success_rate["chat"]
        mgr.record_attempt("chat", success=True)
        # EMA: 0.5 * 0.9 + 1.0 * 0.1 = 0.55
        assert mgr.state.worker_success_rate["chat"] > initial_rate

    def test_record_attempt_failure_updates_rate(self):
        mgr = self._make_manager()
        initial_rate = mgr.state.worker_success_rate["chat"]
        mgr.record_attempt("chat", success=False, error_msg="timeout")
        # EMA: 0.5 * 0.9 + 0.0 * 0.1 = 0.45
        assert mgr.state.worker_success_rate["chat"] < initial_rate
        assert "timeout" in mgr.state.worker_errors["chat"]

    # --- get_fallback_worker ---

    def test_get_fallback_returns_configured_fallback(self):
        mgr = self._make_manager()
        fallback = mgr.get_fallback_worker("sqler", called_workers=set(), last_worker="sqler")
        # sqler 的 fallback_workers=["vec_kg", "chat"]，按 priority 排序：chat(8) > vec_kg(6)
        assert fallback == "chat"

    def test_get_fallback_skips_already_called(self):
        mgr = self._make_manager()
        fallback = mgr.get_fallback_worker("sqler", called_workers={"vec_kg"}, last_worker="sqler")
        assert fallback == "chat"  # vec_kg 已调用，回退到 chat

    def test_get_fallback_returns_none_if_all_called(self):
        mgr = self._make_manager()
        fallback = mgr.get_fallback_worker("sqler", called_workers={"vec_kg", "chat"}, last_worker="sqler")
        assert fallback is None

    def test_get_fallback_unknown_worker(self):
        mgr = self._make_manager()
        fallback = mgr.get_fallback_worker("nonexistent", called_workers=set(), last_worker="x")
        assert fallback is None

    # --- should_force_finish ---

    def test_should_force_finish_all_exhausted(self):
        mgr = self._make_manager()
        # 用完所有 worker 的尝试次数
        mgr.record_attempt("chat", success=True)
        mgr.record_attempt("chat", success=True)
        mgr.record_attempt("sqler", success=True)
        mgr.record_attempt("sqler", success=True)
        mgr.record_attempt("vec_kg", success=True)
        mgr.record_attempt("graph_kg", success=True)
        assert mgr.should_force_finish(called_workers=set(), next_worker="chat") is True

    def test_should_force_finish_next_unavailable(self):
        mgr = self._make_manager()
        mgr.record_attempt("vec_kg", success=True)  # vec_kg max=1, 已用完
        assert mgr.should_force_finish(called_workers=set(), next_worker="vec_kg") is True

    def test_should_not_force_finish_still_available(self):
        mgr = self._make_manager()
        assert mgr.should_force_finish(called_workers=set(), next_worker="chat") is False

    # --- tick & reset ---

    def test_tick_increments_turns(self):
        mgr = self._make_manager()
        assert mgr.state.total_turns == 0
        mgr.tick()
        mgr.tick()
        assert mgr.state.total_turns == 2

    def test_reset_for_new_query_clears_attempts(self):
        mgr = self._make_manager()
        mgr.record_attempt("chat", success=True)
        mgr.record_attempt("sqler", success=False, error_msg="err")
        mgr.reset_for_new_query()
        assert mgr.state.worker_attempts == {}
        # 成功率和错误历史应保留
        assert "chat" in mgr.state.worker_success_rate


class TestFactoryFunctions:
    """测试工厂函数"""

    def test_create_default_policy(self):
        mgr = create_default_policy()
        assert isinstance(mgr, RoutingPolicyManager)
        assert "chat" in mgr.worker_configs
        assert "sqler" in mgr.worker_configs
        assert "graph_kg" in mgr.worker_configs
        assert "vec_kg" in mgr.worker_configs

    def test_create_conservative_policy(self):
        mgr = create_conservative_policy()
        assert isinstance(mgr, RoutingPolicyManager)
        # 保守策略：所有 worker 最多尝试 1 次
        for cfg in mgr.worker_configs.values():
            assert cfg.max_attempts == 1
