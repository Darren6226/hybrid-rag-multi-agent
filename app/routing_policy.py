"""
路由策略管理器
提供可配置的循环检测、代理重试和回退策略
"""

from typing import Dict, List, Set, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
import re


class RetryStrategy(Enum):
    """重试策略"""
    ONCE = "once"           # 只尝试一次
    TWICE = "twice"         # 尝试两次
    EXPONENTIAL = "exp"     # 指数退避
    NEVER = "never"         # 永不重试


class FallbackOrder(Enum):
    """回退顺序"""
    SEQUENTIAL = "seq"      # 按列表顺序
    PRIORITY = "prio"       # 按优先级权重
    SCORE_BASED = "score"   # 基于历史成功率


@dataclass
class WorkerConfig:
    """单个代理的配置"""
    name: str
    max_attempts: int = 1          # 最大尝试次数
    retry_strategy: RetryStrategy = RetryStrategy.ONCE
    priority: int = 5              # 优先级 (1-10, 越高越优先)
    fallback_workers: List[str] = field(default_factory=list)
    error_keywords: List[str] = field(default_factory=list)  # 触发回退的错误关键词


@dataclass
class RoutingState:
    """路由状态跟踪"""
    worker_attempts: Dict[str, int] = field(default_factory=dict)
    worker_errors: Dict[str, List[str]] = field(default_factory=dict)
    worker_success_rate: Dict[str, float] = field(default_factory=dict)
    total_turns: int = 0


class RoutingPolicyManager:
    """路由策略管理器 - 核心改进"""

    def __init__(self, worker_configs: List[WorkerConfig]):
        self.worker_configs: Dict[str, WorkerConfig] = {
            cfg.name: cfg for cfg in worker_configs
        }
        self.state = RoutingState()
        self._init_success_rates()

    def _init_success_rates(self):
        """初始化成功率 (可以从持久化存储加载)"""
        for name in self.worker_configs:
            self.state.worker_success_rate[name] = 0.5  # 默认50%

    def can_call_worker(self, worker_name: str) -> bool:
        """检查是否可以调用该代理"""
        if worker_name not in self.worker_configs:
            return False

        config = self.worker_configs[worker_name]

        # 检查尝试次数
        attempts = self.state.worker_attempts.get(worker_name, 0)
        return attempts < config.max_attempts

    def record_attempt(self, worker_name: str, success: bool = True, error_msg: str = ""):
        """记录代理调用结果"""
        self.state.worker_attempts[worker_name] = \
            self.state.worker_attempts.get(worker_name, 0) + 1

        if success:
            # 更新成功率 (简单移动平均)
            current_rate = self.state.worker_success_rate.get(worker_name, 0.5)
            self.state.worker_success_rate[worker_name] = (current_rate * 0.9 + 1.0 * 0.1)
        else:
            current_rate = self.state.worker_success_rate.get(worker_name, 0.5)
            self.state.worker_success_rate[worker_name] = (current_rate * 0.9 + 0.0 * 0.1)

            # 记录错误
            if worker_name not in self.state.worker_errors:
                self.state.worker_errors[worker_name] = []
            self.state.worker_errors[worker_name].append(error_msg)

    def get_fallback_worker(
        self,
        failed_worker: str,
        called_workers: Set[str],
        last_worker: str
    ) -> Optional[str]:
        """获取回退代理"""
        if failed_worker not in self.worker_configs:
            return None

        config = self.worker_configs[failed_worker]

        # 按优先级排序候选代理
        candidates = []
        for fallback_name in config.fallback_workers:
            if fallback_name in self.worker_configs:
                if fallback_name not in called_workers:
                    fallback_cfg = self.worker_configs[fallback_name]
                    candidates.append((
                        fallback_name,
                        fallback_cfg.priority,
                        self.state.worker_success_rate.get(fallback_name, 0.5)
                    ))

        if not candidates:
            return None

        # 排序: 优先级 > 成功率 > 未调用
        candidates.sort(key=lambda x: (-x[1], -x[2]))
        return candidates[0][0] if candidates else None

    def should_force_finish(
        self,
        called_workers: Set[str],
        next_worker: str
    ) -> bool:
        """判断是否应该强制结束"""
        # 如果所有可用代理都尝试过
        available_workers = {
            name for name in self.worker_configs
            if self.can_call_worker(name)
        }

        # 没有可用代理了
        if not available_workers:
            return True

        # 下一个代理不可用
        if next_worker not in available_workers:
            return True

        return False

    def tick(self):
        """轮次递进"""
        self.state.total_turns += 1

    def reset_for_new_query(self):
        """为新问题重置部分状态"""
        self.state.worker_attempts.clear()
        self.state.total_turns = 0
        # 保留成功率和历史错误用于学习

    def get_routing_summary(self) -> Dict:
        """获取路由状态摘要"""
        return {
            "total_turns": self.state.total_turns,
            "worker_attempts": self.state.worker_attempts.copy(),
            "worker_success_rates": self.state.worker_success_rate.copy(),
        }


def create_default_policy() -> RoutingPolicyManager:
    """创建默认的路由策略配置"""
    worker_configs = [
        WorkerConfig(
            name="chat",
            max_attempts=2,
            retry_strategy=RetryStrategy.TWICE,
            priority=8,
            fallback_workers=["sqler", "coder"]
        ),
        WorkerConfig(
            name="coder",
            max_attempts=1,
            retry_strategy=RetryStrategy.ONCE,
            priority=7,
            fallback_workers=["sqler"]
        ),
        WorkerConfig(
            name="sqler",
            max_attempts=2,
            retry_strategy=RetryStrategy.TWICE,
            priority=9,
            fallback_workers=["graph_kg", "vec_kg", "chat"],
            error_keywords=["error", "错误", "timeout", "超时"]
        ),
        WorkerConfig(
            name="graph_kg",
            max_attempts=1,
            retry_strategy=RetryStrategy.ONCE,
            priority=6,
            fallback_workers=["vec_kg", "sqler"],
            error_keywords=[
                "syntax", "error", "错误", "connection", "timeout",
                "unknown", "不知道", "无法回答"
            ]
        ),
        WorkerConfig(
            name="vec_kg",
            max_attempts=1,
            retry_strategy=RetryStrategy.ONCE,
            priority=6,
            fallback_workers=["graph_kg", "sqler", "chat"],
            error_keywords=[
                "error", "错误", "timeout", "不知道", "无法回答"
            ]
        ),
    ]

    return RoutingPolicyManager(worker_configs)


def create_conservative_policy() -> RoutingPolicyManager:
    """创建保守的路由策略 (更严格的循环检测)"""
    worker_configs = [
        WorkerConfig(
            name="chat",
            max_attempts=1,
            retry_strategy=RetryStrategy.ONCE,
            priority=10
        ),
        WorkerConfig(
            name="coder",
            max_attempts=1,
            retry_strategy=RetryStrategy.NEVER,
            priority=5
        ),
        WorkerConfig(
            name="sqler",
            max_attempts=1,
            retry_strategy=RetryStrategy.ONCE,
            priority=9
        ),
        WorkerConfig(
            name="graph_kg",
            max_attempts=1,
            retry_strategy=RetryStrategy.NEVER,
            priority=4,
            fallback_workers=["vec_kg", "sqler"]
        ),
        WorkerConfig(
            name="vec_kg",
            max_attempts=1,
            retry_strategy=RetryStrategy.NEVER,
            priority=4,
            fallback_workers=["sqler", "chat"]
        ),
    ]

    return RoutingPolicyManager(worker_configs)
