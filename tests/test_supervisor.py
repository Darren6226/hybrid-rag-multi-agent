"""
测试 Supervisor 路由决策逻辑

使用 mock LLM，不依赖真实 API。
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END

from app.supervisor import (
    supervisor,
    members,
    Router,
    _extract_worker_response_content,
    _check_worker_error,
    _analyze_response_quality,
    policy_manager,
    reset_supervisor,
)


@pytest.fixture(autouse=True)
def _reset_policy():
    """每个测试前重置路由策略状态"""
    reset_supervisor()
    yield
    reset_supervisor()


class TestHelperFunctions:
    """测试辅助函数"""

    def test_extract_worker_response_content_found(self):
        state = {
            "messages": [
                HumanMessage(content="问题"),
                AIMessage(content="vec_kg 的回答", name="vec_kg"),
            ]
        }
        content = _extract_worker_response_content(state, "vec_kg")
        assert content == "vec_kg 的回答"

    def test_extract_worker_response_content_not_found(self):
        state = {"messages": [HumanMessage(content="问题")]}
        content = _extract_worker_response_content(state, "nonexistent")
        assert content == ""

    def test_check_worker_error_positive(self):
        assert _check_worker_error("I don't know the answer", ["don't know", "不知道"]) is True

    def test_check_worker_error_negative(self):
        assert _check_worker_error("小米公司是一家科技企业", ["don't know", "error"]) is False

    def test_check_worker_error_case_insensitive(self):
        assert _check_worker_error("ERROR: connection failed", ["error"]) is True


class TestAnalyzeResponseQuality:
    """测试响应质量分析"""

    def test_empty_response_is_failure(self):
        state = {"messages": [HumanMessage(content="问题")]}
        success, msg = _analyze_response_quality(state, "vec_kg")
        assert success is False
        assert "No response" in msg

    def test_normal_response_is_success(self):
        state = {
            "messages": [
                HumanMessage(content="问题"),
                AIMessage(content="小米公司专注于智能手机和IoT", name="vec_kg"),
            ]
        }
        success, msg = _analyze_response_quality(state, "vec_kg")
        assert success is True
        assert msg == ""

    def test_failure_indicator_detected(self):
        state = {
            "messages": [
                HumanMessage(content="问题"),
                AIMessage(content="无法找到答案", name="vec_kg"),
            ]
        }
        success, msg = _analyze_response_quality(state, "vec_kg")
        assert success is False

    def test_chat_agent_tolerates_failure_indicators(self):
        """chat 代理的 '不知道' 不应被视为错误"""
        state = {
            "messages": [
                HumanMessage(content="问题"),
                AIMessage(content="根据已有信息，无法找到答案", name="chat"),
            ]
        }
        success, msg = _analyze_response_quality(state, "chat")
        assert success is True


class TestSupervisorRouting:
    """测试 Supervisor 路由决策"""

    @patch("app.supervisor.supervisor_llm")
    def test_routes_to_llm_decision(self, mock_llm):
        """LLM 决策应被尊重（当无循环时）"""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"next": "vec_kg"}
        mock_llm.with_structured_output.return_value = mock_structured

        state = {"messages": [HumanMessage(content="小米有哪些技术？")]}
        result = supervisor(state)
        assert result["next"] == "vec_kg"

    @patch("app.supervisor.supervisor_llm")
    def test_finish_routes_to_end(self, mock_llm):
        """FINISH 应转为 END"""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"next": "FINISH"}
        mock_llm.with_structured_output.return_value = mock_structured

        state = {"messages": [HumanMessage(content="你好")]}
        result = supervisor(state)
        assert result["next"] == END

    @patch("app.supervisor.supervisor_llm")
    def test_prevents_consecutive_same_worker(self, mock_llm):
        """连续调用同一 worker 应被拦截并回退"""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"next": "vec_kg"}
        mock_llm.with_structured_output.return_value = mock_structured

        state = {
            "messages": [
                HumanMessage(content="问题"),
                AIMessage(content="vec_kg 的回答", name="vec_kg"),
            ]
        }
        result = supervisor(state)
        # vec_kg 刚被调用过，应被重定向
        assert result["next"] != "vec_kg"


class TestMembers:
    """测试成员列表配置"""

    def test_members_contains_all_workers(self):
        assert "chat" in members
        assert "coder" in members
        assert "sqler" in members
        assert "graph_kg" in members
        assert "vec_kg" in members

    def test_members_length(self):
        assert len(members) == 5
