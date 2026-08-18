"""
测试多轮对话记忆的摘要压缩逻辑（summarize_and_trim）

使用 mock LLM，不依赖真实 API。
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage

from app.state import (
    summarize_and_trim,
    trim_history,
    SUMMARY_TRIGGER_THRESHOLD,
    SUMMARY_BATCH_SIZE,
    SUMMARY_MIN_BATCH,
    MAX_HISTORY_MESSAGES,
    _find_last_user_message_index,
)


def _make_user(content: str, idx: str) -> HumanMessage:
    """用户消息：name=None，用于 _find_last_user_message_index 识别轮次边界"""
    return HumanMessage(content=content, id=idx)


def _make_worker(content: str, name: str, idx: str) -> HumanMessage:
    """Worker 消息：name 设为 agent 名称"""
    return HumanMessage(content=content, name=name, id=idx)


def _make_mock_llm(response_text: str = "这是摘要内容"):
    """构造 mock LLM，invoke 返回带 .content 的 mock 响应"""
    resp = MagicMock()
    resp.content = response_text
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=resp)
    return llm


def _make_mock_llm_error():
    """构造 mock LLM，invoke 抛异常"""
    llm = MagicMock()
    llm.invoke = MagicMock(side_effect=RuntimeError("API timeout"))
    return llm


def _build_conversation(turns: int) -> list:
    """生成 N 轮对话消息，每轮 = [user, vec_kg, chat]"""
    messages = []
    for t in range(turns):
        messages.append(_make_user(f"用户问题{t}", f"u{t}"))
        messages.append(_make_worker(f"vec_kg回答{t}", "vec_kg", f"v{t}"))
        messages.append(_make_worker(f"chat回答{t}", "chat", f"c{t}"))
    return messages


class TestFindLastUserMessageIndex:
    """测试轮次边界定位"""

    def test_single_user_message(self):
        msgs = [_make_user("问题", "u0")]
        assert _find_last_user_message_index(msgs) == 0

    def test_user_at_end(self):
        msgs = [
            _make_worker("旧回答", "vec_kg", "v0"),
            _make_user("新问题", "u1"),
        ]
        assert _find_last_user_message_index(msgs) == 1

    def test_multiple_turns(self):
        msgs = _build_conversation(5)  # 15 条，最后一条 user 在 idx=12
        assert _find_last_user_message_index(msgs) == 12

    def test_no_user_message(self):
        msgs = [_make_worker("回答", "vec_kg", "v0")]
        assert _find_last_user_message_index(msgs) == -1


class TestSummarizeAndTrim:
    """测试摘要压缩主逻辑"""

    def test_below_threshold_no_summarization(self):
        """消息数 ≤ 阈值时不触发摘要"""
        msgs = _build_conversation(4)  # 12 条 < 15
        effective, state_updates = summarize_and_trim(msgs, _make_mock_llm())
        assert effective == msgs
        assert state_updates is None

    def test_at_threshold_no_summarization(self):
        """消息数恰好等于阈值时不触发摘要"""
        msgs = _build_conversation(5)  # 15 条 == 15
        effective, state_updates = summarize_and_trim(msgs, _make_mock_llm())
        assert effective == msgs
        assert state_updates is None

    def test_above_threshold_triggers_summarization(self):
        """消息数 > 阈值时触发摘要"""
        msgs = _build_conversation(6)  # 18 条 > 15
        effective, state_updates = summarize_and_trim(msgs, _make_mock_llm("摘要文本"))

        assert state_updates is not None
        # 有效消息 = 1 条摘要 + 剩余消息
        assert len(effective) == 1 + (len(msgs) - SUMMARY_BATCH_SIZE)
        # 第一条是摘要 SystemMessage
        assert isinstance(effective[0], SystemMessage)
        assert "摘要文本" in effective[0].content

    def test_state_updates_contain_remove_messages(self):
        """state_updates 包含 RemoveMessage（删除旧消息）"""
        msgs = _build_conversation(6)  # 18 条
        _, state_updates = summarize_and_trim(msgs, _make_mock_llm())

        remove_msgs = [m for m in state_updates if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == SUMMARY_BATCH_SIZE
        # 验证删除的是最早的消息 ID
        removed_ids = {m.id for m in remove_msgs}
        expected_ids = {msgs[i].id for i in range(SUMMARY_BATCH_SIZE)}
        assert removed_ids == expected_ids

    def test_state_updates_contain_summary_systemmessage(self):
        """state_updates 最后一条是摘要 SystemMessage"""
        msgs = _build_conversation(6)
        _, state_updates = summarize_and_trim(msgs, _make_mock_llm("摘要结果"))

        assert isinstance(state_updates[-1], SystemMessage)
        assert "摘要结果" in state_updates[-1].content

    def test_recent_messages_preserved_intact(self):
        """近期消息（含当前轮次）完整保留，内容不变"""
        msgs = _build_conversation(6)  # 18 条
        effective, _ = summarize_and_trim(msgs, _make_mock_llm())

        # effective[0] 是摘要，effective[1:] 是近期消息
        recent_effective = effective[1:]
        recent_original = msgs[SUMMARY_BATCH_SIZE:]
        assert len(recent_effective) == len(recent_original)
        for orig, eff in zip(recent_original, recent_effective):
            assert orig.content == eff.content
            assert orig.id == eff.id

    def test_current_turn_not_summarized(self):
        """当前轮次的消息绝不被摘要"""
        # 构造：6 条旧消息 + 12 条当前轮次消息（含最后一条用户消息）
        msgs = []
        for i in range(6):
            msgs.append(_make_worker(f"旧回答{i}", "vec_kg", f"old{i}"))
        msgs.append(_make_user("当前用户问题", "cur_user"))
        for i in range(11):
            msgs.append(_make_worker(f"当前回答{i}", "chat", f"cur{i}"))
        # 共 18 条，last_user_idx = 6

        effective, state_updates = summarize_and_trim(msgs, _make_mock_llm())

        # 摘要了 6 条旧消息（min(6, 6) = 6 >= 3）
        assert state_updates is not None
        remove_msgs = [m for m in state_updates if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == 6

        # 当前轮次消息（idx 6-17）全部保留
        recent_ids = {m.id for m in effective[1:]}
        for i in range(6, 18):
            assert f"cur_user" in recent_ids or f"cur{i-7}" in recent_ids

    def test_too_few_summarizable_messages(self):
        """可摘要消息不足 SUMMARY_MIN_BATCH 时不触发"""
        # 16 条消息，但 last_user_idx = 2（只有 2 条可摘要）
        msgs = []
        msgs.append(_make_worker("旧1", "vec_kg", "old0"))
        msgs.append(_make_worker("旧2", "vec_kg", "old1"))
        msgs.append(_make_user("用户问题", "u0"))
        for i in range(13):
            msgs.append(_make_worker(f"回答{i}", "chat", f"c{i}"))
        # 共 16 条 > 15，但 summarizable_count = min(2, 6) = 2 < 3

        effective, state_updates = summarize_and_trim(msgs, _make_mock_llm())
        assert state_updates is None
        assert effective == msgs

    def test_llm_failure_falls_back_to_hard_truncation(self):
        """LLM 调用失败时回退到硬截断"""
        msgs = _build_conversation(6)  # 18 条
        effective, state_updates = summarize_and_trim(msgs, _make_mock_llm_error())

        assert state_updates is None
        assert len(effective) == MAX_HISTORY_MESSAGES
        # 硬截断保留的是最后 MAX_HISTORY_MESSAGES 条
        assert effective[0].content == msgs[-MAX_HISTORY_MESSAGES].content

    def test_no_user_message_no_summarization(self):
        """没有用户消息时不触发摘要（无法定位轮次边界）"""
        msgs = []
        for i in range(20):
            msgs.append(_make_worker(f"回答{i}", "vec_kg", f"v{i}"))

        effective, state_updates = summarize_and_trim(msgs, _make_mock_llm())
        assert state_updates is None


class TestTrimHistoryFallback:
    """测试 trim_history 硬截断兜底"""

    def test_short_history_unchanged(self):
        msgs = _build_conversation(3)  # 9 条 < 12
        result = trim_history(msgs)
        assert result == msgs

    def test_long_history_truncated(self):
        msgs = _build_conversation(6)  # 18 条 > 12
        result = trim_history(msgs)
        assert len(result) == MAX_HISTORY_MESSAGES
        assert result[0].content == msgs[-MAX_HISTORY_MESSAGES].content


# ============================================================
# chat 节点记忆策略一致性（问题①修复）
# ============================================================

class TestChatNodeMemoryOverride:
    """验证 chat_node 与 Supervisor 采用统一摘要策略，不再用旧的 trim_history 硬截断"""

    def _run_chat(self, state_messages, llm, summary_llm):
        """真实运行 chat_node，mock 其依赖的 LLM"""
        from app.nodes import chat_node
        with patch("app.nodes.llm", llm), patch("app.nodes.summary_llm", summary_llm):
            return chat_node(state_messages)

    def _make_state(self, messages):
        return {"messages": messages}

    def test_chat_receives_summary_instead_of_hard_truncation(self):
        """
        消息超阈值时，chat 节点接收到的应是摘要 SystemMessage + 近期完整，
        而不是旧的 trim_history(12) 硬截断丢开头。
        """
        # 18 条消息：6 条旧 worker + 1 条用户消息 + 11 条近期 worker
        msgs = []
        for i in range(6):
            msgs.append(_make_worker(f"旧数据{i}", "sqler" if i % 2 == 0 else "vec_kg", f"old{i}"))
        msgs.append(_make_user("当前问题", "cur_user"))
        for i in range(11):
            msgs.append(_make_worker(f"近期数据{i}", "vec_kg", f"recent{i}"))

        # mock chat 推理 LLM：捕获输入
        chat_resp = MagicMock()
        chat_resp.content = "chat 回答"
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=chat_resp)

        # mock 摘要 LLM
        summary_resp = MagicMock()
        summary_resp.content = "远期摘要文本"
        mock_summary = MagicMock()
        mock_summary.invoke = MagicMock(return_value=summary_resp)

        state = self._make_state(msgs)
        result = self._run_chat(state, mock_llm, mock_summary)

        # 1. chat 推理 LLM 被调用，输入含摘要 SystemMessage
        sent_messages = mock_llm.invoke.call_args[0][0]
        system_prompt = sent_messages[0]  # 第一个是 chat 的 system_prompt SystemMessage
        post_system = sent_messages[1:]   # 之后是 effective_messages

        # 第一位是摘要 SystemMessage 而非旧消息
        assert isinstance(post_system[0], SystemMessage)
        assert "远期摘要文本" in post_system[0].content

        # 2. 旧消息(前6条)不在 chat 输入中（未被 trim_history 丢弃而是被摘要替换）
        all_contents = [m.content for m in post_system]
        assert not any("旧数据" in c for c in all_contents)

        # 3. 近期消息完整保留（含当前用户问题）
        assert "当前问题" in all_contents
        assert "近期数据10" in all_contents

        # 4. 摘要 LLM 被调用一次
        assert mock_summary.invoke.call_count == 1

        # 5. 返回消息 = chat 回复 + RemoveMessage + 摘要 SystemMessage
        returned = result["messages"]
        assert returned[0].content == "chat 回答"
        remove_msgs = [m for m in returned if isinstance(m, RemoveMessage)]
        assert len(remove_msgs) == 6

    def test_chat_below_threshold_uses_messages_unchanged(self):
        """消息未超阈值时，chat 直接使用原始消息，不触发摘要 LLM"""
        msgs = _build_conversation(4)  # 12 条 ≤ 15

        chat_resp = MagicMock()
        chat_resp.content = "chat 回答"
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=chat_resp)

        mock_summary = MagicMock()
        mock_summary.invoke = MagicMock(return_value=MagicMock())

        state = self._make_state(msgs)
        result = self._run_chat(state, mock_llm, mock_summary)

        # 摘要 LLM 未被调用
        mock_summary.invoke.assert_not_called()
        # chat 输入包含全部原始消息
        sent_messages = mock_llm.invoke.call_args[0][0]
        assert len(sent_messages) == 1 + 12  # system + 12条
        # 返回不含 RemoveMessage
        assert not any(isinstance(m, RemoveMessage) for m in result["messages"])
