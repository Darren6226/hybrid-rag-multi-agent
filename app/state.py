import logging
from typing import Annotated, Sequence, Literal, Tuple, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

class AgentState(MessagesState):
    next: str

class ReActAgentState(TypedDict):
    messages: Annotated[Sequence[AIMessage | HumanMessage | SystemMessage | ToolMessage], add_messages]


# ==================== 滑动窗口兜底 ====================
# 硬截断上限：当摘要压缩未触发或失败时，作为最后一道防线防止 token 溢出。
MAX_HISTORY_MESSAGES = 12


def trim_history(messages: list) -> list:
    """硬截断兜底：超过上限时只保留最近 MAX_HISTORY_MESSAGES 条消息"""
    if len(messages) > MAX_HISTORY_MESSAGES:
        return list(messages[-MAX_HISTORY_MESSAGES:])
    return list(messages)


# ==================== 近期完整 + 远期摘要 ====================
# 当消息总数超过此阈值时触发摘要压缩（≈ 3-4 轮对话后）
SUMMARY_TRIGGER_THRESHOLD = 15
# 每次摘要压缩的旧消息条数（触发后 15 → 15-6+1 = 10，降至安全线以下）
SUMMARY_BATCH_SIZE = 6
# 少于此条数时不值得摘要（信息量太少）
SUMMARY_MIN_BATCH = 3


def _find_last_user_message_index(messages: list) -> int:
    """找到最后一条用户消息的索引（当前轮次边界），确保不摘要当前轮次"""
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "name", None) is None:
            return i
    return -1


def summarize_and_trim(messages: list, llm) -> Tuple[list, Optional[List]]:
    """
    近期完整 + 远期摘要：超过阈值时将最早 N 条压缩为摘要 SystemMessage。

    仅压缩当前轮次（最后一条用户消息）之前的历史消息，当前轮次始终完整保留。
    摘要通过 RemoveMessage 从 checkpoint 删除旧消息，并插入摘要 SystemMessage。

    Args:
        messages: 当前 state 中的完整消息列表
        llm: 摘要专用 LLM（建议用 qwen-turbo 等轻量模型）

    Returns:
        (effective_messages, state_updates)
        - effective_messages: 用于 LLM 调用的消息列表（摘要 + 近期完整）
        - state_updates: 需要写入 checkpoint 的消息操作列表
                        （RemoveMessage × N + 摘要 SystemMessage），未触发时为 None
    """
    if len(messages) <= SUMMARY_TRIGGER_THRESHOLD:
        return list(messages), None

    # 定位当前轮次边界，确保不摘要当前轮次的消息
    last_user_idx = _find_last_user_message_index(messages)
    if last_user_idx < 0:
        return list(messages), None

    # 只摘要 last_user_idx 之前的消息，不超过 BATCH_SIZE
    summarizable_count = min(last_user_idx, SUMMARY_BATCH_SIZE)
    if summarizable_count < SUMMARY_MIN_BATCH:
        return list(messages), None

    to_summarize = messages[:summarizable_count]
    recent = messages[summarizable_count:]

    # 构建摘要输入文本
    conversation_text = "\n".join(
        f"[{getattr(m, 'name', 'user') or 'user'}] {m.content}"
        for m in to_summarize
    )

    summary_prompt = (
        "请将以下多轮对话历史压缩为简洁摘要（200字以内）。\n"
        "保留要点：用户的核心问题和意图、各 Agent 返回的关键结论与数据。\n"
        "丢弃：中间过程细节（SQL 语句、图谱查询过程、向量检索中间结果）、重复描述。\n\n"
        f"对话历史：\n{conversation_text}"
    )

    try:
        summary = llm.invoke(summary_prompt)
        summary_text = summary.content
        logger.info(
            "[Memory] 摘要压缩：%d 条旧消息 → 1 条摘要（%d字），近期 %d 条完整保留",
            summarizable_count, len(summary_text), len(recent)
        )
    except Exception as e:
        logger.warning("[Memory] 摘要生成失败: %s，回退到硬截断", str(e)[:100])
        return list(messages[-MAX_HISTORY_MESSAGES:]), None

    # 有效消息 = 摘要 SystemMessage + 近期完整消息
    summary_msg = SystemMessage(content=f"[对话历史摘要]\n{summary_text}")
    effective = [summary_msg] + list(recent)

    # state 更新 = RemoveMessage(每条旧消息) + 摘要 SystemMessage
    state_updates = []
    for m in to_summarize:
        if m.id:
            state_updates.append(RemoveMessage(id=m.id))
    state_updates.append(summary_msg)

    return effective, state_updates
