from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

class AgentState(MessagesState):
    next: str

class ReActAgentState(TypedDict):
    messages: Annotated[Sequence[AIMessage | HumanMessage | SystemMessage | ToolMessage], add_messages]


# 滑动窗口大小：保留最近 N 条消息（约 4-6 轮对话），防止多轮历史导致 token 溢出。
# MAX_TURNS=3 限制单轮 worker 调用约 4-5 条消息，窗口 12 足以容纳完整当前轮次。
MAX_HISTORY_MESSAGES = 12


def trim_history(messages: list) -> list:
    """滑动窗口裁剪：超过上限时只保留最近 MAX_HISTORY_MESSAGES 条消息"""
    if len(messages) > MAX_HISTORY_MESSAGES:
        return list(messages[-MAX_HISTORY_MESSAGES:])
    return list(messages)
