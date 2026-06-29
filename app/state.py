from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

class AgentState(MessagesState):
    next: str

class ReActAgentState(TypedDict):
    messages: Annotated[Sequence[AIMessage | HumanMessage | SystemMessage | ToolMessage], add_messages]
