"""
Pydantic 请求/响应模型
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 ID（可选，用于多轮对话追踪）")


class ChatEvent(BaseModel):
    """SSE 流式事件"""
    node: str = Field(..., description="产生事件的节点名称（supervisor/vec_kg/chat 等）")
    content: str = Field("", description="消息内容")
    done: bool = Field(False, description="是否为最终事件")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    milvus: str = "unknown"
    neo4j: str = "unknown"
    mysql: str = "unknown"
