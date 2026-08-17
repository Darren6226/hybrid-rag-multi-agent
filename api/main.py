"""
FastAPI 应用入口

接口：
  POST /chat       — SSE 流式对话
  GET  /health     — 依赖服务健康检查
  GET  /docs       — Swagger UI（FastAPI 内置）

启动方式：
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest, ChatEvent, HealthResponse
from api.service import chat_stream, check_health

logger = logging.getLogger(__name__)

app = FastAPI(
    title="混合 RAG 多代理智能体系统",
    description="基于 LangGraph Supervisor-Worker 架构的混合知识库检索系统",
    version="2.0.0",
)

# CORS（允许前端跨域调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["运维"])
async def health():
    """健康检查：返回各依赖服务的连通状态"""
    return check_health()


@app.post("/chat", tags=["对话"])
async def chat(request: ChatRequest):
    """
    SSE 流式对话接口

    请求体：
        {"query": "小米公司有哪些技术？", "session_id": "可选"}

    响应：text/event-stream，每个事件格式：
        data: {"node": "vec_kg", "content": "...", "done": false}
    """

    def event_generator():
        for event in chat_stream(request.query, request.session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
