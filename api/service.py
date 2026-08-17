"""
服务层 — 封装 LangGraph 编排逻辑

将 Graph 构建与业务调用分离，供 FastAPI 路由层调用。
"""

import logging
import uuid
from typing import Generator, Optional

from app.rag import init_rag
from app.graph_builder import build_graph
from app.supervisor import reset_supervisor
from app.stream_utils import sse_stream
from app.config import MILVUS_URI, NEO4J_URL, DATABASE_URI

logger = logging.getLogger(__name__)

# 全局编译好的 Graph（惰性构建）
_compiled_graph = None


def get_compiled_graph():
    """获取编译好的 Graph（懒加载）"""
    global _compiled_graph
    if _compiled_graph is None:
        init_rag()  # 确保 RAG 系统已初始化
        _compiled_graph = build_graph(use_checkpointer=True)  # MemorySaver 支持多轮对话
        logger.info("LangGraph 编排图已构建")
    return _compiled_graph


def chat_stream(query: str, session_id: Optional[str] = None) -> Generator[dict, None, None]:
    """
    流式处理用户查询，逐节点产出事件。

    Args:
        query: 用户问题
        session_id: 会话 ID。传入相同值时，LangGraph checkpointer 按 thread_id
                    恢复历史消息，实现多轮对话；不传时生成随机 ID（单轮无状态）

    Yields:
        {"node": str, "content": str, "done": bool}
    """
    graph = get_compiled_graph()
    reset_supervisor()  # 重置路由状态（worker_attempts/total_turns），历史消息由 checkpointer 保留

    # 路由状态按轮重置，对话历史按 thread_id 持久化 —— 两层状态职责分离
    thread_id = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logger.info("开始处理查询: %s (thread_id=%s)", query, thread_id)
    yield from sse_stream(graph, query, config)


def check_health() -> dict:
    """检查各依赖服务的连通性"""
    health = {"status": "ok", "milvus": "unknown", "neo4j": "unknown", "mysql": "unknown"}

    # Milvus
    try:
        from pymilvus import connections, utility
        host, port = _parse_milvus_uri(MILVUS_URI)
        connections.connect("default", host=host, port=port, timeout=5)
        utility.list_collections()
        connections.disconnect("default")
        health["milvus"] = "ok"
    except Exception as e:
        health["milvus"] = f"error: {e}"
        health["status"] = "degraded"

    # Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URL, auth=("neo4j", "password"))
        with driver.session() as session:
            session.run("RETURN 1").single()
        driver.close()
        health["neo4j"] = "ok"
    except Exception as e:
        health["neo4j"] = f"error: {e}"
        health["status"] = "degraded"

    # MySQL
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(DATABASE_URI, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        health["mysql"] = "ok"
    except Exception as e:
        health["mysql"] = f"error: {e}"
        health["status"] = "degraded"

    return health


def _parse_milvus_uri(uri: str) -> tuple[str, int]:
    """从 Milvus URI 解析 host 和 port"""
    from urllib.parse import urlparse
    parsed = urlparse(uri)
    return parsed.hostname or "localhost", parsed.port or 19530
