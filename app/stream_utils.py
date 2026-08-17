"""
流式输出工具函数

提供统一的 graph.stream 消息遍历和 SSE 事件生成逻辑，
消除 main.py / main_fast.py / hybrid_rag_supervisor.py / api/service.py 的重复代码。
"""

from typing import Generator, Optional

from langchain_core.messages import HumanMessage, AIMessage


def iter_stream_messages(graph, query: str, config: Optional[dict] = None):
    """
    遍历 graph.stream 产出的消息，返回 (name, content) 元组。

    Args:
        graph: 编译好的 LangGraph CompiledGraph
        query: 用户输入的查询文本
        config: LangGraph 调用配置（含 thread_id 时启用多轮对话，
                上一轮的历史消息不重复产出）

    Yields:
        (node_name: str, content: str)
    """
    # 多轮对话：先取 checkpoint 中的存量消息数作为基线，只产出本轮新增消息
    messages_seen = 0
    if config is not None:
        snapshot = graph.get_state(config)
        messages_seen = len(snapshot.values.get("messages", [])) if snapshot.values else 0

    for chunk in graph.stream({"messages": query}, config, stream_mode="values"):
        messages = chunk.get("messages", [])
        if not messages:
            continue

        while messages_seen < len(messages):
            msg = messages[messages_seen]
            name = getattr(msg, "name", None)
            if not name:
                if isinstance(msg, HumanMessage):
                    name = "user"
                elif isinstance(msg, AIMessage):
                    name = "assistant"
                else:
                    name = "system"

            yield name, msg.content
            messages_seen += 1


def print_stream(graph, query: str, config: Optional[dict] = None):
    """
    将 graph.stream 结果打印到控制台（供 CLI 入口使用）。

    Args:
        graph: 编译好的 LangGraph CompiledGraph
        query: 用户查询
        config: LangGraph 调用配置（可选）
    """
    for name, content in iter_stream_messages(graph, query, config):
        print(f"\n[{name}]: {content}", flush=True)


def sse_stream(graph, query: str, config: Optional[dict] = None) -> Generator[dict, None, None]:
    """
    将 graph.stream 结果转换为 SSE 事件字典（供 FastAPI 使用）。

    Yields:
        {"node": str, "content": str, "done": bool}
    """
    try:
        for name, content in iter_stream_messages(graph, query, config):
            yield {"node": name, "content": content, "done": False}

        yield {"node": "system", "content": "", "done": True}

    except Exception as e:
        yield {"node": "error", "content": f"处理失败: {str(e)}", "done": True}
