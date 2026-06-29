"""
流式输出工具函数

提供统一的 graph.stream 消息遍历和 SSE 事件生成逻辑，
消除 main.py / main_fast.py / hybrid_rag_supervisor.py / api/service.py 的重复代码。
"""

from typing import Generator, Optional

from langchain_core.messages import HumanMessage, AIMessage


def iter_stream_messages(graph, query: str):
    """
    遍历 graph.stream 产出的消息，返回 (name, content) 元组。

    Args:
        graph: 编译好的 LangGraph CompiledGraph
        query: 用户输入的查询文本

    Yields:
        (node_name: str, content: str)
    """
    messages_seen = 0
    for chunk in graph.stream({"messages": query}, stream_mode="values"):
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


def print_stream(graph, query: str):
    """
    将 graph.stream 结果打印到控制台（供 CLI 入口使用）。

    Args:
        graph: 编译好的 LangGraph CompiledGraph
        query: 用户查询
    """
    for name, content in iter_stream_messages(graph, query):
        print(f"\n[{name}]: {content}", flush=True)


def sse_stream(graph, query: str) -> Generator[dict, None, None]:
    """
    将 graph.stream 结果转换为 SSE 事件字典（供 FastAPI 使用）。

    Yields:
        {"node": str, "content": str, "done": bool}
    """
    try:
        for name, content in iter_stream_messages(graph, query):
            yield {"node": name, "content": content, "done": False}

        yield {"node": "system", "content": "", "done": True}

    except Exception as e:
        yield {"node": "error", "content": f"处理失败: {str(e)}", "done": True}
