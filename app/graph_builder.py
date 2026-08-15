"""
共享的 LangGraph 图构建模块

消除 main.py / main_fast.py / api/service.py / hybrid_rag_supervisor.py
中重复的 StateGraph 构建逻辑。
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END

from app.state import AgentState
from app.supervisor import supervisor, members
from app.nodes import sqler_node, coder_node, graph_kg_node, vec_kg_node, chat_node

logger = logging.getLogger(__name__)


def build_graph(skip_graph_kg: bool = False):
    """
    构建并编译多代理 StateGraph。

    Args:
        skip_graph_kg: 若为 True，将 graph_kg 路由重定向到 vec_kg
                       （用于快速启动模式，跳过 Neo4j 图索引构建）

    Returns:
        编译好的 CompiledGraph
    """
    builder = StateGraph(AgentState)

    # 注册所有节点
    builder.add_node("supervisor", supervisor)
    builder.add_node("chat", chat_node)
    builder.add_node("coder", coder_node)
    builder.add_node("sqler", sqler_node)
    builder.add_node("vec_kg", vec_kg_node)

    if skip_graph_kg:
        # 快速模式：不注册 graph_kg 节点，路由重定向
        builder.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "chat": "chat",
                "coder": "coder",
                "sqler": "sqler",
                "graph_kg": "vec_kg",  # 重定向
                "vec_kg": "vec_kg",
                END: END,
            },
        )
    else:
        builder.add_node("graph_kg", graph_kg_node)
        builder.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "chat": "chat",
                "coder": "coder",
                "sqler": "sqler",
                "graph_kg": "graph_kg",
                "vec_kg": "vec_kg",
                END: END,
            },
        )

    # 每个子代理完成后向 supervisor 汇报
    for member in members:
        if skip_graph_kg and member == "graph_kg":
            continue
        builder.add_edge(member, "supervisor")

    builder.add_edge(START, "supervisor")

    graph = builder.compile()
    logger.info("LangGraph 编排图已构建 (skip_graph_kg=%s)", skip_graph_kg)
    return graph
