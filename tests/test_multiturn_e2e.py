"""
多轮对话端到端集成测试

真实组件: LangGraph 图编排 + MemorySaver checkpointer + Supervisor 完整路由逻辑(含三层兜底)
Mock 组件: supervisor LLM 决策(脚本化路由序列) + 各 Worker 节点函数

覆盖场景:
1. 跨轮状态持久化 —— 同 thread_id 两轮对话，消息历史正确累积
2. SSE 增量推送 —— 第二轮只产出本轮新增消息，不重复推送历史
3. called_workers 轮次隔离 —— 上一轮的数据源"无数据"不应触发本轮兜底A
4. thread_id 会话隔离 —— 不同 thread 互不可见
5. reset_supervisor —— 路由状态(轮次计数)按轮重置，不跨轮累积
6. 摘要压缩 —— 6轮对话后触发摘要，旧消息被 RemoveMessage 删除，近期完整保留
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph_builder import build_graph
from app.supervisor import reset_supervisor, policy_manager
from app.stream_utils import iter_stream_messages


# ============================================================
# 测试辅助
# ============================================================

class ScriptedRouter:
    """脚本化 Supervisor LLM：按预设序列返回路由决策"""

    def __init__(self, routes):
        self.routes = list(routes)
        self.calls = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        return {"next": self.routes.pop(0)}


def make_worker(name, content):
    """构造 mock Worker 节点：返回带 name 标记的固定回复"""
    def node(state):
        return {"messages": [HumanMessage(content=content, name=name)]}
    return node


def patch_workers(vec_content="vec_kg 检索结果", graph_content="graph_kg 检索结果",
                  sql_content="sqler 查询结果", chat_content="chat 综合回答"):
    """批量 patch graph_builder 中的全部 Worker 节点"""
    return [
        patch("app.graph_builder.vec_kg_node", make_worker("vec_kg", vec_content)),
        patch("app.graph_builder.graph_kg_node", make_worker("graph_kg", graph_content)),
        patch("app.graph_builder.sqler_node", make_worker("sqler", sql_content)),
        patch("app.graph_builder.coder_node", make_worker("coder", "coder 执行结果")),
        patch("app.graph_builder.chat_node", make_worker("chat", chat_content)),
    ]


def run_with_workers(routes, turns):
    """
    构建带 checkpointer 的图并依次执行多轮对话。

    Args:
        routes: 每轮的脚本化路由序列，如 [["vec_kg","chat","FINISH"], ["sqler","chat","FINISH"]]
        turns: 每轮的 (query, thread_id)

    Returns:
        (graph, router) —— 用于后续断言
    """
    router = ScriptedRouter([r for turn_routes in routes for r in turn_routes])
    patches = patch_workers()
    with patch("app.supervisor.supervisor_llm", router):
        for p in patches:
            p.start()
        try:
            graph = build_graph(use_checkpointer=True)
            for query, thread_id in turns:
                reset_supervisor()  # 模拟 api/service.py 的每轮行为
                config = {"configurable": {"thread_id": thread_id}}
                graph.invoke({"messages": query}, config)
            return graph, router
        finally:
            for p in patches:
                p.stop()


def get_thread_messages(graph, thread_id):
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    return snapshot.values.get("messages", [])


# ============================================================
# 场景 1: 跨轮状态持久化
# ============================================================

class TestMultiTurnPersistence:

    def test_history_accumulates_across_turns(self):
        """同 thread_id 两轮对话后，消息历史包含两轮全部消息且顺序正确"""
        graph, _ = run_with_workers(
            routes=[["vec_kg", "chat", "FINISH"], ["sqler", "chat", "FINISH"]],
            turns=[("小米的快充技术?", "s1"), ("那总销售额呢?", "s1")],
        )
        msgs = get_thread_messages(graph, "s1")
        names = [getattr(m, "name", None) for m in msgs]

        # 6 条消息: q1 + vec_kg + chat + q2 + sqler + chat
        assert len(msgs) == 6
        assert names == [None, "vec_kg", "chat", None, "sqler", "chat"]
        assert msgs[0].content == "小米的快充技术?"
        assert msgs[3].content == "那总销售额呢?"

    def test_supervisor_sees_full_history_in_turn2(self):
        """第二轮 Supervisor 决策时能看到第一轮的完整历史"""
        graph, router = run_with_workers(
            routes=[["vec_kg", "chat", "FINISH"], ["sqler", "chat", "FINISH"]],
            turns=[("小米的快充技术?", "s2"), ("那总销售额呢?", "s2")],
        )
        # 两轮共 6 次 supervisor 调用: 3 + 3
        assert len(router.calls) == 6
        # 第二轮第一次决策(第4次调用)时，LLM 输入应包含前3条历史消息
        turn2_first_call = router.calls[3]
        history = turn2_first_call[1:]  # 去掉 system prompt
        assert len(history) == 4  # q1 + vec_kg + chat + q2


# ============================================================
# 场景 2: SSE 增量推送
# ============================================================

class TestSseIncrementalStream:

    def test_turn2_yields_only_new_messages(self):
        """第二轮流式输出只包含本轮新增消息，不重复推送第一轮历史"""
        router = ScriptedRouter(["vec_kg", "chat", "FINISH", "sqler", "chat", "FINISH"])
        patches = patch_workers()
        with patch("app.supervisor.supervisor_llm", router):
            for p in patches:
                p.start()
            try:
                graph = build_graph(use_checkpointer=True)
                config = {"configurable": {"thread_id": "s3"}}

                # 第一轮
                reset_supervisor()
                graph.invoke({"messages": "小米的快充技术?"}, config)

                # 第二轮: 通过流式接口驱动，收集产出的事件
                reset_supervisor()
                events = [
                    (name, content)
                    for name, content in iter_stream_messages(graph, "那总销售额呢?", config)
                ]

                # 只应产出第二轮的 3 条新消息: user(q2) + sqler + chat
                assert len(events) == 3
                assert events[0][0] == "user"
                assert events[0][1] == "那总销售额呢?"
                assert events[1][0] == "sqler"
                assert events[2][0] == "chat"
            finally:
                for p in patches:
                    p.stop()


# ============================================================
# 场景 3: called_workers 轮次隔离（回归测试）
# ============================================================

class TestTurnBoundaryIsolation:

    def test_turn1_no_data_does_not_block_turn2_dispatch(self):
        """
        回归测试: 第一轮 vec_kg/graph_kg 都返回"无数据"，
        第二轮 LLM 建议调度 sqler 时，兜底A不应被第一轮的历史误触发。
        (旧逻辑 called_workers 会跨轮回溯污染，导致 sqler 被强制替换为 chat)
        """
        router = ScriptedRouter(["vec_kg", "graph_kg", "chat", "FINISH",
                                 "sqler", "chat", "FINISH"])
        patches = [
            patch("app.graph_builder.vec_kg_node",
                  make_worker("vec_kg", "文档中暂无相关信息")),
            patch("app.graph_builder.graph_kg_node",
                  make_worker("graph_kg", "知识图谱中未找到相关信息")),
            patch("app.graph_builder.sqler_node",
                  make_worker("sqler", "小米总销售额 29990 元")),
            patch("app.graph_builder.coder_node",
                  make_worker("coder", "coder 执行结果")),
            patch("app.graph_builder.chat_node",
                  make_worker("chat", "chat 综合回答")),
        ]
        with patch("app.supervisor.supervisor_llm", router):
            for p in patches:
                p.start()
            try:
                graph = build_graph(use_checkpointer=True)
                config = {"configurable": {"thread_id": "s4"}}

                reset_supervisor()
                graph.invoke({"messages": "小米的快充技术?"}, config)

                reset_supervisor()
                graph.invoke({"messages": "那总销售额呢?"}, config)

                msgs = get_thread_messages(graph, "s4")
                names = [getattr(m, "name", None) for m in msgs]

                # 关键断言: 第二轮 sqler 真的被调度了（而非被兜底A替换为 chat）
                assert "sqler" in names
                # 完整顺序: q1, vec_kg, graph_kg, chat, q2, sqler, chat
                assert names == [None, "vec_kg", "graph_kg", "chat",
                                 None, "sqler", "chat"]
            finally:
                for p in patches:
                    p.stop()


# ============================================================
# 场景 4: thread 隔离
# ============================================================

class TestThreadIsolation:

    def test_different_threads_are_isolated(self):
        """不同 thread_id 的会话互不可见"""
        graph, _ = run_with_workers(
            routes=[["vec_kg", "chat", "FINISH"], ["vec_kg", "chat", "FINISH"]],
            turns=[("会话A的问题", "thread-a"), ("会话B的问题", "thread-b")],
        )
        msgs_a = get_thread_messages(graph, "thread-a")
        msgs_b = get_thread_messages(graph, "thread-b")

        assert len(msgs_a) == 3  # q + vec_kg + chat
        assert len(msgs_b) == 3
        assert msgs_a[0].content == "会话A的问题"
        assert msgs_b[0].content == "会话B的问题"


# ============================================================
# 场景 5: 路由状态按轮重置
# ============================================================

class TestSupervisorStateReset:

    def test_total_turns_reset_between_queries(self):
        """reset_supervisor 后轮次计数归零，兜底B不跨轮累积"""
        router = ScriptedRouter(["vec_kg", "chat", "FINISH", "vec_kg", "chat", "FINISH"])
        patches = patch_workers()
        with patch("app.supervisor.supervisor_llm", router):
            for p in patches:
                p.start()
            try:
                graph = build_graph(use_checkpointer=True)
                config = {"configurable": {"thread_id": "s5"}}

                reset_supervisor()
                graph.invoke({"messages": "第一轮问题"}, config)
                assert policy_manager.state.total_turns == 3  # 第一轮结束: 3 次 supervisor

                reset_supervisor()
                assert policy_manager.state.total_turns == 0  # 已重置

                graph.invoke({"messages": "第二轮问题"}, config)
                assert policy_manager.state.total_turns == 3  # 第二轮重新计数
            finally:
                for p in patches:
                    p.stop()


# ============================================================
# 场景 6: 摘要压缩（近期完整 + 远期摘要）
# ============================================================

class TestSummaryCompression:
    """测试摘要压缩在真实 LangGraph 图中的端到端行为"""

    def test_summary_triggered_after_6_turns(self):
        """
        6轮对话后触发摘要压缩：
        - 每轮 vec_kg→chat→FINISH = 3条消息
        - 5轮 = 15条（==阈值不触发）
        - 第6轮首条消息时 = 16条 > 15 → 触发摘要
        - 摘要最早6条（第1-2轮），RemoveMessage从checkpoint删除
        - 最终: 1摘要 + 4轮×3 = 13条（而非18条）
        """
        routes_per_turn = [["vec_kg", "chat", "FINISH"]] * 6
        all_routes = [r for turn_routes in routes_per_turn for r in turn_routes]
        router = ScriptedRouter(all_routes)

        # Mock summary_llm：记录调用并返回固定摘要
        summary_resp = MagicMock()
        summary_resp.content = "用户问了小米技术和销售额，Agent返回了相关结果"
        summary_invoke_count = [0]

        def mock_summary_invoke(prompt):
            summary_invoke_count[0] += 1
            return summary_resp

        mock_summary_llm = MagicMock()
        mock_summary_llm.invoke = mock_summary_invoke

        patches = patch_workers()
        with patch("app.supervisor.supervisor_llm", router), \
             patch("app.supervisor.summary_llm", mock_summary_llm):
            for p in patches:
                p.start()
            try:
                graph = build_graph(use_checkpointer=True)
                config = {"configurable": {"thread_id": "sum-test"}}

                for t in range(6):
                    reset_supervisor()
                    graph.invoke({"messages": f"第{t}轮问题"}, config)

                msgs = get_thread_messages(graph, "sum-test")
                contents = [m.content for m in msgs]

                # 1. 摘要 LLM 被调用1次（第6轮首次supervisor时触发）
                assert summary_invoke_count[0] == 1

                # 2. checkpoint 包含摘要 SystemMessage
                summary_msgs = [
                    m for m in msgs
                    if isinstance(m, SystemMessage) and "对话历史摘要" in m.content
                ]
                assert len(summary_msgs) == 1
                assert "小米" in summary_msgs[0].content

                # 3. 消息总数 < 18（未压缩应为6轮×3=18条）
                assert len(msgs) < 18

                # 4. 旧消息被删除（第1-2轮用户问题不在checkpoint中）
                assert "第0轮问题" not in contents
                assert "第1轮问题" not in contents

                # 5. 近期消息完整保留（第3-6轮用户问题在checkpoint中）
                for t in range(2, 6):
                    assert f"第{t}轮问题" in contents
            finally:
                for p in patches:
                    p.stop()
