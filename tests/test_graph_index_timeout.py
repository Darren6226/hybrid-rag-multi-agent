"""
测试 _build_graph_index 的并行处理逻辑

用 mock 模拟 LLM 调用慢/失败的文档，验证：
1. 并行模式下错误文档不阻塞其他文档
2. 慢文档不阻塞其他文档（核心：解决原来串行卡死的问题）
3. 超出总时间预算时终止剩余任务
4. 正常文档全部处理
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def make_docs_with_behavior(behaviors):
    """构造带行为标记的测试文档，behavior 编码在 content 中实现顺序无关"""
    return [
        Document(page_content=f"doc {b} {i}", metadata={"source": f"test_{i}.txt"})
        for i, b in enumerate(behaviors)
    ]


def fake_graph_conn():
    conn = MagicMock()
    conn.add_graph_documents = MagicMock()
    return conn


class TestGraphIndexParallel:

    @patch("langchain_experimental.graph_transformers.LLMGraphTransformer")
    def test_error_doc_does_not_block_others(self, mock_cls):
        """错误文档不阻塞其他文档"""
        from app.rag import _build_graph_index

        class FakeTransformer:
            def convert_to_graph_documents(self, docs):
                content = docs[0].page_content
                if 'error' in content:
                    raise RuntimeError("LLM API error")
                return [MagicMock()]

        mock_cls.return_value = FakeTransformer()
        conn = fake_graph_conn()
        docs = make_docs_with_behavior(['ok', 'error', 'ok', 'error', 'ok'])

        result = _build_graph_index(conn, docs)

        assert len(result) == 3
        assert conn.add_graph_documents.call_count == 3

    @patch("langchain_experimental.graph_transformers.LLMGraphTransformer")
    def test_slow_doc_does_not_block_others(self, mock_cls):
        """慢文档不阻塞其他文档（核心：解决原来串行卡死的问题）"""
        from app.rag import _build_graph_index

        class FakeTransformer:
            def convert_to_graph_documents(self, docs):
                content = docs[0].page_content
                if 'slow' in content:
                    time.sleep(3)
                if 'error' in content:
                    raise RuntimeError("LLM API error")
                return [MagicMock()]

        mock_cls.return_value = FakeTransformer()
        conn = fake_graph_conn()
        docs = make_docs_with_behavior(['ok', 'slow', 'ok', 'error', 'ok'])

        start = time.time()
        result = _build_graph_index(conn, docs)
        elapsed = time.time() - start

        # 3 个 ok + 1 个 slow 文档都成功返回（共 4 个），error 文档被跳过
        assert len(result) == 4
        assert conn.add_graph_documents.call_count == 4
        # 并行模式：慢文档不阻塞其他文档，总耗时约 3s（slow 的 sleep 时间）
        # 串行模式下总耗时会是 3s + 其他文档处理时间
        assert elapsed < 5

    @patch("langchain_experimental.graph_transformers.LLMGraphTransformer")
    def test_all_normal_docs_processed(self, mock_cls):
        """所有正常文档都被处理"""
        from app.rag import _build_graph_index

        class FakeTransformer:
            def convert_to_graph_documents(self, docs):
                return [MagicMock()]

        mock_cls.return_value = FakeTransformer()
        conn = fake_graph_conn()
        docs = make_docs_with_behavior(['ok', 'ok', 'ok'])

        result = _build_graph_index(conn, docs)

        assert len(result) == 3
        assert conn.add_graph_documents.call_count == 3

    @patch("app.rag._GRAPH_TOTAL_BUDGET", 1)
    @patch("langchain_experimental.graph_transformers.LLMGraphTransformer")
    def test_total_budget_terminates_early(self, mock_cls):
        """超出总时间预算时终止剩余任务"""
        from app.rag import _build_graph_index

        class SlowTransformer:
            def convert_to_graph_documents(self, docs):
                time.sleep(2)
                return [MagicMock()]

        mock_cls.return_value = SlowTransformer()
        conn = fake_graph_conn()
        docs = make_docs_with_behavior(['ok'] * 5)

        start = time.time()
        result = _build_graph_index(conn, docs)
        elapsed = time.time() - start

        # 预算 1s，每个文档 2s，全部未完成就超时
        assert len(result) == 0
        assert elapsed < 3
