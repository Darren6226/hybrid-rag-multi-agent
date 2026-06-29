"""
公共 fixtures — mock LLM、mock embeddings、sample state
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# 确保项目根目录在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ============================================================
# Mock Embeddings（避免真实 API 调用）
# ============================================================

class FakeEmbeddings:
    """模拟 Embeddings，返回固定维度的随机向量"""

    def embed_documents(self, texts):
        return [[0.1 * i] * 10 for i in range(len(texts))]

    def embed_query(self, text):
        return [0.1] * 10


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


# ============================================================
# Mock LLM
# ============================================================

class FakeLLM:
    """模拟 LLM，返回预设响应"""

    def __init__(self, response_content="fake response"):
        self._response = response_content

    def invoke(self, messages, **kwargs):
        msg = MagicMock()
        msg.content = self._response
        msg.name = None
        return msg

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        """返回一个 callable，模拟 supervisor 的结构化输出"""
        outer = self

        class FakeStructuredLLM:
            def invoke(self, messages, **kwargs):
                msg = MagicMock()
                msg.__getitem__ = lambda s, k: "chat" if k == "next" else None
                msg.__contains__ = lambda s, k: k == "next"
                # 直接返回 dict 以便 supervisor 使用
                return {"next": "chat"}

        return FakeStructuredLLM()


@pytest.fixture
def fake_llm():
    return FakeLLM()


# ============================================================
# Sample AgentState
# ============================================================

@pytest.fixture
def sample_state():
    """构造一个包含用户消息的 AgentState"""
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content="小米公司有哪些技术？")],
        "next": "",
    }
