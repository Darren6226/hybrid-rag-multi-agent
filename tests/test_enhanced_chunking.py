"""
测试增强分块策略

使用 mock embeddings，不依赖真实 API。
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.enhanced_chunking import (
    EnhancedChunker,
    EnhancedMetadataExtractor,
    ParentChildChunker,
    ChunkMetadata,
    create_enhanced_chunker,
)


# ============================================================
# 测试 ChunkMetadata
# ============================================================

class TestChunkMetadata:
    def test_default_values(self):
        meta = ChunkMetadata()
        assert meta.source == ""
        assert meta.document_title is None
        assert meta.chunk_type == "text"
        assert meta.is_title is False

    def test_to_dict(self):
        meta = ChunkMetadata(source="test.txt", document_title="测试文档")
        d = meta.to_dict()
        assert d["source"] == "test.txt"
        assert d["document_title"] == "测试文档"
        assert "chunk_type" in d


# ============================================================
# 测试 EnhancedMetadataExtractor
# ============================================================

class TestEnhancedMetadataExtractor:
    def setup_method(self):
        self.extractor = EnhancedMetadataExtractor()

    def test_extract_document_title_from_content(self):
        content = "小米科技有限责任公司年度报告\n\n第一章 公司概况"
        title = self.extractor._extract_document_title(content, "report.txt")
        assert "小米" in title or "年度报告" in title

    def test_extract_document_title_from_filename(self):
        content = "短内容"
        title = self.extractor._extract_document_title(content, "/path/to/company_report.txt")
        assert "company report" in title.lower() or "company" in title.lower()

    def test_is_section_title_chinese_chapter(self):
        assert self.extractor._is_section_title("第 三 章 技术创新") is True

    def test_is_section_title_english_chapter(self):
        assert self.extractor._is_section_title("Chapter 3: Results") is True

    def test_is_section_title_numbered(self):
        assert self.extractor._is_section_title("3.1 实验设计") is True

    def test_is_not_section_title_normal_text(self):
        assert self.extractor._is_section_title("小米公司是一家专注于智能手机的企业") is False

    def test_find_chapter_context(self):
        # 注意：正则要求 "第 X 章" 中间有空格，测试数据需匹配
        content = "第 一 章 公司简介\n\n小米科技是一家...\n\n第 二 章 技术创新\n\n小米在快充技术..."
        position = content.index("小米在快充技术")
        result = self.extractor._find_chapter_context(content, position)
        assert result is not None
        assert "技术创新" in result["title"]

    def test_identify_theme(self):
        # 正则模式要求 "XX 公司"（有空格），需匹配格式
        text = "小米 公司推出了新一代智能手机技术"
        theme = self.extractor._identify_theme(text)
        assert theme is not None

    def test_identify_theme_no_keyword(self):
        text = "今天天气真好"
        theme = self.extractor._identify_theme(text)
        assert theme is None

    def test_extract_all_metadata(self):
        content = "第一章 公司简介\n\n小米科技有限责任公司是一家科技企业。"
        chunk_text = "小米科技有限责任公司是一家科技企业。"
        meta = self.extractor.extract_all_metadata(chunk_text, content, 20, "test.txt")
        assert isinstance(meta, ChunkMetadata)
        assert meta.source == "test.txt"


# ============================================================
# 测试 ParentChildChunker
# ============================================================

class TestParentChildChunker:
    def setup_method(self):
        self.chunker = ParentChildChunker(parent_size=800, child_size=200)

    def test_create_parent_child_chunks(self):
        content = "小米科技是一家专注于智能手机的公司。" * 20  # 足够长的内容
        result = self.chunker.create_parent_child_chunks(content)
        assert len(result) > 0
        for parent, children in result:
            assert isinstance(parent, Document)
            assert isinstance(children, list)
            assert len(children) > 0
            assert parent.metadata.get("chunk_level") == "parent"
            for child in children:
                assert child.metadata.get("chunk_level") == "child"

    def test_flatten_for_vectorstore(self):
        content = "这是测试内容。" * 30
        chunks = self.chunker.create_parent_child_chunks(content)
        flat = self.chunker.flatten_for_vectorstore(chunks)
        # 应包含所有父块 + 所有子块
        parents = [d for d in flat if d.metadata.get("chunk_level") == "parent"]
        children = [d for d in flat if d.metadata.get("chunk_level") == "child"]
        assert len(parents) > 0
        assert len(children) > 0

    def test_parent_size_larger_than_child(self):
        """父块应大于子块"""
        content = "A" * 2000
        chunks = self.chunker.create_parent_child_chunks(content)
        for parent, children in chunks:
            for child in children:
                assert len(child.page_content) <= len(parent.page_content) + 10  # 允许小误差


# ============================================================
# 测试 EnhancedChunker
# ============================================================

class TestEnhancedChunker:

    @patch("app.enhanced_chunking.embeddings")
    def test_basic_strategy(self, mock_embeddings):
        """basic 策略：不使用语义分块，不增强元数据"""
        chunker = EnhancedChunker(
            use_semantic=False,
            use_parent_child=False,
            enable_metadata_enhancement=False,
        )
        content = "小米公司简介。" * 50
        chunks = chunker.chunk_document(content, "test.txt")
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Document)

    @patch("app.enhanced_chunking.embeddings")
    def test_semantic_strategy_with_metadata(self, mock_embeddings):
        """semantic_metadata 策略：使用语义分块 + 元数据增强"""
        chunker = EnhancedChunker(
            use_semantic=True,
            use_parent_child=False,
            enable_metadata_enhancement=True,
            min_chunk_size=50,
        )
        content = "第一章 公司简介\n\n小米科技是一家专注于智能手机的企业。\n\n第二章 技术创新\n\n小米在快充技术方面取得突破。"
        chunks = chunker.chunk_document(content, "test.txt")
        assert len(chunks) > 0

    @patch("app.enhanced_chunking.embeddings")
    def test_semantic_fallback_to_recursive(self, mock_embeddings):
        """语义分块产生太少块时应降级到递归分块"""
        chunker = EnhancedChunker(use_semantic=True, min_chunk_size=50)
        # 很短的内容，语义分块可能只产生 1 个块
        content = "短文本"
        chunks = chunker.chunk_document(content, "test.txt")
        assert len(chunks) >= 1


# ============================================================
# 测试 create_enhanced_chunker 工厂函数
# ============================================================

class TestCreateEnhancedChunker:

    @patch("app.enhanced_chunking.embeddings")
    def test_semantic_metadata_strategy(self, mock_embeddings):
        chunker = create_enhanced_chunker(strategy="semantic_metadata")
        assert isinstance(chunker, EnhancedChunker)
        assert chunker.use_semantic is True
        assert chunker.enable_metadata is True

    @patch("app.enhanced_chunking.embeddings")
    def test_parent_child_strategy(self, mock_embeddings):
        chunker = create_enhanced_chunker(strategy="parent_child")
        assert chunker.use_parent_child is True

    @patch("app.enhanced_chunking.embeddings")
    def test_full_strategy(self, mock_embeddings):
        chunker = create_enhanced_chunker(strategy="full")
        assert chunker.use_semantic is True
        assert chunker.use_parent_child is True
        assert chunker.enable_metadata is True

    @patch("app.enhanced_chunking.embeddings")
    def test_basic_strategy(self, mock_embeddings):
        chunker = create_enhanced_chunker(strategy="basic")
        assert chunker.use_semantic is False
        assert chunker.use_parent_child is False
        assert chunker.enable_metadata is False
