"""
增强的分块策略模块：语义分块 + 父子分块 + 元信息增强

核心功能:
1. 语义分块：基于语义完整性进行智能分割
2. 元信息增强：自动提取文档标题、章节标题、段落编号等元数据
3. 父子分块：支持父块 (大上下文) 和子块 (精细检索) 的双层结构
4. 多文档类型适配：TXT、PDF、Markdown 等

符合 RAG 系统多格式文档处理规范与分块策略要求
"""

import os
import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from app.config import embeddings

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """分块元数据结构"""
    source: str = ""
    document_title: Optional[str] = None
    chapter_title: Optional[str] = None
    chapter_number: Optional[str] = None
    section_title: Optional[str] = None
    paragraph_number: Optional[int] = None
    chunk_type: str = "text"
    theme: Optional[str] = None
    is_title: bool = False
    is_section_start: bool = False
    is_section_end: bool = False
    parent_chunk_id: Optional[str] = None
    has_children: bool = False
    # 自定义扩展字段
    extra_metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "source": self.source,
            "document_title": self.document_title,
            "chapter_title": self.chapter_title,
            "chapter_number": self.chapter_number,
            "section_title": self.section_title,
            "paragraph_number": self.paragraph_number,
            "chunk_type": self.chunk_type,
            "theme": self.theme,
            "is_title": self.is_title,
            "is_section_start": self.is_section_start,
            "is_section_end": self.is_section_end,
            "parent_chunk_id": self.parent_chunk_id,
            "has_children": self.has_children,
            **self.extra_metadata
        }


class EnhancedMetadataExtractor:
    """增强的元数据提取器"""
    
    def __init__(self):
        # 章节标题模式
        self.chapter_patterns = [
            (r'^第 ([一二三四五六七八九十百\d]+) 章 [.：:\s]*(.+)$', 'chinese_chapter'),
            (r'^Chapter\s+(\d+)[.:]\s*(.+)$', 'english_chapter'),
            (r'^(\d+)\s+[\.、]\s*(.+)$', 'numbered_section'),
            (r'^(\d+\.\d+)\s+[\.、]?\s*(.+)$', 'subsection'),
            (r'^第 ([一二三四五六七八九十百\d]+) 节 [.：:\s]*(.+)$', 'chinese_section'),
        ]
        
        # 特殊章节标记
        self.special_sections = ['摘要', '关键词', '引言', '绪论', '方法', '结果', '讨论', 
                                '结论', '参考文献', '附录', '致谢']
    
    def extract_all_metadata(self, chunk_text: str, full_content: str, 
                            chunk_position: int, file_path: str) -> ChunkMetadata:
        """提取完整的元数据"""
        metadata = ChunkMetadata(source=file_path)
        
        # 1. 提取文档标题
        metadata.document_title = self._extract_document_title(full_content, file_path)
        
        # 2. 查找章节上下文
        chapter_info = self._find_chapter_context(full_content, chunk_position)
        if chapter_info:
            metadata.chapter_title = chapter_info['title']
            metadata.chapter_number = chapter_info['number']
        
        # 3. 检查是否为章节标题
        if self._is_section_title(chunk_text):
            metadata.is_title = True
            metadata.is_section_start = True
            section_info = self._parse_section_title(chunk_text)
            if section_info:
                metadata.section_title = section_info['title']
        
        # 4. 提取段落编号
        metadata.paragraph_number = self._estimate_paragraph_number(full_content, chunk_position)
        
        # 5. 识别主题
        metadata.theme = self._identify_theme(chunk_text)
        
        return metadata
    
    def _extract_document_title(self, content: str, file_path: str) -> Optional[str]:
        """从内容或文件名提取文档标题"""
        # 尝试从内容前几行提取
        lines = content.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                # 排除明显的章节标题
                if not any(re.match(p[0], line) for p in self.chapter_patterns):
                    return line
        
        # 从文件名推断
        basename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(basename)[0]
        return name_without_ext.replace('_', ' ').replace('-', ' ')
    
    def _find_chapter_context(self, content: str, position: int) -> Optional[Dict]:
        """在指定位置附近查找章节信息"""
        preceding = content[max(0, position-3000):position]
        lines = preceding.split('\n')
        
        # 从后向前查找最近的章节标题
        for i in range(len(lines)-1, max(-1, len(lines)-50), -1):
            line = lines[i].strip()
            if len(line) < 5 or len(line) > 100:
                continue
                
            for pattern, chapter_type in self.chapter_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    return {
                        'title': line,
                        'number': match.group(1),
                        'type': chapter_type
                    }
            
            # 检查特殊章节
            for special in self.special_sections:
                if line == special or line.startswith(special + ' '):
                    return {'title': line, 'number': special, 'type': 'special'}
        
        return None
    
    def _is_section_title(self, text: str) -> bool:
        """判断是否为章节标题"""
        text = text.strip()
        if len(text) > 100 or len(text) < 3:
            return False
        
        # 检查是否匹配章节模式
        for pattern, _ in self.chapter_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # 检查是否为特殊章节
        if text in self.special_sections:
            return True
        
        return False
    
    def _parse_section_title(self, text: str) -> Optional[Dict]:
        """解析章节标题"""
        text = text.strip()
        for pattern, chapter_type in self.chapter_patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                return {
                    'title': text,
                    'number': match.group(1),
                    'type': chapter_type
                }
        return None
    
    def _estimate_paragraph_number(self, content: str, position: int) -> Optional[int]:
        """估算段落编号"""
        preceding = content[:position]
        # 简单估算：计算前面的空行数
        paragraph_count = preceding.count('\n\n') + 1
        return paragraph_count if paragraph_count > 0 else None
    
    def _identify_theme(self, text: str) -> Optional[str]:
        """识别文本主题关键词"""
        # 简单实现：提取高频名词
        # TODO: 可以使用 LLM 或关键词提取算法优化
        keywords = []
        
        # 查找可能的公司名、技术术语等
        patterns = [
            r'([A-Za-z\u4e00-\u9fa5]+) 公司',
            r'([A-Za-z\u4e00-\u9fa5]+) 技术',
            r'([A-Za-z\u4e00-\u9fa5]+) 系统',
            r'([A-Za-z\u4e00-\u9fa5]+) 方法',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)
        
        return keywords[0] if keywords else None


class ParentChildChunker:
    """父子分块器：搜子块（精确匹配）→ 返回父块（完整上下文）"""

    def __init__(self, parent_size: int = 2000, child_size: int = 400,
                 parent_overlap: int = 200, child_overlap: int = 50):
        """
        初始化父子分块器
        
        Args:
            parent_size: 父块大小（字符）
            child_size: 子块大小（字符）
            parent_overlap: 父块重叠
            child_overlap: 子块重叠
        """
        self.parent_size = parent_size
        self.child_size = child_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", "，", " ", ""]
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=["。", "！", "？", ".", "!", "?", "；", "，", " ", ""]
        )
    
    def create_parent_child_chunks(self, content: str, base_metadata: Dict = None) -> List[Tuple[Document, List[Document]]]:
        """
        创建父子分块结构
        
        Returns:
            List[Tuple[父块文档，子块文档列表]]
        """
        parent_chunks = self.parent_splitter.split_text(content)
        
        result = []
        for parent_idx, parent_text in enumerate(parent_chunks):
            # 为每个父块生成子块
            child_chunks = self.child_splitter.split_text(parent_text)
            
            # 创建父块文档
            parent_meta = {
                **(base_metadata or {}),
                "chunk_level": "parent",
                "chunk_index": parent_idx,
                "has_children": True,
                "num_children": len(child_chunks)
            }
            parent_doc = Document(page_content=parent_text, metadata=parent_meta)
            
            # 创建子块文档
            child_docs = []
            for child_idx, child_text in enumerate(child_chunks):
                child_meta = {
                    **(base_metadata or {}),
                    "chunk_level": "child",
                    "parent_index": parent_idx,
                    "child_index": child_idx,
                    "parent_chunk_id": f"p{parent_idx}"
                }
                child_doc = Document(page_content=child_text, metadata=child_meta)
                child_docs.append(child_doc)
            
            result.append((parent_doc, child_docs))
        
        return result
    
    def flatten_for_vectorstore(self, chunks: List[Tuple[Document, List[Document]]]) -> List[Document]:
        """
        将父子分块展平用于向量存储（旧版，全部展平，不推荐）
        """
        all_docs = []
        for parent_doc, child_docs in chunks:
            all_docs.append(parent_doc)
            all_docs.extend(child_docs)
        return all_docs

    def get_children_and_parent_lookup(
        self, chunks: List[Tuple[Document, List[Document]]]
    ) -> Tuple[List[Document], Dict[str, str]]:
        """
        正确的父子分块检索方式：只存子块到向量库，父块内容存到 lookup dict。

        Returns:
            (child_docs, parent_lookup)
            - child_docs: 子块列表，存入 Milvus 向量库
            - parent_lookup: {parent_chunk_id: parent_content}，检索到子块后查此 dict 返回父块
        """
        child_docs = []
        parent_lookup = {}

        for parent_doc, children in chunks:
            parent_id = parent_doc.metadata.get("chunk_index", f"p{len(parent_lookup)}")
            parent_lookup[f"p{parent_id}"] = parent_doc.page_content
            for child in children:
                child.metadata["parent_chunk_id"] = f"p{parent_id}"
                child_docs.append(child)

        return child_docs, parent_lookup


class EnhancedChunker:
    """增强分块器 - 整合语义分块、元信息增强、父子分块"""
    
    def __init__(
        self,
        use_semantic: bool = True,
        use_parent_child: bool = False,
        enable_metadata_enhancement: bool = True,
        max_chunk_size: int = 400,
        min_chunk_size: int = 200
    ):
        """
        初始化增强分块器
        
        Args:
            use_semantic: 是否使用语义分块
            use_parent_child: 是否使用父子分块
            enable_metadata_enhancement: 是否启用元信息增强
            max_chunk_size: 最大分块大小
            min_chunk_size: 最小分块大小
        """
        self.use_semantic = use_semantic
        self.use_parent_child = use_parent_child
        self.enable_metadata = enable_metadata_enhancement
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # 语义分块器（优化参数，避免过度分割）
        self.semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=92,  # 高阈值，减少分割点
            min_chunk_size=min_chunk_size     # 保证最小语义单元
        )
        
        # 递归分块器（备用方案）
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=60,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", "，", " ", ""]
        )
        
        # 父子分块器（parent=2000, child=800，比例 2.5:1）
        # child_size 不能太小，否则子块 embedding 质量差，检索效果反而不如 basic
        self.parent_child_chunker = ParentChildChunker(
            parent_size=2000,
            child_size=800,
            parent_overlap=200,
            child_overlap=100
        )
        self.parent_lookup: Dict[str, str] = {}  # 父块内容 lookup
        
        # 元数据提取器
        self.metadata_extractor = EnhancedMetadataExtractor()
    
    def chunk_document(self, content: str, file_path: str = "unknown") -> List[Document]:
        """
        执行增强分块
        
        Args:
            content: 文档内容
            file_path: 文件路径
            
        Returns:
            分块后的文档列表
        """
        logger.info("开始增强分块处理：%s", os.path.basename(file_path))
        logger.info("配置：语义分块=%s, 父子分块=%s, 元信息增强=%s",
                     self.use_semantic, self.use_parent_child, self.enable_metadata)

        # 步骤 1: 基础分块
        if self.use_semantic:
            logger.info("应用语义分块策略...")
            chunks = self._semantic_chunking_safe(content)
        else:
            logger.info("应用递归分块策略...")
            chunks = self.recursive_splitter.split_text(content)
            chunks = [Document(page_content=text, metadata={}) for text in chunks]

        logger.info("基础分块完成，生成 %d 个块", len(chunks))

        # 步骤 2: 元信息增强
        if self.enable_metadata:
            logger.info("提取元信息...")
            chunks = self._enhance_all_metadata(chunks, content, file_path)
            logger.info("元信息增强完成")

        # 步骤 3: 父子分块转换
        if self.use_parent_child:
            logger.info("构建父子分块结构...")
            chunks = self._create_parent_child_structure(chunks, file_path)
            logger.info("父子分块构建完成")

        logger.info("增强分块处理完成，最终生成 %d 个文档块", len(chunks))
        return chunks
    
    def _local_semantic_split(self, content: str) -> List[str]:
        """本地语义分块（不依赖 API），基于段落边界和句子结构"""
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current = ""
        target_size = 1500  # 目标块大小

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) < target_size:
                current += ("\n\n" + para if current else para)
            else:
                if current:
                    chunks.append(current)
                if len(para) > target_size * 1.5:
                    # 大段落按句子切分
                    sentences = re.split(r'(?<=[。！？.!?])', para)
                    temp = ""
                    for sent in sentences:
                        if len(temp) + len(sent) < target_size:
                            temp += sent
                        else:
                            if temp:
                                chunks.append(temp)
                            temp = sent
                    current = temp if temp else ""
                else:
                    current = para
        if current:
            chunks.append(current)
        return chunks

    def _semantic_chunking_safe(self, content: str) -> List[Document]:
        """安全的语义分块（API 优先，超时降级到本地语义分块）"""
        try:
            semantic_chunks = self.semantic_splitter.split_text(content)
            if len(semantic_chunks) < 3 and len(content) > 1000:
                logger.warning("语义分块仅产生 %d 个块，降级到本地语义分块", len(semantic_chunks))
                texts = self._local_semantic_split(content)
                return [Document(page_content=text, metadata={}) for text in texts]
            return [Document(page_content=text, metadata={}) for text in semantic_chunks]
        except Exception as e:
            logger.warning("API 语义分块失败（%s），降级到本地语义分块", str(e)[:80])
            texts = self._local_semantic_split(content)
            return [Document(page_content=text, metadata={}) for text in texts]
    
    def _enhance_all_metadata(self, chunks: List[Document], content: str, 
                             file_path: str) -> List[Document]:
        """为所有分块增强元数据"""
        enhanced = []
        
        for i, chunk in enumerate(chunks):
            # 估算 chunk 在原文中的位置
            chunk_position = sum(len(c.page_content) for c in chunks[:i])
            
            # 提取元数据
            meta = self.metadata_extractor.extract_all_metadata(
                chunk.page_content, content, chunk_position, file_path
            )
            
            # 合并原有元数据
            merged_meta = {**chunk.metadata, **meta.to_dict()}
            
            # 将元信息融入文本内容（关键改进）
            enhanced_content = self._prepend_metadata_to_content(chunk.page_content, meta)
            
            enhanced.append(Document(
                page_content=enhanced_content,
                metadata=merged_meta
            ))
        
        return enhanced
    
    def _prepend_metadata_to_content(self, content: str, meta: ChunkMetadata) -> str:
        """将元信息作为前缀添加到内容中"""
        context_parts = []
        
        if meta.document_title:
            context_parts.append(f"[文档:{meta.document_title}]")
        if meta.chapter_title:
            context_parts.append(f"[章节:{meta.chapter_title}]")
        if meta.section_title:
            context_parts.append(f"[小节:{meta.section_title}]")
        
        if context_parts:
            return " ".join(context_parts) + "\n" + content
        return content
    
    def _create_parent_child_structure(self, chunks: List[Document],
                                       file_path: str) -> List[Document]:
        """创建父子分块结构，只返回子块（用于向量检索），父块存到 self.parent_lookup"""
        all_parent_child_chunks = []

        for chunk in chunks:
            parent_child_result = self.parent_child_chunker.create_parent_child_chunks(
                chunk.page_content,
                base_metadata=chunk.metadata
            )
            all_parent_child_chunks.extend(parent_child_result)

        # 正确方式：只存子块到向量库，父块存到 lookup
        child_docs, parent_lookup = self.parent_child_chunker.get_children_and_parent_lookup(
            all_parent_child_chunks
        )
        self.parent_lookup = parent_lookup  # 暴露给外部使用
        logger.info("父子分块：子块 %d 个（入向量库），父块 %d 个（lookup）",
                     len(child_docs), len(parent_lookup))
        return child_docs


# 便捷函数
def create_enhanced_chunker(
    strategy: str = "semantic_metadata",
    **kwargs
) -> EnhancedChunker:
    """
    创建增强分块器的工厂函数
    
    Args:
        strategy: 分块策略
            - "semantic_metadata": 语义分块 + 元信息增强
            - "parent_child": 父子分块
            - "full": 全部功能
            - "basic": 仅基础分块
    
    Returns:
        EnhancedChunker 实例
    """
    if strategy == "semantic_metadata":
        return EnhancedChunker(
            use_semantic=True,
            use_parent_child=False,
            enable_metadata_enhancement=True,
            **kwargs
        )
    elif strategy == "parent_child":
        return EnhancedChunker(
            use_semantic=True,
            use_parent_child=True,
            enable_metadata_enhancement=True,
            **kwargs
        )
    elif strategy == "full":
        return EnhancedChunker(
            use_semantic=True,
            use_parent_child=True,
            enable_metadata_enhancement=True,
            **kwargs
        )
    else:  # basic
        return EnhancedChunker(
            use_semantic=False,
            use_parent_child=False,
            enable_metadata_enhancement=False,
            **kwargs
        )
