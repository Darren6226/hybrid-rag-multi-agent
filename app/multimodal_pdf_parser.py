"""
多模态PDF解析器 - 专为RAG优化的增强方案

核心功能：
1. 同时提取文本、表格、图片
2. 建立内容关联（图片-文本上下文）
3. 生成图片描述用于向量检索
4. 智能分块策略优化检索准确性
"""

import os
import io
import base64
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json


@dataclass
class MultimodalContent:
    """多模态内容基类"""
    content_type: str          # text, table, image, page
    page_num: int
    content: Any               # 实际内容
    metadata: Dict = field(default_factory=dict)
    
    def to_text(self) -> str:
        """转换为文本表示（用于索引）"""
        raise NotImplementedError


@dataclass
class TextContent(MultimodalContent):
    """文本内容"""
    content_type: str = field(default="text", init=False)
    text: str = ""
    
    def __post_init__(self):
        if not self.text and isinstance(self.content, str):
            self.text = self.content
    
    def to_text(self) -> str:
        return self.text


@dataclass
class TableContent(MultimodalContent):
    """表格内容"""
    content_type: str = field(default="table", init=False)
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    caption: str = ""
    
    def to_text(self) -> str:
        """将表格转换为结构化文本"""
        text_parts = ["表格内容:"]
        if self.caption:
            text_parts.append(f"标题: {self.caption}")
        if self.headers:
            text_parts.append("表头: " + " | ".join(self.headers))
        for i, row in enumerate(self.rows[:10], 1):
            text_parts.append(f"第{i}行: " + " | ".join(str(cell) for cell in row))
        if len(self.rows) > 10:
            text_parts.append(f"... (共{len(self.rows)}行)")
        return "\n".join(text_parts)


@dataclass
class ImageContent(MultimodalContent):
    """图片内容"""
    content_type: str = field(default="image", init=False)
    image_data: bytes = field(default_factory=bytes)
    ext: str = "png"
    width: int = 0
    height: int = 0
    caption: str = ""           # 图片标题/描述
    context_text: str = ""      # 上下文文本
    ocr_text: str = ""          # OCR识别文本
    
    @property
    def base64(self) -> str:
        return base64.b64encode(self.image_data).decode('utf-8') if self.image_data else ""
    
    def to_text(self) -> str:
        """生成图片的文本表示用于检索"""
        parts = ["图片内容:"]
        if self.caption:
            parts.append(f"描述: {self.caption}")
        if self.context_text:
            parts.append(f"上下文: {self.context_text[:200]}")
        if self.ocr_text:
            parts.append(f"图中文字: {self.ocr_text[:300]}")
        parts.append(f"尺寸: {self.width}x{self.height}像素")
        return "\n".join(parts)


@dataclass
class ParsedDocument:
    """解析后的文档"""
    source_path: str
    total_pages: int = 0
    contents: List[MultimodalContent] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def get_text_contents(self) -> List[TextContent]:
        return [c for c in self.contents if isinstance(c, TextContent)]
    
    def get_table_contents(self) -> List[TableContent]:
        return [c for c in self.contents if isinstance(c, TableContent)]
    
    def get_image_contents(self) -> List[ImageContent]:
        return [c for c in self.contents if isinstance(c, ImageContent)]
    
    def to_chunks(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
        """将文档内容转换为检索块"""
        chunks = []
        
        for content in self.contents:
            base_metadata = {
                "source": self.source_path,
                "page": content.page_num,
                "type": content.content_type,
                **content.metadata
            }
            
            if content.content_type == "text":
                # 文本分块
                text_chunks = self._split_text(content.to_text(), chunk_size, chunk_overlap)
                for i, text_chunk in enumerate(text_chunks):
                    chunks.append({
                        "content": text_chunk,
                        "content_type": "text",
                        "metadata": {**base_metadata, "chunk_index": i}
                    })
                    
            elif content.content_type == "table":
                # 表格作为独立块
                chunks.append({
                    "content": content.to_text(),
                    "content_type": "table",
                    "metadata": base_metadata
                })
                
            elif content.content_type == "image":
                # 图片内容生成文本描述块
                img_text = content.to_text()
                if img_text:
                    chunks.append({
                        "content": img_text,
                        "content_type": "image_description",
                        "metadata": {
                            **base_metadata,
                            "has_image": True,
                            "image_base64": content.base64[:100] + "..." if len(content.base64) > 100 else content.base64
                        }
                    })
        
        return chunks
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """智能文本分块"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        separators = ["\n\n", "\n", "。", ".", " ", ""]
        
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # 寻找最佳分割点
            if end < len(text):
                for sep in separators:
                    pos = text.rfind(sep, start, end)
                    if pos > start + chunk_size // 2:
                        end = pos + len(sep)
                        break
            
            chunks.append(text[start:end].strip())
            start = end - overlap if end < len(text) else end
        
        return chunks


class MultimodalPDFParser:
    """多模态PDF解析器"""
    
    def __init__(self, enable_ocr: bool = False, min_image_size: int = 100):
        """
        初始化解析器
        
        Args:
            enable_ocr: 是否启用OCR识别图片中的文字
            min_image_size: 最小图片尺寸（过滤小图标）
        """
        self.enable_ocr = enable_ocr
        self.min_image_size = min_image_size
        self._init_dependencies()
    
    def _init_dependencies(self):
        """初始化依赖"""
        # PyMuPDF用于图片提取
        try:
            import fitz
            self.has_fitz = True
        except ImportError:
            self.has_fitz = False
            raise ImportError("请安装 PyMuPDF: pip install PyMuPDF")
        
        # pdfplumber用于表格提取
        try:
            import pdfplumber
            self.has_pdfplumber = True
        except ImportError:
            self.has_pdfplumber = False
        
        # OCR依赖
        if self.enable_ocr:
            try:
                import pytesseract
                from PIL import Image
                self.has_ocr = True
            except ImportError:
                self.has_ocr = False
                print("⚠ 警告: OCR依赖未安装，图片文字识别功能将禁用")
    
    def parse(self, file_path: str) -> ParsedDocument:
        """
        解析PDF文档，提取所有多模态内容
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            解析后的文档对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        import fitz
        
        doc = ParsedDocument(
            source_path=file_path,
            metadata={
                "parsed_at": datetime.now().isoformat(),
                "file_size": os.path.getsize(file_path)
            }
        )
        
        print(f"📄 开始解析PDF: {os.path.basename(file_path)}")
        
        with fitz.open(file_path) as pdf:
            doc.total_pages = len(pdf)
            print(f"   共 {len(pdf)} 页")
            
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                
                # 1. 提取页面文本
                text = page.get_text()
                if text.strip():
                    doc.contents.append(TextContent(
                        page_num=page_num,
                        content=text,
                        text=text,
                        metadata={"char_count": len(text)}
                    ))
                
                # 2. 提取图片
                images = self._extract_images_from_page(pdf, page, page_num)
                for img in images:
                    doc.contents.append(img)
                
                # 3. 建立图片与文本的关联
                self._link_images_with_context(doc, page_num)
        
        # 4. 提取表格（使用pdfplumber）
        if self.has_pdfplumber:
            tables = self._extract_tables(file_path)
            for table in tables:
                # 找到合适的插入位置
                doc.contents.append(table)
            
            # 按页码排序
            doc.contents.sort(key=lambda x: x.page_num)
        
        print(f"✅ 解析完成: {len(doc.get_text_contents())}段文本, "
              f"{len(doc.get_table_contents())}个表格, "
              f"{len(doc.get_image_contents())}张图片")
        
        return doc
    
    def _extract_images_from_page(self, pdf, page, page_num: int) -> List[ImageContent]:
        """从页面提取图片"""
        import fitz
        
        images = []
        img_list = page.get_images(full=True)
        
        for img_index, img in enumerate(img_list):
            xref = img[0]
            try:
                pix = fitz.Pixmap(pdf, xref)
                
                # 处理CMYK图片
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                # 尺寸过滤
                if pix.width < self.min_image_size or pix.height < self.min_image_size:
                    pix = None
                    continue
                
                # 转换为PNG
                img_data = pix.tobytes("png")
                
                # OCR识别（如果启用）
                ocr_text = ""
                if self.enable_ocr and self.has_ocr:
                    ocr_text = self._ocr_image(img_data)
                
                image_content = ImageContent(
                    page_num=page_num,
                    content=img_data,
                    image_data=img_data,
                    ext="png",
                    width=pix.width,
                    height=pix.height,
                    ocr_text=ocr_text,
                    metadata={
                        "xref": xref,
                        "image_index": img_index
                    }
                )
                
                images.append(image_content)
                pix = None
                
            except Exception as e:
                print(f"   图片提取失败 (xref={xref}): {e}")
                continue
        
        return images
    
    def _extract_tables(self, file_path: str) -> List[TableContent]:
        """提取表格"""
        import pdfplumber
        
        tables = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                
                for table_idx, table_data in enumerate(page_tables):
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    # 提取表头和数据
                    headers = [str(cell or "") for cell in table_data[0]]
                    rows = []
                    for row in table_data[1:]:
                        rows.append([str(cell or "") for cell in row])
                    
                    table_content = TableContent(
                        page_num=page_num,
                        content=table_data,
                        headers=headers,
                        rows=rows,
                        metadata={
                            "table_index": table_idx,
                            "row_count": len(rows),
                            "col_count": len(headers)
                        }
                    )
                    
                    tables.append(table_content)
        
        return tables
    
    def _link_images_with_context(self, doc: ParsedDocument, page_num: int):
        """建立图片与周围文本的关联"""
        # 获取当前页面的所有内容
        page_contents = [c for c in doc.contents if c.page_num == page_num]
        images = [c for c in page_contents if isinstance(c, ImageContent)]
        texts = [c for c in page_contents if isinstance(c, TextContent)]
        
        # 为每张图片找到最近的文本上下文
        for img in images:
            # 简单策略：使用页面前面的文本作为上下文
            context_parts = []
            for text_content in texts:
                text = text_content.text.strip()
                if len(text) > 20:  # 过滤短文本
                    # 提取可能相关的句子（包含"图"、"Figure"等关键词）
                    lines = text.split('\n')
                    for line in lines[-5:]:  # 取最后几行
                        if any(kw in line for kw in ['图', 'Figure', '表', 'Table', '如图', '如下']):
                            context_parts.append(line)
            
            if context_parts:
                img.context_text = ' '.join(context_parts[-3:])  # 最多取3行
            
            # 生成简单描述
            if not img.caption:
                img.caption = f"第{page_num+1}页的图片 ({img.width}x{img.height}像素)"
    
    def _ocr_image(self, image_data: bytes) -> str:
        """对图片进行OCR识别"""
        if not self.has_ocr:
            return ""
        
        try:
            from PIL import Image
            import pytesseract
            import io
            
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            return text.strip()
        except Exception as e:
            return ""


class MultimodalIndexer:
    """多模态内容索引器 - 为RAG准备向量数据"""
    
    def __init__(self):
        self.chunks = []
    
    def index_document(self, doc: ParsedDocument):
        """索引解析后的文档"""
        chunks = doc.to_chunks()
        
        for chunk in chunks:
            # 生成唯一ID
            content_hash = hashlib.md5(
                chunk['content'].encode()
            ).hexdigest()[:12]
            
            chunk_id = f"{chunk['content_type']}_{chunk['metadata']['page']}_{content_hash}"
            
            indexed_chunk = {
                "id": chunk_id,
                "content": chunk['content'],
                "content_type": chunk['content_type'],
                "metadata": chunk['metadata']
            }
            
            self.chunks.append(indexed_chunk)
        
        print(f"✅ 已索引 {len(chunks)} 个块")
        return self.chunks
    
    def save_index(self, output_path: str):
        """保存索引到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"✅ 索引已保存: {output_path}")
    
    def get_statistics(self) -> Dict:
        """获取索引统计信息"""
        type_counts = {}
        for chunk in self.chunks:
            ctype = chunk['content_type']
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
        
        return {
            "total_chunks": len(self.chunks),
            "type_distribution": type_counts,
            "avg_chunk_length": sum(len(c['content']) for c in self.chunks) / len(self.chunks) if self.chunks else 0
        }


# 便捷函数
def parse_pdf_for_rag(file_path: str, enable_ocr: bool = False) -> Tuple[ParsedDocument, List[Dict]]:
    """
    一站式PDF解析函数
    
    Args:
        file_path: PDF文件路径
        enable_ocr: 是否启用OCR
        
    Returns:
        (解析后的文档, 索引块列表)
    """
    # 解析文档
    parser = MultimodalPDFParser(enable_ocr=enable_ocr)
    doc = parser.parse(file_path)
    
    # 创建索引
    indexer = MultimodalIndexer()
    chunks = indexer.index_document(doc)
    
    return doc, chunks


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        print("\n" + "="*60)
        print("多模态PDF解析器测试")
        print("="*60)
        
        try:
            doc, chunks = parse_pdf_for_rag(pdf_path, enable_ocr=False)
            
            print("\n📊 解析统计:")
            print(f"   总页数: {doc.total_pages}")
            print(f"   文本块: {len(doc.get_text_contents())}")
            print(f"   表格数: {len(doc.get_table_contents())}")
            print(f"   图片数: {len(doc.get_image_contents())}")
            
            print("\n📊 索引统计:")
            indexer = MultimodalIndexer()
            indexer.chunks = chunks
            stats = indexer.get_statistics()
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            # 保存示例
            indexer.save_index("multimodal_index.json")
            
            # 显示前3个块
            print("\n📝 前3个索引块预览:")
            for chunk in chunks[:3]:
                print(f"\n   [{chunk['content_type']}] {chunk['id'][:20]}...")
                preview = chunk['content'][:150].replace('\n', ' ')
                print(f"   {preview}...")
                
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("用法: python multimodal_pdf_parser.py <pdf_file_path>")
