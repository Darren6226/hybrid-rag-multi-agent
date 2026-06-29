"""
增强的PDF文本提取模块

功能：
1. 支持多种PDF解析库（pdfminer, PyPDF2）
2. 表格内容提取（可选）
3. 智能文本清理
4. 结构信息保留
"""

import re
from typing import Tuple, List, Optional
import os


class PDFExtractor:
    """PDF文本提取器 - 支持多种解析方式"""

    def __init__(self, enable_tables: bool = False):
        """
        初始化PDF提取器

        Args:
            enable_tables: 是否启用表格提取（需要pdfplumber）
        """
        self.enable_tables = enable_tables
        self._check_dependencies()

    def _check_dependencies(self):
        """检查依赖库"""
        try:
            from pdfminer.high_level import extract_text
            self.has_pdfminer = True
        except ImportError:
            self.has_pdfminer = False

        try:
            import PyPDF2
            self.has_pypdf2 = True
        except ImportError:
            self.has_pypdf2 = False

        if self.enable_tables:
            try:
                import pdfplumber
                self.has_pdfplumber = True
            except ImportError:
                print("⚠ 警告: pdfplumber未安装，表格提取功能将被禁用")
                self.enable_tables = False
                self.has_pdfplumber = False
        else:
            self.has_pdfplumber = False

    def extract_text(self, file_path: str, method: str = 'auto') -> Tuple[str, dict]:
        """
        提取PDF文本内容

        Args:
            file_path: PDF文件路径
            method: 提取方法 ('auto', 'pdfminer', 'pypdf2', 'pdfplumber')

        Returns:
            (提取的文本, 元数据字典)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")

        metadata = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'extraction_method': method
        }

        # 自动选择最佳方法
        if method == 'auto':
            if self.enable_tables and self.has_pdfplumber:
                return self._extract_with_pdfplumber(file_path, metadata)
            elif self.has_pdfminer:
                return self._extract_with_pdfminer(file_path, metadata)
            elif self.has_pypdf2:
                return self._extract_with_pypdf2(file_path, metadata)
            else:
                raise ImportError("没有可用的PDF解析库，请安装 pdfminer 或 PyPDF2")
        elif method == 'pdfminer' and self.has_pdfminer:
            return self._extract_with_pdfminer(file_path, metadata)
        elif method == 'pypdf2' and self.has_pypdf2:
            return self._extract_with_pypdf2(file_path, metadata)
        elif method == 'pdfplumber' and self.has_pdfplumber:
            return self._extract_with_pdfplumber(file_path, metadata)
        else:
            raise ValueError(f"不支持的提取方法: {method}")

    def _extract_with_pdfminer(self, file_path: str, metadata: dict) -> Tuple[str, dict]:
        """使用pdfminer提取文本"""
        try:
            from pdfminer.high_level import extract_text
            print("   使用pdfminer提取文本...")

            # 尝试使用不同编码
            codecs = ['utf-8', 'gbk', 'latin-1']
            text = None
            used_codec = None

            for codec in codecs:
                try:
                    text = extract_text(file_path, codec=codec)
                    used_codec = codec
                    break
                except Exception:
                    continue

            if text is None:
                text = extract_text(file_path)
                used_codec = 'default'

            # 清理文本
            text = self._clean_extracted_text(text)

            metadata['extraction_method'] = 'pdfminer'
            metadata['codec'] = used_codec
            metadata['text_length'] = len(text)

            print(f"   提取完成，使用编码: {used_codec}, 文本长度: {len(text)} 字符")
            return text, metadata

        except Exception as e:
            print(f"   pdfminer提取失败: {e}")
            raise

    def _extract_with_pypdf2(self, file_path: str, metadata: dict) -> Tuple[str, dict]:
        """使用PyPDF2提取文本（降级方案）"""
        try:
            import PyPDF2
            print("   使用PyPDF2提取文本（降级方案）...")

            text_parts = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            text = '\n\n'.join(text_parts)
            text = self._clean_extracted_text(text)

            metadata['extraction_method'] = 'pypdf2'
            metadata['num_pages'] = num_pages
            metadata['text_length'] = len(text)

            print(f"   提取完成，共 {num_pages} 页，文本长度: {len(text)} 字符")
            return text, metadata

        except Exception as e:
            print(f"   PyPDF2提取失败: {e}")
            raise

    def _extract_with_pdfplumber(self, file_path: str, metadata: dict) -> Tuple[str, dict]:
        """使用pdfplumber提取文本和表格"""
        try:
            import pdfplumber
            print("   使用pdfplumber提取文本和表格...")

            text_parts = []
            tables = []
            num_pages = 0

            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages):
                    # 提取文本
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

                    # 提取表格
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table_idx, table in enumerate(page_tables):
                            table_text = self._table_to_text(table, page_num, table_idx)
                            if table_text:
                                tables.append(table_text)

            # 合并文本和表格
            if tables:
                text_parts.insert(0, "\n=== 表格内容 ===\n")
                text_parts.extend(tables)

            text = '\n\n'.join(text_parts)
            text = self._clean_extracted_text(text)

            metadata['extraction_method'] = 'pdfplumber'
            metadata['num_pages'] = num_pages
            metadata['num_tables'] = len(tables)
            metadata['text_length'] = len(text)

            print(f"   提取完成，共 {num_pages} 页，{len(tables)} 个表格，文本长度: {len(text)} 字符")
            return text, metadata

        except Exception as e:
            print(f"   pdfplumber提取失败: {e}")
            raise

    def _table_to_text(self, table: List[list], page_num: int, table_idx: int) -> str:
        """
        将表格转换为文本描述

        Args:
            table: 表格数据（二维数组）
            page_num: 页码
            table_idx: 表格索引

        Returns:
            表格文本描述
        """
        if not table or not table[0]:
            return ""

        table_text = f"\n表格 {table_idx + 1} (第 {page_num + 1} 页):\n"

        # 提取表头
        header = table[0]
        table_text += "表头: " + " | ".join([str(cell) for cell in header if cell]) + "\n"

        # 提取数据行（限制显示行数）
        data_rows = table[1:11]  # 最多显示10行数据
        for row_idx, row in enumerate(data_rows):
            row_text = " | ".join([str(cell) for cell in row if cell])
            table_text += f"行 {row_idx + 1}: {row_text}\n"

        if len(table) > 11:
            table_text += f"...(省略 {len(table) - 11} 行数据)\n"

        return table_text

    def _clean_extracted_text(self, text: str) -> str:
        """
        清理提取的文本

        改进：
        1. 移除过多空行
        2. 修复连字符断行
        3. 统一换行符
        4. 移除页码、页眉页脚
        """
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 移除页码（常见的页码格式）
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # 单独一行的数字
        text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', text, flags=re.IGNORECASE)

        # 修复连字符断行
        text = re.sub(r'-\n', '', text)

        # 移除过多的空行（保留最多2个连续空行）
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 移除每行开头和结尾的空白
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)

        # 移除可能的页眉页脚（短行且在页面开头/结尾）
        # 这里简单处理：移除过短的行（可能是页码）
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            # 保留有意义的行
            if len(line) >= 10 or line == '' or re.match(r'^[一二三四五六七八九十]+[\.、]', line):
                filtered_lines.append(line)

        text = '\n'.join(filtered_lines)

        # 清理多余空格
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def extract_with_structure(self, file_path: str) -> Tuple[str, List[dict]]:
        """
        提取PDF并保留结构信息

        Returns:
            (文本, 结构信息列表)
        """
        text, metadata = self.extract_text(file_path)

        # 尝试提取章节结构
        structure = self._extract_structure(text)

        return text, structure

    def _extract_structure(self, text: str) -> List[dict]:
        """
        从文本中提取结构信息（章节、标题等）

        Returns:
            结构信息列表
        """
        structure = []
        lines = text.split('\n')

        for line_idx, line in enumerate(lines):
            line = line.strip()

            # 检测可能的标题
            title_patterns = [
                (r'^第[一二三四五六七八九十]+章\s+.+$', 'chapter'),
                (r'^\d+[\.\)]\s+[A-Z][a-zA-Z\s\-]+$', 'section'),
                (r'^\d+\.\d+[\.\)]\s+.+$', 'subsection'),
                (r'^摘要$|^关键词$|^引言$|^结论$|^参考文献$', 'special_section'),
            ]

            for pattern, section_type in title_patterns:
                if re.match(pattern, line):
                    structure.append({
                        'line_number': line_idx,
                        'type': section_type,
                        'title': line,
                        'content_start': line_idx + 1
                    })
                    break

        return structure


def extract_pdf_text(file_path: str, enable_tables: bool = False) -> str:
    """
    便捷函数：提取PDF文本

    Args:
        file_path: PDF文件路径
        enable_tables: 是否启用表格提取

    Returns:
        提取的文本
    """
    extractor = PDFExtractor(enable_tables=enable_tables)
    text, _ = extractor.extract_text(file_path)
    return text


def extract_pdf_with_images(
    file_path: str, 
    enable_tables: bool = False,
    enable_images: bool = True,
    image_output_dir: Optional[str] = None
) -> dict:
    """
    便捷函数：提取PDF文本、表格和图片

    Args:
        file_path: PDF文件路径
        enable_tables: 是否启用表格提取
        enable_images: 是否启用图片提取
        image_output_dir: 图片保存目录（None则不保存到文件）

    Returns:
        包含文本、元数据、图片等信息的字典
    """
    from app.pdf_image_extractor import PDFImageExtractor
    
    result = {
        "text": "",
        "metadata": {},
        "images": [],
        "image_summary": {},
        "structure": []
    }
    
    # 提取文本
    extractor = PDFExtractor(enable_tables=enable_tables)
    text, metadata = extractor.extract_text(file_path)
    result["text"] = text
    result["metadata"] = metadata
    
    # 提取结构
    _, structure = extractor.extract_with_structure(file_path)
    result["structure"] = structure
    
    # 提取图片
    if enable_images:
        try:
            image_extractor = PDFImageExtractor()
            images = image_extractor.extract_images(file_path)
            result["images"] = images
            result["image_summary"] = image_extractor.get_image_summary(images)
            
            if image_output_dir and images:
                image_extractor.save_images(images, image_output_dir)
        except ImportError:
            print("⚠ 警告: PyMuPDF未安装，跳过图片提取")
        except Exception as e:
            print(f"⚠ 图片提取失败: {e}")
    
    return result


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        # 检查是否需要提取图片
        extract_images = "--images" in sys.argv or "-i" in sys.argv
        
        if extract_images:
            print("🚀 开始提取PDF文本和图片...")
            result = extract_pdf_with_images(
                pdf_path, 
                enable_tables=True, 
                enable_images=True,
                image_output_dir="./extracted_images" if extract_images else None
            )
            
            print(f"\n提取完成:")
            print(f"- 文件: {result['metadata'].get('file_path')}")
            print(f"- 文本长度: {len(result['text'])} 字符")
            print(f"- 图片数量: {len(result['images'])}")
            
            if result['image_summary']:
                print(f"\n图片统计:")
                for key, value in result['image_summary'].items():
                    print(f"  {key}: {value}")
        else:
            extractor = PDFExtractor(enable_tables=True)
            text, metadata = extractor.extract_text(pdf_path)
            print(f"提取完成:")
            print(f"- 文件: {metadata['file_path']}")
            print(f"- 方法: {metadata['extraction_method']}")
            print(f"- 长度: {metadata['text_length']} 字符")
            print(f"\n文本预览 (前500字符):")
            print(text[:500])
            
            print("\n💡 提示: 添加 --images 或 -i 参数可提取图片")
    else:
        print("用法: python pdf_extractor.py <pdf_file_path> [--images|-i]")
        print("\n示例:")
        print("  python pdf_extractor.py document.pdf")
        print("  python pdf_extractor.py document.pdf --images")
