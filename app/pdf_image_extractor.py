"""
PDF图片提取模块

功能：
1. 从PDF中提取所有图片
2. 支持多种图片格式（PNG, JPEG, BMP, etc.）
3. 自动处理图片格式转换
4. 提取图片元数据（位置、尺寸、页码等）
5. 支持批量处理和多线程优化
"""

import os
import io
import base64
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


@dataclass
class ExtractedImage:
    """提取的图片数据结构"""
    image_data: bytes          # 图片二进制数据
    ext: str                   # 图片格式扩展名
    page_num: int              # 所在页码（从0开始）
    index: int                 # 图片在页面中的索引
    width: int                 # 图片宽度
    height: int                # 图片高度
    name: Optional[str] = None # 图片名称（如果有）
    
    @property
    def size(self) -> int:
        """获取图片大小（字节）"""
        return len(self.image_data)
    
    @property
    def base64(self) -> str:
        """获取Base64编码的图片数据"""
        return base64.b64encode(self.image_data).decode('utf-8')
    
    def save(self, output_path: str) -> str:
        """保存图片到指定路径"""
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # 添加扩展名
        if not output_path.lower().endswith(f'.{self.ext.lower()}'):
            output_path = f"{output_path}.{self.ext}"
        
        with open(output_path, 'wb') as f:
            f.write(self.image_data)
        
        return output_path


class PDFImageExtractor:
    """PDF图片提取器 - 使用PyMuPDF (fitz)"""
    
    def __init__(self, dpi: int = 150, max_workers: int = 4):
        """
        初始化图片提取器
        
        Args:
            dpi: 渲染DPI（用于将PDF页面转换为图片时）
            max_workers: 多线程最大工作线程数
        """
        self.dpi = dpi
        self.max_workers = max_workers
        self._check_dependencies()
    
    def _check_dependencies(self):
        """检查依赖库"""
        try:
            import fitz  # PyMuPDF
            self.has_fitz = True
        except ImportError:
            self.has_fitz = False
            raise ImportError(
                "PyMuPDF (fitz) 未安装，请运行: pip install PyMuPDF\n"
                "或者: pip install -r requirements.txt"
            )
        
        try:
            from PIL import Image
            self.has_pil = True
        except ImportError:
            self.has_pil = False
            print("⚠ 警告: Pillow未安装，部分功能可能受限")
    
    def extract_images(
        self, 
        file_path: str, 
        min_width: int = 100,
        min_height: int = 100,
        max_images: Optional[int] = None
    ) -> List[ExtractedImage]:
        """
        从PDF中提取所有图片
        
        Args:
            file_path: PDF文件路径
            min_width: 最小图片宽度（过滤小图标）
            min_height: 最小图片高度（过滤小图标）
            max_images: 最大提取图片数量（None表示不限制）
            
        Returns:
            提取的图片列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")
        
        if not self.has_fitz:
            raise ImportError("PyMuPDF (fitz) 未安装")
        
        import fitz
        
        extracted_images = []
        
        print(f"📄 开始提取PDF图片: {os.path.basename(file_path)}")
        
        with fitz.open(file_path) as doc:
            print(f"   PDF共 {len(doc)} 页")
            
            for page_num in range(len(doc)):
                if max_images and len(extracted_images) >= max_images:
                    print(f"   已达到最大提取数量 {max_images}，停止提取")
                    break
                
                page = doc[page_num]
                page_images = self._extract_images_from_page(
                    page, page_num, min_width, min_height, max_images
                )
                extracted_images.extend(page_images)
                
                if page_images:
                    print(f"   第 {page_num + 1} 页: 提取 {len(page_images)} 张图片")
        
        print(f"✅ 图片提取完成，共 {len(extracted_images)} 张")
        return extracted_images
    
    def _extract_images_from_page(
        self, 
        page, 
        page_num: int,
        min_width: int,
        min_height: int,
        max_images: Optional[int]
    ) -> List[ExtractedImage]:
        """从单个页面提取图片"""
        import fitz
        
        images = []
        img_list = page.get_images(full=True)
        
        for img_index, img in enumerate(img_list):
            if max_images and len(images) >= max_images:
                break
            
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)
            
            # 跳过CMYK图片的转换问题
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            
            # 检查图片尺寸
            if pix.width < min_width or pix.height < min_height:
                pix = None
                continue
            
            # 转换为PNG格式
            img_data = pix.tobytes("png")
            
            extracted_img = ExtractedImage(
                image_data=img_data,
                ext="png",
                page_num=page_num,
                index=img_index,
                width=pix.width,
                height=pix.height,
                name=f"page_{page_num + 1}_img_{img_index + 1}"
            )
            
            images.append(extracted_img)
            pix = None
        
        return images
    
    def extract_pages_as_images(
        self, 
        file_path: str,
        page_numbers: Optional[List[int]] = None,
        zoom: float = 2.0
    ) -> List[ExtractedImage]:
        """
        将PDF页面转换为图片（适用于扫描件或无法提取嵌入图片的情况）
        
        Args:
            file_path: PDF文件路径
            page_numbers: 要转换的页码列表（None表示全部）
            zoom: 缩放倍数（2.0表示2倍DPI）
            
        Returns:
            页面图片列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")
        
        import fitz
        
        images = []
        
        print(f"📄 将PDF页面转换为图片: {os.path.basename(file_path)}")
        
        with fitz.open(file_path) as doc:
            # 确定要处理的页面
            if page_numbers is None:
                page_numbers = list(range(len(doc)))
            
            print(f"   将处理 {len(page_numbers)} 页")
            
            # 设置渲染矩阵
            mat = fitz.Matrix(zoom, zoom)
            
            for page_num in page_numbers:
                if page_num >= len(doc):
                    print(f"   ⚠ 页码 {page_num + 1} 超出范围，跳过")
                    continue
                
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat)
                
                img_data = pix.tobytes("png")
                
                extracted_img = ExtractedImage(
                    image_data=img_data,
                    ext="png",
                    page_num=page_num,
                    index=0,
                    width=pix.width,
                    height=pix.height,
                    name=f"page_{page_num + 1}_render"
                )
                
                images.append(extracted_img)
                pix = None
                
                if (page_num + 1) % 10 == 0:
                    print(f"   已处理 {page_num + 1}/{len(doc)} 页")
        
        print(f"✅ 页面转换完成，共 {len(images)} 页")
        return images
    
    def save_images(
        self, 
        images: List[ExtractedImage], 
        output_dir: str,
        naming_pattern: str = "{name}_{index:03d}"
    ) -> List[str]:
        """
        批量保存提取的图片
        
        Args:
            images: 提取的图片列表
            output_dir: 输出目录
            naming_pattern: 命名模式，可用变量: {name}, {index}, {page}, {ext}
            
        Returns:
            保存的文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        
        for i, img in enumerate(images):
            filename = naming_pattern.format(
                name=img.name or "image",
                index=i + 1,
                page=img.page_num + 1,
                ext=img.ext
            )
            
            output_path = os.path.join(output_dir, f"{filename}.{img.ext}")
            saved_path = img.save(output_path)
            saved_paths.append(saved_path)
        
        print(f"✅ 已保存 {len(saved_paths)} 张图片到: {output_dir}")
        return saved_paths
    
    def get_image_summary(self, images: List[ExtractedImage]) -> Dict:
        """
        获取图片提取的统计摘要
        
        Args:
            images: 提取的图片列表
            
        Returns:
            统计信息字典
        """
        if not images:
            return {"total": 0}
        
        total_size = sum(img.size for img in images)
        
        # 按页面统计
        pages_with_images = set(img.page_num for img in images)
        
        # 尺寸分布
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        
        return {
            "total": len(images),
            "pages_with_images": len(pages_with_images),
            "page_numbers": sorted(pages_with_images),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "avg_width": round(sum(widths) / len(widths), 0),
            "avg_height": round(sum(heights) / len(heights), 0),
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights)
        }


class IntegratedPDFExtractor:
    """集成的PDF提取器 - 同时提取文本、表格和图片"""
    
    def __init__(self, enable_tables: bool = False, enable_images: bool = True):
        """
        初始化集成提取器
        
        Args:
            enable_tables: 是否启用表格提取
            enable_images: 是否启用图片提取
        """
        self.enable_tables = enable_tables
        self.enable_images = enable_images
        
        # 导入文本提取器
        from app.pdf_extractor import PDFExtractor
        self.text_extractor = PDFExtractor(enable_tables=enable_tables)
        
        # 初始化图片提取器
        if enable_images:
            try:
                self.image_extractor = PDFImageExtractor()
                self.has_image_support = True
            except ImportError:
                print("⚠ 警告: PyMuPDF未安装，图片提取功能将被禁用")
                self.has_image_support = False
    
    def extract_all(
        self, 
        file_path: str,
        extract_images: bool = True,
        min_image_width: int = 100,
        min_image_height: int = 100
    ) -> Dict:
        """
        提取PDF中的所有内容（文本、表格、图片）
        
        Args:
            file_path: PDF文件路径
            extract_images: 是否提取图片
            min_image_width: 最小图片宽度
            min_image_height: 最小图片高度
            
        Returns:
            包含所有内容的字典
        """
        result = {
            "text": "",
            "metadata": {},
            "images": [],
            "image_summary": {},
            "structure": []
        }
        
        print(f"\n🚀 开始全面提取PDF内容: {os.path.basename(file_path)}")
        print("=" * 60)
        
        # 1. 提取文本
        print("\n📄 步骤1: 提取文本内容...")
        try:
            text, metadata = self.text_extractor.extract_text(file_path)
            result["text"] = text
            result["metadata"] = metadata
            print(f"   ✅ 文本提取完成: {len(text)} 字符")
        except Exception as e:
            print(f"   ❌ 文本提取失败: {e}")
        
        # 2. 提取结构
        print("\n📄 步骤2: 提取文档结构...")
        try:
            _, structure = self.text_extractor.extract_with_structure(file_path)
            result["structure"] = structure
            print(f"   ✅ 结构提取完成: {len(structure)} 个章节")
        except Exception as e:
            print(f"   ❌ 结构提取失败: {e}")
        
        # 3. 提取图片
        if extract_images and self.has_image_support:
            print("\n📄 步骤3: 提取图片...")
            try:
                images = self.image_extractor.extract_images(
                    file_path,
                    min_width=min_image_width,
                    min_height=min_image_height
                )
                result["images"] = images
                result["image_summary"] = self.image_extractor.get_image_summary(images)
                print(f"   ✅ 图片提取完成: {len(images)} 张")
            except Exception as e:
                print(f"   ❌ 图片提取失败: {e}")
        elif extract_images:
            print("\n📄 步骤3: 跳过图片提取（未安装PyMuPDF）")
        
        print("\n" + "=" * 60)
        print("🎉 PDF内容提取完成!")
        
        return result


# 便捷函数
def extract_pdf_images(
    file_path: str,
    output_dir: Optional[str] = None,
    min_width: int = 100,
    min_height: int = 100
) -> List[ExtractedImage]:
    """
    便捷函数：提取PDF中的图片
    
    Args:
        file_path: PDF文件路径
        output_dir: 图片保存目录（None则不保存）
        min_width: 最小图片宽度
        min_height: 最小图片高度
        
    Returns:
        提取的图片列表
    """
    extractor = PDFImageExtractor()
    images = extractor.extract_images(file_path, min_width, min_height)
    
    if output_dir and images:
        extractor.save_images(images, output_dir)
    
    return images


def extract_pdf_all(file_path: str) -> Dict:
    """
    便捷函数：提取PDF中的所有内容
    
    Args:
        file_path: PDF文件路径
        
    Returns:
        包含文本、图片等内容的字典
    """
    extractor = IntegratedPDFExtractor(enable_tables=True, enable_images=True)
    return extractor.extract_all(file_path)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        
        print("\n" + "=" * 60)
        print("PDF图片提取测试")
        print("=" * 60)
        
        # 测试图片提取
        extractor = PDFImageExtractor()
        images = extractor.extract_images(pdf_path)
        
        if images:
            print("\n📊 提取摘要:")
            summary = extractor.get_image_summary(images)
            for key, value in summary.items():
                print(f"   {key}: {value}")
            
            if output_dir:
                extractor.save_images(images, output_dir)
        else:
            print("\n⚠ 未找到图片，尝试将页面转换为图片...")
            page_images = extractor.extract_pages_as_images(pdf_path)
            if page_images:
                print(f"✅ 已将 {len(page_images)} 页转换为图片")
                if output_dir:
                    extractor.save_images(page_images, output_dir)
    else:
        print("用法: python pdf_image_extractor.py <pdf_file_path> [output_directory]")
        print("\n示例:")
        print("  python pdf_image_extractor.py document.pdf")
        print("  python pdf_image_extractor.py document.pdf ./extracted_images")
