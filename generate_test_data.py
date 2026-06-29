"""
RAGAS 测试数据生成脚本
使用 LLM 从源文档批量生成 RAGAS 评估测试问题、标准答案和参考上下文。

使用方法:
    python generate_test_data.py --source company --count 20
    python generate_test_data.py --source dnngp --count 30
    python generate_test_data.py --source all
"""

import argparse
import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 常量
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(PROJECT_ROOT, "doc")
PDF_DIR = os.path.join(PROJECT_ROOT, "pdf")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmark_data")

COMPANY_TXT = os.path.join(DOC_DIR, "company.txt")
DNNGP_PDF = os.path.join(
    PDF_DIR,
    "DNNGP, a deep neural network-based method for genomic prediction "
    "using multi-omics data in plants(简短版）.pdf"
)

MAX_DOC_CHARS = 15000


def read_company_document() -> str:
    """读取 company.txt 文档内容。"""
    if not os.path.exists(COMPANY_TXT):
        raise FileNotFoundError(f"文档不存在: {COMPANY_TXT}")
    with open(COMPANY_TXT, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  文档长度: {len(text)} 字符")
    return text[:MAX_DOC_CHARS]


def read_dnngp_document() -> str:
    """读取 DNNGP PDF 文档内容。"""
    if not os.path.exists(DNNGP_PDF):
        raise FileNotFoundError(f"PDF 文件不存在: {DNNGP_PDF}")
    from app.pdf_extractor import PDFExtractor
    extractor = PDFExtractor(enable_tables=False)
    text, _ = extractor.extract_text(DNNGP_PDF, method="auto")
    print(f"  文档长度: {len(text)} 字符")
    return text[:MAX_DOC_CHARS]


GENERATION_PROMPT_TEMPLATE = """\
你是一个 RAG（检索增强生成）系统的测试数据生成专家。
以下是一篇文档的内容，请你基于这篇文档，生成 {count} 个高质量的测试问题。

每个问题需要包含：
1. question: 一个具体的、有明确答案的问题（用中文提问）
2. ground_truth: 基于文档内容的标准答案（用中文回答）
3. contexts: 从文档中提取的、与该问题直接相关的文本片段（1-3个片段），作为 RAG 系统应该检索到的参考上下文

要求：
- 问题应覆盖文档的不同部分和主题，不要集中在同一个话题
- 问题难度应多样化，包含事实性问题和需要综合理解的问题
- ground_truth 必须准确反映文档内容，不要编造信息
- contexts 必须是文档中的原始文本片段，不要修改
- 每个问题的 contexts 应包含足够信息来回答该问题

请直接输出 JSON 数组，不要包含任何其他文字说明。格式如下：
[
  {{
    "question": "问题内容",
    "ground_truth": "标准答案",
    "contexts": ["参考文档片段1", "参考文档片段2"]
  }}
]

文档内容：
---
{document}
---
"""


def generate_test_data_with_llm(document: str, count: int, source: str) -> list:
    """调用 LLM 生成测试数据。"""
    from app.config import llm

    prompt = GENERATION_PROMPT_TEMPLATE.format(
        count=count, document=document
    )

    print(f"  正在调用 LLM 生成 {count} 个测试问题...")
    response = llm.invoke(prompt)
    raw_text = response.content

    # 从 LLM 响应中提取 JSON 数组
    return parse_json_array(raw_text)


def parse_json_array(text: str) -> list:
    """从 LLM 响应文本中解析 JSON 数组。

    通过查找第一个 '[' 和最后一个 ']' 来定位 JSON 内容，
    以应对 LLM 可能在 JSON 前后添加说明文字的情况。
    """
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")

    if first_bracket == -1 or last_bracket == -1:
        raise ValueError(
            "LLM 响应中未找到有效的 JSON 数组。\n"
            f"响应前 500 字符: {text[:500]}"
        )

    json_str = text[first_bracket:last_bracket + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON 解析失败: {e}\n"
            f"提取的 JSON 字符串前 500 字符: {json_str[:500]}"
        )

    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 数组，但得到 {type(data).__name__}")

    return data


def validate_test_data(data: list, source: str) -> list:
    """验证生成的测试数据格式是否正确。"""
    valid_items = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"  警告: 第 {i + 1} 项不是字典，已跳过")
            continue
        if "question" not in item or "ground_truth" not in item:
            print(f"  警告: 第 {i + 1} 项缺少必要字段，已跳过")
            continue
        if "contexts" not in item or not isinstance(item["contexts"], list):
            item["contexts"] = []
        valid_items.append(item)

    print(f"  有效问题数: {len(valid_items)} / {len(data)}")
    return valid_items


def save_test_data(data: list, source: str) -> str:
    """保存测试数据到 benchmark_data 目录。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{source}_test_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


def generate_for_source(source: str, count: int) -> str:
    """为指定数据源生成测试数据。

    Returns:
        保存的文件路径
    """
    print(f"\n{'=' * 60}")
    print(f"正在为数据源 '{source}' 生成测试数据")
    print(f"{'=' * 60}")

    # 读取文档
    if source == "company":
        print("读取 company.txt ...")
        document = read_company_document()
    elif source == "dnngp":
        print("读取 DNNGP PDF ...")
        document = read_dnngp_document()
    else:
        raise ValueError(f"未知数据源: {source}")

    # 调用 LLM 生成
    data = generate_test_data_with_llm(document, count, source)

    # 验证
    data = validate_test_data(data, source)

    if not data:
        print("  错误: 未生成任何有效的测试数据")
        return ""

    # 保存
    output_path = save_test_data(data, source)
    print(f"\n已保存 {len(data)} 个测试问题到: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="使用 LLM 批量生成 RAGAS 评估测试数据"
    )
    parser.add_argument(
        "--source",
        choices=["company", "dnngp", "all"],
        default="all",
        help="数据源: company / dnngp / all (default: all)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="每个数据源生成的问题数量 (default: 20)",
    )
    args = parser.parse_args()

    sources = ["company", "dnngp"] if args.source == "all" else [args.source]

    print("=" * 60)
    print("RAGAS 测试数据生成器")
    print(f"数据源: {sources}")
    print(f"每个数据源生成问题数: {args.count}")
    print("=" * 60)

    saved_paths = []
    for source in sources:
        try:
            path = generate_for_source(source, args.count)
            if path:
                saved_paths.append(path)
        except FileNotFoundError as e:
            print(f"\n错误: {e}")
        except ValueError as e:
            print(f"\n数据生成错误: {e}")
        except Exception as e:
            print(f"\n未预期的错误 ({source}): {e}")

    print(f"\n{'=' * 60}")
    if saved_paths:
        print("生成完成！输出文件:")
        for p in saved_paths:
            print(f"  {p}")
    else:
        print("未能生成任何测试数据。")
    print("=" * 60)


if __name__ == "__main__":
    main()
