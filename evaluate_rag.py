"""
RAGAS Evaluation Runner
RAG 系统评估运行脚本

使用方法:
    # 单次评估
    python evaluate_rag.py --source company --retriever vec
    python evaluate_rag.py --source company --retriever vec --model deepseek

    # A/B 对比：baseline vs optimized
    python evaluate_rag.py --source company --retriever vec --mode compare
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_test_data(source: str) -> list:
    """加载测试数据"""
    data_dir = os.path.join(os.path.dirname(__file__), "evaluation_test_data")

    if source == "company":
        path = os.path.join(data_dir, "company_test_data.json")
    elif source == "dnngp":
        path = os.path.join(data_dir, "dnngp_test_data.json")
    else:
        raise ValueError(f"未知数据源: {source}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_baseline_vectorstore(embeddings):
    """构建 baseline 向量库：简单递归分块，无元信息增强"""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_milvus import Milvus
    from pymilvus import connections, utility, Collection

    current_dir = os.path.dirname(os.path.abspath(__file__))
    company_txt_path = os.path.join(current_dir, "doc", "company.txt")
    dnngp_pdf_path = os.path.join(current_dir, "pdf",
        "DNNGP, a deep neural network-based method for genomic prediction using multi-omics data in plants(简短版）.pdf")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", "。", ".", "!", "?", "；", "，", " ", ""]
    )

    all_documents = []

    # 处理 company.txt
    if os.path.exists(company_txt_path):
        with open(company_txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        texts = splitter.split_text(content)
        for text in texts:
            all_documents.append(Document(
                page_content=text,
                metadata={"source": company_txt_path, "chunk_type": "text"}
            ))
        print(f"   📄 company.txt → {len(texts)} 个块（递归分块）")

    # 处理 PDF（简化：只提取文本）
    if os.path.exists(dnngp_pdf_path):
        try:
            from app.pdf_extractor import PDFExtractor
            pdf_extractor = PDFExtractor(enable_tables=False)
            pdf_text, _ = pdf_extractor.extract_text(dnngp_pdf_path, method='auto')
            texts = splitter.split_text(pdf_text)
            for text in texts:
                all_documents.append(Document(
                    page_content=text,
                    metadata={"source": dnngp_pdf_path, "chunk_type": "text"}
                ))
            print(f"   📄 DNNGP.pdf → {len(texts)} 个块（递归分块）")
        except Exception as e:
            print(f"   ⚠ PDF 处理失败: {e}")

    if not all_documents:
        raise RuntimeError("无文档可索引")

    # 构建 Milvus 向量库
    connections.connect("default", host="localhost", port="19530", timeout=30)
    collection_name = "baseline_milvus"
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()

    vectorstore = Milvus.from_documents(
        documents=all_documents,
        collection_name=collection_name,
        embedding=embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        drop_old=False,
        enable_dynamic_field=True,
        index_params={
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX",
        },
    )
    print(f"   ✅ Baseline 向量库构建完成，共 {len(all_documents)} 个文档块")
    return vectorstore


def get_retriever(retriever_type: str):
    """获取检索器（optimized 模式）"""
    from app.rag import init_rag, get_vectorstore, get_cypher_chain
    init_rag()
    vectorstore = get_vectorstore()
    cypher_chain = get_cypher_chain()

    if retriever_type == "vec":
        if vectorstore is None:
            raise RuntimeError("Milvus 向量存储未初始化，请检查 Milvus 服务")
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    elif retriever_type == "graph":
        if cypher_chain is None:
            raise RuntimeError("Neo4j 图数据库未初始化，请检查 Neo4j 服务")
        print("⚠ GraphCypherQAChain 不支持标准检索接口，将使用预定义上下文")
        return None
    else:
        raise ValueError(f"未知检索器类型: {retriever_type}")


def run_single_evaluation(evaluator, retriever, sources, retriever_type):
    """运行单次评估"""
    all_results = []
    for source in sources:
        print(f"\n{'#'*60}")
        print(f"# 评估数据源: {source}")
        print(f"{'#'*60}")

        test_data = load_test_data(source)
        result = evaluator.evaluate(
            test_data=test_data,
            dataset_name=source,
            retriever=retriever,
            retriever_type=retriever_type,
        )
        all_results.append(result)
    return all_results


def print_comparison(baseline_results, optimized_results):
    """打印对比结果"""
    print("\n" + "=" * 70)
    print("📊 Baseline vs Optimized 对比结果")
    print("=" * 70)

    for b_res, o_res in zip(baseline_results, optimized_results):
        print(f"\n数据源: {b_res.dataset_name}")
        print("-" * 60)
        print(f"{'指标':<25} {'Baseline':>12} {'Optimized':>12} {'提升':>10}")
        print("-" * 60)

        b_metrics = b_res.metrics
        o_metrics = o_res.metrics
        for key in b_metrics:
            b_val = b_metrics[key]
            o_val = o_metrics[key]
            diff = o_val - b_val
            sign = "+" if diff >= 0 else ""
            print(f"{key:<25} {b_val:>12.4f} {o_val:>12.4f} {sign}{diff:>9.4f}")

        b_overall = sum(b_metrics.values()) / len(b_metrics)
        o_overall = sum(o_metrics.values()) / len(o_metrics)
        diff = o_overall - b_overall
        sign = "+" if diff >= 0 else ""
        print("-" * 60)
        print(f"{'overall_score':<25} {b_overall:>12.4f} {o_overall:>12.4f} {sign}{diff:>9.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RAGAS RAG 评估工具")
    parser.add_argument(
        "--source",
        choices=["company", "dnngp", "all"],
        default="all",
        help="评估数据源 (default: all)"
    )
    parser.add_argument(
        "--retriever",
        choices=["vec", "graph"],
        default="vec",
        help="检索器类型 (default: vec)"
    )
    parser.add_argument(
        "--model",
        choices=["dashscope", "deepseek"],
        default="dashscope",
        help="LLM 模型 (default: dashscope)"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized", "compare"],
        default="optimized",
        help="评估模式: baseline(无优化), optimized(有优化), compare(对比两者) (default: optimized)"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results",
        help="结果输出目录 (default: evaluation_results)"
    )
    args = parser.parse_args()

    # 导入评估模块
    from app.evaluation import RAGASEvaluator, generate_report
    from app.config import llm, embeddings, deepseek_llm

    # 选择 LLM
    if args.model == "deepseek":
        if deepseek_llm is None:
            print("❌ DeepSeek 模型未配置，请检查 .env 中的 SENSENOVA_API_KEY")
            return
        eval_llm = deepseek_llm
        print("🤖 使用模型: SenseNova DeepSeek (deepseek-v4-flash)")
    else:
        eval_llm = llm
        print("🤖 使用模型: DashScope (qwen-plus)")

    # 初始化评估器
    evaluator = RAGASEvaluator(llm=eval_llm, embeddings=embeddings)

    # 确定要评估的数据源
    sources = ["company", "dnngp"] if args.source == "all" else [args.source]

    if args.mode == "compare":
        # ===== A/B 对比模式 =====
        print("\n🔬 运行 A/B 对比评估...")

        # Baseline
        print("\n" + "=" * 50)
        print("📌 Phase 1: Baseline（递归分块，无元信息）")
        print("=" * 50)
        baseline_vs = build_baseline_vectorstore(embeddings)
        baseline_retriever = baseline_vs.as_retriever(search_kwargs={"k": 5})
        baseline_results = run_single_evaluation(evaluator, baseline_retriever, sources, args.retriever)

        # Optimized
        print("\n" + "=" * 50)
        print("📌 Phase 2: Optimized（语义分块 + 元信息增强）")
        print("=" * 50)
        optimized_retriever = get_retriever(args.retriever)
        optimized_results = run_single_evaluation(evaluator, optimized_retriever, sources, args.retriever)

        # 对比
        print_comparison(baseline_results, optimized_results)

        # 保存对比报告
        report = {
            "comparison": "baseline vs optimized",
            "model": args.model,
            "retriever": args.retriever,
            "results": []
        }
        for b_res, o_res in zip(baseline_results, optimized_results):
            report["results"].append({
                "source": b_res.dataset_name,
                "baseline": b_res.metrics,
                "optimized": o_res.metrics,
            })

        os.makedirs(args.output, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(args.output, f"comparison_{ts}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 对比报告已保存至: {report_path}")

    else:
        # ===== 单次评估模式 =====
        if args.mode == "baseline":
            print("\n📌 模式: Baseline（递归分块，无元信息）")
            baseline_vs = build_baseline_vectorstore(embeddings)
            retriever = baseline_vs.as_retriever(search_kwargs={"k": 5})
        else:
            print(f"\n🔍 初始化检索器: {args.retriever}")
            retriever = get_retriever(args.retriever)

        all_results = run_single_evaluation(evaluator, retriever, sources, args.retriever)
        report_path = generate_report(all_results, output_dir=args.output)
        print(f"\n✅ 评估完成！报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
