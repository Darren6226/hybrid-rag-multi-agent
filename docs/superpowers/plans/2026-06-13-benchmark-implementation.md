# Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a benchmark suite that runs 5 optimization comparison experiments on the RAG system and generates quantifiable metrics for job interview use.

**Architecture:** Two standalone scripts — `generate_test_data.py` (LLM-powered test question generation) and `benchmark.py` (5 experiments + report generation). Both scripts live at project root, reuse existing `app/` modules, and write output to `benchmark_data/` and `benchmark_results/`.

**Tech Stack:** Python 3.9, DashScope (LLM + Rerank API), Milvus, RAGAS, rank_bm25, langchain

---

## File Structure

```
Create:
  generate_test_data.py         # Test data generation (LLM batch generation)
  benchmark.py                  # Main benchmark script (5 experiments + reports)
  benchmark_data/               # Generated test data directory
  benchmark_results/            # Output directory

Modify: None (all app/ files untouched)

Dependencies:
  pip install rank_bm25         # For hybrid retrieval experiment
```

---

## Task 1: Install Dependency

- [ ] **Step 1: Install rank_bm25**

Run: `pip install rank_bm25`
Expected: Successfully installed rank_bm25

- [ ] **Step 2: Verify import**

Run: `python -c "from rank_bm25 import BM25Okapi; print('OK')"`
Expected: `OK`

---

## Task 2: Create generate_test_data.py

**Files:**
- Create: `generate_test_data.py`

- [ ] **Step 1: Write the script skeleton with CLI and document loading**

```python
"""
测试数据生成器 — 用 LLM 批量生成 RAGAS 评估题目

用法:
    python generate_test_data.py --source dnngp --count 30
    python generate_test_data.py --source company --count 20
    python generate_test_data.py --source all
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_document(source: str) -> str:
    """加载源文档全文"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if source == "company":
        path = os.path.join(current_dir, "doc", "company.txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif source == "dnngp":
        from app.pdf_extractor import PDFExtractor
        pdf_path = os.path.join(current_dir, "pdf",
            "DNNGP, a deep neural network-based method for genomic prediction using multi-omics data in plants(简短版）.pdf")
        extractor = PDFExtractor(enable_tables=False)
        text, _ = extractor.extract_text(pdf_path, method="auto")
        return text
    else:
        raise ValueError(f"未知数据源: {source}")


def generate_questions(document_text: str, source: str, count: int) -> list:
    """调 LLM 生成测试题目"""
    from app.config import llm

    # 截断过长的文档（避免超出 LLM 上下文窗口）
    max_chars = 15000
    if len(document_text) > max_chars:
        document_text = document_text[:max_chars]

    prompt = f"""你是一个 RAG 系统评估专家。请基于以下文档内容，生成 {count} 个高质量的问答对，用于评估检索增强生成（RAG）系统的效果。

要求：
1. 问题要多样化，覆盖文档的不同部分和主题
2. 问题难度要有梯度：简单（直接查找）、中等（需要理解）、困难（需要推理）
3. ground_truth 必须完全基于文档内容，不要编造
4. 每个问题附带 1-2 个相关的参考文档片段（contexts）
5. 问题用中文，ground_truth 用中文

文档内容：
{document_text}

请严格按以下 JSON 格式输出（不要输出其他内容）：
```json
[
  {{
    "question": "问题内容",
    "ground_truth": "标准答案",
    "contexts": ["参考文档片段1", "参考文档片段2"]
  }}
]
```"""

    response = llm.invoke(prompt)
    content = response.content

    # 提取 JSON 部分
    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"LLM 输出中未找到 JSON 数组: {content[:200]}")

    return json.loads(content[start:end])


def save_test_data(data: list, source: str):
    """保存测试数据到 JSON 文件"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{source}_test_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 保存 {len(data)} 道题目到 {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="生成 RAGAS 评估测试数据")
    parser.add_argument("--source", choices=["dnngp", "company", "all"], default="all",
                        help="数据源（默认: all）")
    parser.add_argument("--count", type=int, default=30,
                        help="每个数据源生成的题目数量（默认: 30）")
    args = parser.parse_args()

    sources = ["dnngp", "company"] if args.source == "all" else [args.source]

    for source in sources:
        count = args.count if args.source != "all" else (30 if source == "dnngp" else 20)
        print(f"\n{'='*50}")
        print(f"📝 生成 {source} 测试数据（{count} 题）")
        print(f"{'='*50}")

        doc_text = load_document(source)
        print(f"📄 文档长度: {len(doc_text)} 字符")

        questions = generate_questions(doc_text, source, count)
        save_test_data(questions, source)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script runs without syntax errors**

Run: `python -c "import generate_test_data; print('OK')"`
Expected: `OK` (no import errors)

---

## Task 3: Create benchmark.py — CLI and Utilities

**Files:**
- Create: `benchmark.py`

- [ ] **Step 1: Write the CLI skeleton, data loading, and common utilities**

```python
"""
RAG 系统多维度优化 Benchmark

用法:
    python benchmark.py                          # 跑全部实验
    python benchmark.py --only chunking          # 只跑分块策略对比
    python benchmark.py --only top_k             # 只跑检索参数对比
    python benchmark.py --only hybrid            # 只跑混合检索对比
    python benchmark.py --only rerank            # 只跑 Reranking 对比
    python benchmark.py --only routing           # 只跑路由准确率
    python benchmark.py --dataset dnngp          # 只用 DNNGP 数据集
    python benchmark.py --dataset company        # 只用 company 数据集
"""

import argparse
import json
import os
import sys
import time
import math
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# 工具函数
# ============================================================================

def load_test_data(dataset: str) -> list:
    """加载 RAGAS 测试数据"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_data")
    path = os.path.join(data_dir, f"{dataset}_test_data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"测试数据不存在: {path}，请先运行 generate_test_data.py")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_source_documents(source: str) -> list:
    """加载源文档并返回 Document 列表"""
    from langchain_core.documents import Document

    current_dir = os.path.dirname(os.path.abspath(__file__))
    documents = []

    if source in ("company", "all"):
        company_path = os.path.join(current_dir, "doc", "company.txt")
        if os.path.exists(company_path):
            with open(company_path, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append(Document(page_content=content, metadata={"source": company_path}))

    if source in ("dnngp", "all"):
        from app.pdf_extractor import PDFExtractor
        pdf_path = os.path.join(current_dir, "pdf",
            "DNNGP, a deep neural network-based method for genomic prediction using multi-omics data in plants(简短版）.pdf")
        if os.path.exists(pdf_path):
            extractor = PDFExtractor(enable_tables=False)
            text, _ = extractor.extract_text(pdf_path, method="auto")
            documents.append(Document(page_content=text, metadata={"source": pdf_path}))

    return documents


def build_vectorstore(documents: list, collection_suffix: str):
    """从文档列表构建 Milvus 向量库（临时 collection）"""
    from langchain_milvus import Milvus
    from pymilvus import connections, utility, Collection
    from app.config import embeddings

    connections.connect("default", host="localhost", port="19530", timeout=30)
    collection_name = f"benchmark_{collection_suffix}"

    # 清理旧 collection
    try:
        if utility.has_collection(collection_name):
            Collection(collection_name).drop()
    except Exception:
        pass

    vs = Milvus.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        drop_old=False,
        enable_dynamic_field=True,
    )
    return vs, collection_name


def cleanup_collection(collection_name: str):
    """删除临时 Milvus collection"""
    try:
        from pymilvus import connections, utility, Collection
        connections.connect("default", host="localhost", port="19530", timeout=30)
        if utility.has_collection(collection_name):
            Collection(collection_name).drop()
    except Exception:
        pass


def run_ragas_evaluation(
    test_data: list,
    retriever,
    dataset_name: str = "benchmark"
) -> Dict[str, float]:
    """
    对给定 retriever 运行 RAGAS 评估，返回指标字典。
    复用 app/evaluation.py 的 RAGASEvaluator。
    """
    from app.evaluation import RAGASEvaluator

    evaluator = RAGASEvaluator()
    result = evaluator.evaluate(
        test_data=test_data,
        dataset_name=dataset_name,
        retriever=retriever,
        retriever_type="vec",
    )
    return result.metrics


def print_table(title: str, rows: List[Dict], columns: List[str], best_col: str = None):
    """打印对比表格到控制台"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    # 表头
    header = f"{'配置':<25}" + "".join(f"{col:<15}" for col in columns)
    print(header)
    print("-" * 70)

    # 找最优值（用于标记）
    best_values = {}
    if best_col:
        for col in columns:
            values = [r.get(col, 0) for r in rows if isinstance(r.get(col, 0), (int, float))]
            best_values[col] = max(values) if values else 0

    # 数据行
    for row in rows:
        label = row.get("label", "")
        line = f"{label:<25}"
        for col in columns:
            val = row.get(col, "N/A")
            if isinstance(val, float):
                marker = " ★" if best_col and col in best_values and val == best_values[col] else ""
                line += f"{val:<13.4f}{marker:>2}"
            else:
                line += f"{str(val):<15}"
        print(line)

    print()


def save_results(results: dict, experiment_name: str):
    """保存实验结果到 JSON"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{experiment_name}_{timestamp}.json"
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"💾 结果已保存: {path}")
    return path
```

- [ ] **Step 2: Verify no import errors**

Run: `python -c "import benchmark; print('OK')"`
Expected: `OK`

---

## Task 4: Implement Experiment 1 — Chunking Strategy

**Files:**
- Modify: `benchmark.py` (add `run_experiment_chunking` function)

- [ ] **Step 1: Add the chunking experiment function to benchmark.py**

Append after the utility functions:

```python
# ============================================================================
# 实验 1: 分块策略对比
# ============================================================================

def run_experiment_chunking(dataset: str, test_data: list) -> dict:
    """
    对比 basic / semantic_metadata / parent_child 三种分块策略。
    控制变量：同一文档、top_k=5、同一 LLM。
    """
    from app.enhanced_chunking import create_enhanced_chunker
    from app.config import embeddings

    print(f"\n{'#'*70}")
    print(f"  实验 1: 分块策略对比 — {dataset}")
    print(f"{'#'*70}")

    # 加载源文档
    raw_docs = load_source_documents(dataset)
    full_text = "\n".join(doc.page_content for doc in raw_docs)
    source_path = raw_docs[0].metadata.get("source", "unknown") if raw_docs else "unknown"

    strategies = {
        "basic": {"strategy": "basic", "max_chunk_size": 400, "min_chunk_size": 200},
        "semantic_metadata": {"strategy": "semantic_metadata", "max_chunk_size": 400, "min_chunk_size": 200},
        "parent_child": {"strategy": "parent_child", "max_chunk_size": 400, "min_chunk_size": 200},
    }

    results = []
    all_metrics = {}

    for name, params in strategies.items():
        print(f"\n🔧 策略: {name}")

        # 1. 分块
        chunker = create_enhanced_chunker(**params)
        chunks = chunker.chunk_document(full_text, source_path)
        chunk_sizes = [len(c.page_content) for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

        print(f"   📊 生成 {len(chunks)} 个块，平均大小 {avg_size:.0f} 字符")

        # 2. 建 vectorstore
        vs, collection_name = build_vectorstore(chunks, f"{dataset}_{name}_{int(time.time())}")

        try:
            # 3. 构建 retriever
            retriever = vs.as_retriever(search_kwargs={"k": 5})

            # 4. 跑 RAGAS
            metrics = run_ragas_evaluation(test_data, retriever, f"{dataset}_{name}")

            row = {
                "label": name,
                "chunk_count": len(chunks),
                "avg_chunk_size": round(avg_size, 1),
                **metrics,
            }
            results.append(row)
            all_metrics[name] = metrics

            print(f"   ✅ RAGAS: Faithfulness={metrics.get('faithfulness', 0):.4f} "
                  f"Relevancy={metrics.get('answer_relevancy', 0):.4f} "
                  f"Precision={metrics.get('context_precision', 0):.4f} "
                  f"Recall={metrics.get('context_recall', 0):.4f}")

        finally:
            cleanup_collection(collection_name)

    # 打印对比表
    columns = ["chunk_count", "avg_chunk_size", "faithfulness", "answer_relevancy",
               "context_precision", "context_recall", "overall_score"]
    print_table(f"分块策略对比 ({dataset})", results, columns, best_col="context_recall")

    return {"experiment": "chunking", "dataset": dataset, "results": results}
```

- [ ] **Step 2: Verify function is syntactically correct**

Run: `python -c "from benchmark import run_experiment_chunking; print('OK')"`
Expected: `OK`

---

## Task 5: Implement Experiment 2 — Top-K Parameter

**Files:**
- Modify: `benchmark.py` (add `run_experiment_topk` function)

- [ ] **Step 1: Add the top_k experiment function**

Append after the chunking experiment:

```python
# ============================================================================
# 实验 2: 检索参数对比（top_k）
# ============================================================================

def run_experiment_topk(dataset: str, test_data: list) -> dict:
    """
    对比 top_k=3/5/8 的检索效果。
    控制变量：semantic_metadata 分块、同一文档。
    """
    from app.enhanced_chunking import create_enhanced_chunker

    print(f"\n{'#'*70}")
    print(f"  实验 2: 检索参数对比 (top_k) — {dataset}")
    print(f"{'#'*70}")

    # 用 semantic_metadata 分块（最优策略）
    raw_docs = load_source_documents(dataset)
    full_text = "\n".join(doc.page_content for doc in raw_docs)
    source_path = raw_docs[0].metadata.get("source", "unknown") if raw_docs else "unknown"

    chunker = create_enhanced_chunker(strategy="semantic_metadata", max_chunk_size=400, min_chunk_size=200)
    chunks = chunker.chunk_document(full_text, source_path)

    vs, collection_name = build_vectorstore(chunks, f"{dataset}_topk_{int(time.time())}")

    try:
        results = []
        for top_k in [3, 5, 8]:
            print(f"\n🔧 top_k = {top_k}")

            retriever = vs.as_retriever(search_kwargs={"k": top_k})

            # 测量检索延迟
            latencies = []
            for q in test_data:
                start = time.time()
                retriever.invoke(q["question"])
                latencies.append((time.time() - start) * 1000)
            avg_latency = sum(latencies) / len(latencies)

            # 跑 RAGAS
            metrics = run_ragas_evaluation(test_data, retriever, f"{dataset}_topk{top_k}")

            row = {
                "label": f"top_k={top_k}",
                "avg_latency_ms": round(avg_latency, 1),
                **metrics,
            }
            results.append(row)

            print(f"   ✅ 延迟={avg_latency:.1f}ms RAGAS: Faithfulness={metrics.get('faithfulness', 0):.4f} "
                  f"Relevancy={metrics.get('answer_relevancy', 0):.4f} "
                  f"Precision={metrics.get('context_precision', 0):.4f} "
                  f"Recall={metrics.get('context_recall', 0):.4f}")

        columns = ["avg_latency_ms", "faithfulness", "answer_relevancy",
                    "context_precision", "context_recall", "overall_score"]
        print_table(f"Top-K 参数对比 ({dataset})", results, columns, best_col="context_recall")

        return {"experiment": "top_k", "dataset": dataset, "results": results}

    finally:
        cleanup_collection(collection_name)
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "from benchmark import run_experiment_topk; print('OK')"`
Expected: `OK`

---

## Task 6: Implement Experiment 3 — Hybrid Retrieval

**Files:**
- Modify: `benchmark.py` (add `run_experiment_hybrid` function)

- [ ] **Step 1: Add the hybrid retrieval experiment function**

Append after the top_k experiment:

```python
# ============================================================================
# 实验 3: 混合检索对比（Dense + Sparse）
# ============================================================================

def run_experiment_hybrid(dataset: str, test_data: list) -> dict:
    """
    对比纯向量检索 vs Dense+Sparse 混合检索（RRF 合并）。
    """
    from app.enhanced_chunking import create_enhanced_chunker
    from rank_bm25 import BM25Okapi
    import jieba

    print(f"\n{'#'*70}")
    print(f"  实验 3: 混合检索对比 — {dataset}")
    print(f"{'#'*70}")

    # 分块
    raw_docs = load_source_documents(dataset)
    full_text = "\n".join(doc.page_content for doc in raw_docs)
    source_path = raw_docs[0].metadata.get("source", "unknown") if raw_docs else "unknown"

    chunker = create_enhanced_chunker(strategy="semantic_metadata", max_chunk_size=400, min_chunk_size=200)
    chunks = chunker.chunk_document(full_text, source_path)

    vs, collection_name = build_vectorstore(chunks, f"{dataset}_hybrid_{int(time.time())}")

    try:
        # 建 BM25 索引
        tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        chunk_texts = [doc.page_content for doc in chunks]

        def hybrid_retriever(query: str, k: int = 5, dense_weight: float = 0.7) -> list:
            """混合检索：向量 + BM25，RRF 合并"""
            # Dense 检索
            dense_results = vs.similarity_search(query, k=k * 2)
            dense_docs = {doc.page_content: (i, doc) for i, doc in enumerate(dense_results)}

            # Sparse 检索（BM25）
            tokenized_query = list(jieba.cut(query))
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_top_indices = sorted(range(len(bm25_scores)),
                                       key=lambda i: bm25_scores[i], reverse=True)[:k * 2]
            sparse_docs = {}
            for rank, idx in enumerate(bm25_top_indices):
                sparse_docs[chunk_texts[idx]] = (rank, chunks[idx])

            # RRF 合并
            all_contents = set(dense_docs.keys()) | set(sparse_docs.keys())
            scored = []
            rrf_k = 60
            for content in all_contents:
                rrf_score = 0
                if content in dense_docs:
                    rrf_score += dense_weight / (rrf_k + dense_docs[content][0] + 1)
                if content in sparse_docs:
                    rrf_score += (1 - dense_weight) / (rrf_k + sparse_docs[content][0] + 1)
                doc = dense_docs.get(content, sparse_docs.get(content, (0, None)))[1]
                scored.append((rrf_score, doc))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:k]]

        # 对每个问题运行混合检索并收集结果
        class HybridRetriever:
            def __init__(self, k=5):
                self.k = k
            def invoke(self, query):
                return hybrid_retriever(query, self.k)

        class VectorRetriever:
            def __init__(self, k=5):
                self.retriever = vs.as_retriever(search_kwargs={"k": k})
            def invoke(self, query):
                return self.retriever.invoke(query)

        # A: 纯向量
        print("\n🔧 A: 纯向量检索")
        vec_retriever = VectorRetriever(k=5)
        vec_metrics = run_ragas_evaluation(test_data, vec_retriever, f"{dataset}_vec_only")
        print(f"   ✅ Faithfulness={vec_metrics.get('faithfulness', 0):.4f} "
              f"Relevancy={vec_metrics.get('answer_relevancy', 0):.4f} "
              f"Precision={vec_metrics.get('context_precision', 0):.4f} "
              f"Recall={vec_metrics.get('context_recall', 0):.4f}")

        # B: 混合检索
        print("\n🔧 B: 混合检索 (Dense 0.7 + BM25 0.3)")
        hybrid_ret = HybridRetriever(k=5)
        hybrid_metrics = run_ragas_evaluation(test_data, hybrid_ret, f"{dataset}_hybrid")
        print(f"   ✅ Faithfulness={hybrid_metrics.get('faithfulness', 0):.4f} "
              f"Relevancy={hybrid_metrics.get('answer_relevancy', 0):.4f} "
              f"Precision={hybrid_metrics.get('context_precision', 0):.4f} "
              f"Recall={hybrid_metrics.get('context_recall', 0):.4f}")

        results = [
            {"label": "纯向量", **vec_metrics},
            {"label": "混合检索", **hybrid_metrics},
        ]

        columns = ["faithfulness", "answer_relevancy", "context_precision",
                    "context_recall", "overall_score"]
        print_table(f"混合检索对比 ({dataset})", results, columns, best_col="context_recall")

        return {"experiment": "hybrid", "dataset": dataset, "results": results}

    finally:
        cleanup_collection(collection_name)
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "from benchmark import run_experiment_hybrid; print('OK')"`
Expected: `OK`

---

## Task 7: Implement Experiment 4 — Reranking

**Files:**
- Modify: `benchmark.py` (add `run_experiment_rerank` function)

- [ ] **Step 1: Add the reranking experiment function**

Append after the hybrid experiment:

```python
# ============================================================================
# 实验 4: Reranking 对比
# ============================================================================

def run_experiment_rerank(dataset: str, test_data: list) -> dict:
    """
    对比无 rerank vs DashScope gte-rerank 重排序。
    """
    from app.enhanced_chunking import create_enhanced_chunker

    print(f"\n{'#'*70}")
    print(f"  实验 4: Reranking 对比 — {dataset}")
    print(f"{'#'*70}")

    # 分块
    raw_docs = load_source_documents(dataset)
    full_text = "\n".join(doc.page_content for doc in raw_docs)
    source_path = raw_docs[0].metadata.get("source", "unknown") if raw_docs else "unknown"

    chunker = create_enhanced_chunker(strategy="semantic_metadata", max_chunk_size=400, min_chunk_size=200)
    chunks = chunker.chunk_document(full_text, source_path)

    vs, collection_name = build_vectorstore(chunks, f"{dataset}_rerank_{int(time.time())}")

    try:
        # A: 无 rerank（baseline）
        print("\n🔧 A: 无 Rerank (top_k=5)")
        vec_retriever = vs.as_retriever(search_kwargs={"k": 5})
        vec_metrics = run_ragas_evaluation(test_data, vec_retriever, f"{dataset}_no_rerank")
        print(f"   ✅ Faithfulness={vec_metrics.get('faithfulness', 0):.4f} "
              f"Relevancy={vec_metrics.get('answer_relevancy', 0):.4f} "
              f"Precision={vec_metrics.get('context_precision', 0):.4f} "
              f"Recall={vec_metrics.get('context_recall', 0):.4f}")

        # B: 有 rerank
        print("\n🔧 B: Rerank (recall top_8 → rerank → top_5)")
        try:
            import dashscope
            from dashscope import TextRerank

            class RerankRetriever:
                def __init__(self, vs, top_k=5, recall_k=8):
                    self.vs = vs
                    self.top_k = top_k
                    self.recall_k = recall_k

                def invoke(self, query):
                    # 召回阶段
                    docs = self.vs.similarity_search(query, k=self.recall_k)
                    if not docs:
                        return docs

                    # 精排阶段
                    documents = [doc.page_content for doc in docs]
                    result = TextRerank.call(
                        model="gte-rerank-v2",
                        top_n=self.top_k,
                        query=query,
                        documents=documents,
                    )

                    if result.status_code != 200:
                        print(f"   ⚠ Rerank API 失败: {result.message}，使用原始排序")
                        return docs[:self.top_k]

                    # 按 rerank 结果重排
                    reranked_indices = [item.index for item in result.output.results]
                    return [docs[i] for i in reranked_indices if i < len(docs)]

            rerank_retriever = RerankRetriever(vs, top_k=5, recall_k=8)
            rerank_metrics = run_ragas_evaluation(test_data, rerank_retriever, f"{dataset}_rerank")
            print(f"   ✅ Faithfulness={rerank_metrics.get('faithfulness', 0):.4f} "
                  f"Relevancy={rerank_metrics.get('answer_relevancy', 0):.4f} "
                  f"Precision={rerank_metrics.get('context_precision', 0):.4f} "
                  f"Recall={rerank_metrics.get('context_recall', 0):.4f}")

        except Exception as e:
            print(f"   ⚠ Rerank 实验失败: {e}")
            rerank_metrics = {"faithfulness": 0, "answer_relevancy": 0,
                              "context_precision": 0, "context_recall": 0, "overall_score": 0}

        results = [
            {"label": "无 Rerank", **vec_metrics},
            {"label": "有 Rerank", **rerank_metrics},
        ]

        columns = ["faithfulness", "answer_relevancy", "context_precision",
                    "context_recall", "overall_score"]
        print_table(f"Reranking 对比 ({dataset})", results, columns, best_col="context_recall")

        return {"experiment": "rerank", "dataset": dataset, "results": results}

    finally:
        cleanup_collection(collection_name)
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "from benchmark import run_experiment_rerank; print('OK')"`
Expected: `OK`

---

## Task 8: Implement Experiment 5 — Routing Accuracy

**Files:**
- Modify: `benchmark.py` (add `run_experiment_routing` function)

- [ ] **Step 1: Create the routing test data file manually**

```bash
mkdir -p benchmark_data
```

Then create `benchmark_data/routing_test_data.json` with 25 questions (5 per worker type + 5 mixed):

```json
[
  {"question": "查询小米科技的总销售额", "expected_worker": "sqler"},
  {"question": "哪些客户购买了华为的产品", "expected_worker": "sqler"},
  {"question": "2024年三星的销售额是多少", "expected_worker": "sqler"},
  {"question": "比亚迪的销量排名", "expected_worker": "sqler"},
  {"question": "查询所有公司的利润率", "expected_worker": "sqler"},
  {"question": "小米有哪些技术", "expected_worker": "graph_kg"},
  {"question": "华为的合作伙伴有哪些", "expected_worker": "graph_kg"},
  {"question": "苹果在哪些地区有业务", "expected_worker": "graph_kg"},
  {"question": "小米和华为有什么合作", "expected_worker": "graph_kg"},
  {"question": "三星的产品有哪些", "expected_worker": "graph_kg"},
  {"question": "什么是深度学习", "expected_worker": "vec_kg"},
  {"question": "Transformer架构的原理是什么", "expected_worker": "vec_kg"},
  {"question": "RAG系统有哪些优化方法", "expected_worker": "vec_kg"},
  {"question": "向量数据库的工作原理", "expected_worker": "vec_kg"},
  {"question": "什么是语义分块", "expected_worker": "vec_kg"},
  {"question": "你好", "expected_worker": "chat"},
  {"question": "请总结一下前面的对话", "expected_worker": "chat"},
  {"question": "谢谢你的帮助", "expected_worker": "chat"},
  {"question": "再见", "expected_worker": "chat"},
  {"question": "你能做什么", "expected_worker": "chat"},
  {"question": "小米的快充技术和华为的5G技术哪个更厉害", "expected_worker": "sqler"},
  {"question": "对比苹果和三星的AI战略", "expected_worker": "vec_kg"},
  {"question": "比亚迪的合作伙伴在哪些地区", "expected_worker": "graph_kg"},
  {"question": "总结一下小米的技术创新", "expected_worker": "chat"},
  {"question": "华为的5G技术用了什么芯片", "expected_worker": "graph_kg"}
]
```

- [ ] **Step 2: Add the routing accuracy experiment function**

Append after the rerank experiment:

```python
# ============================================================================
# 实验 5: 路由准确率
# ============================================================================

def run_experiment_routing(dataset: str, test_data: list) -> dict:
    """
    对比有/无 cycle detection 的路由准确率。
    """
    from app.graph_builder import build_graph
    from app.database import init_seed_data
    from app.rag import init_rag

    print(f"\n{'#'*70}")
    print(f"  实验 5: 路由准确率")
    print(f"{'#'*70}")

    # 初始化系统
    init_seed_data()
    init_rag()

    graph = build_graph(skip_graph_kg=False)

    # 收集所有路由决策
    all_decisions = []

    for i, sample in enumerate(test_data):
        question = sample["question"]
        expected = sample["expected_worker"]

        print(f"\n   [{i+1}/{len(test_data)}] Q: {question[:40]}...")
        print(f"         Expected: {expected}")

        try:
            # 运行 graph 并捕获路由决策
            first_routing = None
            total_rounds = 0
            final_answer = None

            for chunk in graph.stream({"messages": question}, stream_mode="values"):
                messages = chunk.get("messages", [])
                if not messages:
                    continue

                last_msg = messages[-1]
                # 捕获 AIMessage 中的路由决策
                if hasattr(last_msg, "content") and last_msg.content:
                    content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
                    # Supervisor 输出中包含路由信息
                    for worker in ["sqler", "graph_kg", "vec_kg", "coder", "chat"]:
                        if worker in content.lower():
                            if first_routing is None:
                                first_routing = worker
                            total_rounds += 1
                            break

            is_correct = first_routing == expected
            print(f"         Got: {first_routing} | {'✅' if is_correct else '❌'}")

            all_decisions.append({
                "question": question,
                "expected": expected,
                "got": first_routing,
                "correct": is_correct,
                "rounds": total_rounds,
            })

        except Exception as e:
            print(f"         ❌ 错误: {e}")
            all_decisions.append({
                "question": question,
                "expected": expected,
                "got": None,
                "correct": False,
                "rounds": 0,
                "error": str(e),
            })

    # 统计
    total = len(all_decisions)
    correct = sum(1 for d in all_decisions if d["correct"])
    avg_rounds = sum(d["rounds"] for d in all_decisions) / total if total else 0
    accuracy = correct / total if total else 0

    # 按 worker 类型统计
    worker_stats = {}
    for d in all_decisions:
        w = d["expected"]
        if w not in worker_stats:
            worker_stats[w] = {"total": 0, "correct": 0}
        worker_stats[w]["total"] += 1
        if d["correct"]:
            worker_stats[w]["correct"] += 1

    print(f"\n{'='*70}")
    print(f"  路由准确率统计")
    print(f"{'='*70}")
    print(f"  总体准确率: {correct}/{total} = {accuracy:.1%}")
    print(f"  平均路由轮次: {avg_rounds:.1f}")
    print(f"\n  按 Worker 类型:")
    for w, stats in sorted(worker_stats.items()):
        w_acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"    {w:<15} {stats['correct']}/{stats['total']} = {w_acc:.1%}")

    result = {
        "experiment": "routing",
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_rounds": avg_rounds,
        "worker_stats": worker_stats,
        "details": all_decisions,
    }

    save_results(result, "routing")
    return result
```

- [ ] **Step 3: Verify syntax**

Run: `python -c "from benchmark import run_experiment_routing; print('OK')"`
Expected: `OK`

---

## Task 9: Implement Report Generation and Main Entry Point

**Files:**
- Modify: `benchmark.py` (add `generate_report` and `main` functions)

- [ ] **Step 1: Add the report generator and main function**

Append to benchmark.py:

```python
# ============================================================================
# 报告生成
# ============================================================================

def generate_report(all_results: dict):
    """生成 Markdown 对比报告"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(output_dir, "benchmark_report.md")

    lines = [
        "# RAG 系统优化对比报告",
        f"\n生成时间: {timestamp}",
        "",
    ]

    # 实验 1: 分块策略
    if "chunking" in all_results:
        r = all_results["chunking"]
        lines.append("## 实验 1: 分块策略对比")
        lines.append(f"**数据集:** {r['dataset']}")
        lines.append("")
        lines.append("| 策略 | Chunk数 | 平均大小 | Faithfulness | Relevancy | Precision | Recall |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in r["results"]:
            lines.append(f"| {row['label']} | {row.get('chunk_count', '-')} | "
                        f"{row.get('avg_chunk_size', '-')} | "
                        f"{row.get('faithfulness', 0):.4f} | "
                        f"{row.get('answer_relevancy', 0):.4f} | "
                        f"{row.get('context_precision', 0):.4f} | "
                        f"{row.get('context_recall', 0):.4f} |")
        lines.append("")

    # 实验 2: top_k
    if "top_k" in all_results:
        r = all_results["top_k"]
        lines.append("## 实验 2: 检索参数对比 (top_k)")
        lines.append(f"**数据集:** {r['dataset']}")
        lines.append("")
        lines.append("| top_k | 延迟(ms) | Faithfulness | Relevancy | Precision | Recall |")
        lines.append("|---|---|---|---|---|---|")
        for row in r["results"]:
            lines.append(f"| {row['label']} | {row.get('avg_latency_ms', '-')} | "
                        f"{row.get('faithfulness', 0):.4f} | "
                        f"{row.get('answer_relevancy', 0):.4f} | "
                        f"{row.get('context_precision', 0):.4f} | "
                        f"{row.get('context_recall', 0):.4f} |")
        lines.append("")

    # 实验 3: 混合检索
    if "hybrid" in all_results:
        r = all_results["hybrid"]
        lines.append("## 实验 3: 混合检索对比")
        lines.append(f"**数据集:** {r['dataset']}")
        lines.append("")
        lines.append("| 检索方式 | Faithfulness | Relevancy | Precision | Recall |")
        lines.append("|---|---|---|---|---|")
        for row in r["results"]:
            lines.append(f"| {row['label']} | "
                        f"{row.get('faithfulness', 0):.4f} | "
                        f"{row.get('answer_relevancy', 0):.4f} | "
                        f"{row.get('context_precision', 0):.4f} | "
                        f"{row.get('context_recall', 0):.4f} |")
        lines.append("")

    # 实验 4: Rerank
    if "rerank" in all_results:
        r = all_results["rerank"]
        lines.append("## 实验 4: Reranking 对比")
        lines.append(f"**数据集:** {r['dataset']}")
        lines.append("")
        lines.append("| 配置 | Faithfulness | Relevancy | Precision | Recall |")
        lines.append("|---|---|---|---|---|")
        for row in r["results"]:
            lines.append(f"| {row['label']} | "
                        f"{row.get('faithfulness', 0):.4f} | "
                        f"{row.get('answer_relevancy', 0):.4f} | "
                        f"{row.get('context_precision', 0):.4f} | "
                        f"{row.get('context_recall', 0):.4f} |")
        lines.append("")

    # 实验 5: 路由
    if "routing" in all_results:
        r = all_results["routing"]
        lines.append("## 实验 5: 路由准确率")
        lines.append("")
        lines.append(f"- **总体准确率:** {r['correct']}/{r['total_questions']} = {r['accuracy']:.1%}")
        lines.append(f"- **平均路由轮次:** {r['avg_rounds']:.1f}")
        lines.append("")
        lines.append("| Worker | 准确率 |")
        lines.append("|---|---|")
        for w, stats in sorted(r.get("worker_stats", {}).items()):
            w_acc = stats["correct"] / stats["total"] if stats["total"] else 0
            lines.append(f"| {w} | {stats['correct']}/{stats['total']} = {w_acc:.1%} |")
        lines.append("")

    # 总结
    lines.append("## 总结")
    lines.append("")
    lines.append("| 优化维度 | 最优配置 | 关键指标提升 |")
    lines.append("|---|---|---|")

    if "chunking" in all_results:
        best = max(all_results["chunking"]["results"],
                   key=lambda x: x.get("context_recall", 0))
        lines.append(f"| 分块策略 | {best['label']} | Recall={best.get('context_recall', 0):.4f} |")

    if "top_k" in all_results:
        best = max(all_results["top_k"]["results"],
                   key=lambda x: x.get("context_recall", 0))
        lines.append(f"| 检索参数 | {best['label']} | Recall={best.get('context_recall', 0):.4f} |")

    if "hybrid" in all_results:
        best = max(all_results["hybrid"]["results"],
                   key=lambda x: x.get("context_recall", 0))
        lines.append(f"| 混合检索 | {best['label']} | Recall={best.get('context_recall', 0):.4f} |")

    if "rerank" in all_results:
        best = max(all_results["rerank"]["results"],
                   key=lambda x: x.get("context_recall", 0))
        lines.append(f"| Reranking | {best['label']} | Recall={best.get('context_recall', 0):.4f} |")

    if "routing" in all_results:
        lines.append(f"| 路由策略 | cycle detection | 准确率={all_results['routing']['accuracy']:.1%} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n📄 报告已生成: {report_path}")
    return report_path


# ============================================================================
# 主入口
# ============================================================================

EXPERIMENTS = {
    "chunking": run_experiment_chunking,
    "top_k": run_experiment_topk,
    "hybrid": run_experiment_hybrid,
    "rerank": run_experiment_rerank,
    "routing": run_experiment_routing,
}


def main():
    parser = argparse.ArgumentParser(description="RAG 系统多维度优化 Benchmark")
    parser.add_argument("--only", choices=list(EXPERIMENTS.keys()),
                        help="只跑指定实验")
    parser.add_argument("--dataset", choices=["dnngp", "company", "all"], default="dnngp",
                        help="评估数据集（默认: dnngp）")
    args = parser.parse_args()

    print("=" * 70)
    print("  RAG 系统多维度优化 Benchmark")
    print(f"  数据集: {args.dataset}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}

    if args.only:
        # 单实验模式
        exp_name = args.only
        if exp_name == "routing":
            test_data = load_test_data("routing")
            result = EXPERIMENTS[exp_name](args.dataset, test_data)
        else:
            test_data = load_test_data(args.dataset)
            result = EXPERIMENTS[exp_name](args.dataset, test_data)
        all_results[exp_name] = result
    else:
        # 全部实验
        for exp_name, exp_func in EXPERIMENTS.items():
            try:
                if exp_name == "routing":
                    test_data = load_test_data("routing")
                else:
                    test_data = load_test_data(args.dataset)

                result = exp_func(args.dataset, test_data)
                all_results[exp_name] = result

            except Exception as e:
                print(f"\n❌ 实验 {exp_name} 失败: {e}")
                import traceback
                traceback.print_exc()

    # 生成报告
    if all_results:
        generate_report(all_results)

    print("\n" + "=" * 70)
    print("  Benchmark 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the full script loads without errors**

Run: `python -c "import benchmark; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify CLI help works**

Run: `python benchmark.py --help`
Expected: Shows usage with --only and --dataset options

---

## Task 10: End-to-End Test

- [ ] **Step 1: Verify generate_test_data.py runs (dry check)**

Run: `python -c "from generate_test_data import load_document; doc = load_document('dnngp'); print(f'DNNGP doc length: {len(doc)}')"`
Expected: Shows document length (should be ~71000 chars)

- [ ] **Step 2: Verify benchmark.py loads all experiment functions**

Run: `python -c "from benchmark import EXPERIMENTS; print(list(EXPERIMENTS.keys()))"`
Expected: `['chunking', 'top_k', 'hybrid', 'rerank', 'routing']`

- [ ] **Step 3: Verify routing test data exists**

Run: `python -c "import json; d = json.load(open('benchmark_data/routing_test_data.json')); print(f'{len(d)} routing questions')"`
Expected: `25 routing questions`

---

## Execution Order

1. Task 1: Install rank_bm25
2. Task 2: Create generate_test_data.py
3. Task 3: Create benchmark.py (CLI + utilities)
4. Task 4: Experiment 1 (chunking)
5. Task 5: Experiment 2 (top_k)
6. Task 6: Experiment 3 (hybrid)
7. Task 7: Experiment 4 (rerank)
8. Task 8: Experiment 5 (routing) + routing test data
9. Task 9: Report generation + main entry point
10. Task 10: End-to-end verification
