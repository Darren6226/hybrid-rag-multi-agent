"""
Benchmark Suite for Hybrid RAG Multi-Agent System

Usage:
    python benchmark.py                          # Run all experiments on default dataset
    python benchmark.py --only chunking          # Run single experiment
    python benchmark.py --dataset company        # Use company dataset
    python benchmark.py --dataset all            # Run on both datasets
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for emoji characters (RAGAS library outputs emoji)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = None  # lazy-init after config import


# ============================================================================
# Logging helper (deferred to avoid config side-effects at import time)
# ============================================================================

def _get_logger():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger


# ============================================================================
# CLI Argument Parsing
# ============================================================================

EXPERIMENT_CHOICES = ["chunking", "top_k", "hybrid", "rerank", "routing"]
DATASET_CHOICES = ["dnngp", "company", "all"]


def parse_args(argv=None):
    """Parse command-line arguments for the benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Benchmark suite for Hybrid RAG Multi-Agent System"
    )
    parser.add_argument(
        "--only",
        choices=EXPERIMENT_CHOICES,
        default=None,
        help="Run a single experiment (default: run all)",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="dnngp",
        help="Dataset to evaluate on (default: dnngp)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of test samples per dataset (default: 5, use 0 for all)",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Comma-separated strategy names to skip (e.g. 'basic' or 'basic,semantic_metadata')",
    )
    return parser.parse_args(argv)


# ============================================================================
# Utility Functions
# ============================================================================

def load_test_data(dataset: str) -> list:
    """
    Load RAGAS test data from benchmark_data/<dataset>_test_data.json.
    Falls back to evaluation_test_data/ if benchmark_data/ does not exist.
    Also supports loading routing_test_data.json for the routing experiment.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    log = _get_logger()

    if dataset == "routing":
        # Try benchmark_data first, then evaluation_test_data
        for subdir in ("benchmark_data", "evaluation_test_data"):
            path = os.path.join(project_root, subdir, "routing_test_data.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                log.info("Loaded routing test data from %s (%d samples)", path, len(data))
                return data
        raise FileNotFoundError(
            "routing_test_data.json not found in benchmark_data/ or evaluation_test_data/"
        )

    # Standard datasets
    for subdir in ("benchmark_data", "evaluation_test_data"):
        path = os.path.join(project_root, subdir, f"{dataset}_test_data.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info("Loaded %s test data from %s (%d samples)", dataset, path, len(data))
            return data

    raise FileNotFoundError(
        f"{dataset}_test_data.json not found in benchmark_data/ or evaluation_test_data/"
    )


class ParentRetriever:
    """检索子块，返回对应父块内容的 Retriever 包装器。

    实现正确的父子分块检索：搜子块（精确匹配）→ 返回父块（完整上下文）。
    """

    def __init__(self, base_retriever, parent_lookup: dict):
        self.base_retriever = base_retriever
        self.parent_lookup = parent_lookup  # {parent_chunk_id: parent_content}

    def invoke(self, query: str, **kwargs):
        child_docs = self.base_retriever.invoke(query, **kwargs)
        seen_parents = set()
        result = []
        for doc in child_docs:
            parent_id = doc.metadata.get("parent_chunk_id")
            if parent_id and parent_id in self.parent_lookup:
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    from langchain_core.documents import Document
                    result.append(Document(
                        page_content=self.parent_lookup[parent_id],
                        metadata={**doc.metadata, "chunk_level": "parent", "retrieved_via": "child"},
                    ))
            else:
                # 没有 parent 映射的块直接返回
                result.append(doc)
        return result

    def get_relevant_documents(self, query: str, **kwargs):
        """兼容旧版 LangChain retriever 接口"""
        return self.invoke(query, **kwargs)


def load_source_documents(source: str) -> list:
    """
    Load source documents and return a list of LangChain Document objects.

    Args:
        source: 'company' or 'dnngp'

    Returns:
        list of Document with metadata={"source": file_path}
    """
    from langchain_core.documents import Document

    project_root = os.path.dirname(os.path.abspath(__file__))
    log = _get_logger()
    documents = []

    if source == "company":
        txt_path = os.path.join(project_root, "doc", "company.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Company text file not found: {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append(Document(
            page_content=content,
            metadata={"source": txt_path},
        ))
        log.info("Loaded company.txt (%d chars)", len(content))

    elif source == "dnngp":
        pdf_dir = os.path.join(project_root, "pdf")
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

        from app.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(enable_tables=False)

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                pdf_text, _ = extractor.extract_text(pdf_path, method="auto")
                if pdf_text and pdf_text.strip():
                    documents.append(Document(
                        page_content=pdf_text,
                        metadata={"source": pdf_path},
                    ))
                    log.info("Loaded PDF: %s (%d chars)", pdf_file, len(pdf_text))
            except Exception as e:
                log.warning("Failed to extract PDF %s: %s", pdf_file, e)
    else:
        raise ValueError(f"Unknown source: {source}. Expected 'company' or 'dnngp'.")

    if not documents:
        raise RuntimeError(f"No documents loaded for source '{source}'")

    return documents


def _split_oversized_chunks(documents: list, max_chars: int = 7500) -> list:
    """Split chunks that exceed the embedding model's input limit."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    result = []
    for doc in documents:
        if len(doc.page_content) > max_chars:
            splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=200)
            for sub in splitter.split_documents([doc]):
                result.append(sub)
        else:
            result.append(doc)
    return result


def build_vectorstore(documents: list, collection_suffix: str):
    """
    Build a Milvus vectorstore from a list of Document objects.

    Args:
        documents: List of LangChain Document objects
        collection_suffix: Suffix for the collection name (e.g. 'chunking_basic')

    Returns:
        (vectorstore, collection_name) tuple
    """
    from langchain_milvus import Milvus
    from pymilvus import connections, utility, Collection
    from app.config import embeddings

    log = _get_logger()
    collection_name = f"benchmark_{collection_suffix}"

    # Split oversized chunks to stay within embedding model limits
    documents = _split_oversized_chunks(documents)

    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530", timeout=30)

    # Drop old collection if it exists
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()
        log.info("Dropped existing collection: %s", collection_name)

    # Build vectorstore
    vectorstore = Milvus.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        drop_old=False,
        enable_dynamic_field=True,
    )
    log.info(
        "Built vectorstore collection '%s' with %d documents",
        collection_name,
        len(documents),
    )
    return vectorstore, collection_name


def cleanup_collection(collection_name: str):
    """Drop a Milvus collection. Errors are caught and logged."""
    log = _get_logger()
    try:
        from pymilvus import connections, utility, Collection

        connections.connect("default", host="localhost", port="19530", timeout=30)
        if utility.has_collection(collection_name):
            Collection(collection_name).drop()
            log.info("Cleaned up collection: %s", collection_name)
        else:
            log.info("Collection %s does not exist, nothing to clean up", collection_name)
    except Exception as e:
        log.warning("Failed to clean up collection %s: %s", collection_name, e)


def run_ragas_evaluation(test_data, retriever, dataset_name) -> dict:
    """
    Run RAGAS evaluation and return a metrics dict.

    Args:
        test_data: List of QA test data dicts
        retriever: LangChain Retriever instance
        dataset_name: Name for logging / reporting

    Returns:
        dict with keys like 'faithfulness', 'answer_relevancy', etc.
    """
    from app.evaluation import RAGASEvaluator

    log = _get_logger()
    log.info("Starting RAGAS evaluation for '%s' (%d samples)", dataset_name, len(test_data))

    # Use ModelScope as fallback if DashScope free tier is exhausted
    from app.config import llm as dashscope_llm, modelscope_llm, embeddings
    eval_llm = dashscope_llm
    try:
        # Quick test if DashScope is available
        eval_llm.invoke("hi")
    except Exception as e:
        if 'FreeTierOnly' in str(e) or '403' in str(e):
            if modelscope_llm:
                log.warning("DashScope free tier exhausted, falling back to ModelScope")
                print("  ⚠ DashScope 额度用完，切换到 ModelScope")
                eval_llm = modelscope_llm
            else:
                raise
    evaluator = RAGASEvaluator(llm=eval_llm, embeddings=embeddings)
    result = evaluator.evaluate(
        test_data=test_data,
        dataset_name=dataset_name,
        retriever=retriever,
        retriever_type="vec",
    )
    return result.metrics


def print_table(title: str, rows: list, columns: list, best_col: str = None):
    """
    Print a comparison table to the console.

    Args:
        title: Table title printed above the table
        rows: List of dicts, each representing a row
        columns: List of column keys to display (must match dict keys)
        best_col: If set, mark the row with the highest value in this column with a star
    """
    if not rows:
        print(f"\n{title}\n  (no data)\n")
        return

    # Determine column widths
    col_widths = {}
    for col in columns:
        col_widths[col] = max(len(str(col)), max(len(str(row.get(col, ""))) for row in rows))
        col_widths[col] = max(col_widths[col], 8)  # minimum width

    # Header
    sep = "+" + "+".join("-" * (col_widths[c] + 2) for c in columns) + "+"
    header = "|" + "|".join(f" {str(c):^{col_widths[c]}} " for c in columns) + "|"

    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep.replace("-", "="))

    # Find best row index if best_col is specified
    best_idx = -1
    if best_col and best_col in columns:
        max_val = float("-inf")
        for i, row in enumerate(rows):
            val = row.get(best_col, 0)
            if isinstance(val, (int, float)) and val > max_val:
                max_val = val
                best_idx = i

    # Data rows
    for i, row in enumerate(rows):
        marker = " *" if i == best_idx else "  "
        cells = "|".join(f" {str(row.get(c, '')):>{col_widths[c]}} " for c in columns)
        print(f"|{cells}|{marker}")
    print(sep)


def save_results(results: dict, experiment_name: str) -> str:
    """
    Save benchmark results to benchmark_results/benchmark_{name}_{timestamp}.json.

    Returns:
        Path to the saved JSON file.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    _get_logger().info("Results saved to %s", filepath)
    print(f"\nResults saved to: {filepath}")
    return filepath


# ============================================================================
# Experiment Stubs (to be implemented in Tasks 4-8)
# ============================================================================

def run_experiment_chunking(dataset: str, test_data: list, skip: set = None) -> dict:
    """
    Experiment 1: Compare chunking strategies.
    Control variables: same document, top_k=5, same LLM.
    """
    from app.enhanced_chunking import create_enhanced_chunker

    print(f"\n{'#'*70}")
    print(f"  Experiment 1: Chunking Strategy Comparison — {dataset}")
    print(f"{'#'*70}")

    # Load source documents
    raw_docs = load_source_documents(dataset)
    full_text = "\n".join(doc.page_content for doc in raw_docs)
    source_path = raw_docs[0].metadata.get("source", "unknown") if raw_docs else "unknown"

    strategies = {
        "basic": {"strategy": "basic", "max_chunk_size": 400, "min_chunk_size": 200},
        "semantic_metadata": {"strategy": "semantic_metadata", "max_chunk_size": 400, "min_chunk_size": 200},
        "parent_child": {"strategy": "parent_child", "max_chunk_size": 400, "min_chunk_size": 200},
    }

    results = []

    for name, params in strategies.items():
        if skip and name in skip:
            print(f"\n  ⏭ Skipping {name} (already completed)")
            continue
        print(f"\n  Strategy: {name}")

        # 1. Chunk
        chunker = create_enhanced_chunker(**params)
        chunks = chunker.chunk_document(full_text, source_path)
        chunk_sizes = [len(c.page_content) for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        print(f"    Chunks: {len(chunks)}, Avg size: {avg_size:.0f} chars")

        # For parent_child: chunks are child chunks, parent_lookup has parent content
        parent_lookup = getattr(chunker, 'parent_lookup', None)
        if parent_lookup:
            print(f"    Parent lookup: {len(parent_lookup)} parent chunks")

        # 2. Build temporary vectorstore
        suffix = f"{dataset}_chunking_{name}_{int(time.time())}"
        vs, collection_name = build_vectorstore(chunks, suffix)

        try:
            # 3. Build retriever
            base_retriever = vs.as_retriever(search_kwargs={"k": 5})

            # For parent_child: wrap retriever to return parent content
            if parent_lookup:
                retriever = ParentRetriever(base_retriever, parent_lookup)
            else:
                retriever = base_retriever

            # 4. Run RAGAS evaluation
            metrics = run_ragas_evaluation(test_data, retriever, f"{dataset}_{name}")

            row = {
                "label": name,
                "chunk_count": len(chunks),
                "avg_chunk_size": round(avg_size, 1),
                **metrics,
            }
            results.append(row)

            print(f"    RAGAS: F={metrics.get('faithfulness',0):.4f} "
                  f"R={metrics.get('answer_relevancy',0):.4f} "
                  f"P={metrics.get('context_precision',0):.4f} "
                  f"Rec={metrics.get('context_recall',0):.4f}")

        finally:
            cleanup_collection(collection_name)

    # Print comparison table
    columns = ["chunk_count", "avg_chunk_size", "faithfulness", "answer_relevancy",
               "context_precision", "context_recall", "overall_score"]
    print_table(f"Chunking Strategy Comparison ({dataset})", results, columns, best_col="context_recall")

    return {"experiment": "chunking", "dataset": dataset, "results": results}


def run_experiment_topk(dataset: str, test_data: list) -> dict:
    """
    Experiment 2: Compare retrieval top_k parameter.
    Control variables: semantic_metadata chunking, same document.
    """
    from app.enhanced_chunking import create_enhanced_chunker

    print(f"\n{'#'*70}")
    print(f"  Experiment 2: Top-K Parameter Comparison — {dataset}")
    print(f"{'#'*70}")

    # Load source and chunk with semantic_metadata
    raw_docs = load_source_documents(dataset)
    full_text = "\n".join(doc.page_content for doc in raw_docs)
    source_path = raw_docs[0].metadata.get("source", "unknown") if raw_docs else "unknown"

    chunker = create_enhanced_chunker(strategy="semantic_metadata", max_chunk_size=400, min_chunk_size=200)
    chunks = chunker.chunk_document(full_text, source_path)

    suffix = f"{dataset}_topk_{int(time.time())}"
    vs, collection_name = build_vectorstore(chunks, suffix)

    try:
        results = []
        for top_k in [3, 5, 8]:
            print(f"\n  top_k = {top_k}")

            retriever = vs.as_retriever(search_kwargs={"k": top_k})

            # Measure retrieval latency
            latencies = []
            for q in test_data:
                start = time.time()
                retriever.invoke(q["question"])
                latencies.append((time.time() - start) * 1000)
            avg_latency = sum(latencies) / len(latencies)

            # Run RAGAS
            metrics = run_ragas_evaluation(test_data, retriever, f"{dataset}_topk{top_k}")

            row = {
                "label": f"top_k={top_k}",
                "avg_latency_ms": round(avg_latency, 1),
                **metrics,
            }
            results.append(row)

            print(f"    Latency={avg_latency:.1f}ms RAGAS: F={metrics.get('faithfulness',0):.4f} "
                  f"R={metrics.get('answer_relevancy',0):.4f} "
                  f"P={metrics.get('context_precision',0):.4f} "
                  f"Rec={metrics.get('context_recall',0):.4f}")

        columns = ["avg_latency_ms", "faithfulness", "answer_relevancy",
                    "context_precision", "context_recall", "overall_score"]
        print_table(f"Top-K Parameter Comparison ({dataset})", results, columns, best_col="context_recall")

        return {"experiment": "top_k", "dataset": dataset, "results": results}

    finally:
        cleanup_collection(collection_name)


def run_experiment_hybrid(dataset: str, test_data: list, skip: set = None) -> dict:
    """
    Experiment 3: Compare pure vector vs Dense+Sparse hybrid retrieval.
    Hybrid uses RRF (Reciprocal Rank Fusion) to merge results.
    """
    from app.enhanced_chunking import create_enhanced_chunker
    from rank_bm25 import BM25Okapi
    import jieba

    print(f"\n{'#'*70}")
    print(f"  Experiment 3: Hybrid Retrieval Comparison — {dataset}")
    print(f"{'#'*70}")

    # Load source and chunk
    raw_docs = load_source_documents(dataset)
    full_text = "\n".join(doc.page_content for doc in raw_docs)
    source_path = raw_docs[0].metadata.get("source", "unknown") if raw_docs else "unknown"

    chunker = create_enhanced_chunker(strategy="semantic_metadata", max_chunk_size=400, min_chunk_size=200)
    chunks = chunker.chunk_document(full_text, source_path)

    suffix = f"{dataset}_hybrid_{int(time.time())}"
    vs, collection_name = build_vectorstore(chunks, suffix)

    try:
        # Build BM25 index
        tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        chunk_texts = [doc.page_content for doc in chunks]

        def hybrid_search(query, k=5, dense_weight=0.7):
            """Hybrid retrieval: vector + BM25, RRF merge."""
            # Dense retrieval
            dense_results = vs.similarity_search(query, k=k * 2)
            dense_docs = {doc.page_content: (i, doc) for i, doc in enumerate(dense_results)}

            # Sparse retrieval (BM25)
            tokenized_query = list(jieba.cut(query))
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_top = sorted(range(len(bm25_scores)),
                              key=lambda i: bm25_scores[i], reverse=True)[:k * 2]
            sparse_docs = {}
            for rank, idx in enumerate(bm25_top):
                sparse_docs[chunk_texts[idx]] = (rank, chunks[idx])

            # RRF merge
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

        # Wrapper retriever classes for RAGAS compatibility
        class VectorRetriever:
            def __init__(self, vs, k=5):
                self.retriever = vs.as_retriever(search_kwargs={"k": k})
            def invoke(self, query):
                return self.retriever.invoke(query)

        class HybridRetriever:
            def __init__(self, vs, k=5):
                self.k = k
                self._vs = vs
            def invoke(self, query):
                return hybrid_search(query, self.k)

        # A: Pure vector
        if skip and "vec_only" in skip:
            print("\n  ⏭ Skipping Pure Vector (already completed)")
            vec_metrics = {"faithfulness": 0, "answer_relevancy": 0,
                           "context_precision": 0, "context_recall": 0, "overall_score": 0}
        else:
            print("\n  A: Pure Vector Retrieval")
            vec_ret = VectorRetriever(vs, k=5)
            vec_metrics = run_ragas_evaluation(test_data, vec_ret, f"{dataset}_vec_only")
        print(f"    RAGAS: F={vec_metrics.get('faithfulness',0):.4f} "
              f"R={vec_metrics.get('answer_relevancy',0):.4f} "
              f"P={vec_metrics.get('context_precision',0):.4f} "
              f"Rec={vec_metrics.get('context_recall',0):.4f}")

        # B: Hybrid retrieval
        if skip and "hybrid" in skip:
            print("\n  ⏭ Skipping Hybrid (already completed)")
            hybrid_metrics = {"faithfulness": 0, "answer_relevancy": 0,
                              "context_precision": 0, "context_recall": 0, "overall_score": 0}
        else:
            print("\n  B: Hybrid Retrieval (Dense 0.7 + BM25 0.3)")
            hybrid_ret = HybridRetriever(vs, k=5)
            hybrid_metrics = run_ragas_evaluation(test_data, hybrid_ret, f"{dataset}_hybrid")
        print(f"    RAGAS: F={hybrid_metrics.get('faithfulness',0):.4f} "
              f"R={hybrid_metrics.get('answer_relevancy',0):.4f} "
              f"P={hybrid_metrics.get('context_precision',0):.4f} "
              f"Rec={hybrid_metrics.get('context_recall',0):.4f}")

        results = [
            {"label": "Pure Vector", **vec_metrics},
            {"label": "Hybrid (D+S)", **hybrid_metrics},
        ]

        columns = ["faithfulness", "answer_relevancy", "context_precision",
                    "context_recall", "overall_score"]
        print_table(f"Hybrid Retrieval Comparison ({dataset})", results, columns, best_col="context_recall")

        return {"experiment": "hybrid", "dataset": dataset, "results": results}

    finally:
        cleanup_collection(collection_name)


def run_experiment_rerank(dataset: str, test_data: list, skip: set = None) -> dict:
    """
    Experiment 4: Compare no-rerank vs DashScope qwen3-rerank reranking.
    """
    from app.enhanced_chunking import create_enhanced_chunker

    print(f"\n{'#'*70}")
    print(f"  Experiment 4: Reranking Comparison — {dataset}")
    print(f"{'#'*70}")

    # Load source and chunk
    raw_docs = load_source_documents(dataset)
    full_text = "\n".join(doc.page_content for doc in raw_docs)
    source_path = raw_docs[0].metadata.get("source", "unknown") if raw_docs else "unknown"

    chunker = create_enhanced_chunker(strategy="semantic_metadata", max_chunk_size=400, min_chunk_size=200)
    chunks = chunker.chunk_document(full_text, source_path)

    suffix = f"{dataset}_rerank_{int(time.time())}"
    vs, collection_name = build_vectorstore(chunks, suffix)

    try:
        # A: No rerank (baseline)
        if skip and "no_rerank" in skip:
            print("\n  ⏭ Skipping No Rerank (already completed)")
            vec_metrics = {"faithfulness": 0, "answer_relevancy": 0,
                           "context_precision": 0, "context_recall": 0, "overall_score": 0}
        else:
            print("\n  A: No Rerank (top_k=5)")
            vec_retriever = vs.as_retriever(search_kwargs={"k": 5})
            vec_metrics = run_ragas_evaluation(test_data, vec_retriever, f"{dataset}_no_rerank")
        print(f"    RAGAS: F={vec_metrics.get('faithfulness',0):.4f} "
              f"R={vec_metrics.get('answer_relevancy',0):.4f} "
              f"P={vec_metrics.get('context_precision',0):.4f} "
              f"Rec={vec_metrics.get('context_recall',0):.4f}")

        # B: With rerank
        if skip and "rerank" in skip:
            print("\n  ⏭ Skipping Rerank (already completed)")
            rerank_metrics = {"faithfulness": 0, "answer_relevancy": 0,
                              "context_precision": 0, "context_recall": 0, "overall_score": 0}
        else:
            print("\n  B: Rerank (recall top_8 -> rerank -> top_5)")
            try:
                from dashscope.rerank.text_rerank import TextReRank as TextRerank

                class RerankRetriever:
                    def __init__(self, vs, top_k=5, recall_k=8):
                        self.vs = vs
                        self.top_k = top_k
                        self.recall_k = recall_k

                    def invoke(self, query):
                        # Recall phase
                        docs = self.vs.similarity_search(query, k=self.recall_k)
                        if not docs:
                            return docs

                        # Rerank phase
                        documents = [doc.page_content for doc in docs]
                        result = TextRerank.call(
                            model="qwen3-rerank",
                            top_n=self.top_k,
                            query=query,
                            documents=documents,
                        )

                        if result.status_code != 200:
                            print(f"    ⚠ Rerank API failed: {result.message}, using original order")
                            return docs[:self.top_k]

                        reranked_indices = [item.index for item in result.output.results]
                        return [docs[i] for i in reranked_indices if i < len(docs)]

                rerank_retriever = RerankRetriever(vs, top_k=5, recall_k=8)
                rerank_metrics = run_ragas_evaluation(test_data, rerank_retriever, f"{dataset}_rerank")
                print(f"    RAGAS: F={rerank_metrics.get('faithfulness',0):.4f} "
                      f"R={rerank_metrics.get('answer_relevancy',0):.4f} "
                      f"P={rerank_metrics.get('context_precision',0):.4f} "
                      f"Rec={rerank_metrics.get('context_recall',0):.4f}")

            except Exception as e:
                print(f"    ⚠ Rerank experiment failed: {e}")
                rerank_metrics = {"faithfulness": 0, "answer_relevancy": 0,
                                  "context_precision": 0, "context_recall": 0, "overall_score": 0}

        results = [
            {"label": "No Rerank", **vec_metrics},
            {"label": "With Rerank", **rerank_metrics},
        ]

        columns = ["faithfulness", "answer_relevancy", "context_precision",
                    "context_recall", "overall_score"]
        print_table(f"Reranking Comparison ({dataset})", results, columns, best_col="context_recall")

        return {"experiment": "rerank", "dataset": dataset, "results": results}

    finally:
        cleanup_collection(collection_name)


def run_experiment_routing(dataset: str, test_data: list, skip: set = None) -> dict:
    """
    Experiment 5: Evaluate supervisor routing accuracy.
    Compares with/without cycle detection by analyzing routing decisions.
    Uses routing_test_data.json (has expected_worker field), not the dataset's QA data.
    """
    from app.graph_builder import build_graph
    from app.database import init_seed_data
    from app.rag import init_rag

    print(f"\n{'#'*70}")
    print(f"  Experiment 5: Routing Accuracy")
    print(f"{'#'*70}")

    # Routing experiment uses its own test data with expected_worker field
    test_data = load_test_data("routing")

    # Initialize system
    init_seed_data()
    init_rag()

    graph = build_graph(skip_graph_kg=False)

    all_decisions = []

    for i, sample in enumerate(test_data):
        question = sample["question"]
        expected = sample["expected_worker"]

        print(f"\n  [{i+1}/{len(test_data)}] Q: {question[:40]}...")
        print(f"         Expected: {expected}")

        try:
            first_routing = None
            total_rounds = 0

            for chunk in graph.stream({"messages": question}, stream_mode="values"):
                # Routing decision is in state's 'next' field, not message content
                next_worker = chunk.get("next", "")
                if next_worker and next_worker != "__end__":
                    if first_routing is None:
                        first_routing = next_worker
                    total_rounds += 1

            is_correct = first_routing == expected
            print(f"         Got: {first_routing} | {'OK' if is_correct else 'MISS'}")

            all_decisions.append({
                "question": question,
                "expected": expected,
                "got": first_routing,
                "correct": is_correct,
                "rounds": total_rounds,
            })

        except Exception as e:
            print(f"         ERROR: {e}")
            all_decisions.append({
                "question": question,
                "expected": expected,
                "got": None,
                "correct": False,
                "rounds": 0,
                "error": str(e),
            })

    # Compute stats
    total = len(all_decisions)
    correct = sum(1 for d in all_decisions if d["correct"])
    avg_rounds = sum(d["rounds"] for d in all_decisions) / total if total else 0
    accuracy = correct / total if total else 0

    # Per-worker stats
    worker_stats = {}
    for d in all_decisions:
        w = d["expected"]
        if w not in worker_stats:
            worker_stats[w] = {"total": 0, "correct": 0}
        worker_stats[w]["total"] += 1
        if d["correct"]:
            worker_stats[w]["correct"] += 1

    print(f"\n{'='*70}")
    print(f"  Routing Accuracy Summary")
    print(f"{'='*70}")
    print(f"  Overall: {correct}/{total} = {accuracy:.1%}")
    print(f"  Avg rounds: {avg_rounds:.1f}")
    print(f"\n  Per Worker:")
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


# ============================================================================
# Experiment Registry
# ============================================================================

EXPERIMENTS = {
    "chunking": run_experiment_chunking,
    "top_k": run_experiment_topk,
    "hybrid": run_experiment_hybrid,
    "rerank": run_experiment_rerank,
    "routing": run_experiment_routing,
}


# ============================================================================
# Report Generator Stub
# ============================================================================

def generate_report(all_results: dict) -> str:
    """
    Generate a Markdown comparison report from all experiment results.
    Saves to benchmark_results/benchmark_report.md and returns the path.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(output_dir, "benchmark_report.md")

    lines = [
        "# RAG System Optimization Benchmark Report",
        f"\nGenerated: {timestamp}",
        "",
    ]

    # Experiment 1: Chunking
    chunking_key = next((k for k in all_results if "chunking" in k), None)
    if chunking_key:
        r = all_results[chunking_key]
        if r and "results" in r:
            lines.append("## Experiment 1: Chunking Strategy Comparison")
            lines.append(f"**Dataset:** {r.get('dataset', 'N/A')}")
            lines.append("")
            lines.append("| Strategy | Chunks | Avg Size | Faithfulness | Relevancy | Precision | Recall |")
            lines.append("|---|---|---|---|---|---|---|")
            for row in r["results"]:
                lines.append(f"| {row.get('label','')} | {row.get('chunk_count','-')} | "
                            f"{row.get('avg_chunk_size','-')} | "
                            f"{row.get('faithfulness',0):.4f} | "
                            f"{row.get('answer_relevancy',0):.4f} | "
                            f"{row.get('context_precision',0):.4f} | "
                            f"{row.get('context_recall',0):.4f} |")
            lines.append("")

    # Experiment 2: Top-K
    topk_key = next((k for k in all_results if "top_k" in k), None)
    if topk_key:
        r = all_results[topk_key]
        if r and "results" in r:
            lines.append("## Experiment 2: Top-K Parameter Comparison")
            lines.append(f"**Dataset:** {r.get('dataset', 'N/A')}")
            lines.append("")
            lines.append("| Top-K | Latency(ms) | Faithfulness | Relevancy | Precision | Recall |")
            lines.append("|---|---|---|---|---|---|")
            for row in r["results"]:
                lines.append(f"| {row.get('label','')} | {row.get('avg_latency_ms','-')} | "
                            f"{row.get('faithfulness',0):.4f} | "
                            f"{row.get('answer_relevancy',0):.4f} | "
                            f"{row.get('context_precision',0):.4f} | "
                            f"{row.get('context_recall',0):.4f} |")
            lines.append("")

    # Experiment 3: Hybrid
    hybrid_key = next((k for k in all_results if "hybrid" in k), None)
    if hybrid_key:
        r = all_results[hybrid_key]
        if r and "results" in r:
            lines.append("## Experiment 3: Hybrid Retrieval Comparison")
            lines.append(f"**Dataset:** {r.get('dataset', 'N/A')}")
            lines.append("")
            lines.append("| Method | Faithfulness | Relevancy | Precision | Recall |")
            lines.append("|---|---|---|---|---|")
            for row in r["results"]:
                lines.append(f"| {row.get('label','')} | "
                            f"{row.get('faithfulness',0):.4f} | "
                            f"{row.get('answer_relevancy',0):.4f} | "
                            f"{row.get('context_precision',0):.4f} | "
                            f"{row.get('context_recall',0):.4f} |")
            lines.append("")

    # Experiment 4: Rerank
    rerank_key = next((k for k in all_results if "rerank" in k), None)
    if rerank_key:
        r = all_results[rerank_key]
        if r and "results" in r:
            lines.append("## Experiment 4: Reranking Comparison")
            lines.append(f"**Dataset:** {r.get('dataset', 'N/A')}")
            lines.append("")
            lines.append("| Config | Faithfulness | Relevancy | Precision | Recall |")
            lines.append("|---|---|---|---|---|")
            for row in r["results"]:
                lines.append(f"| {row.get('label','')} | "
                            f"{row.get('faithfulness',0):.4f} | "
                            f"{row.get('answer_relevancy',0):.4f} | "
                            f"{row.get('context_precision',0):.4f} | "
                            f"{row.get('context_recall',0):.4f} |")
            lines.append("")

    # Experiment 5: Routing
    routing_key = next((k for k in all_results if "routing" in k), None)
    if routing_key:
        r = all_results[routing_key]
        if r and "accuracy" in r:
            lines.append("## Experiment 5: Routing Accuracy")
            lines.append("")
            lines.append(f"- **Overall Accuracy:** {r['correct']}/{r['total_questions']} = {r['accuracy']:.1%}")
            lines.append(f"- **Avg Routing Rounds:** {r['avg_rounds']:.1f}")
            lines.append("")
            lines.append("| Worker | Accuracy |")
            lines.append("|---|---|")
            for w, stats in sorted(r.get("worker_stats", {}).items()):
                w_acc = stats["correct"] / stats["total"] if stats["total"] else 0
                lines.append(f"| {w} | {stats['correct']}/{stats['total']} = {w_acc:.1%} |")
            lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Optimization | Best Config | Key Metric |")
    lines.append("|---|---|---|")

    if chunking_key and all_results.get(chunking_key, {}).get("results"):
        best = max(all_results[chunking_key]["results"], key=lambda x: x.get("context_recall", 0))
        lines.append(f"| Chunking | {best.get('label','')} | Recall={best.get('context_recall',0):.4f} |")

    if topk_key and all_results.get(topk_key, {}).get("results"):
        best = max(all_results[topk_key]["results"], key=lambda x: x.get("context_recall", 0))
        lines.append(f"| Top-K | {best.get('label','')} | Recall={best.get('context_recall',0):.4f} |")

    if hybrid_key and all_results.get(hybrid_key, {}).get("results"):
        best = max(all_results[hybrid_key]["results"], key=lambda x: x.get("context_recall", 0))
        lines.append(f"| Hybrid | {best.get('label','')} | Recall={best.get('context_recall',0):.4f} |")

    if rerank_key and all_results.get(rerank_key, {}).get("results"):
        best = max(all_results[rerank_key]["results"], key=lambda x: x.get("context_recall", 0))
        lines.append(f"| Rerank | {best.get('label','')} | Recall={best.get('context_recall',0):.4f} |")

    if routing_key and all_results.get(routing_key, {}).get("accuracy") is not None:
        lines.append(f"| Routing | cycle detection | Accuracy={all_results[routing_key]['accuracy']:.1%} |")

    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved to: {report_path}")
    return report_path


# ============================================================================
# Main Entry Point
# ============================================================================

def main(argv=None):
    """Parse CLI args, run selected experiments, and generate a report."""
    args = parse_args(argv)
    log = _get_logger()

    print("=" * 60)
    print("Hybrid RAG Multi-Agent Benchmark Suite")
    print(f"Dataset: {args.dataset}")
    print(f"Experiment: {args.only or 'all'}")
    if args.skip:
        print(f"Skip: {args.skip}")
    print("=" * 60)

    # Parse --skip into a set
    skip_set = set(args.skip.split(",")) if args.skip else set()

    # Determine datasets to run on
    if args.dataset == "all":
        datasets = ["company", "dnngp"]
    else:
        datasets = [args.dataset]

    # Determine experiments to run
    if args.only:
        experiment_names = [args.only]
    else:
        experiment_names = list(EXPERIMENTS.keys())

    all_results = {}

    for ds in datasets:
        print(f"\n{'#' * 60}")
        print(f"# Dataset: {ds}")
        print(f"{'#' * 60}")

        for exp_name in experiment_names:
            # Load test data for this dataset
            try:
                test_data = load_test_data(ds)
            except FileNotFoundError as e:
                log.warning("Skipping %s/%s: %s", exp_name, ds, e)
                print(f"  Skipping {exp_name}: {e}")
                continue

            # Limit test samples if --samples is set
            if args.samples > 0 and len(test_data) > args.samples:
                print(f"  Using {args.samples}/{len(test_data)} test samples")
                test_data = test_data[:args.samples]

            # Run the experiment
            func = EXPERIMENTS[exp_name]
            result_key = f"{exp_name}_{ds}"
            try:
                result = func(ds, test_data, skip=skip_set) if skip_set else func(ds, test_data)
                all_results[result_key] = result
            except Exception as e:
                log.error("Experiment %s failed on %s: %s", exp_name, ds, e)
                print(f"  ERROR in {exp_name}: {e}")
                all_results[result_key] = {"error": str(e)}

    # Generate summary report
    generate_report(all_results)

    return all_results


if __name__ == "__main__":
    main()
