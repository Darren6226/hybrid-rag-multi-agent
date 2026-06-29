"""
快速测试两个修复:
1. Context Precision = 0 问题 (RAGAS v2 字段名)
2. parent_child 优化 (缩小 parent 尺寸 + 增大 k)

只用 3 个 company 样本，快速验证可行性。
"""

import sys
import os
import time
import types

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix RAGAS vertexai import
if 'langchain_community.chat_models.vertexai' not in sys.modules:
    mock_module = types.ModuleType('langchain_community.chat_models.vertexai')
    mock_module.ChatVertexAI = None
    sys.modules['langchain_community.chat_models.vertexai'] = mock_module

import json
from datasets import Dataset
from ragas.metrics import (
    _Faithfulness as Faithfulness,
    _AnswerRelevancy as AnswerRelevancy,
    _ContextPrecision as ContextPrecision,
    _ContextRecall as ContextRecall,
)
from ragas.evaluation import evaluate as ragas_evaluate
from ragas.run_config import RunConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def load_test_data(n=3):
    """Load first n samples from company test data."""
    with open("evaluation_test_data/company_test_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:n]


def load_docs_text():
    """Load company document text."""
    with open("doc/company.txt", "r", encoding="utf-8") as f:
        return f.read()


def build_vectorstore_and_retriever(chunks, collection_suffix, k=5):
    """Build Milvus vectorstore and return retriever."""
    from app.config import embeddings
    from langchain_milvus import Milvus

    vs = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"quick_test_{collection_suffix}",
        connection_args={"host": "localhost", "port": "19530"},
        drop_old=True,
        enable_dynamic_field=True,
    )
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return vs, retriever


def generate_answer(llm, question, context_str):
    """Generate answer using LLM."""
    prompt = PromptTemplate(
        template="""基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法找到答案。

Question: {question}
Context: {context}

Answer:""",
        input_variables=["question", "context"],
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context_str})


def build_dataset_v2(test_data, retriever, llm):
    """
    修复版: 使用 RAGAS v2 字段名 (user_input, retrieved_contexts, reference)
    不再同时放 ground_truth 和 reference。
    """
    questions = []
    references = []
    contexts_list = []
    answers = []

    for sample in test_data:
        q = sample["question"]
        gt = sample["ground_truth"]

        # 检索
        docs = retriever.invoke(q)
        contexts = [doc.page_content for doc in docs]

        # 生成答案
        context_str = "\n\n".join(contexts)
        answer = generate_answer(llm, q, context_str)

        questions.append(q)
        references.append(gt)
        contexts_list.append(contexts)
        answers.append(answer)

    # 使用 RAGAS v2 字段名
    dataset = Dataset.from_dict({
        "user_input": questions,            # v2: user_input (was "question")
        "response": answers,                # v2: response (was "answer")
        "retrieved_contexts": contexts_list, # v2: retrieved_contexts (was "contexts")
        "reference": references,             # v2: reference (was "ground_truth")
        # 不再包含 "ground_truth" — 避免重复列
    })
    return dataset


def run_ragas(dataset, llm, embeddings):
    """Run RAGAS evaluation and return metrics."""
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]
    result = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(max_workers=1, max_wait=300),  # 单线程避免并发问题
    )
    return result


def print_metrics(result, label):
    """Print metrics from RAGAS result."""
    all_scores = result.scores
    metric_names = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    print(f"\n  === {label} ===")
    for name in metric_names:
        values = [s[name] for s in all_scores if name in s and s[name] is not None]
        avg = sum(values) / len(values) if values else 0.0
        print(f"  {name}: {avg:.4f}")


def main():
    from app.config import llm, embeddings

    print("=" * 60)
    print("快速验证测试 (3 samples, company dataset)")
    print("=" * 60)

    test_data = load_test_data(3)
    full_text = load_docs_text()
    print(f"Loaded {len(test_data)} test samples, doc length: {len(full_text)} chars")

    # ====================================================================
    # 测试 1: Context Precision 修复 (用 basic 策略对比 v1 vs v2 字段名)
    # ====================================================================
    print(f"\n{'─' * 60}")
    print("测试 1: Context Precision 修复验证")
    print("  对比 v1 字段名(question/contexts/ground_truth) vs v2 字段名")
    print(f"{'─' * 60}")

    from app.enhanced_chunking import create_enhanced_chunker

    # 用 basic 策略产生 chunks
    chunker = create_enhanced_chunker(strategy="basic", max_chunk_size=400, min_chunk_size=200)
    chunks = chunker.chunk_document(full_text, "doc/company.txt")
    print(f"  Basic chunks: {len(chunks)}")

    vs, retriever = build_vectorstore_and_retriever(chunks, "basic_v1v2_test", k=5)

    try:
        # --- v1 (原始方式) ---
        print("\n  [v1] 生成数据集 (question/contexts/ground_truth + reference)...")
        questions_v1, gts_v1, ctxs_v1, answers_v1 = [], [], [], []
        for sample in test_data:
            q = sample["question"]
            gt = sample["ground_truth"]
            docs = retriever.invoke(q)
            contexts = [d.page_content for d in docs]
            answer = generate_answer(llm, q, "\n\n".join(contexts))
            questions_v1.append(q)
            gts_v1.append(gt)
            ctxs_v1.append(contexts)
            answers_v1.append(answer)

        dataset_v1 = Dataset.from_dict({
            "question": questions_v1,
            "answer": answers_v1,
            "contexts": ctxs_v1,
            "ground_truth": gts_v1,
            "reference": gts_v1,
        })

        print("  [v1] 运行 RAGAS...")
        result_v1 = run_ragas(dataset_v1, llm, embeddings)
        print_metrics(result_v1, "v1 字段名 (原始)")

        # --- v2 (修复方式) ---
        print("\n  [v2] 生成数据集 (user_input/retrieved_contexts/reference)...")
        dataset_v2 = build_dataset_v2(test_data, retriever, llm)

        print("  [v2] 运行 RAGAS...")
        result_v2 = run_ragas(dataset_v2, llm, embeddings)
        print_metrics(result_v2, "v2 字段名 (修复)")

    finally:
        vs.col.drop() if hasattr(vs, 'col') else None
        print("\n  Cleaned up test collection")

    # ====================================================================
    # 测试 2: parent_child 优化 (缩小 parent + 增大 k)
    # ====================================================================
    print(f"\n{'─' * 60}")
    print("测试 2: parent_child 优化验证")
    print("  对比 原始(2000/400, k=5) vs 优化(1000/400, k=10)")
    print(f"{'─' * 60}")

    # --- 原始 parent_child ---
    print("\n  [原始] parent_child (parent=2000, child=400, k=5)")
    from app.enhanced_chunking import EnhancedChunker
    chunker_orig = EnhancedChunker(
        use_semantic=False,  # 跳过语义分块，直接做 parent_child
        use_parent_child=True,
        enable_metadata_enhancement=True,
    )
    # 手动设置 ParentChildChunker 参数
    from app.enhanced_chunking import ParentChildChunker
    chunker_orig.parent_child_chunker = ParentChildChunker(
        parent_size=2000, child_size=400, parent_overlap=200, child_overlap=50
    )
    chunks_orig = chunker_orig.chunk_document(full_text, "doc/company.txt")
    parent_lookup_orig = getattr(chunker_orig, 'parent_lookup', None)
    print(f"    Child chunks: {len(chunks_orig)}, Parents: {len(parent_lookup_orig) if parent_lookup_orig else 0}")

    vs_orig, base_ret_orig = build_vectorstore_and_retriever(chunks_orig, "pc_orig", k=5)
    try:
        from benchmark import ParentRetriever
        retriever_orig = ParentRetriever(base_ret_orig, parent_lookup_orig) if parent_lookup_orig else base_ret_orig

        print("  [原始] 生成数据集并评估...")
        dataset_orig = build_dataset_v2(test_data, retriever_orig, llm)
        result_orig = run_ragas(dataset_orig, llm, embeddings)
        print_metrics(result_orig, "原始 parent_child (2000/400, k=5)")
    finally:
        vs_orig.col.drop() if hasattr(vs_orig, 'col') else None

    # --- 优化 parent_child ---
    print("\n  [优化] parent_child (parent=1000, child=400, k=10)")
    chunker_opt = EnhancedChunker(
        use_semantic=False,
        use_parent_child=True,
        enable_metadata_enhancement=True,
    )
    chunker_opt.parent_child_chunker = ParentChildChunker(
        parent_size=1000, child_size=400, parent_overlap=100, child_overlap=50
    )
    chunks_opt = chunker_opt.chunk_document(full_text, "doc/company.txt")
    parent_lookup_opt = getattr(chunker_opt, 'parent_lookup', None)
    print(f"    Child chunks: {len(chunks_opt)}, Parents: {len(parent_lookup_opt) if parent_lookup_opt else 0}")

    vs_opt, base_ret_opt = build_vectorstore_and_retriever(chunks_opt, "pc_opt", k=10)
    try:
        retriever_opt = ParentRetriever(base_ret_opt, parent_lookup_opt) if parent_lookup_opt else base_ret_opt

        print("  [优化] 生成数据集并评估...")
        dataset_opt = build_dataset_v2(test_data, retriever_opt, llm)
        result_opt = run_ragas(dataset_opt, llm, embeddings)
        print_metrics(result_opt, "优化 parent_child (1000/400, k=10)")
    finally:
        vs_opt.col.drop() if hasattr(vs_opt, 'col') else None

    # ====================================================================
    # 测试 3: basic 作为 baseline 对比
    # ====================================================================
    print(f"\n{'─' * 60}")
    print("测试 3: basic baseline 对比")
    print(f"{'─' * 60}")

    chunker_basic = create_enhanced_chunker(strategy="basic", max_chunk_size=400, min_chunk_size=200)
    chunks_basic = chunker_basic.chunk_document(full_text, "doc/company.txt")
    print(f"  Basic chunks: {len(chunks_basic)}")

    vs_basic, retriever_basic = build_vectorstore_and_retriever(chunks_basic, "basic_bl", k=5)
    try:
        print("  [basic] 生成数据集并评估...")
        dataset_basic = build_dataset_v2(test_data, retriever_basic, llm)
        result_basic = run_ragas(dataset_basic, llm, embeddings)
        print_metrics(result_basic, "basic baseline (k=5)")
    finally:
        vs_basic.col.drop() if hasattr(vs_basic, 'col') else None

    # ====================================================================
    # 汇总
    # ====================================================================
    print(f"\n{'=' * 60}")
    print("汇总对比")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
