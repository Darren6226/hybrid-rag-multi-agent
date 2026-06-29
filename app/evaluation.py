"""
RAGAS Evaluation Module for RAG Systems
基于 RAGAS 框架的 RAG 系统评估模块

RAGAS 指标:
1. Faithfulness — 答案是否忠实于检索到的上下文（防幻觉）
2. Answer Relevancy — 答案是否与问题相关
3. Context Precision — 检索结果中相关文档的排名是否靠前
4. Context Recall — 是否检索到了回答问题所需的所有信息
"""

import sys
import os
import json
import math
import time
import threading
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict

# 修复 RAGAS 的 vertexai 导入问题
import types
if 'langchain_community.chat_models.vertexai' not in sys.modules:
    mock_module = types.ModuleType('langchain_community.chat_models.vertexai')
    mock_module.ChatVertexAI = None
    sys.modules['langchain_community.chat_models.vertexai'] = mock_module

from datasets import Dataset
from ragas.metrics import (
    _Faithfulness as Faithfulness,
    _AnswerRelevancy as AnswerRelevancy,
    _ContextPrecision as ContextPrecision,
    _ContextRecall as ContextRecall,
)
from ragas.evaluation import evaluate as ragas_evaluate

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# ============================================================================
# 限速重试 LLM 包装器
# ============================================================================

class RateLimitedChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI 子类，为 _generate 添加限速和重试逻辑。
    继承 ChatOpenAI 保持类型兼容，RAGAS 内部类型检查不会出问题。
    """
    _rate_limit_lock: Any = None
    _last_call_time: float = 0.0
    _min_interval: float = 1.5
    _max_retries: int = 5
    _base_wait: float = 5.0

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        min_interval = kwargs.pop('min_interval', 1.5)
        max_retries = kwargs.pop('max_retries', 5)
        base_wait = kwargs.pop('base_wait', 5.0)
        super().__init__(**kwargs)
        object.__setattr__(self, '_rate_limit_lock', threading.Lock())
        object.__setattr__(self, '_min_interval', min_interval)
        object.__setattr__(self, '_max_retries', max_retries)
        object.__setattr__(self, '_base_wait', base_wait)
        object.__setattr__(self, '_last_call_time', 0.0)

    def _wait_for_rate_limit(self):
        with self._rate_limit_lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.time()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """重写 _generate，添加限速 + 重试"""
        for attempt in range(self._max_retries):
            try:
                self._wait_for_rate_limit()
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                error_msg = str(e)
                is_rate_limit = '429' in error_msg or 'rpm exhausted' in error_msg or 'rate' in error_msg.lower()
                if is_rate_limit and attempt < self._max_retries - 1:
                    wait_time = self._base_wait * (2 ** attempt)
                    print(f"  ⚠ API 限流，等待 {wait_time:.0f}s 后重试 ({attempt+1}/{self._max_retries})")
                    time.sleep(wait_time)
                else:
                    raise


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class EvalSample:
    """单个评估样本"""
    question: str
    ground_truth: str
    contexts: List[str]
    answer: str = ""
    retrieval_context: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """评估结果"""
    dataset_name: str
    retriever_type: str
    metrics: Dict[str, float]
    samples: List[Dict]
    timestamp: str = ""


# ============================================================================
# RAGAS 评估器
# ============================================================================

class RAGASEvaluator:
    """基于 RAGAS 的 RAG 评估器"""

    def __init__(self, llm=None, embeddings=None):
        """
        初始化评估器

        Args:
            llm: LangChain ChatOpenAI 实例（用于生成答案和 RAGAS 评估）
            embeddings: LangChain Embeddings 实例（用于 RAGAS 评估）
        """
        if llm is None:
            from app.config import llm as default_llm
            llm = default_llm
        if embeddings is None:
            from app.config import embeddings as default_embeddings
            embeddings = default_embeddings

        self.llm = llm
        self.embeddings = embeddings

        # 仅对 DeepSeek (SenseNova) 模型启用限速，其他模型直接使用
        base_url = getattr(llm, 'openai_api_base', '') or ''
        is_deepseek = 'sensenova' in base_url.lower() or 'deepseek' in getattr(llm, 'model_name', '').lower()

        if is_deepseek:
            eval_llm = RateLimitedChatOpenAI(
                model=llm.model_name,
                api_key=llm.openai_api_key,
                base_url=llm.openai_api_base,
                temperature=llm.temperature,
                min_interval=1.5,
                max_retries=5,
                base_wait=5.0,
            )
        else:
            eval_llm = llm

        # 初始化 RAGAS 指标
        self.metrics = [
            Faithfulness(llm=eval_llm),
            AnswerRelevancy(llm=eval_llm, embeddings=embeddings),
            ContextPrecision(llm=eval_llm),
            ContextRecall(llm=eval_llm),
        ]

        # RAG 生成 prompt
        self.generation_prompt = PromptTemplate(
            template="""基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法找到答案。

Question: {question}
Context: {context}

Answer:""",
            input_variables=["question", "context"],
        )

    def generate_answer(self, question: str, context: str) -> str:
        """使用 RAG 链生成答案"""
        chain = self.generation_prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})

    def prepare_dataset(
        self,
        test_data: List[Dict],
        retriever=None,
        retriever_type: str = "vector"
    ) -> Dataset:
        """
        准备 RAGAS 评估数据集

        Args:
            test_data: QA 测试数据列表
            retriever: LangChain Retriever 实例（可选，如果提供则使用检索结果）
            retriever_type: 检索器类型标识

        Returns:
            HuggingFace Dataset 格式的评估数据
        """
        questions = []
        ground_truths = []
        contexts_list = []
        answers = []

        for sample in test_data:
            question = sample["question"]
            ground_truth = sample["ground_truth"]
            predefined_contexts = sample.get("contexts", [])

            # 如果提供了检索器，使用检索结果
            if retriever is not None:
                try:
                    docs = retriever.invoke(question)
                    retrieved_contexts = [doc.page_content for doc in docs]
                except Exception as e:
                    print(f"  ⚠ 检索失败: {e}，使用预定义上下文")
                    retrieved_contexts = predefined_contexts
            else:
                retrieved_contexts = predefined_contexts

            # 使用检索到的上下文生成答案
            context_str = "\n\n".join(retrieved_contexts)
            answer = self.generate_answer(question, context_str)

            questions.append(question)
            ground_truths.append(ground_truth)  # RAGAS 0.4.x 要求 string 格式
            contexts_list.append(retrieved_contexts)
            answers.append(answer)

        # 构建 RAGAS Dataset — 使用 RAGAS 0.4.x v2 字段名
        # 重要：不能同时包含 ground_truth 和 reference，否则会创建重复列
        dataset = Dataset.from_dict({
            "user_input": questions,            # v2: user_input (was "question")
            "response": answers,                # v2: response (was "answer")
            "retrieved_contexts": contexts_list, # v2: retrieved_contexts (was "contexts")
            "reference": ground_truths,         # v2: reference (was "ground_truth")
        })

        return dataset

    def evaluate(
        self,
        test_data: List[Dict],
        dataset_name: str = "unknown",
        retriever=None,
        retriever_type: str = "vector"
    ) -> EvalResult:
        """
        运行完整评估流程

        Args:
            test_data: QA 测试数据
            dataset_name: 数据集名称
            retriever: 检索器实例
            retriever_type: 检索器类型

        Returns:
            EvalResult 评估结果
        """
        from datetime import datetime

        print(f"\n{'='*60}")
        print(f"📊 RAGAS 评估: {dataset_name} ({retriever_type})")
        print(f"{'='*60}")
        print(f"📝 测试样本数: {len(test_data)}")

        # 准备数据集
        print(f"\n🔨 准备评估数据集...")
        dataset = self.prepare_dataset(test_data, retriever, retriever_type)
        print(f"✅ 数据集准备完成")

        # 打印数据集信息
        print(f"\n📋 数据集预览:")
        for i, row in enumerate(dataset):
            print(f"  [{i+1}] Q: {row['user_input'][:50]}...")
            print(f"      A: {row['response'][:80]}...")

        # 运行 RAGAS 评估（max_workers=3 并行执行，提速）
        from ragas.run_config import RunConfig
        run_config = RunConfig(max_workers=1, max_wait=600, max_retries=5)

        print(f"\n🔄 正在运行 RAGAS 评估（并行模式，max_workers=3）...")
        result = ragas_evaluate(
            dataset=dataset,
            metrics=self.metrics,
            run_config=run_config,
        )

        # 提取结果 — result.scores 是 list[dict]，每个样本一个 dict
        all_scores = result.scores  # List[Dict[str, float]]
        metric_names = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

        metrics = {}
        for name in metric_names:
            values = [s[name] for s in all_scores if name in s and s[name] is not None and not math.isnan(s[name])]
            metrics[name] = sum(values) / len(values) if values else 0.0

        # 计算综合分（仅使用有效指标）
        valid_metrics = [v for v in metrics.values() if v > 0]
        metrics['overall_score'] = sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0.0

        # 打印结果
        print(f"\n{'─'*60}")
        print(f"📊 评估结果:")
        print(f"{'─'*60}")
        metric_names = {
            'faithfulness': 'Faithfulness (忠实度)',
            'answer_relevancy': 'Answer Relevancy (答案相关性)',
            'context_precision': 'Context Precision (上下文精确度)',
            'context_recall': 'Context Recall (上下文召回率)',
            'overall_score': 'Overall Score (综合分)',
        }
        for key, value in metrics.items():
            name = metric_names.get(key, key)
            print(f"  {name}: {value:.4f}")
        print(f"{'─'*60}")

        # 收集样本详情
        sample_details = []
        for i, row in enumerate(dataset):
            sample_details.append({
                "question": row["user_input"],
                "answer": row["response"],
                "ground_truth": row["reference"],
                "contexts": row["retrieved_contexts"],
            })

        eval_result = EvalResult(
            dataset_name=dataset_name,
            retriever_type=retriever_type,
            metrics=metrics,
            samples=sample_details,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        return eval_result


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(
    results: List[EvalResult],
    output_dir: str = "evaluation_results"
) -> str:
    """
    生成 JSON + 文本评估报告

    Args:
        results: 评估结果列表
        output_dir: 输出目录

    Returns:
        报告文件路径
    """
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)

    # JSON 报告
    report_data = {
        "evaluation_summary": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_evaluations": len(results),
            "results": []
        }
    }

    for result in results:
        report_data["evaluation_summary"]["results"].append({
            "dataset": result.dataset_name,
            "retriever": result.retriever_type,
            "metrics": result.metrics,
            "sample_count": len(result.samples),
        })

    # 保存 JSON
    json_path = os.path.join(output_dir, f"ragas_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"📊 评估汇总报告")
    print(f"{'='*60}")

    for result in results:
        print(f"\n🎯 {result.dataset_name} ({result.retriever_type}):")
        for key, value in result.metrics.items():
            print(f"   {key}: {value:.4f}")

    print(f"\n📄 JSON 报告: {json_path}")
    print(f"{'='*60}")

    return json_path
