# RAGAS 评估实现详解

## 目录

- [一、评估思路](#一评估思路)
- [二、核心代码文件](#二核心代码文件)
- [三、测试数据文件](#三测试数据文件)
- [四、评估结果](#四评估结果)
- [五、文件清单](#五文件清单)
- [六、使用方法](#六使用方法)

---

## 一、评估思路

### 1.1 为什么替换旧框架

旧的 `llm_based_evaluation.py` 存在三个根本问题：

| 问题 | 说明 |
|------|------|
| **自我评估偏差** | 用同一个 LLM（qwen-plus）生成答案和评估答案，LLM 倾向于给自己的输出打高分 |
| **指标定义模糊** | `context_relevance` vs `context_completeness`、`answer_faithfulness` vs `answer_accuracy` 边界不清 |
| **无统计意义** | 几个样本的平均分没有置信区间，且 `semantic_similarity` 用 LLM 文本判断而非 embedding 向量 |

### 1.2 RAGAS 的 4 个核心指标

RAGAS（Retrieval Augmented Generation Assessment）是 RAG 评估领域的标准开源框架，使用 4 个精确指标：

| 指标 | 含义 | 评估方式 |
|------|------|----------|
| **Faithfulness** | 答案是否忠实于检索到的上下文（防幻觉） | 将答案拆分为原子陈述，逐一验证是否可从上下文推导 |
| **Answer Relevancy** | 答案是否与问题相关 | 从答案反向生成问题，计算与原问题的语义相似度 |
| **Context Precision** | 检索结果中相关文档的排名是否靠前 | 基于排序的精确率，相关文档位置越靠前得分越高 |
| **Context Recall** | 是否检索到了回答问题所需的所有信息 | 将 ground_truth 拆分为句子，检查被上下文覆盖的比例 |

### 1.3 评估流程

```
                    Ground Truth QA 测试集
                    (question + ground_truth + contexts)
                              │
                              ▼
                    ┌───────────────────┐
                    │  RAGASEvaluator   │
                    │  .prepare_dataset()│
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        使用 Retriever    调用 LLM       构建 HuggingFace
        检索真实上下文    生成答案        Dataset
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ragas_evaluate()
                    计算 4 个指标
                              │
                              ▼
                    生成 JSON 评估报告
```

**流程说明：**

1. **加载测试数据**：从 JSON 文件读取 `question`、`ground_truth`、`contexts`
2. **检索上下文**：如果提供了 Retriever（如 Milvus 向量检索），使用它检索真实上下文；否则使用预定义的 contexts
3. **生成答案**：调用 LLM 基于检索到的上下文生成答案
4. **构建数据集**：将 question/answer/contexts/ground_truth 组装为 HuggingFace Dataset 格式
5. **RAGAS 评估**：调用 `ragas_evaluate()` 计算 4 个指标
6. **生成报告**：输出 JSON 格式的评估结果

---

## 二、核心代码文件

### 2.1 评估模块：`app/evaluation.py`

**类结构：**

```python
class RAGASEvaluator:
    def __init__(self, llm, embeddings)           # 初始化 RAGAS 指标
    def generate_answer(self, question, context)   # RAG 链生成答案
    def prepare_dataset(self, test_data, retriever)# 构建 RAGAS 数据集
    def evaluate(self, test_data, dataset_name, retriever)  # 完整评估流程
```

**关键实现细节：**

#### ① 修复 RAGAS 的 vertexai 导入问题

RAGAS 0.4.x 内部尝试导入 `langchain_community.chat_models.vertexai`，但该模块不存在。使用 mock 模块修复：

```python
import types
if 'langchain_community.chat_models.vertexai' not in sys.modules:
    mock_module = types.ModuleType('langchain_community.chat_models.vertexai')
    mock_module.ChatVertexAI = None
    sys.modules['langchain_community.chat_models.vertexai'] = mock_module
```

#### ② 使用旧式指标类

RAGAS 0.4.x 的 `ragas.metrics.collections` 新式指标要求 `InstructorLLM`，不兼容 LangChain 的 `ChatOpenAI`。使用旧式指标类（带下划线前缀）解决：

```python
from ragas.metrics import (
    _Faithfulness as Faithfulness,
    _AnswerRelevancy as AnswerRelevancy,
    _ContextPrecision as ContextPrecision,
    _ContextRecall as ContextRecall,
)
```

#### ③ 数据集格式适配

RAGAS 0.4.x 要求 `ground_truth` 为 string 格式（而非 list），且使用 `reference` 字段：

```python
ground_truths.append(ground_truth)  # string，不是 [ground_truth]

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts_list,
    "ground_truth": ground_truths,
    "reference": ground_truths,  # RAGAS 0.4.x 使用 reference 字段
})
```

#### ④ 结果提取处理 nan

RAGAS 评估可能因 API 超时返回 nan，需要过滤：

```python
all_scores = result.scores  # List[Dict[str, float]]
for name in metric_names:
    values = [s[name] for s in all_scores
              if name in s and s[name] is not None and not math.isnan(s[name])]
    metrics[name] = sum(values) / len(values) if values else 0.0
```

### 2.2 命令行入口：`evaluate_rag.py`

提供 CLI 接口，支持以下参数：

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `--source` | `company`, `dnngp`, `all` | 评估哪个知识库 |
| `--retriever` | `vec`, `graph` | 使用哪个检索器 |
| `--output` | 目录路径 | 结果输出目录，默认 `evaluation_results` |

**核心流程：**

```python
def main():
    # 1. 解析命令行参数
    args = parser.parse_args()

    # 2. 初始化评估器（使用项目的 LLM 和 Embeddings）
    evaluator = RAGASEvaluator(llm=llm, embeddings=embeddings)

    # 3. 获取检索器（Milvus 向量检索 or Neo4j 图检索）
    retriever = get_retriever(args.retriever)

    # 4. 逐数据源评估
    for source in sources:
        test_data = load_test_data(source)
        result = evaluator.evaluate(test_data, retriever=retriever)
        all_results.append(result)

    # 5. 生成报告
    generate_report(all_results, output_dir=args.output)
```

---

## 三、测试数据文件

### 3.1 公司知识库：`evaluation_test_data/company_test_data.json`

基于 `doc/company.txt`（全球科技企业竞争格局分析报告）构建，**12 个 QA 对**：

| # | 问题 | 类型 |
|---|------|------|
| 1 | 小米的快充技术发展经历了哪些关键节点？ | 单事实检索 |
| 2 | 华为在5G领域有哪些重要成就？ | 单事实检索 |
| 3 | 苹果的Vision Pro混合现实头显有哪些技术特点？ | 单事实检索 |
| 4 | 三星在半导体领域有哪些核心业务？ | 单事实检索 |
| 5 | 比亚迪的刀片电池有什么技术优势？ | 单事实检索 |
| 6 | 小米SU7电动汽车有哪些核心技术？ | 单事实检索 |
| 7 | 华为的鸿蒙操作系统与小米的澎湃OS有什么区别？ | 多段落对比 |
| 8 | 比亚迪的DM-i超级混动系统有什么特点？ | 单事实检索 |
| 9 | 五家企业在AI大模型领域各自的布局是什么？ | 跨段落综合 |
| 10 | 华为智选车模式包含哪些合作品牌？ | 单事实检索 |
| 11 | 比亚迪的全球化扩张策略是什么？ | 跨段落综合 |
| 12 | 苹果的芯片策略与三星和华为有什么不同？ | 多段落对比 |

**数据格式：**

```json
{
  "question": "小米的快充技术发展经历了哪些关键节点？",
  "ground_truth": "小米在2021年率先量产120W有线快充方案，实测15分钟即可将手机电池从零充满...",
  "contexts": [
    "快充领域是小米最具辨识度的技术标签之一...",
    "快充技术并非小米孤立开发的成果..."
  ]
}
```

- `question`：用户问题
- `ground_truth`：标准答案（从原文提取，用于 Context Recall 评估）
- `contexts`：期望检索到的上下文（当 Retriever 不可用时作为 fallback）

### 3.2 DNNGP 论文：`evaluation_test_data/dnngp_test_data.json`

基于 DNNGP 学术论文构建，**12 个 QA 对**，覆盖论文核心内容：

| # | 问题 | 覆盖内容 |
|---|------|----------|
| 1 | DNNGP 是什么？它的主要用途是什么？ | 方法定义 |
| 2 | DNNGP 相比传统方法有哪些创新点？ | 5 个创新点 |
| 3 | DNNGP 的神经网络结构是怎样的？ | 网络架构 |
| 4 | DNNGP 在实验中与哪些方法进行了对比？结果如何？ | GBLUP, LightGBM, SVR 等 |
| 5 | DNNGP 使用了哪些多组学数据？ | 基因组、转录组等 |
| 6 | DNNGP 的计算性能如何？相比 DeepGS 有什么优势？ | 计算效率对比 |
| 7 | DNNGP 如何防止过拟合？ | 批量归一化、早停 |
| 8 | DNNGP 在小数据集和大数据集上的表现有何不同？ | 性能差异 |
| 9 | DNNGP 的超参数如何调优？有什么优势？ | 本地批量调优 |
| 10 | DNNGP 在植物育种中的实际应用价值是什么？ | 基因组选择平台 |
| 11 | 传统基因组预测方法有什么局限性？ | 线性模型不足 |
| 12 | DNNGP 适用于哪些作物和研究场景？ | 小麦、玉米等 |

---

## 四、评估结果

### 公司知识库评估（已完成）

| 指标 | 得分 | 含义 |
|------|------|------|
| Faithfulness | **1.0000** | 答案完全忠实于检索上下文，无幻觉 |
| Answer Relevancy | **0.8981** | 答案与问题高度相关 |
| Context Precision | **1.0000** | 检索到的上下文完全正确 |
| Context Recall | **1.0000** | 检索到了所有必要信息 |
| **Overall Score** | **0.9745** | 综合分 |

**结果分析：**
- RAG 系统检索到了正确的上下文（Context Precision/Recall = 1.0）
- 生成的答案忠实于上下文（Faithfulness = 1.0）
- 答案与问题相关性高（Answer Relevancy = 0.898），略有提升空间

### DNNGP 论文评估

DNNGP 评估因 DashScope API 账户欠费失败（`Arrearage` 错误），需充值后重试。

### 常见评估错误及处理

| 错误类型 | 原因 | 处理方式 |
|----------|------|----------|
| `RateLimitError (429)` | DashScope API 请求频率超限 | RAGAS 自动重试，不影响最终结果 |
| `TimeoutError` | LLM 响应超时 | 该指标计为 nan，计算时自动过滤 |
| `ValidationError (JSON)` | LLM 返回的 JSON 格式不规范 | 该样本该指标跳过，不影响其他样本 |
| `Arrearage (400)` | DashScope 账户欠费 | 需充值后重试 |

---

## 五、文件清单

| 文件 | 作用 |
|------|------|
| `app/evaluation.py` | RAGAS 评估核心模块（`RAGASEvaluator` 类 + `generate_report` 函数） |
| `evaluate_rag.py` | 命令行入口（`--source` / `--retriever` / `--output`） |
| `evaluation_test_data/company_test_data.json` | 公司知识库 Ground Truth（12 QA 对） |
| `evaluation_test_data/dnngp_test_data.json` | DNNGP 论文 Ground Truth（12 QA 对） |
| `evaluation_results/ragas_evaluation_*.json` | 评估结果报告（自动生成） |

---

## 六、使用方法

### 前置条件

```bash
# 1. 启动 Docker 服务
docker-compose -f docker-compose-rag.yml up -d

# 2. 安装依赖
pip install ragas>=0.4.0 datasets>=4.0.0
```

### 运行评估

```bash
# 评估公司知识库（向量检索）
python evaluate_rag.py --source company --retriever vec

# 评估 DNNGP 论文（向量检索）
python evaluate_rag.py --source dnngp --retriever vec

# 评估全部数据源
python evaluate_rag.py --source all --retriever vec

# 指定输出目录
python evaluate_rag.py --source all --retriever vec --output my_results
```

### 输出示例

```
============================================================
📊 RAGAS 评估: company (vec)
============================================================
📝 测试样本数: 12

🔨 准备评估数据集...
✅ 数据集准备完成

🔄 正在运行 RAGAS 评估...
Evaluating: 100%|██████████| 48/48 [06:49<00:00,  8.53s/it]

────────────────────────────────────────────────────────────
📊 评估结果:
────────────────────────────────────────────────────────────
  Faithfulness (忠实度): 1.0000
  Answer Relevancy (答案相关性): 0.8981
  Context Precision (上下文精确度): 1.0000
  Context Recall (上下文召回率): 1.0000
  Overall Score (综合分): 0.9745
────────────────────────────────────────────────────────────

📄 JSON 报告: evaluation_results\ragas_evaluation_20260608_220125.json
```

---

## 七、扩展指南

### 添加新的测试数据集

1. 在 `evaluation_test_data/` 目录下创建新的 JSON 文件
2. 按照以下格式组织数据：

```json
[
  {
    "question": "问题文本",
    "ground_truth": "标准答案",
    "contexts": ["期望检索到的上下文1", "期望检索到的上下文2"]
  }
]
```

3. 在 `evaluate_rag.py` 的 `load_test_data()` 函数中添加新数据源的加载逻辑

### 自定义评估指标

在 `app/evaluation.py` 的 `RAGASEvaluator.__init__()` 中修改 `self.metrics` 列表：

```python
from ragas.metrics import _Faithfulness, _AnswerRelevancy, _ContextPrecision, _ContextRecall

self.metrics = [
    _Faithfulness(llm=llm),
    _AnswerRelevancy(llm=llm, embeddings=embeddings),
    _ContextPrecision(llm=llm),
    _ContextRecall(llm=llm),
    # 添加更多指标...
]
```
