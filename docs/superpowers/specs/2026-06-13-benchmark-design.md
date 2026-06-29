# RAG 系统多维度优化 Benchmark 设计文档

## 目标

为面试准备量化数据，证明系统在多个层面做了有数据支撑的优化。输出可执行的 benchmark 脚本 + 自动生成的对比报告。

## 核心原则

- 不修改 `app/` 下的现有代码
- benchmark.py 是独立脚本，复用现有模块
- 每个实验控制变量，只改一个因素
- 输出控制台表格 + JSON + Markdown 报告

## 测试数据集

### 数据来源

| 数据集 | 来源 | 文本特点 | 用途 |
|---|---|---|---|
| DNNGP | `pdf/DNNGP*.pdf` | 学术论文，术语密集，长段落 | RAGAS 主评估 |
| Company | `doc/company.txt` | 结构化描述，短段落 | RAGAS 辅评估 |

### 测试题数量

| 类型 | 数量 | 生成方式 |
|---|---|---|
| DNNGP RAGAS 题 | 30 题 | LLM 自动生成 + 人工审核 |
| Company RAGAS 题 | 20 题 | LLM 自动生成 + 人工审核 |
| 路由测试题 | 25 题 | 手动出题，覆盖 4 种 worker 类型 |

### 测试数据格式

RAGAS 题（复用现有格式）：
```json
[
  {
    "question": "DNNGP 的核心创新点是什么？",
    "ground_truth": "DNNGP 提出了...",
    "contexts": ["参考文档片段..."]
  }
]
```

路由题：
```json
[
  {
    "question": "查询华为的销售额",
    "expected_worker": "sqler"
  }
]
```

## 实验设计

### 实验 1：分块策略对比

**目的：** 证明语义分块比固定分块更适合复杂文档

**控制变量：** 同一文档、同一检索参数（top_k=5）、同一 LLM

**对比项：**

| 策略 | 分块方式 | 元信息增强 | 父子结构 |
|---|---|---|---|
| basic | RecursiveCharacterTextSplitter(chunk_size=400) | ❌ | ❌ |
| semantic_metadata | SemanticChunker + metadata | ✅ | ❌ |
| parent_child | SemanticChunker + metadata | ✅ | ✅ |

**指标：** RAGAS 4 项 + chunk 数量 + 平均 chunk 大小（字符数）

**实现流程：**
1. 用 `create_enhanced_chunker(strategy)` 分块
2. 建临时 Milvus collection（`benchmark_{strategy}_{timestamp}`）
3. 调 `RAGASEvaluator` 跑 RAGAS
4. 清理临时 collection

### 实验 2：检索参数对比

**目的：** 找到 precision/recall 的最优平衡点

**控制变量：** 同一分块策略（semantic_metadata）、同一文档

**对比项：** top_k = 3 / 5 / 8

**指标：** RAGAS 4 项 + 平均检索延迟（ms）

**实现流程：**
1. 复用实验 1 的 semantic_metadata vectorstore
2. 对每个 top_k，创建 retriever 并计时
3. 调 `RAGASEvaluator` 跑 RAGAS

### 实验 3：混合检索对比

**目的：** 证明 Dense+Sparse 混合检索比纯向量检索更好

**控制变量：** 同一分块策略、同一 top_k（5）

**对比项：**
- A：纯向量检索（baseline）
- B：混合检索（向量 0.7 + BM25 0.3，RRF 合并）

**指标：** RAGAS 4 项

**实现流程：**
1. 从 vectorstore 提取所有文档
2. 用 `rank_bm25` 建 BM25 索引
3. 向量检索取 top 10 + BM25 取 top 10
4. RRF 合并（k=60），取 top 5
5. 调 `RAGASEvaluator` 跑 RAGAS

**依赖：** `pip install rank_bm25`

### 实验 4：Reranking 对比

**目的：** 证明两阶段检索（先召回再精排）能提升质量

**控制变量：** 同一分块策略、同一文档

**对比项：**
- A：无 rerank，retriever(k=5)
- B：retriever(k=8) → DashScope gte-rerank → top_5

**指标：** RAGAS 4 项 + rerank 额外延迟（ms）

**实现流程：**
1. 向量检索取 top 8
2. 调 DashScope `TextRerank` API 重排
3. 取 top 5 作为最终上下文
4. 调 `RAGASEvaluator` 跑 RAGAS

**依赖：** DashScope API（已有）

### 实验 5：路由准确率

**目的：** 证明 cycle detection + EMA 策略的有效性

**控制变量：** 同一组测试题

**对比项：**
- A：无 cycle detection（模拟原始 Supervisor）
- B：有 cycle detection + fallback（当前实现）

**指标：** 首次命中率 + 平均路由轮次 + 循环率

**实现流程：**
1. 跑 25 个测试题
2. 捕获每次 Supervisor 的路由决策（state["next"]）
3. 对比 expected_worker
4. 统计首次命中率、总轮次、循环次数

**注意：** 需要模拟"无 cycle detection"的场景——可以通过临时禁用 RoutingPolicyManager 实现

## 文件结构

```
benchmark.py                  # 主脚本（所有实验 + 报告生成）
generate_test_data.py         # 测试数据生成脚本（LLM 批量出题）
benchmark_data/               # 测试数据目录
├── dnngp_test_data.json
├── company_test_data.json
└── routing_test_data.json
benchmark_results/            # 输出目录
├── benchmark_<timestamp>.json
└── benchmark_report.md
```

## 输出格式

### 控制台

每个实验输出一张 Markdown 格式的对比表，带最优值标记。

### Markdown 报告结构

```markdown
# RAG 系统优化对比报告
生成时间: YYYY-MM-DD HH:MM:SS
数据集: DNNGP (30 questions)

## 实验 1: 分块策略对比
**问题：** ...
**方案：** ...
**结果：** [对比表格]
**结论：** ...

## 实验 2: 检索参数对比
...（同上结构）

## 实验 3: 混合检索对比
...

## 实验 4: Reranking 对比
...

## 实验 5: 路由准确率
...

## 总结
各维度最优配置汇总表
```

## 用法

```bash
# 生成测试数据
python generate_test_data.py --source dnngp --count 30
python generate_test_data.py --source company --count 20

# 跑全部实验
python benchmark.py

# 只跑单个实验
python benchmark.py --only chunking
python benchmark.py --only top_k
python benchmark.py --only hybrid
python benchmark.py --only rerank
python benchmark.py --only routing

# 指定数据集
python benchmark.py --dataset dnngp
python benchmark.py --dataset company
python benchmark.py --dataset all
```

## 依赖

| 依赖 | 用途 | 状态 |
|---|---|---|
| rank_bm25 | BM25 稀疏检索 | 需安装 |
| dashscope | Rerank API | 已有 |
| ragas | 评估框架 | 已有 |
| langchain / milvus | 向量存储 | 已有 |

## 风险

1. **API 限流：** RAGAS 评估调用频繁，可能触发 DashScope 限流 → 用现有 `RateLimitedChatOpenAI` 解决
2. **Milvus collection 冲突：** 多次实验可能命名冲突 → 用时间戳后缀避免
3. **DNNGP PDF 解析失败：** 依赖 pdfplumber → 有 fallback 到 PyMuPDF
4. **测试题质量差：** LLM 生成的题可能偏离文档 → 人工审核环节
