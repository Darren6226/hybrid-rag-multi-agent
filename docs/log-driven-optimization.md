# 日志驱动调优（Log-Driven Optimization）实现方案

> 基于 hybrid-rag-multi-agent 项目的现状，设计一套可落地的日志采集 → 错误聚类 → 反馈优化的闭环体系。

---

## 目录

1. [核心理念](#1-核心理念)
2. [现状分析](#2-现状分析)
3. [总体架构](#3-总体架构)
4. [Phase 1：结构化日志采集](#4-phase-1结构化日志采集)
5. [Phase 2：Badcase 采集与存储](#5-phase-2badcase-采集与存储)
6. [Phase 3：错误聚类与分析](#6-phase-3错误聚类与分析)
7. [Phase 4：自动化反馈优化](#7-phase-4自动化反馈优化)
8. [Phase 5：可视化与监控](#8-phase-5可视化与监控)
9. [落地路线图](#9-落地路线图)
10. [附录：关键代码示例](#10-附录关键代码示例)

---

## 1. 核心理念

所有真实项目走到最后，都要落到这一点：

```
用户查询 → Agent 路由 → 工具执行 → LLM 生成 → 输出响应
                                              ↓
                                        记录日志 ← ← ← ← ← ←
                                              ↓
                                        聚类分析 ← ← ← ← ← ←
                                              ↓
                             反推 Prompt / Schema / 路由优化
                                              ↓
                                         重新评估
                                              ↓
                                         部署上线 → → → → →
```

记录的核心字段（每条请求一条记录）：

| 字段 | 说明 | 来源 |
|------|------|------|
| `trace_id` | 请求追踪 ID | 入口生成 |
| `user_query` | 用户原始问题 | 入口捕获 |
| `session_id` | 会话 ID（可选） | 请求参数 |
| `supervisor_decision` | 路由决策序列 | supervisor 输出 |
| `workers_called` | 实际调用的 worker 列表 | 各 node 记录 |
| `worker_results` | 各 worker 的输出摘要 | 各 node 回调 |
| `tool_chosen` | tool 名称 + 参数 | agent 的 tool_calls |
| `llm_model` | 使用的模型名称 | config 记录 |
| `llm_prompt` | 完整 prompt（可选，按需开启） | LLM 调用前捕获 |
| `llm_response` | 完整响应（可选，按需开启） | LLM 调用后捕获 |
| `latency_ms` | 各阶段耗时 | 时间戳差值 |
| `token_usage` | token 消耗 | LLM 返回的 usage |
| `error_type` | 错误类型（聚类用） | 错误分类器 |
| `error_detail` | 错误详情 | 异常捕获 |
| `user_feedback` | 用户反馈（点赞/点踩） | 用户主动提交 |
| `success` | 是否成功 | 综合判定 |

经过聚类你会发现典型分布：

```
30% 的错是日期解析 / SQL 语法
20% 的错是酒店 location 语义不明
10% 的错是工具名太相似
 5% 的错是模型乱输出
...

```

---

## 2. 现状分析

### 2.1 已存在的

| 模块 | 说明 |
|------|------|
| `RoutingPolicyManager` | 内存级的路由状态追踪（attempts / errors / success_rate） |
| `_analyze_response_quality` | 响应质量分析（空值、错误关键词、失败模式） |
| `RAGASEvaluator` | 离线 RAGAS 评估（Faithfulness / Relevancy / Precision / Recall） |
| `benchmark.py` | 离线 benchmark 实验 |
| `api/schemas.py` | `ChatRequest.session_id` 字段已定义但未使用 |

### 2.2 缺失的

| 缺失能力 | 影响 |
|---------|------|
| ❌ 无持久化日志存储 | 进程重启后所有历史丢失 |
| ❌ 无 `user_query` 日志 | 无法回溯用户提了什么问题 |
| ❌ 无 LLM prompt/completion 日志 | 无法分析模型行为 |
| ❌ 无 tool call 参数日志 | 无法定位工具调用错误 |
| ❌ 无请求耗时追踪 | 无法做性能分析 |
| ❌ 无错误聚类 | 无法发现高频错误模式 |
| ❌ 无用户反馈收集 | 无法从用户侧判断回答质量 |
| ❌ 无自动反馈回路 | 所有优化靠手动分析 + 离线评估 |
| ❌ 无 trace_id | 无法关联单次请求的全链路数据 |

---

## 3. 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        业务层（现有）                            │
│  main.py / api/main.py → graph_builder → supervisor → nodes    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 埋点采集（最小侵入）
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Phase 1: 结构化日志采集                        │
│                                                                 │
│  TraceContext(trace_id, start_time) 贯穿全链路                    │
│  ├─ 入口: 生成 trace_id, 记录 user_query, session_id            │
│  ├─ supervisor: 记录 routing 决策                                │
│  ├─ worker: 记录调用结果、tool_calls                             │
│  └─ LLM: 记录 prompt、completion、token_usage（可选）            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Phase 2: Badcase 存储                          │
│                                                                 │
│  SQLite / MySQL 日志表（轻量级方案）                              │
│  或 本地 JSONL 文件（极简方案）                                  │
│  或 外部存储（ElasticSearch / ClickHouse 进阶方案）               │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Phase 3: 错误聚类与分析（离线 / 定时任务）           │
│                                                                 │
│  1. 提取 error_type / error_detail                              │
│  2. 关键词聚类（规则） → LLM 聚类（语义） → 人工确认             │
│  3. 输出: 高频错误 Top-N + 根因分析                             │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Phase 4: 自动化反馈优化                             │
│                                                                 │
│  聚类结果 → 针对性地:                                           │
│  ├─ Prompt 优化（更新 agent system prompt）                     │
│  ├─ Schema 优化（更新 few-shot example / tool 描述）             │
│  ├─ 路由优化（调整 WorkerConfig.priority / fallback）            │
│  └─ 生成优化建议报告（人工审核后上线）                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1：结构化日志采集

### 4.1 核心数据结构

```python
# app/tracing.py
import uuid
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """单次 tool call 的记录"""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: str  # 截断后的结果摘要
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class WorkerRecord:
    """单个 worker 的执行记录"""
    worker_name: str
    success: bool
    content_summary: str  # 输出内容摘要（前 200 字符）
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class TraceRecord:
    """单次请求的全链路记录"""
    trace_id: str
    user_query: str
    session_id: Optional[str] = None
    timestamp: float = 0.0
    supervisor_decisions: List[str] = field(default_factory=list)  # 路由决策序列
    workers: List[WorkerRecord] = field(default_factory=list)
    llm_model: str = ""
    total_latency_ms: float = 0.0
    total_token_usage: int = 0
    user_feedback: Optional[int] = None  # 1=赞, -1=踩, None=未评价
    success: bool = True
    error_type: Optional[str] = None
    error_detail: Optional[str] = None

    def to_log_dict(self) -> dict:
        """转为可序列化的字典（用于写入日志存储）"""
        return asdict(self)
```

### 4.2 TraceContext 上下文

```python
# app/tracing.py (续)

class TraceContext:
    """
    贯穿单次请求全链路的追踪上下文。
    使用 contextvars 确保异步场景下线程安全。
    """
    _contextvar = None  # 暂用 threading.local

    @classmethod
    def _get_storage(cls):
        if not hasattr(cls, "_storage"):
            cls._storage = threading.local()
        return cls._storage

    @classmethod
    def begin(cls, user_query: str, session_id: Optional[str] = None) -> str:
        """开始一个新的追踪，返回 trace_id"""
        storage = cls._get_storage()
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        storage.current = TraceRecord(
            trace_id=trace_id,
            user_query=user_query,
            session_id=session_id,
            timestamp=time.time(),
        )
        logger.debug("[Trace %s] 开始追踪: %s", trace_id, user_query[:80])
        return trace_id

    @classmethod
    def current(cls) -> Optional[TraceRecord]:
        """获取当前请求的追踪记录"""
        storage = cls._get_storage()
        return getattr(storage, "current", None)

    @classmethod
    def add_supervisor_decision(cls, decision: str):
        """记录 supervisor 的一次路由决策"""
        record = cls.current()
        if record:
            record.supervisor_decisions.append(decision)

    @classmethod
    def add_worker(cls, worker_name: str, success: bool, content: str,
                   error: Optional[str] = None, latency_ms: float = 0.0):
        """记录一个 worker 的执行结果"""
        record = cls.current()
        if record:
            record.workers.append(WorkerRecord(
                worker_name=worker_name,
                success=success,
                content_summary=content[:200],
                error=error,
                latency_ms=latency_ms,
            ))

    @classmethod
    def add_tool_call(cls, worker_name: str, tool_name: str,
                      tool_args: dict, tool_result: str,
                      error: Optional[str] = None, latency_ms: float = 0.0):
        """记录 worker 内部的一次 tool call"""
        record = cls.current()
        if record:
            for w in record.workers:
                if w.worker_name == worker_name:
                    w.tool_calls.append(ToolCallRecord(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_result=tool_result[:300],
                        error=error,
                        latency_ms=latency_ms,
                    ))
                    break

    @classmethod
    def finish(cls, success: bool = True, error_type: Optional[str] = None,
               error_detail: Optional[str] = None):
        """结束追踪，设置最终状态"""
        record = cls.current()
        if record:
            record.total_latency_ms = (time.time() - record.timestamp) * 1000
            record.success = success
            record.error_type = error_type
            record.error_detail = error_detail
            logger.debug("[Trace %s] 完成追踪, 耗时 %.0fms",
                         record.trace_id, record.total_latency_ms)

    @classmethod
    def end_and_export(cls) -> Optional[TraceRecord]:
        """结束追踪并导出记录（清空上下文）"""
        record = cls.current()
        if record:
            cls.finish()
            storage = cls._get_storage()
            result = record
            storage.current = None
            return result
        return None
```

### 4.3 侵入点对接

在现有代码中，只需在**关键节点添加 3~5 行代码**即可完成全链路日志：

#### 入口（`api/service.py` / `main.py`）

```python
# api/service.py 的 chat_stream 函数
def chat_stream(query: str) -> Generator[dict, None, None]:
    trace_id = TraceContext.begin(query)
    graph = get_compiled_graph()
    reset_supervisor()

    # 注入 trace_id 到 streaming 上下文（若需要）
    yield {"node": "system", "content": "", "done": False,
           "trace_id": trace_id}

    for event in sse_stream(graph, query):
        yield event

    # 导出一条日志（异步写入存储）
    record = TraceContext.end_and_export()
    if record:
        LogStorage.write(record)
```

#### Supervisor（`app/supervisor.py`）

```python
# 在 supervisor 函数的 LLM 决策后添加
TraceContext.add_supervisor_decision(next_)

# 在 _analyze_response_quality 或 record_attempt 后
# 分析结果已经蕴含在 policy_manager.record_attempt 中
```

#### Worker 节点（`app/nodes.py`）

```python
# 在每个 worker 函数的开头 + 结尾
def sqler_node(state: AgentState):
    t0 = time.time()
    try:
        result = db_agent.invoke(state)
        content = result["messages"][-1].content
        # 记录 worker 执行结果
        TraceContext.add_worker("sqler", True, content,
                                latency_ms=(time.time() - t0) * 1000)
        return {"messages": [HumanMessage(content=content, name="sqler")]}
    except Exception as e:
        TraceContext.add_worker("sqler", False, "", str(e),
                                latency_ms=(time.time() - t0) * 1000)
        raise
```

#### Agent Tool 调用（`app/agents.py` 的 `tool_node`）

```python
def tool_node(state: ReActAgentState):
    last_msg = state["messages"][-1]
    tool_messages = []
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tool_call in last_msg.tool_calls:
            t0 = time.time()
            try:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                res = tool_map[tool_name].invoke(tool_args)
                tool_messages.append(...)
                TraceContext.add_tool_call(
                    worker_name="?",  # 需从上下文推断当前 worker
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=str(res)[:300],
                    latency_ms=(time.time() - t0) * 1000,
                )
            except Exception as e:
                ...
    return {"messages": tool_messages}
```

> **注意**：`tool_node` 需要知道当前是哪个 worker 在执行。可以通过 `state["messages"]` 之前的 worker 消息推断，或者在 `create_react_agent_v1` 的 `think_node` 中设置一个上下文标记。

---

## 5. Phase 2：Badcase 采集与存储

### 5.1 存储层抽象

先定义一个存储接口，支持多种后端：

```python
# app/log_storage.py
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
from app.tracing import TraceRecord


class LogStorage(ABC):
    """日志存储抽象接口"""

    @abstractmethod
    def write(self, record: TraceRecord):
        ...

    @abstractmethod
    def query(self, **filters) -> list[TraceRecord]:
        ...

    @abstractmethod
    def get_error_stats(self, since: Optional[str] = None) -> dict:
        """获取错误统计（用于聚类分析）"""
        ...
```

### 5.2 极简方案：JSONL 文件

```python
# app/log_storage.py (续)

class JsonlLogStorage(LogStorage):
    """
    JSONL 文件存储（极简方案，零依赖）

    优点：无需外部存储，可直接用于小规模项目
    缺点：不支持高效查询，不适合大规模生产
    适用：单机开发 / 日均 < 1000 请求
    """

    def __init__(self, log_dir: str = "logs/traces"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _get_today_file(self) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.log_dir, f"traces_{date_str}.jsonl")

    def write(self, record: TraceRecord):
        line = json.dumps(record.to_log_dict(), ensure_ascii=False)
        with open(self._get_today_file(), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def query(self, **filters) -> list[TraceRecord]:
        # 简单实现：逐行读取并按 filters 过滤
        # 生产环境请替换为 ElasticSearch / SQL
        results = []
        # ... 略
        return results

    def get_error_stats(self, since: Optional[str] = None) -> dict:
        # 遍历日志，聚合 error_type 计数
        # ... 略
        return {}
```

### 5.3 轻量方案：SQLite / MySQL

使用 SQLite（零部署）或项目现成的 MySQL，建表：

```sql
-- 日志主表
CREATE TABLE IF NOT EXISTS trace_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT NOT NULL UNIQUE,
    user_query      TEXT NOT NULL,
    session_id      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 路由信息
    supervisor_decisions TEXT,      -- JSON array: ["vec_kg", "sqler", "FINISH"]

    -- 性能
    total_latency_ms    REAL,
    total_token_usage   INTEGER,

    -- LLM
    llm_model       TEXT,

    -- 结果
    success         BOOLEAN DEFAULT 1,
    error_type      TEXT,           -- 聚类用的错误类别
    error_detail    TEXT,

    -- 反馈
    user_feedback   INTEGER         -- 1=赞, -1=踩, NULL=未评价
);

-- worker 明细表（一对多）
CREATE TABLE IF NOT EXISTS trace_workers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT NOT NULL REFERENCES trace_logs(trace_id),
    worker_name     TEXT NOT NULL,
    success         BOOLEAN DEFAULT 1,
    content_summary TEXT,
    error           TEXT,
    latency_ms      REAL
);

-- tool_call 明细表（多对多）
CREATE TABLE IF NOT EXISTS trace_tool_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT NOT NULL REFERENCES trace_logs(trace_id),
    worker_name     TEXT NOT NULL,
    tool_name       TEXT NOT NULL,
    tool_args       TEXT,           -- JSON
    tool_result     TEXT,
    error           TEXT,
    latency_ms      REAL
);
```

### 5.4 用户反馈接口

```python
# api/service.py 新增

def submit_feedback(trace_id: str, feedback: int) -> bool:
    """
    用户对某次回答进行反馈。

    Args:
        trace_id: 追踪 ID
        feedback: 1=点赞, -1=点踩

    Returns:
        是否提交成功
    """
    # 更新 trace_logs 的 user_feedback 字段
    return LogStorage.update_feedback(trace_id, feedback)


# api/main.py 新增端点
@app.post("/feedback", tags=["对话"])
async def feedback(trace_id: str, feedback: int = Body(...)):
    """用户反馈接口"""
    success = submit_feedback(trace_id, feedback)
    return {"ok": success}
```

---

## 6. Phase 3：错误聚类与分析

### 6.1 错误分类器

```python
# app/error_classifier.py
"""
错误分类器：将原始错误信息映射到预定义的 error_type 类别。

设计思路：
  1. 关键词规则（快速、准确率高、可解释）→ 覆盖 80% 的常见错误
  2. LLM 分类（处理规则无法覆盖的边界情况）
  3. 人工审核聚类结果 → 补充规则 → 迭代
"""

import re
from typing import Optional

# 预定义错误类别 + 匹配规则
ERROR_PATTERNS = [
    # ---- SQL 相关 ----
    (r"(?i)(SQL syntax|You have an error in your SQL|Unknown column|Table .* doesn't exist)",
     "sql_syntax_error"),
    (r"(?i)(not allowed|dangerous pattern|only select)", "sql_security_blocked"),
    (r"(?i)(connection refused|can\'t connect to.*mysql|timeout.*db)", "sql_connection_error"),

    # ---- 知识图谱相关 ----
    (r"(?i)(cypher.*syntax|neo4j.*error|node.*not found|relationship.*not found)",
     "graph_kg_query_error"),
    (r"(?i)(neo4j.*connection refused|bolt.*failed|graph.*unavailable)", "graph_kg_connection_error"),

    # ---- 向量检索相关 ----
    (r"(?i)(milvus.*error|collection.*not found|vector.*dimension)", "vec_kg_error"),
    (r"(?i)(no.*chunk|embedding.*fail|vector.*timeout)", "vec_kg_retrieval_error"),

    # ---- Python 沙箱相关 ----
    (r"(?i)(SyntaxError.*sandbox|RestrictedPython|compilation error)", "python_compile_error"),
    (r"(?i)(Execution error:|not allowed|import.*not allowed)", "python_execution_error"),

    # ---- LLM 相关 ----
    (r"(?i)(rate limit|429|too many requests|rpm exhausted)", "llm_rate_limit"),
    (r"(?i)(context length|token limit|maximum context)", "llm_context_overflow"),
    (r"(?i)(api key|unauthorized|403|401|authentication)", "llm_auth_error"),
    (r"(?i)(timeout.*llm|llm.*timeout|request timed out)", "llm_timeout"),
    (r"(?i)(empty response|don\'t know|我不知道|cannot answer|no information)",
     "llm_no_answer"),

    # ---- 路由相关 ----
    (r"(?i)(no worker|all workers failed|fallback exhausted)", "routing_no_worker"),
    (r"(?i)(loop detected|cycle|calling same worker)", "routing_loop"),

    # ---- 输入相关 ----
    (r"(?i)(query too long|query too short|invalid input)", "input_validation_error"),
]

# 兜底类别
FALLBACK_CATEGORY = "unknown_error"


def classify_error(error_detail: str, worker_name: str = "",
                   worker_content: str = "") -> str:
    """
    对错误进行分类。

    Args:
        error_detail: 原始错误信息或异常字符串
        worker_name: 出错的 worker 名称（辅助判断）
        worker_content: worker 的输出内容（辅助判断）

    Returns:
        错误类别字符串
    """
    text = f"{error_detail} {worker_content} {worker_name}"

    for pattern, category in ERROR_PATTERNS:
        if re.search(pattern, text):
            return category

    # 检查是否有明显的 SQL 但未匹配到模式
    if "SQL" in text or "execute_sql" in text:
        return "sql_unknown_error"
    if "cypher" in text or "graph" in text:
        return "graph_kg_unknown_error"

    return FALLBACK_CATEGORY
```

### 6.2 聚类分析脚本

```python
# scripts/error_clustering.py
"""
错误聚类分析脚本（离线运行）。

用法：
    python scripts/error_clustering.py [--days 7]

输出：
    1. 控制台打印：高频错误 Top-N
    2. 生成报告文件：logs/cluster_reports/YYYYMMDD_cluster_report.json
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict


def run_clustering(days: int = 7) -> dict:
    """
    对最近 days 天的日志进行聚类分析。

    返回:
        {
            "period": "2026-06-24 ~ 2026-06-30",
            "total_requests": 1000,
            "error_rate": 0.15,
            "top_errors": [
                {"error_type": "sql_syntax_error", "count": 30, "ratio": 0.20,
                 "sample_queries": ["...", "..."]},
                ...
            ],
            "worker_error_rates": {
                "sqler": 0.25,
                "graph_kg": 0.18,
                ...
            },
            "trending": {
                "sql_syntax_error": {"today": 5, "yesterday": 2, "change": "+150%"},
                ...
            }
        }
    """
    # 1. 从存储读取日志
    logs = _load_logs(days)

    # 2. 整体统计
    total = len(logs)
    error_logs = [log for log in logs if not log.get("success", True)]
    error_rate = len(error_logs) / total if total > 0 else 0

    # 3. 按 error_type 聚类
    error_counter = Counter()
    error_samples = defaultdict(list)  # 每个类别保留几个样本

    for log in error_logs:
        et = log.get("error_type", "unknown")
        error_counter[et] += 1
        if len(error_samples[et]) < 3:
            error_samples[et].append(log.get("user_query", "")[:100])

    # 4. 按 worker 聚合错误率
    worker_errors = defaultdict(int)
    worker_total = defaultdict(int)
    for log in logs:
        for w in log.get("workers", []):
            wname = w.get("worker_name", "?")
            worker_total[wname] += 1
            if not w.get("success", True):
                worker_errors[wname] += 1

    worker_error_rates = {
        name: round(worker_errors[name] / total, 4) if total > 0 else 0
        for name, total in worker_total.items()
    }

    # 5. 构建 top errors
    top_errors = []
    for et, count in error_counter.most_common(15):
        top_errors.append({
            "error_type": et,
            "count": count,
            "ratio": round(count / len(error_logs), 4) if error_logs else 0,
            "sample_queries": error_samples.get(et, []),
        })

    return {
        "period": f"{(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')} ~ {datetime.now().strftime('%Y-%m-%d')}",
        "total_requests": total,
        "error_count": len(error_logs),
        "error_rate": round(error_rate, 4),
        "worker_error_rates": worker_error_rates,
        "top_errors": top_errors,
    }


def generate_report(cluster_result: dict, output_dir: str = "logs/cluster_reports"):
    """生成聚类报告文件"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(
        output_dir,
        f"cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cluster_result, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"  错误聚类分析报告")
    print(f"{'='*60}")
    print(f"  统计周期: {cluster_result['period']}")
    print(f"  总请求: {cluster_result['total_requests']}")
    print(f"  错误数: {cluster_result['error_count']}")
    print(f"  错误率: {cluster_result['error_rate']*100:.2f}%")
    print(f"\n  Top 错误:")
    for i, et in enumerate(cluster_result['top_errors'][:10], 1):
        print(f"  {i:2d}. [{et['error_type']}] x{et['count']} "
              f"({et['ratio']*100:.1f}%)")
        for sq in et['sample_queries']:
            print(f"      例: {sq}")
    print(f"\n  Worker 错误率:")
    for w, r in sorted(cluster_result['worker_error_rates'].items(),
                       key=lambda x: -x[1]):
        print(f"    {w}: {r*100:.1f}%")
    print(f"\n  报告已保存: {path}")
    print(f"{'='*60}\n")

    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    result = run_clustering(days=args.days)
    generate_report(result)
```

### 6.3 典型聚类结果示例

```
============================================================
  错误聚类分析报告
============================================================
  统计周期: 2026-06-23 ~ 2026-06-30
  总请求: 1250
  错误数: 187
  错误率: 14.96%

  Top 错误:
   1. [sql_syntax_error] x56 (29.9%)
       例: "2025年第二季度销售额最高的产品是什么？"
       例: "上海地区市场份额最高的竞争对手是谁？"
   2. [graph_kg_connection_error] x38 (20.3%)
       例: "小米和华为有哪些合作关系？"
   3. [llm_no_answer] x21 (11.2%)
       例: "请帮我画一个饼图展示市场份额分布"
   4. [python_execution_error] x15 (8.0%)
       例: "用matplotlib画一张柱状图"
   5. [routing_loop] x12 (6.4%)
       例: "对比分析三家公司"
   ...

  Worker 错误率:
    graph_kg: 22.5%
    sqler: 18.3%
    coder: 12.1%
    vec_kg: 5.2%
    chat: 1.8%

  >>> 优化建议:
  1. [sql_syntax_error] 占比 30% → 优化 sqler System Prompt 中的 few-shot SQL 示例
  2. [graph_kg_connection_error] 20% → 增加 graph_kg 连通性健康检查 + 快速降级到 vec_kg
  3. [llm_no_answer] → 优化 routing 策略，对"画图"类请求优先路由到 coder
```

---

## 7. Phase 4：自动化反馈优化

这是整个体系的"闭环"环节。聚类结果不能只停留在报表里，要**反哺到系统配置**中。

### 7.1 优化建议自动生成

```python
# app/optimization_advisor.py
"""
基于聚类结果，自动生成优化建议。

每个建议包含:
  - 问题描述（来自聚类数据）
  - 建议的修改内容
  - 影响范围
  - 风险评估
"""

from typing import List, Dict


# 错误类型 → 优化策略映射
OPTIMIZATION_MAP = {
    "sql_syntax_error": {
        "action": "update_prompt",
        "target": "db_agent_system_prompt",
        "description": "SQL 语法错误高频 → 在 System Prompt 中增加更多 few-shot SQL 示例",
        "risk": "low",
    },
    "sql_security_blocked": {
        "action": "update_config",
        "target": "tools._DANGEROUS_PATTERNS / _validate_sql",
        "description": "SQL 被安全规则拦截 → 检查是否需要放宽规则或优化查询",
        "risk": "medium",
    },
    "graph_kg_connection_error": {
        "action": "update_routing",
        "target": "WorkerConfig(graph_kg).fallback_workers",
        "description": "图数据库连接失败高频 → 将 vec_kg 设为 graph_kg 的首选 fallback",
        "risk": "low",
    },
    "llm_no_answer": {
        "action": "update_routing",
        "target": "supervisor system prompt / routing policy",
        "description": "LLM 频繁表示不知道 → 优化路由策略，减少 LLM 直接回答的场景",
        "risk": "medium",
    },
    "python_execution_error": {
        "action": "update_prompt",
        "target": "code_agent system prompt",
        "description": "Python 执行错误多 → 增加常见错误的规避说明和代码模板",
        "risk": "low",
    },
    "routing_loop": {
        "action": "update_config",
        "target": "RoutingPolicyManager / max_attempts",
        "description": "路由循环 → 降低 max_attempts 或增加 fallback 选项",
        "risk": "low",
    },
}


def generate_optimization_suggestions(cluster_result: dict) -> List[Dict]:
    """
    根据聚类结果生成优化建议列表。

    Args:
        cluster_result: run_clustering() 的输出

    Returns:
        [{"error_type": ..., "count": ..., "action": ..., "target": ...,
          "description": ..., "risk": ..., "suggested_change": ...}, ...]
    """
    suggestions = []

    for top_error in cluster_result.get("top_errors", []):
        et = top_error["error_type"]
        if et not in OPTIMIZATION_MAP:
            continue

        # 只在错误比例 > 5% 时生成建议
        if top_error["ratio"] < 0.05:
            continue

        base = OPTIMIZATION_MAP[et]
        suggestions.append({
            "error_type": et,
            "count": top_error["count"],
            "ratio": top_error["ratio"],
            **base,
        })

    return suggestions
```

### 7.2 Prompt 自动优化（示例：SQL Agent）

当聚类发现 `sql_syntax_error` 高发时，自动在 `db_agent_system_prompt` 追加高频错误的正确写法示例：

```python
# app/prompt_optimizer.py

def inject_few_shot_examples(prompt: str, error_type: str, examples: List[dict]) -> str:
    """
    在 System Prompt 中注入 few-shot 示例。
    仅追加新的示例，不修改原有内容。

    Args:
        prompt: 原 System Prompt
        error_type: 触发优化的错误类型
        examples: [{"question": "...", "correct_sql": "..."}, ...]

    Returns:
        增强后的 System Prompt
    """
    section_title = f"\n\n[优化示例 - 针对高频错误: {error_type}]\n"
    section_body = "以下是根据历史错误优化后的正确查询示例：\n\n"

    for ex in examples:
        section_body += f"问题: {ex['question']}\n"
        section_body += f"正确 SQL: {ex['correct_sql']}\n\n"

    section_body += "请参考以上示例，避免类似错误。\n"

    # 检查是否已存在该章节（避免重复注入）
    if section_title.strip() not in prompt:
        return prompt + section_title + section_body
    return prompt


def auto_update_db_agent_prompt(cluster_result: dict):
    """
    自动生成 SQL few-shot 示例并更新 db_agent_system_prompt。
    此函数仅在人工审核后执行（不自动部署）。
    """
    sql_errors = [e for e in cluster_result.get("top_errors", [])
                  if e["error_type"] == "sql_syntax_error"]

    if not sql_errors or sql_errors[0]["count"] < 5:
        return None  # 错误量不足，暂不优化

    # 根据常见的错误 SQL 模式，生成修正后的示例
    examples = [
        {
            "question": "2025年第二季度销售额最高的产品是什么？",
            "correct_sql": (
                "SELECT p.product_name, SUM(s.amount) as total_amount "
                "FROM sales_data s "
                "JOIN product_information p ON s.product_id = p.product_id "
                "WHERE s.sale_date BETWEEN '2025-04-01' AND '2025-06-30' "
                "GROUP BY p.product_name "
                "ORDER BY total_amount DESC "
                "LIMIT 1"
            ),
        },
        {
            "question": "上海地区市场份额最高的竞争对手是谁？",
            "correct_sql": (
                "SELECT competitor_name, market_share "
                "FROM competitor_analysis "
                "WHERE region = '上海' "
                "ORDER BY market_share DESC "
                "LIMIT 1"
            ),
        },
        {
            "question": "上季度各产品类别的销售总额",
            "correct_sql": (
                "SELECT p.category, SUM(s.amount) as total "
                "FROM sales_data s "
                "JOIN product_information p ON s.product_id = p.product_id "
                "WHERE s.sale_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH) "
                "GROUP BY p.category "
                "ORDER BY total DESC"
            ),
        },
    ]

    # 打印建议（等待人工确认后手动执行）
    print(f"\n{'='*60}")
    print(f"  Prompt 优化建议")
    print(f"{'='*60}")
    print(f"  发现 {sql_errors[0]['count']} 个 SQL 语法错误")
    print(f"  建议在 db_agent_system_prompt 中添加以下 few-shot 示例:\n")
    for ex in examples:
        print(f"  问题: {ex['question']}")
        print(f"  SQL: {ex['correct_sql']}\n")

    return examples
```

### 7.3 路由策略自动优化

```python
# app/routing_optimizer.py

def auto_adjust_routing(cluster_result: dict, current_policy: RoutingPolicyManager) -> list:
    """
    根据聚类结果生成路由策略调整建议。

    调整策略示例:
      - graph_kg 连接失败率高 → 降低其 priority，增加 fallback
      - 某个 worker 成功率极低 → 考虑将其替换或禁用
      - 路由循环多 → 增加循环检测的严格度（降低 max_attempts）

    Returns:
        [{"worker": ..., "field": ..., "old_value": ..., "new_value": ..., "reason": ...}, ...]
    """
    adjustments = []
    worker_rates = cluster_result.get("worker_error_rates", {})

    # 示例：graph_kg 连接失败率高，调整路由策略
    graph_kg_rate = worker_rates.get("graph_kg", 0)
    if graph_kg_rate > 0.2:  # 错误率 > 20%
        # 建议增加 vec_kg 作为替代
        adjustments.append({
            "worker": "graph_kg",
            "field": "fallback_workers",
            "old_value": current_policy.worker_configs["graph_kg"].fallback_workers,
            "new_value": ["vec_kg", "sqler"],  # 优先 fallback 到 vec_kg
            "reason": f"graph_kg 错误率 {graph_kg_rate*100:.0f}% 过高，应优先切换到 vec_kg",
        })

    # 示例：routing_loop 多，降低全局 max_attempts
    routing_errors = [e for e in cluster_result.get("top_errors", [])
                      if e["error_type"] == "routing_loop"]
    if routing_errors and routing_errors[0]["count"] > 5:
        adjustments.append({
            "worker": "all",
            "field": "max_attempts",
            "old_value": "1~2",
            "new_value": "1 (统一降为 1)",
            "reason": f"路由循环出现 {routing_errors[0]['count']} 次，应减少每个 worker 的重试次数",
        })

    return adjustments
```

### 7.4 完整优化工作流

```
每日/每周定时任务:
  ┌──────────────────────────────┐
  │  1. 运行错误聚类分析         │
  │     python scripts/error_clustering.py --days 7
  └──────────────┬───────────────┘
                 ▼
  ┌──────────────────────────────┐
  │  2. 生成优化建议             │
  │     ├─ Prompt 优化建议       │
  │     ├─ 路由策略调整建议      │
  │     └─ 配置文件变更建议      │
  └──────────────┬───────────────┘
                 ▼
  ┌──────────────────────────────┐
  │  3. 人工审核（关键步骤！）   │
  │     ├─ 确认聚类结果无误       │
  │     ├─ 评估优化风险           │
  │     └─ 决定是否应用           │
  └──────────────┬───────────────┘
                 ▼
  ┌──────────────────────────────┐
  │  4. 应用优化                 │
  │     ├─ 更新 System Prompt    │
  │     ├─ 调整 routing_policy   │
  │     └─ 重新评估（A/B Test）  │
  └──────────────┬───────────────┘
                 ▼
  ┌──────────────────────────────┐
  │  5. 验证效果                 │
  │     ├─ 对比优化前后错误率     │
  │     ├─ 运行离线 RAGAS 评估   │
  │     └─ 确认无误后部署上线     │
  └──────────────────────────────┘
```

> **核心原则**：自动化提供**建议**，人工做**决策**。避免自动修改生产配置带来的风险。

---

## 8. Phase 5：可视化与监控

### 8.1 简易 CLI 看板

```python
# scripts/dashboard.py
"""
简易 CLI 看板，每日运行查看系统健康度。

用法：
    python scripts/dashboard.py [--days 7]
"""

def print_dashboard(days: int = 1):
    result = run_clustering(days=days)

    print(f"\n╔{'═'*58}╗")
    print(f"║  📊 系统运行看板 (过去 {days} 天)            ║")
    print(f"╠{'═'*58}╣")
    print(f"║  总请求: {result['total_requests']:<8d}  "
          f"错误数: {result['error_count']:<5d}  "
          f"错误率: {result['error_rate']*100:.1f}%    ║")
    print(f"╠{'═'*58}╣")

    print(f"║  Top 错误:                                     ║")
    for i, et in enumerate(result['top_errors'][:5], 1):
        bar_len = int(et['ratio'] * 40)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        print(f"║  {i}. {et['error_type']:<25s} {et['count']:>4d} {bar}║")

    print(f"╠{'═'*58}╣")
    print(f"║  Worker 错误率:                                ║")
    for w, r in sorted(result['worker_error_rates'].items(),
                       key=lambda x: -x[1]):
        bar_len = int(r * 40)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        print(f"║    {w:<25s} {r*100:>5.1f}% {bar}║")

    print(f"╚{'═'*58}╝")
```

### 8.2 Grafana + Prometheus（进阶方案）

```
架构:
  app/metrics.py (Prometheus client)
      ↓
  Prometheus (抓取 /metrics 端点)
      ↓
  Grafana (可视化看板)

关键指标:
  - hybrid_rag_requests_total{status="success|error"}
  - hybrid_rag_request_duration_seconds
  - hybrid_rag_worker_duration_seconds{worker="sqler|..."}
  - hybrid_rag_error_type_total{type="sql_syntax_error|..."}
  - hybrid_rag_user_feedback{feedback="1|-1"}
```

---

## 9. 落地路线图

### Phase 1（1~2 天）：结构化日志

| 任务 | 文件 | 工作量 |
|------|------|--------|
| 实现 `TraceContext` / `TraceRecord` | 新建 `app/tracing.py` | ~80 行 |
| 实现 `LogStorage` + JSONL 存储 | 新建 `app/log_storage.py` | ~60 行 |
| 入口接入：`api/service.py` + `main.py` | 修改 2 个文件，各 ~5 行 | 10 分钟 |
| supervisor 接入：记录路由决策 | 修改 `app/supervisor.py` | ~3 行 |
| worker 节点接入：记录执行结果 | 修改 `app/nodes.py` | ~15 行 |
| tool_node 接入：记录 tool call | 修改 `app/agents.py` | ~10 行 |

**产出**：每次请求生成一条完整的 JSONL 日志，包含 query → routing → worker → tool_call 全链路。

### Phase 2（1 天）：Badcase 采集

| 任务 | 文件 | 工作量 |
|------|------|--------|
| 实现错误分类器 `classify_error()` | 新建 `app/error_classifier.py` | ~60 行 |
| 在 `TraceContext.finish()` 中自动分类 | 修改 `app/tracing.py` | ~5 行 |
| 实现用户反馈端点 | 修改 `api/main.py` + `api/service.py` | ~20 行 |

**产出**：每条日志带有 `error_type` 标签，用户可对回答点赞/点踩。

### Phase 3（1~2 天）：聚类分析

| 任务 | 文件 | 工作量 |
|------|------|--------|
| 实现聚类分析脚本 | 新建 `scripts/error_clustering.py` | ~120 行 |
| 实现 CLI 看板 | 新建 `scripts/dashboard.py` | ~80 行 |

**产出**：每日自动生成聚类报告，可视化 Top 错误和 Worker 错误率。

### Phase 4（2~3 天）：自动化反馈

| 任务 | 文件 | 工作量 |
|------|------|--------|
| 实现 `optimization_advisor` | 新建 `app/optimization_advisor.py` | ~80 行 |
| 实现 prompt 注入优化 | 新建 `app/prompt_optimizer.py` | ~60 行 |
| 实现路由策略调整 | 新建 `app/routing_optimizer.py` | ~60 行 |
| 定时任务脚本 | 新建 `scripts/auto_optimization_pipeline.py` | ~80 行 |

**产出**：定时运行分析 pipeline，生成优化建议报告供人工审核。

### Phase 5（可选）

| 任务 | 工作量 |
|------|--------|
| ElasticSearch 替换 JSONL | 1~2 天 |
| Grafana + Prometheus 集成 | 2~3 天 |
| 基于 LLM 的语义聚类（替代关键词规则） | 1~2 天 |
| A/B Test 框架（优化前后效果对比） | 2~3 天 |

---

## 10. 附录：关键代码示例

### 10.1 快速接入指南

> 从零到完整的日志驱动调优，最快 1 天可跑通基础版。

**步骤：**

```
1. 复制 app/tracing.py → 项目中
2. 复制 app/log_storage.py → 项目中
3. 复制 app/error_classifier.py → 项目中
4. 在 api/service.py 的 chat_stream 开头/结尾添加 TraceContext.begin()/end_and_export()
5. 在 app/nodes.py 各 worker 函数开头记录耗时 + 结尾记录 TraceContext.add_worker()
6. 运行几次请求，验证 logs/traces/ 下生成 JSONL 文件
7. 复制 scripts/error_clustering.py → 运行 python scripts/error_clustering.py --days 1
```

### 10.2 关键文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `app/tracing.py` | **新增** | TraceContext + TraceRecord 核心 |
| `app/log_storage.py` | **新增** | 日志存储（JSONL / SQLite） |
| `app/error_classifier.py` | **新增** | 错误分类器（关键词规则） |
| `app/optimization_advisor.py` | **新增** | 优化建议生成 |
| `app/prompt_optimizer.py` | **新增** | Prompt 自动增强 |
| `app/routing_optimizer.py` | **新增** | 路由策略自动调整 |
| `scripts/error_clustering.py` | **新增** | 聚类分析脚本 |
| `scripts/dashboard.py` | **新增** | CLI 看板 |
| `app/supervisor.py` | **修改** | 追加 ~3 行 TraceContext 调用 |
| `app/nodes.py` | **修改** | 追加 ~15 行 TraceContext 调用 |
| `app/agents.py` | **修改** | 追加 ~10 行 TraceContext 调用 |
| `api/service.py` | **修改** | 追加 ~5 行 TraceContext.begin/end |
| `api/main.py` | **修改** | 追加反馈端点 |

### 10.3 持续迭代原则

1. **先跑通，再完善**：先用 JSONL 存储跑通全链路，再考虑 ES 或 ClickHouse
2. **规则先于模型**：错误分类先用关键词规则（可解释、无成本），规则覆盖不到的再用 LLM
3. **人工审核不可跳过**：自动化生成的优化建议必须经过人工确认才能上线
4. **效果可衡量**：每次优化前后对比 `error_rate` + RAGAS 指标
5. **渐进式投入**：错误率降到 5% 以下前，集中精力解决 Top 3 错误即可覆盖大部分问题

---

> **最后**：这套体系的本质是将"凭感觉优化"变为"用数据驱动优化"。初期投入不大（1~2 天即可跑通基础链路），但长期收益显著——每次优化都有数据支撑，每个决策都可追溯效果。
