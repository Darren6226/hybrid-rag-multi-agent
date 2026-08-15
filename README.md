# 混合 RAG 多智能体系统

基于 **LangGraph Supervisor-Worker 架构**的企业级混合知识库检索系统。通过 LLM 动态路由，将用户问题分发给 5 个专职 Agent（SQL 查询 / 图检索 / 向量检索 / 代码执行 / 通用对话），融合**结构化查询、图多跳推理、语义向量召回**三种异构数据通路，实现跨源协同推理。

## 核心亮点

- **多智能体调度架构**：Supervisor 双层路由——LLM 结构化输出（JSON Schema）做初步决策，`RoutingPolicyManager` 策略层二次校验，通过循环检测、EMA 成功率追踪、错误关键词识别和自动退避机制防止 Agent 死循环与重复调度，**路由准确率达 96%**
- **异构数据融合**：打通 MySQL（交易/客户/产品/竞品数据）、Neo4j（企业股权/供应链/技术图谱）、Milvus（文档向量嵌入）三种数据后端，Supervisor 按问题类型动态决定并行或串行调用多个 Agent
- **工程化与安全**：抽象 ReAct Agent 工厂方法统一创建 5 个 Agent、消除重复代码；SQL 执行层做多因素安全校验（SELECT 白名单 + 只读事务 + 危险模式检测 + 长度限制）；Python 执行层基于 RestrictedPython 字节码级沙箱，限制 builtin 函数与白名单模块，隔离文件和网络操作
- **系统评估体系**：基于 RAGAS 搭建评估管线，5 组对照实验（分块策略 / Top-K / 混合检索 / 重排序消融 / 路由准确率）逐模块量化效果
- **可观测与部署**：统一日志格式、懒加载 RAG 管线、Docker Compose 一键部署基础设施、FastAPI SSE 流式服务

## 技术栈

LangGraph · LangChain · DashScope（Qwen3.7-max）· MySQL · Neo4j · Milvus · FastAPI · Docker Compose · RestrictedPython · RAGAS

## 项目成果

- 最优配置（语义元数据分块 + 纯向量检索 + qwen3-rerank 重排序）RAGAS 综合得分 **0.9139**，路由准确率 **96%**
- 重排序使 Context Precision 从 0.7687 提升至 0.9064（**+17.9%**）
- 79 个 pytest 单元测试覆盖路由策略与工具安全

## 系统架构

```
                        ┌─────────────────────────┐
                        │       Supervisor        │
                        │  LLM 路由 + 策略校验     │
                        └───────────┬─────────────┘
                ┌──────────┬────────┼────────┬──────────┐
                ▼          ▼        ▼        ▼          ▼
             sqler     graph_kg  vec_kg    coder       chat
                │          │        │        │          │
          ┌─────┴──┐  ┌────┴───┐ ┌──┴───┐ ┌──┴────┐     │
          │ MySQL  │  │ Neo4j  │ │Milvus│ │Python │     │
          └────────┘  └────────┘ └──────┘ │沙箱   │     │
                                          └───────┘     │
                └────────────── 汇总输出 ────────────────┘
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- Docker（用于启动 MySQL / Neo4j / Milvus 基础设施）
- DashScope API Key（[阿里云百炼](https://dashscope.aliyun.com/)）

### 2. 启动基础设施

```bash
docker-compose -f docker-compose-rag.yml up -d
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

### 5. 运行

**CLI 交互模式（完整模式，含图检索）**

```bash
python main.py
```

**快速模式（跳过 Neo4j，仅向量检索）**

```bash
python main.py --fast
```

**单次提问**

```bash
python main.py -q "小米公司有哪些技术？"
python main.py --fast -q "小米公司有哪些技术？"
```

**FastAPI 服务（SSE 流式接口）**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

接口文档见 `http://localhost:8000/docs`（Swagger UI）。

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat` | SSE 流式对话，请求体 `{"query": "...", "session_id": "可选"}` |
| GET | `/health` | 依赖服务健康检查 |
| GET | `/docs` | Swagger UI 接口文档 |

## 项目结构

```
hybrid-rag-multi-agent/
├── main.py                 # CLI 入口（交互/单次提问）
├── api/                    # FastAPI 服务层
│   ├── main.py             #   API 入口（/chat、/health）
│   ├── schemas.py          #   请求/响应模型
│   └── service.py          #   业务逻辑与流式生成
├── app/                    # 核心逻辑
│   ├── supervisor.py       #   Supervisor 路由节点
│   ├── agents.py           #   ReAct Agent 工厂
│   ├── routing_policy.py   #   路由策略管理（循环检测/退避）
│   ├── rag.py              #   RAG 管线（解析→分块→向量化/建图）
│   ├── graph_builder.py    #   LangGraph 图构建
│   ├── tools.py            #   工具集（SQL/Python 沙箱等）
│   ├── config.py           #   配置与 LLM/Embedding 初始化
│   ├── database.py         #   数据初始化
│   ├── enhanced_chunking.py#   增强分块策略
│   ├── multimodal_pdf_parser.py # PDF 多模态解析
│   └── evaluation.py       #   评估工具
├── tests/                  # pytest 单元测试
├── benchmark.py            # 基准测试脚本
├── docker-compose-rag.yml  # 基础设施编排
└── requirements.txt
```

## 测试

```bash
pytest tests/ -v
```

## 许可证

学习演示项目
