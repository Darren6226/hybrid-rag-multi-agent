# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A hybrid RAG multi-agent system built on LangGraph that routes queries through a Supervisor-Worker topology to 5 specialized agents. Uses Alibaba DashScope (Qwen models) as the primary LLM provider, with SenseNova DeepSeek as an alternative.

## Quick Start

```bash
# Start infrastructure services (MySQL, Neo4j, Milvus)
docker-compose -f docker-compose-rag.yml up -d

# Setup environment
conda create -n llm python=3.9
conda activate llm
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# Configure .env (see Configuration section)

# Run the system
python main.py                          # Full system (all agents + GraphRAG)
python main.py --fast                   # Fast mode (vector search only, skips Neo4j)
python main.py -q "你的问题"            # Single query mode
python main.py --fast -q "你的问题"     # Fast + single query
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload  # API server with SSE streaming
```

## Key Commands

| Action | Command |
|---|---|
| Full system (interactive) | `python main.py` |
| Full system (single query) | `python main.py -q "你的问题"` |
| Fast mode (no Neo4j) | `python main.py --fast` |
| API server (SSE streaming) | `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload` |
| Run all tests | `pytest tests/ -v` |
| Run single test file | `pytest tests/test_routing_policy.py -v` |
| Run single test | `pytest tests/test_tools.py::TestValidateSql::test_reject_non_select -v` |
| Generate graph diagram | `python draw_graph.py` |
| RAGAS evaluation (DashScope) | `python evaluate_rag.py --source company --retriever vec` |
| RAGAS evaluation (DeepSeek) | `python evaluate_rag.py --source company --retriever vec --model deepseek` |
| RAGAS evaluation (all sources) | `python evaluate_rag.py --source all --retriever vec` |
| Benchmark (all experiments) | `python benchmark.py` |
| Benchmark (single experiment) | `python benchmark.py --only chunking` |
| Benchmark (limited samples) | `python benchmark.py --only chunking --samples 5` |
| Benchmark (company dataset) | `python benchmark.py --dataset company` |

## Architecture

### Supervisor-Worker Pattern

```
User Query → Supervisor (LLM router with cycle detection)
  ├→ sqler      — MySQL CRUD via ReAct agent (app/agents.py)
  ├→ graph_kg   — Neo4j Cypher query chain (app/rag.py → GraphCypherQAChain)
  ├→ vec_kg     — Milvus vector retrieval + LLM generation (app/rag.py → vectorstore)
  ├→ coder      — Python REPL via ReAct agent (app/agents.py)
  └→ chat       — Direct LLM conversation / final summary
```

Every worker result flows back to the supervisor, which decides to dispatch again or route to `END`. The supervisor uses `RoutingPolicyManager` (app/routing_policy.py) to prevent infinite loops via cycle detection and fallback logic.

### State Management (`app/state.py`)

- `AgentState`: extends `MessagesState` with a `next` field (routing decision)
- `ReActAgentState`: separate TypedDict for ReAct agent loops (sqler, coder)

### Entry Points

- **`main.py`** — CLI 入口。支持 `--fast`（跳过 Neo4j）和 `-q`（单次提问），省略 `-q` 进入交互模式。
- **`main_fast.py`** — 兼容性包装，等价于 `python main.py --fast`。
- **`hybrid_rag_supervisor.py`** — 兼容性包装，等价于 `python main.py`。
- **`api/main.py`** — FastAPI service. `POST /chat` returns SSE stream (`text/event-stream`), `GET /health` checks Milvus/Neo4j/MySQL connectivity.

### Module Responsibilities

| Module | Role |
|---|---|
| `app/config.py` | Central config: creates LLM instances (`llm`, `graph_llm`, `supervisor_llm`, `deepseek_llm`), custom `DashScopeEmbeddings` class, DB connection URIs, unified logging config. All `.env` loading happens here. |
| `app/graph_builder.py` | Shared graph construction: `build_graph(skip_graph_kg)` builds and compiles the LangGraph `StateGraph`. Eliminates duplication across entry points. |
| `app/stream_utils.py` | Shared streaming utilities: `iter_stream_messages()`, `print_stream()` (CLI), `sse_stream()` (FastAPI). |
| `app/supervisor.py` | LLM-based router with structured output. Builds dynamic prompts from worker history. Integrates `RoutingPolicyManager` for cycle prevention. |
| `app/routing_policy.py` | `RoutingPolicyManager`: tracks per-worker attempt counts, success rates (EMA), error keywords. Provides `get_fallback_worker()` and `should_force_finish()`. |
| `app/agents.py` | `create_react_agent_v1()` factory: builds think-tool-respond ReAct loops. Creates `db_agent` (sqler tools) and `code_agent` (python_repl). |
| `app/nodes.py` | LangGraph node wrappers: `sqler_node`, `coder_node`, `graph_kg_node`, `vec_kg_node`, `chat_node`. Uses lazy-loaded `get_vectorstore()` / `get_cypher_chain()` from `app/rag.py`. |
| `app/rag.py` | **Lazy-loaded** — call `init_rag()` to trigger document loading, Neo4j graph index, and Milvus vector index. Access via `get_vectorstore()`, `get_cypher_chain()`, `get_graph()`. Thread-safe via `threading.Lock`. |
| `app/tools.py` | `@tool`-decorated LangChain tools for MySQL CRUD and Python REPL execution (RestrictedPython sandbox). |
| `app/database.py` | **Lazy-loaded** — SQLAlchemy ORM models (`SalesData`, `CustomerInformation`, `ProductInformation`, `CompetitorAnalysis`). `_get_session()` creates engine on first call. `init_seed_data()` populates sample data. |
| `app/enhanced_chunking.py` | Four strategies via `create_enhanced_chunker(strategy)`: `basic`, `semantic_metadata`, `parent_child`, `full`. |
| `app/multimodal_pdf_parser.py` | Extracts text + tables (pdfplumber) + images (PyMuPDF) from PDFs. Produces `ParsedDocument` with `to_chunks()`. |
| `app/evaluation.py` | RAGAS-based evaluation module. Contains `RAGASEvaluator` class and `RateLimitedChatOpenAI` wrapper for API rate limiting. |
| `evaluate_rag.py` | CLI runner for RAGAS evaluation. Supports `--source`, `--retriever`, `--model` arguments. |
| `api/main.py` | FastAPI app. `POST /chat` (SSE stream), `GET /health` (dependency check). |
| `api/service.py` | Wraps LangGraph graph construction + `chat_stream()` SSE generator + `check_health()`. Uses `app/graph_builder.py` and `app/stream_utils.py`. |
| `api/schemas.py` | Pydantic models: `ChatRequest`, `ChatEvent`, `HealthResponse`. |
| `benchmark.py` | Benchmark suite: 5 experiments (chunking, top_k, hybrid, rerank, routing). Uses temporary Milvus collections. `ParentRetriever` class wraps base retriever to return parent chunk content when using parent_child strategy. |

### Database Ports

- MySQL: 5306
- Neo4j: 17687 (browser: 17474)
- Milvus: 19530

## Configuration

All config lives in `app/config.py`. The `.env` file should contain:
- `DASHSCOPE_API_KEY` — Alibaba DashScope API key
- `DASHSCOPE_API_BASE` — API base URL (ModelScope-compatible endpoint)
- `SENSENOVA_API_KEY` — SenseNova API key for DeepSeek model (optional)
- `modelscope_base_url` — ModelScope API inference endpoint (optional)
- `modelscope_api_key` — ModelScope API key (optional)

Four LLM instances are available: `llm` (qwen3.6-plus, general), `graph_llm` (qwen3.6-plus, GraphRAG/Cypher), `supervisor_llm` (qwen3.6-flash, routing), and `deepseek_llm` (SenseNova DeepSeek deepseek-v4-flash, conditional on `SENSENOVA_API_KEY`).

## Chunking Strategies

The system supports multiple chunking strategies for different document types:
- `basic` — RecursiveCharacterTextSplitter only (chunk_size=400). Simple, fast.
- `semantic_metadata` — SemanticChunker (embedding-based sentence grouping) + metadata enhancement. Best for technical/academic documents. Benchmark winner: Overall 0.8855.
- `parent_child` — Parent (2000 chars) + child (400 chars) dual-layer. Only child chunks go into the vector index; parent content is stored in a lookup dict. Use `ParentRetriever` (in `benchmark.py`) to search children and return parents.
- `full` — Combines all strategies.

Strategy is configured in `app/rag.py` when creating the `EnhancedChunker`.

**Important**: DashScope embedding model has an 8192 char input limit. Use `_split_oversized_chunks()` in `benchmark.py` to split chunks exceeding this limit before embedding.

## Benchmark Experiments

`benchmark.py` runs 5 experiments against temporary Milvus collections:

| Experiment | `--only` value | What it tests |
|---|---|---|
| Chunking | `chunking` | basic vs semantic_metadata vs parent_child |
| Top-K | `top_k` | Retrieval quality at different k values |
| Hybrid retrieval | `hybrid` | Pure vector vs Dense+BM25 hybrid |
| Reranking | `rerank` | No rerank vs qwen3-rerank (top8→5) |
| Routing accuracy | `routing` | Supervisor routing decisions vs expected labels |

Best known configuration: semantic_metadata + pure vector + qwen3-rerank = Overall 0.9139, routing accuracy 96.0%.

## Evaluation

The evaluation framework uses RAGAS (Retrieval Augmented Generation Assessment) with 4 core metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall.

- **`app/evaluation.py`** — Core module with `RAGASEvaluator` class and `RateLimitedChatOpenAI` wrapper
- **`evaluate_rag.py`** — CLI runner with `--source company|dnngp|all`, `--retriever vec|graph`, `--model dashscope|deepseek`
- **`evaluation_test_data/`** — QA test datasets (`company_test_data.json`, `dnngp_test_data.json`)
- Results saved to `evaluation_results/` as JSON

The `RateLimitedChatOpenAI` class (inheriting `ChatOpenAI`) handles SenseNova API rate limits (429 errors) with exponential backoff and RAGAS `RunConfig(max_workers=1)` for sequential execution.

**RAGAS 0.4.x field mapping**: The `convert_v1_to_v2_dataset()` function automatically maps `question`→`user_input`, `contexts`→`retrieved_contexts`, `answer`→`response`, `ground_truth`→`reference`. When building a RAGAS Dataset, use the v1 field names (`question`, `answer`, `contexts`, `ground_truth`). Do NOT include both `ground_truth` and `reference` — it creates duplicate columns. RAGAS 0.4.x also patches a `langchain_community.chat_models.vertexai` import error via `sys.modules` mock in `app/evaluation.py`.

## Conventions

- PEP 8 style
- Logging: use `logging.getLogger(__name__)` — unified format configured in `app/config.py`. Never use `print()` except in CLI entry points (`main.py` user-facing output).
- Tests: `pytest tests/ -v` — 79 tests covering routing policy, supervisor logic, chunking strategies, and tool security. Tests use mock LLM/embeddings (no API calls).
- Doc knowledge base: `doc/company.txt`; PDF knowledge base: `pdf/` directory
- Benchmark data: `benchmark_data/` (DNNGP, company, routing test datasets)
- Benchmark results: `benchmark_results/` (JSON + markdown report)
