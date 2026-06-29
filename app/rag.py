"""
RAG 模块 — 懒加载设计

所有文档加载、索引构建、数据库连接均延迟到首次调用时执行。
通过 get_vectorstore() / get_cypher_chain() / get_graph() 访问，
内部由 threading.Lock 保证线程安全。
"""

import os
import threading
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import graph_llm, llm, embeddings, NEO4J_URL, MILVUS_URI

logger = logging.getLogger(__name__)

# ============================================================
# 模块级占位变量（初始为 None，init_rag() 后赋值）
# ============================================================
vectorstore = None
cypher_chain = None
graph = None
processed_documents: List[Document] = []

# 懒加载控制
_initialized = False
_init_lock = threading.Lock()

# 多模态 PDF 解析支持（可选）
MULTIMODAL_SUPPORT = False
try:
    from app.multimodal_pdf_parser import MultimodalPDFParser, MultimodalIndexer
    MULTIMODAL_SUPPORT = True
except ImportError:
    logger.info("多模态解析模块未加载，将使用传统PDF解析")


# ============================================================
# 文档处理辅助函数
# ============================================================

def _load_documents() -> List[Document]:
    """加载并分块所有文档（company.txt + PDF）"""
    from app.pdf_extractor import PDFExtractor
    from app.enhanced_chunking import create_enhanced_chunker

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    company_txt_path = os.path.normpath(os.path.join(current_dir, "doc", "company.txt"))
    dnngp_pdf_path = os.path.normpath(os.path.join(
        current_dir, "pdf",
        "DNNGP, a deep neural network-based method for genomic prediction using multi-omics data in plants(简短版）.pdf"
    ))

    logger.info("启动增强文档分块系统...")
    logger.info("策略：语义分块 + 元信息增强 | 参数：max_chunk_size=400, min_chunk_size=200, breakpoint_threshold=92")

    enhanced_chunker = create_enhanced_chunker(
        strategy="semantic_metadata",
        max_chunk_size=400,
        min_chunk_size=200,
    )

    # --- 企业文档 ---
    company_documents: List[Document] = []
    if os.path.exists(company_txt_path):
        logger.info("处理企业知识库文档：%s", company_txt_path)
        with open(company_txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        company_documents = enhanced_chunker.chunk_document(content, company_txt_path)
        logger.info("企业文档增强分块完成，生成 %d 个文档块", len(company_documents))
    else:
        logger.warning("找不到企业文档 %s", company_txt_path)

    # --- DNNGP 论文 ---
    dnngp_documents: List[Document] = []
    if os.path.exists(dnngp_pdf_path):
        logger.info("处理学术论文文档：%s", dnngp_pdf_path)
        try:
            if MULTIMODAL_SUPPORT:
                logger.info("使用多模态 PDF 解析器...")
                parser = MultimodalPDFParser(enable_ocr=False, min_image_size=100)
                doc = parser.parse(dnngp_pdf_path)
                multimodal_chunks = doc.to_chunks(chunk_size=500, chunk_overlap=50)
                for chunk in multimodal_chunks:
                    metadata = {
                        "source": dnngp_pdf_path,
                        "page": chunk["metadata"].get("page", 0),
                        "content_type": chunk["content_type"],
                        **{k: v for k, v in chunk["metadata"].items()
                           if k not in ("source", "page", "content_type")},
                    }
                    dnngp_documents.append(Document(page_content=chunk["content"], metadata=metadata))
                logger.info("多模态解析完成，生成 %d 个文档块", len(dnngp_documents))
            else:
                logger.info("使用传统 PDF 提取器...")
                pdf_extractor = PDFExtractor(enable_tables=False)
                pdf_text, pdf_metadata = pdf_extractor.extract_text(dnngp_pdf_path, method="auto")
                logger.info("PDF 提取完成，方法：%s，文本长度：%d 字符",
                            pdf_metadata["extraction_method"], len(pdf_text))
                dnngp_documents = enhanced_chunker.chunk_document(pdf_text, dnngp_pdf_path)
                logger.info("学术论文增强分块完成，生成 %d 个文档块", len(dnngp_documents))
        except Exception as e:
            logger.error("PDF 处理失败: %s", e, exc_info=True)
    else:
        logger.warning("找不到学术论文 %s", dnngp_pdf_path)

    all_documents = company_documents + dnngp_documents
    logger.info("总计文档块数: %d (企业:%d, DNNGP:%d)",
                len(all_documents), len(company_documents), len(dnngp_documents))
    return all_documents


def _build_graph_index(graph_conn, txt_documents: List[Document]) -> list:
    """对 TXT 文档构建 Neo4j 图索引，返回 graph_documents 列表"""
    from langchain_experimental.graph_transformers import LLMGraphTransformer

    logger.info("正在构建图知识库索引... (仅处理 %d 个 TXT 文档块)", len(txt_documents))
    graph_transformer = LLMGraphTransformer(llm=graph_llm, ignore_tool_usage=True)

    batch_size = 10
    all_graph_documents = []
    for i in range(0, len(txt_documents), batch_size):
        batch = txt_documents[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(txt_documents) + batch_size - 1) // batch_size
        logger.info("处理批次 %d/%d (TXT文档块 %d-%d)...",
                     batch_num, total_batches, i + 1, min(i + batch_size, len(txt_documents)))
        try:
            gd = graph_transformer.convert_to_graph_documents(batch)
            all_graph_documents.extend(gd)
            graph_conn.add_graph_documents(gd)
            logger.info("批次完成，提取 %d 个图文档", len(gd))
        except Exception as e:
            logger.warning("批次处理失败: %s，跳过此批次", e)

    logger.info("图索引构建完成！图文档数: %d", len(all_graph_documents))
    return all_graph_documents


def _build_cypher_chain(graph_conn):
    """构建 GraphCypherQAChain"""
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

    cypher_prompt_template = """You are a Neo4j Cypher expert. Generate syntactically correct Cypher queries.

Schema:
{schema}

Important rules:
1. Company names are stored as FULL names (e.g., "小米科技有限责任公司"). ALWAYS use CONTAINS for partial matching
2. Use pattern: WHERE c.id CONTAINS 'keyword' instead of exact match
3. For querying technologies of a company, use: (c:Company)-[:HAS_TECHNOLOGY]->(t:Technology)
4. For querying partners of a company, use: (c:Company)-[:PARTNER_OF]->(p)
5. For querying where a company operates, use: (c:Company)-[:OPERATES_IN]->(l:Location) or (c:Company)-[:EXPANDS_TO]->(l:Location)
6. For querying products, use: (c:Company)-[:HAS_PRODUCT]->(p:Product)
7. NEVER use undefined variables
8. ALWAYS start with MATCH keyword
9. Use OPTIONAL MATCH to handle cases where relationships might not exist
10. When querying technologies, check HAS_TECHNOLOGY and DEVELOPS relationships
11. When querying partnerships, check PARTNER_OF and PARTNERS_WITH relationships
12. Return meaningful columns with aliases

Examples:
{examples}

Question: {question}
Cypher:"""

    improved_examples = [
        {"question": "都有哪些公司？", "query": "MATCH (c:Company) RETURN c.id as company"},
        {"question": "小米在技术创新方面有什么突破？",
         "query": "MATCH (c:Company)-[:HAS_TECHNOLOGY|DEVELOPS]->(t) WHERE c.id CONTAINS '小米' RETURN c.id as company, t.id as technology"},
        {"question": "华为有哪些技术？",
         "query": "MATCH (c:Company)-[:HAS_TECHNOLOGY]->(t:Technology) WHERE c.id CONTAINS '华为' RETURN t.id as technology"},
        {"question": "小米与哪些公司有合作关系？",
         "query": "MATCH (c:Company)-[:PARTNER_OF|PARTNERS_WITH]->(p) WHERE c.id CONTAINS '小米' RETURN p.id as partner"},
        {"question": "苹果和哪些公司有合作？",
         "query": "MATCH (c:Company)-[:PARTNER_OF|PARTNERS_WITH]->(p) WHERE c.id CONTAINS '苹果' RETURN p.id as partner"},
        {"question": "对比小米和华为的技术创新侧重点",
         "query": "MATCH (c:Company)-[:HAS_TECHNOLOGY|DEVELOPS]->(t) WHERE c.id CONTAINS '小米' OR c.id CONTAINS '华为' RETURN c.id as company, collect(t.id) as technologies"},
        {"question": "小米在哪些地区有业务？",
         "query": "MATCH (c:Company)-[:OPERATES_IN|EXPANDS_TO]->(l) WHERE c.id CONTAINS '小米' RETURN l.id as location"},
        {"question": "华为的合作伙伴有哪些？",
         "query": "MATCH (c:Company)-[:PARTNER_OF|PARTNERS_WITH]->(p) WHERE c.id CONTAINS '华为' RETURN p.id as partner, labels(p)[0] as type"},
    ]

    cypher_prompt = PromptTemplate(
        template=cypher_prompt_template,
        input_variables=["schema", "question"],
        partial_variables={
            "examples": "\n".join(
                f"Question: {ex['question']}\nCypher: {ex['query']}" for ex in improved_examples
            )
        },
    )

    return GraphCypherQAChain.from_llm(
        graph=graph_conn,
        cypher_llm=graph_llm,
        qa_llm=llm,
        cypher_prompt=cypher_prompt,
        validate_cypher=True,
        allow_dangerous_requests=True,
        top_k=3,
        return_intermediate_steps=False,
    )


def _build_vectorstore(documents: List[Document]):
    """构建 Milvus 向量索引"""
    from langchain_milvus import Milvus
    from pymilvus import connections, utility, Collection

    logger.info("正在构建 Milvus 向量索引...")
    connections.connect("default", host="localhost", port="19530", timeout=30)
    logger.info("Milvus 默认连接已初始化")

    collection_name = "company_milvus"
    try:
        if utility.has_collection(collection_name):
            logger.info("检测到旧集合 %s，正在删除...", collection_name)
            Collection(collection_name).drop()
            logger.info("旧集合已删除")
    except Exception as cleanup_error:
        logger.warning("清理旧集合时出错（可忽略）: %s", cleanup_error)

    logger.info("准备向量化 %d 个文档块...", len(documents))
    vs = Milvus.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        drop_old=False,
        enable_dynamic_field=True,
    )
    logger.info("向量索引构建完成，共 %d 个文档块", len(documents))
    return vs


# ============================================================
# 核心初始化函数
# ============================================================

def init_rag(force: bool = False):
    """
    初始化 RAG 系统（文档加载 → 图索引 → 向量索引）。

    线程安全，仅执行一次（除非 force=True）。
    """
    global vectorstore, cypher_chain, graph, processed_documents, _initialized

    with _init_lock:
        if _initialized and not force:
            logger.info("RAG 系统已初始化，跳过重复初始化")
            return

        logger.info("=" * 50)
        logger.info("RAG 系统初始化开始")
        logger.info("=" * 50)

        # 1. 加载文档
        all_documents = _load_documents()
        processed_documents = all_documents

        txt_documents = [d for d in all_documents if d.metadata.get("source", "").endswith(".txt")]
        pdf_documents = [d for d in all_documents if not d.metadata.get("source", "").endswith(".txt")]
        logger.info("GraphRAG 构建策略: TXT=%d 块(启用), PDF=%d 块(仅向量)", len(txt_documents), len(pdf_documents))

        # 2. Neo4j 连接 + 图索引
        graph_conn = None
        try:
            from langchain_community.graphs import Neo4jGraph
            graph_conn = Neo4jGraph(url=NEO4J_URL, username="neo4j", password="password", database="neo4j")
            logger.info("Neo4j 连接成功")
        except Exception as e:
            logger.error("Neo4j 连接失败: %s", e)

        graph = graph_conn

        if graph_conn and txt_documents:
            try:
                # 检查 Neo4j 是否已有图数据，避免重复构建
                existing = graph_conn.query("MATCH (n) RETURN count(n) as cnt")
                if existing and existing[0].get("cnt", 0) > 0:
                    logger.info("Neo4j 已有 %d 个节点，跳过图索引构建", existing[0]["cnt"])
                else:
                    _build_graph_index(graph_conn, txt_documents)
            except Exception as e:
                logger.error("图索引构建失败: %s", e, exc_info=True)

        # 3. Cypher QA Chain
        if graph_conn:
            try:
                cypher_chain = _build_cypher_chain(graph_conn)
            except Exception as e:
                logger.error("Cypher Chain 构建失败: %s", e)

        # 4. 向量索引
        if all_documents:
            try:
                vectorstore = _build_vectorstore(all_documents)
            except Exception as e:
                logger.error("向量索引构建失败: %s", e, exc_info=True)
                vectorstore = None
        else:
            logger.warning("无文档，跳过向量索引构建")

        _initialized = True
        logger.info("=" * 50)
        logger.info("RAG 系统初始化完成")
        logger.info("=" * 50)


# ============================================================
# 懒加载访问函数
# ============================================================

def get_vectorstore():
    """获取向量存储（首次调用时自动初始化；若构建失败则重试）"""
    global vectorstore
    if not _initialized:
        init_rag()
    if vectorstore is None and _initialized:
        # 向量索引构建曾失败，尝试单独重建
        logger.warning("向量存储为 None，尝试重新构建...")
        if processed_documents:
            try:
                vectorstore = _build_vectorstore(processed_documents)
                logger.info("向量存储重建成功")
            except Exception as e:
                logger.error("向量存储重建失败: %s", e, exc_info=True)
    return vectorstore


def get_cypher_chain():
    """获取 Cypher QA Chain（首次调用时自动初始化）"""
    if not _initialized:
        init_rag()
    return cypher_chain


def get_graph():
    """获取 Neo4j 图连接（首次调用时自动初始化）"""
    if not _initialized:
        init_rag()
    return graph


def get_processed_documents() -> List[Document]:
    """获取已处理的文档列表"""
    if not _initialized:
        init_rag()
    return processed_documents


# ============================================================
# 独立功能：多模态 PDF 处理（不依赖初始化）
# ============================================================

def process_pdf_with_multimodal(
    pdf_path: str,
    enable_ocr: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """使用多模态解析处理 PDF 文件"""
    if not os.path.exists(pdf_path):
        logger.warning("文件不存在: %s", pdf_path)
        return []

    if not MULTIMODAL_SUPPORT:
        logger.info("多模态解析不可用，使用传统解析方式")
        from app.pdf_extractor import PDFExtractor
        pdf_extractor = PDFExtractor(enable_tables=True)
        pdf_text, _ = pdf_extractor.extract_text(pdf_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "!", "?", "；", "，", " ", ""],
        )
        texts = splitter.split_text(pdf_text)
        return [Document(page_content=t, metadata={"source": pdf_path, "content_type": "text"}) for t in texts]

    logger.info("多模态解析 PDF: %s", os.path.basename(pdf_path))
    try:
        parser = MultimodalPDFParser(enable_ocr=enable_ocr, min_image_size=100)
        doc = parser.parse(pdf_path)
        chunks = doc.to_chunks(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = []
        for chunk in chunks:
            metadata = {
                "source": pdf_path,
                "page": chunk["metadata"].get("page", 0),
                "content_type": chunk["content_type"],
                **{k: v for k, v in chunk["metadata"].items()
                   if k not in ("source", "page", "content_type")},
            }
            documents.append(Document(page_content=chunk["content"], metadata=metadata))
        logger.info("解析完成: %d 个文档块", len(documents))
        return documents
    except Exception as e:
        logger.error("解析失败: %s", e)
        return []


__all__ = [
    "init_rag",
    "get_vectorstore",
    "get_cypher_chain",
    "get_graph",
    "get_processed_documents",
    "process_pdf_with_multimodal",
    # 向后兼容：直接访问模块变量
    "vectorstore",
    "cypher_chain",
    "graph",
    "processed_documents",
]
