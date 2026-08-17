import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.state import AgentState, trim_history
from app.agents import db_agent, code_agent
from app.rag import get_cypher_chain, get_vectorstore
from app.config import llm, graph_llm

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def _extract_user_question(state: AgentState) -> str:
    """从消息历史中提取用户的原始问题（无 name 属性的第一条 HumanMessage）。"""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and not hasattr(msg, "name"):
            return msg.content
    return state["messages"][-1].content if state["messages"] else ""


# ============================================================
# Worker 节点
# ============================================================

def sqler_node(state: AgentState):
    logger.info("[Worker: sqler] 正在执行结构化数据库查询...")
    result = db_agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name="sqler")]}


def coder_node(state: AgentState):
    logger.info("[Worker: coder] 正在执行 Python 代码生成与计算...")
    result = code_agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name="coder")]}


def graph_kg_node(state: AgentState):
    logger.info("[Worker: graph_kg] 正在执行 Neo4j 图知识库路径检索...")
    user_question = _extract_user_question(state)

    cypher_chain = get_cypher_chain()
    if cypher_chain is None:
        return {"messages": [HumanMessage(
            content="图知识库未初始化，请检查 Neo4j 服务是否正常运行。", name="graph_kg"
        )]}

    try:
        response = cypher_chain.invoke(user_question)
        result_content = response["result"]

        if not result_content or result_content.strip() == "":
            result_content = f"知识图谱中未找到与'{user_question}'直接相关的信息。建议：尝试使用向量检索(vec_kg)获取更多上下文。"
        elif "don't know" in result_content.lower() or "不知道" in result_content:
            if "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    if "context" in step and step["context"]:
                        result_content = f"根据知识图谱查询结果：\n{result_content}"
                        break

        return {"messages": [HumanMessage(content=result_content, name="graph_kg")]}

    except Exception as e:
        error_msg = str(e)
        logger.error("[Worker: graph_kg] 错误详情: %s", error_msg)

        if "SyntaxError" in error_msg or "syntax error" in error_msg.lower():
            error_response = "图知识库查询遇到语法错误。建议：1. 尝试重新表述问题 2. 使用更简单的查询语句 3. 切换到向量检索"
        elif "Connection refused" in error_msg or "connection" in error_msg.lower():
            error_response = "无法连接到图数据库服务。请检查 Neo4j 服务是否正常运行。"
        else:
            error_response = f"图知识库检索遇到错误: {error_msg}。建议尝试其他检索方式。"

        return {"messages": [HumanMessage(content=error_response, name="graph_kg")]}


def vec_kg_node(state: AgentState):
    logger.info("[Worker: vec_kg] 正在执行 Milvus 向量语义检索...")
    user_question = _extract_user_question(state)

    vectorstore = get_vectorstore()
    if vectorstore is None:
        error_msg = "向量存储未初始化，请检查 Milvus 数据库连接和文档处理是否成功。"
        logger.error(error_msg)
        return {"messages": [HumanMessage(content=error_msg, name="vec_kg")]}

    prompt = PromptTemplate(
        template=(
            "你是一个知识库检索助手。根据以下参考文档回答用户问题。\n"
            "要求：\n"
            "1. 仅基于参考文档中的信息作答，不要编造\n"
            "2. 如果文档中没有相关信息，明确说明\n"
            "3. 回答要简洁、准确、有条理\n\n"
            "参考文档：\n{context}\n\n"
            "用户问题：{question}\n\n"
            "回答："
        ),
        input_variables=["question", "context"],
    )

    rag_chain = prompt | graph_llm | StrOutputParser()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(user_question)
    generation = rag_chain.invoke({"context": docs, "question": user_question})

    return {"messages": [HumanMessage(content=generation, name="vec_kg")]}


def chat_node(state: AgentState):
    logger.info("[Worker: chat] 正在生成自然语言回复...")

    # 只看当前轮次（最近一条用户提问之后）是否有数据源 Worker 输出，
    # 避免多轮对话中上一轮的数据消息导致误用融合 prompt
    _DATA_WORKERS = {"sqler", "graph_kg", "vec_kg"}
    current_turn_msgs = []
    for msg in reversed(state["messages"]):
        current_turn_msgs.append(msg)
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None) is None:
            break  # 到达本轮用户提问
    data_sources_present = any(
        getattr(msg, "name", None) in _DATA_WORKERS
        for msg in current_turn_msgs
    )

    if data_sources_present:
        # 融合场景：多源信息合成
        system_prompt = (
            "你是一个信息融合助手。以下对话历史中包含来自不同数据源的检索结果，"
            "请综合所有信息，给出完整、准确的回答。\n\n"
            "数据源说明：\n"
            "- name='sqler' 的消息来自 MySQL 结构化数据库（销售额、产品、客户等）\n"
            "- name='graph_kg' 的消息来自 Neo4j 知识图谱（实体关系、合作伙伴等）\n"
            "- name='vec_kg' 的消息来自 Milvus 向量文档库（技术细节、文档描述等）\n\n"
            "融合规则：\n"
            "1. 综合所有数据源的信息，不要遗漏任何相关内容\n"
            "2. 如果不同数据源的信息互补，将它们有机整合为连贯的回答\n"
            "3. 如果不同数据源的信息冲突，以结构化数据(sqler)为准，并注明差异\n"
            "4. 如果某数据源返回了'无数据'或'未找到'，忽略该部分，基于可用信息回答\n"
            "5. 回答使用中文，保持简洁有条理"
        )
    else:
        # 普通对话场景：无数据源信息（如问候、闲聊）
        system_prompt = (
            "你是一个友好的对话助手。请根据对话历史自然地回应用户。"
            "如果没有足够的信息，请坦诚告知。使用中文回答。"
        )

    # 滑动窗口裁剪：与 Supervisor 共用同一窗口策略，防止多轮历史 token 溢出
    messages = [SystemMessage(content=system_prompt)] + trim_history(list(state["messages"]))
    model_response = llm.invoke(messages)
    return {"messages": [HumanMessage(content=model_response.content, name="chat")]}
