"""
Supervisor 节点 - 智能路由管理
使用 RoutingPolicyManager 实现可配置的循环检测和回退策略
"""

import logging
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import END
from app.state import AgentState
from app.config import supervisor_llm
from app.routing_policy import create_default_policy, RoutingPolicyManager
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

__all__ = ["supervisor", "members", "Router", "policy_manager", "reset_supervisor"]

members = ["chat", "coder", "sqler", "graph_kg", "vec_kg"]
options = members + ["FINISH"]

# 全局策略管理器
policy_manager: RoutingPolicyManager = create_default_policy()

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH"""
    next: Literal["chat", "coder", "sqler", "graph_kg", "vec_kg", "FINISH"]


def reset_supervisor():
    """重置 Supervisor 状态（用于新问题）"""
    policy_manager.reset_for_new_query()


def _extract_worker_response_content(state: AgentState, worker_name: str) -> str:
    """提取特定代理的响应内容"""
    for msg in reversed(state["messages"]):
        if getattr(msg, "name", None) == worker_name:
            return msg.content
    return ""


def _check_worker_error(content: str, error_keywords: list) -> bool:
    """检查代理响应是否包含错误"""
    content_lower = content.lower()
    return any(keyword.lower() in content_lower for keyword in error_keywords)


def _analyze_response_quality(state: AgentState, worker_name: str) -> tuple[bool, str]:
    """
    分析代理响应质量
    返回: (是否成功, 错误信息)
    """
    content = _extract_worker_response_content(state, worker_name)

    if not content:
        return False, "No response"

    # 获取代理配置中的错误关键词
    config = policy_manager.worker_configs.get(worker_name)
    if config and config.error_keywords:
        if _check_worker_error(content, config.error_keywords):
            return False, f"Error detected in response: {content[:100]}"

    # 检查明显的失败模式
    failure_indicators = [
        "i don't know", "不知道", "无法回答", "无法找到答案",
        "no information", "没有信息",
        "error", "错误", "failed"
    ]

    for indicator in failure_indicators:
        if indicator.lower() in content.lower():
            # 如果是 chat 代理，这可能不是错误
            if worker_name == "chat":
                continue
            return False, f"Response contains failure indicator: {indicator}"

    return True, ""


def supervisor(state: AgentState):
    """
    Supervisor 路由决策函数
    结合 LLM 智能决策和策略管理器的循环检测
    """
    # 1. 识别已调用的代理
    last_worker = None
    called_workers = set()

    # 逆序遍历找到当前问题轮次中所有参与过的 worker
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and not hasattr(msg, 'name'):
            break
        name = getattr(msg, "name", None)
        if name in members:
            if last_worker is None:
                last_worker = name
            called_workers.add(name)

    # 分析上一个代理的响应质量并记录（修复：必须在 last_worker 赋值之后）
    if last_worker:
        success, error_msg = _analyze_response_quality(state, last_worker)
        policy_manager.record_attempt(last_worker, success, error_msg)

    logger.info("[Supervisor] 正在分析状态...")
    logger.info("  已尝试: %s", list(called_workers) or '无')
    logger.info("  上一步: %s", last_worker or '无')
    logger.info("  策略状态: %s", policy_manager.get_routing_summary())

    # 2. 构建动态 System Prompt
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}.\n\n"
        "You must respond with a valid JSON object containing a 'next' field.\n\n"
        "Each worker has a specific role:\n"
        "- chat: Responds directly to user inputs using natural language.\n"
        "- coder: Run python code to display diagrams or output execution results.\n"
        "- sqler: Handle structured data, including sales records, products, and COMPETITOR ANALYSIS (market share, regions).\n"
        "- graph_kg: Graph-based knowledge retrieval, for broad/relationship questions.\n"
        "- vec_kg: Vector-based semantic retrieval, for detailed/fine-grained questions.\n\n"
        f"Workers already tried in this turn: {list(called_workers)}\n"
        f"Last worker called: {last_worker or 'None'}\n"
        "CRITICAL RULES:\n"
        "1. If a worker's response has fully answered the request, return FINISH.\n"
        "2. If a question involves 'market share' or 'competitors', prefer 'sqler'.\n"
        "3. If a question involves partnerships/合作，try 'graph_kg' first, then fallback to others.\n"
        "4. If a worker indicates it doesn't know, try other available workers.\n"
        "5. Avoid calling the same worker twice unless specifically needed.\n"
    )

    # 添加历史错误信息
    if policy_manager.state.worker_errors:
        error_summary = "\nRecent worker errors:\n"
        for worker, errors in policy_manager.state.worker_errors.items():
            if errors:
                error_summary += f"- {worker}: {errors[-1][:100]}...\n"
        system_prompt += error_summary

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    # 使用 supervisor_llm（支持 JSON 模式）
    response = supervisor_llm.with_structured_output(Router).invoke(messages)
    next_ = response["next"]

    logger.info("[Supervisor] LLM 建议：%s", next_)

    # 3. 策略管理器干预 (仅保留回退逻辑)
    if next_ != "FINISH" and next_ != END:
        # 防止连续调用同一代理
        if next_ == last_worker:
            logger.info("[Supervisor] 防止循环：避免连续调用 %s", next_)
            fallback = policy_manager.get_fallback_worker(
                failed_worker=next_,
                called_workers=called_workers,
                last_worker=last_worker
            )
            if fallback:
                next_ = fallback
            else:
                next_ = "FINISH"

    # 轮次递进
    policy_manager.tick()

    logger.info("[Supervisor] 最终决策: 调度 [%s]", next_)

    if next_ == "FINISH":
        next_ = END

    return {"next": next_}
