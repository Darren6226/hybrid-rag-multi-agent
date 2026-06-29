from typing import List
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from app.state import ReActAgentState
from app.config import llm
from app.tools import add_sale, delete_sale, update_sale, query_sales, execute_sql, python_repl

def create_react_agent_v1(llm, tools: List[BaseTool], system_prompt: str):
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def think_node(state: ReActAgentState):
        if not any(isinstance(m, SystemMessage) for m in state["messages"]):
            messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        else:
            messages = list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_node(state: ReActAgentState):
        last_msg = state["messages"][-1]
        tool_messages = []
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tool_call in last_msg.tool_calls:
                try:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    res = tool_map[tool_name].invoke(tool_args)
                    tool_messages.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
                except Exception as e:
                    tool_messages.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call["id"]))
        return {"messages": tool_messages}

    def should_continue(state: ReActAgentState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tool"
        return END

    builder = StateGraph(ReActAgentState)
    builder.add_node("think", think_node)
    builder.add_node("tool", tool_node)
    builder.add_edge(START, "think")
    builder.add_conditional_edges("think", should_continue, {"tool": "tool", END: END})
    builder.add_edge("tool", "think")
    
    return builder.compile()

db_agent_system_prompt = """You are a database expert. You can perform database operations and provide accurate data.

Available database tables:
1. customer_information (customer_id, customer_name, contact_info, region, customer_type)
2. product_information (product_id, product_name, category, unit_price, stock_level)
3. sales_data (sales_id, product_id, employee_id, customer_id, sale_date, quantity, amount, discount)
4. competitor_analysis (competitor_id, competitor_name, region, market_share)

When user asks about "companies" (公司), query the customer_information table.
When user asks about "products" (产品), query the product_information table.
When user asks about "sales" (销售), query the sales_data table.

Use execute_sql tool with SELECT queries to get data. Always respond in Chinese."""

db_agent = create_react_agent_v1(
    llm=llm,
    tools=[add_sale, delete_sale, update_sale, query_sales, execute_sql],
    system_prompt=db_agent_system_prompt
)

code_agent = create_react_agent_v1(
    llm=llm,
    tools=[python_repl],
    system_prompt=(
        "你是一个 Python 代码执行助手。使用 python_repl 工具运行代码来完成用户的请求。\n"
        "可用库：pandas、numpy、math、matplotlib。\n"
        "注意：代码在受限沙箱中运行，不可使用 os/subprocess/sys 等系统模块。"
    )
)
