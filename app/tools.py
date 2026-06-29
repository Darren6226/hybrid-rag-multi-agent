"""
安全工具模块
- SQL: 结构化白名单验证 + 只读事务，防止注入
- Python REPL: RestrictedPython v8 沙箱，白名单机制
"""

import io
import contextlib
from typing import Annotated
from langchain_core.tools import tool
from pydantic import BaseModel
from app.database import _get_session, SalesData

# ============================================================
# 数据库 CRUD 工具（ORM 操作，天然参数化）
# ============================================================

class AddSaleSchema(BaseModel):
    product_id: int
    employee_id: int
    customer_id: int
    sale_date: str
    quantity: int
    amount: float
    discount: float

class DeleteSaleSchema(BaseModel):
    sales_id: int

class UpdateSaleSchema(BaseModel):
    sales_id: int
    quantity: int
    amount: float

class QuerySalesSchema(BaseModel):
    sales_id: int

@tool(args_schema=AddSaleSchema)
def add_sale(product_id, employee_id, customer_id, sale_date, quantity, amount, discount):
    """Add sale record to the database."""
    session = _get_session()
    try:
        new_sale = SalesData(
            product_id=product_id, employee_id=employee_id,
            customer_id=customer_id, sale_date=sale_date,
            quantity=quantity, amount=amount, discount=discount,
        )
        session.add(new_sale)
        session.commit()
        return {"messages": ["销售记录添加成功。"]}
    except Exception as e:
        session.rollback()
        return {"messages": [f"添加失败，错误原因：{e}"]}
    finally:
        session.close()

@tool(args_schema=DeleteSaleSchema)
def delete_sale(sales_id):
    """Delete sale record from the database."""
    session = _get_session()
    try:
        sale_to_delete = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_to_delete:
            session.delete(sale_to_delete)
            session.commit()
            return {"messages": ["销售记录删除成功。"]}
        return {"messages": [f"未找到销售记录ID：{sales_id}"]}
    except Exception as e:
        session.rollback()
        return {"messages": [f"删除失败，错误原因：{e}"]}
    finally:
        session.close()

@tool(args_schema=UpdateSaleSchema)
def update_sale(sales_id, quantity, amount):
    """Update sale record in the database."""
    session = _get_session()
    try:
        sale_to_update = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_to_update:
            sale_to_update.quantity = quantity
            sale_to_update.amount = amount
            session.commit()
            return {"messages": ["销售记录更新成功。"]}
        return {"messages": [f"未找到销售记录ID：{sales_id}"]}
    except Exception as e:
        session.rollback()
        return {"messages": [f"更新失败，错误原因：{e}"]}
    finally:
        session.close()

@tool(args_schema=QuerySalesSchema)
def query_sales(sales_id):
    """Query sale record from the database."""
    session = _get_session()
    try:
        sale_data = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_data:
            return {
                "sales_id": sale_data.sales_id, "product_id": sale_data.product_id,
                "employee_id": sale_data.employee_id, "customer_id": sale_data.customer_id,
                "sale_date": sale_data.sale_date, "quantity": sale_data.quantity,
                "amount": sale_data.amount, "discount": sale_data.discount,
            }
        return {"messages": [f"未找到销售记录ID：{sales_id}。"]}
    except Exception as e:
        return {"messages": [f"查询失败，错误原因：{e}"]}
    finally:
        session.close()


# ============================================================
# SQL 执行工具 — 结构化白名单验证 + 只读事务
# ============================================================

# 危险模式检测（大小写不敏感，覆盖绕过变体）
_DANGEROUS_PATTERNS = [
    "INTO OUTFILE", "INTO DUMPFILE", "LOAD_FILE",
    "BENCHMARK(", "SLEEP(", "WAITFOR", "PG_SLEEP",
    "EXEC(", "EXECUTE(", "CALL(",
]


def _validate_sql(query: str) -> str | None:
    """
    结构化验证 SQL 查询。
    返回 None 表示通过，返回字符串表示错误信息。
    """
    query = query.strip()

    if len(query) > 500:
        return "Error: Query too long (max 500 characters)."

    if not query.upper().startswith("SELECT"):
        return "Error: Only SELECT queries allowed for safety."

    # 允许末尾分号（LLM 常生成），但中间不允许
    stripped = query.rstrip(";").strip()
    if ";" in stripped:
        return "Error: Multiple statements not allowed. Provide a single SELECT."

    if "--" in query or "/*" in query:
        return "Error: SQL comments are not allowed."

    query_upper = query.upper()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in query_upper:
            return f"Error: Dangerous pattern detected: {pattern}"

    return None


@tool
def execute_sql(query: str):
    """
    Execute a read-only SQL query on the database.
    Use SELECT queries for listing companies, products, sales, competitors, etc.
    The query runs in a read-only transaction — no modifications allowed.
    """
    error = _validate_sql(query)
    if error:
        return error

    session = _get_session()
    try:
        from sqlalchemy import text
        # 启动只读事务（MySQL 8.0+ 支持）
        session.execute(text("START TRANSACTION READ ONLY"))
        # 执行查询
        result = session.execute(text(query))
        rows = [dict(zip(result.keys(), row)) for row in result.fetchall()]
        return rows if rows else "查询结果为空。"
    except Exception as e:
        return f"SQL Error: {e}"
    finally:
        session.close()


# ============================================================
# Python REPL — RestrictedPython v8 安全沙箱
# ============================================================

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import safer_getattr, guarded_setattr
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.Eval import default_guarded_getiter

# 允许导入的模块白名单
_ALLOWED_MODULES = frozenset({
    "pandas", "numpy", "math", "datetime", "json",
    "csv", "re", "collections", "itertools", "functools",
    "statistics", "decimal", "fractions",
})


def _safe_import(name, *args, **kwargs):
    """白名单导入：只允许数据科学相关的安全模块（供 numpy 内部调用）"""
    import importlib
    top = name.split(".")[0]
    if top not in _ALLOWED_MODULES:
        raise ImportError(
            f"Import of '{name}' is not allowed. "
            f"Allowed: {', '.join(sorted(_ALLOWED_MODULES))}"
        )
    return importlib.import_module(name)


def _build_restricted_globals():
    """构建受限的全局命名空间（预注入安全模块，无需用户 import）"""
    import pandas as pd
    import numpy as np
    import math, datetime, json, csv, re
    import collections, itertools, functools, statistics

    g = safe_globals.copy()
    sb = g["__builtins__"]

    # 白名单内置函数（移除所有危险函数）
    safe_builtins = {
        "print", "range", "len", "sum", "max", "min", "abs", "round",
        "sorted", "reversed", "enumerate", "zip", "map", "filter",
        "list", "dict", "set", "tuple", "frozenset",
        "str", "int", "float", "bool", "complex",
        "isinstance", "repr",
        "True", "False", "None",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "ZeroDivisionError", "AttributeError", "RuntimeError",
        "StopIteration", "NotImplemented",
    }
    for attr in list(sb.keys()):
        if attr not in safe_builtins:
            sb.pop(attr, None)

    # 注入受限的 __import__（numpy 内部需要，但只允许白名单模块）
    sb["__import__"] = _safe_import

    # 预注入安全模块到全局命名空间（用户代码直接用 pd/np/math 等）
    g.update({
        "pd": pd, "np": np, "math": math, "datetime": datetime,
        "json": json, "csv": csv, "re": re,
        "collections": collections, "itertools": itertools,
        "functools": functools, "statistics": statistics,
    })

    # RestrictedPython v8 守卫
    g["_getattr_"] = safer_getattr
    g["_setattr_"] = guarded_setattr
    g["_getiter_"] = default_guarded_getiter
    g["_print_"] = PrintCollector
    g["_inplacevar_"] = lambda op, x, y: op(x, y)

    return g


# 预构建全局命名空间（避免每次调用重新构建）
_RESTRICTED_GLOBALS = _build_restricted_globals()


@tool
def python_repl(code: Annotated[str, "The python code to execute to generate your chart."]):
    """
    Execute python code in a RestrictedPython sandbox.
    Allowed: pd (pandas), np (numpy), math, datetime, json, csv, re,
             collections, itertools, functools, statistics, print.
    Forbidden: file ops, network access, system commands, os, sys, subprocess.
    """
    if len(code) > 2000:
        return "Error: Code too long (max 2000 characters). Please keep it concise."

    # 1. RestrictedPython 编译时插入安全守卫
    try:
        byte_code = compile_restricted(code, "<sandbox>", "exec")
    except SyntaxError as e:
        return f"Syntax error: {e}"
    except Exception as e:
        return f"Compilation error: {e}"

    # 2. 在受限环境中执行
    #    RestrictedPython 编译 print(x) 为 _print_(x)，
    #    PrintCollector 实例存入 ctx["_print"]，输出在 ctx["_print"].txt 列表中。
    try:
        exec_ctx = _RESTRICTED_GLOBALS.copy()
        exec(byte_code, exec_ctx)

        # 从 PrintCollector 收集输出
        print_collector = exec_ctx.get("_print")
        if print_collector is not None and hasattr(print_collector, "txt"):
            captured = "".join(print_collector.txt)
        else:
            captured = ""

        if captured:
            return f"Successfully executed:\n```python\n{code}\n```\nOutput:\n{captured}"
        return f"Successfully executed:\n```python\n{code}\n```\n(No output)"

    except Exception as e:
        return f"Execution error: {type(e).__name__}: {e}"
