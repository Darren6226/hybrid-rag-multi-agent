"""
测试安全工具模块

重点测试 SQL 白名单验证和 Python 沙箱隔离。
"""

import pytest
from app.tools import _validate_sql, python_repl


# ============================================================
# 测试 _validate_sql
# ============================================================

class TestValidateSql:
    """SQL 白名单验证测试"""

    def test_valid_select(self):
        """合法 SELECT 应通过"""
        assert _validate_sql("SELECT * FROM sales_data") is None

    def test_valid_select_with_where(self):
        assert _validate_sql("SELECT product_name, unit_price FROM product_information WHERE stock_level < 250") is None

    def test_reject_non_select(self):
        """非 SELECT 语句应被拒绝"""
        assert _validate_sql("INSERT INTO sales_data VALUES (1,2,3)") is not None
        assert _validate_sql("UPDATE sales_data SET amount=100") is not None
        assert _validate_sql("DELETE FROM sales_data") is not None
        assert _validate_sql("DROP TABLE sales_data") is not None

    def test_reject_too_long(self):
        """超长查询应被拒绝"""
        long_query = "SELECT * FROM t WHERE " + "x" * 500
        assert _validate_sql(long_query) is not None

    def test_reject_multiple_statements(self):
        """多语句应被拒绝"""
        assert _validate_sql("SELECT 1; DROP TABLE t") is not None

    def test_reject_comments(self):
        """SQL 注释应被拒绝（防注入）"""
        assert _validate_sql("SELECT * FROM t -- comment") is not None
        assert _validate_sql("SELECT * FROM t /* comment */") is not None

    def test_reject_into_outfile(self):
        """INTO OUTFILE 应被拒绝"""
        assert _validate_sql("SELECT * INTO OUTFILE '/tmp/out.txt' FROM t") is not None

    def test_reject_benchmark(self):
        """BENCHMARK 应被拒绝"""
        assert _validate_sql("SELECT BENCHMARK(1000000, SHA1('test'))") is not None

    def test_reject_sleep(self):
        """SLEEP 应被拒绝"""
        assert _validate_sql("SELECT SLEEP(10)") is not None

    def test_allow_trailing_semicolon(self):
        """末尾分号应允许（LLM 常生成）"""
        assert _validate_sql("SELECT 1;") is None

    def test_reject_empty_query(self):
        """空查询应被拒绝"""
        assert _validate_sql("") is not None
        assert _validate_sql("   ") is not None


# ============================================================
# 测试 python_repl 沙箱
# ============================================================

class TestPythonRepl:
    """Python 沙箱执行测试"""

    def test_simple_print(self):
        """简单 print 应正常执行"""
        result = python_repl.invoke({"code": "print(1 + 2)"})
        assert "3" in result

    def test_pandas_available(self):
        """pd (pandas) 应在沙箱中可用"""
        result = python_repl.invoke({"code": "import pandas as pd\ndf = pd.DataFrame({'a': [1,2,3]})\nprint(df.sum())"})
        assert "6" in result

    def test_numpy_available(self):
        """np (numpy) 应在沙箱中可用"""
        result = python_repl.invoke({"code": "import numpy as np\nprint(np.sum([1,2,3]))"})
        assert "6" in result

    def test_math_available(self):
        """math 应在沙箱中可用"""
        result = python_repl.invoke({"code": "import math\nprint(math.sqrt(16))"})
        assert "4.0" in result

    def test_reject_os_import(self):
        """os 模块应被拒绝"""
        result = python_repl.invoke({"code": "import os\nos.system('ls')"})
        assert "Error" in result or "not allowed" in result.lower()

    def test_reject_subprocess_import(self):
        """subprocess 模块应被拒绝"""
        result = python_repl.invoke({"code": "import subprocess\nsubprocess.run(['ls'])"})
        assert "Error" in result or "not allowed" in result.lower()

    def test_reject_sys_import(self):
        """sys 模块应被拒绝"""
        result = python_repl.invoke({"code": "import sys\nprint(sys.version)"})
        assert "Error" in result or "not allowed" in result.lower()

    def test_reject_file_operations(self):
        """文件操作应被拒绝"""
        result = python_repl.invoke({"code": "open('/etc/passwd', 'r').read()"})
        assert "Error" in result

    def test_code_too_long(self):
        """超长代码应被拒绝"""
        long_code = "x = 1\n" * 2001
        result = python_repl.invoke({"code": long_code})
        assert "Error" in result or "too long" in result.lower()

    def test_syntax_error_reported(self):
        """语法错误应被正确报告"""
        result = python_repl.invoke({"code": "def foo("})
        assert "error" in result.lower()

    def test_runtime_error_reported(self):
        """运行时错误应被正确报告"""
        result = python_repl.invoke({"code": "x = 1 / 0"})
        assert "error" in result.lower() or "ZeroDivision" in result

    def test_no_output_reported(self):
        """无输出的代码应被正确报告"""
        result = python_repl.invoke({"code": "x = 42"})
        assert "No output" in result or "Successfully" in result
