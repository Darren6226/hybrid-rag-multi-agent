"""
快速启动脚本 — 等价于 python main.py --fast

保留此文件以兼容已有文档和习惯。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接调用 main.py 的入口，注入 --fast 参数
sys.argv = [sys.argv[0], "--fast"] + sys.argv[1:]

from main import main

if __name__ == "__main__":
    main()
