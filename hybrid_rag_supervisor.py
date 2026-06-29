"""
Supervisor 演示入口 — 等价于 python main.py

保留此文件以兼容教学文档中的引用。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    main()
