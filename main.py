"""
混合 RAG 多代理系统 — CLI 入口

用法：
    python main.py                          # 完整模式（含 Neo4j 图检索）
    python main.py --fast                   # 快速模式（跳过 Neo4j，仅向量检索）
    python main.py -q "你的问题"             # 单次提问
    python main.py --fast -q "你的问题"      # 快速模式 + 单次提问

使用前请确保 Docker 环境已启动：
    docker-compose -f docker-compose-rag.yml up -d
"""

import sys
import io
import argparse
import logging

from app.database import init_seed_data
from app.rag import init_rag
from app.graph_builder import build_graph
from app.stream_utils import print_stream

# 设置输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="混合 RAG 多代理智能体系统")
    parser.add_argument(
        "--fast", action="store_true",
        help="快速模式：跳过 Neo4j 图索引构建，graph_kg 路由重定向到 vec_kg"
    )
    parser.add_argument(
        "-q", "--query", type=str, default=None,
        help="单次提问（省略则进入交互模式）"
    )
    args = parser.parse_args()

    mode = "快速模式（仅向量检索）" if args.fast else "完整模式（向量 + 图检索）"
    logger.info("启动 %s...", mode)

    # 初始化
    init_seed_data()
    init_rag()

    # 构建图
    graph = build_graph(skip_graph_kg=args.fast)

    print(f"\n{'=' * 60}")
    print(f"混合 RAG 多代理系统已就绪 — {mode}")
    print(f"{'=' * 60}\n", flush=True)

    # 单次提问模式
    if args.query:
        print_stream(graph, args.query)
        return

    # 交互模式
    print("输入问题进行查询，输入 quit/exit 退出。\n", flush=True)
    while True:
        try:
            query = input("提问: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！", flush=True)
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("再见！", flush=True)
            break

        print(f"\n{'=' * 60}", flush=True)
        print_stream(graph, query)
        print(flush=True)


if __name__ == "__main__":
    main()
