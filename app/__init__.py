# LangGraph Supervisor 多代理应用
"""
混合 RAG 多代理智能体系统

模块结构:
- config: 全局配置 (LLM, 数据库连接)
- state: 状态定义
- database: 数据库模型与初始化
- tools: 工具函数定义
- agents: ReAct 代理工厂
- supervisor: Supervisor 路由逻辑
- routing_policy: 可配置的路由策略管理器
- nodes: 各 worker 节点实现
- rag: RAG 链配置
"""

__version__ = "1.0.0"
