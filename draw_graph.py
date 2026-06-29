import os
from hybrid_rag_supervisor import graph

def draw_system_architecture():
    """
    独立生成并保存系统架构图
    """
    print("\n>>> [System] 正在分析系统拓扑结构并生成架构图...", flush=True)
    
    try:
        # 使用 xray=True 展开 ReAct 代理内部结构
        # 此时由于在 hybrid_rag_supervisor.py 中修复了 conditional_edges 的映射，
        # 连线将会正确显示
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        
        output_file = "system_architecture_xray.png"
        with open(output_file, "wb") as f:
            f.write(png_data)
            
        print(f"✅ 架构图生成成功！")
        print(f"📍 文件路径: {os.path.abspath(output_file)}")
        print("\n提示：该图展示了 Supervisor 如何通过条件边调度各个子代理，")
        print("同时也展示了 sqler 和 coder 内部的 ReAct (Think-Tool) 循环。")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        print("请确保网络连接正常（draw_mermaid_png 需要访问渲染服务）。")

if __name__ == "__main__":
    draw_system_architecture()