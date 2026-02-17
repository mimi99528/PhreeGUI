"""
PhreeqcRM 空间扩散模拟主程序 (模块化版本)
====================================================
这是一个模块化的PhreeqcRM pH扩散模拟程序，包含以下主要功能：

功能模块说明：
1. utils.py          - 通用工具函数和常量定义
2. core_simulation.py - 核心模拟逻辑和流程控制  
3. physics.py        - 物理计算（扩散等）
4. data_handling.py  - 数据获取和处理
5. visualization.py  - 可视化功能

修改内容：
1. 示踪剂改为 pH 追踪
2. 在右下角放置等量 HCl
3. 修复了混合机制，使用 SetConcentrations 实现真正的质量传输
4. 修正了多组件数据的数组操作
5. 采用模块化设计，便于维护和扩展
"""

# 导入分离后的模块
from core_simulation import simulate_spatial_diffusion
from visualization import create_interactive_3d_visualization
from utils import NX, NY, NZ

def main():
    """
    主程序入口函数
    """
    # 运行核心模拟
    ph_history = simulate_spatial_diffusion(steps=1000)
    
    # 创建交互式3D可视化
    create_interactive_3d_visualization(ph_history, NX, NY, NZ)

if __name__ == "__main__":
    main()