#!/usr/bin/env python3
"""
化学-物理耦合仿真快速启动脚本
Quick Launch Script for Coupled Chemistry-Physics Simulation
"""

import subprocess
import sys
import os

def main():
    """主启动函数"""
    print("=" * 60)
    print("化学-物理耦合仿真系统启动器")
    print("=" * 60)
    print()
    # 检查必要的模块
    required_modules = ['PySide6', 'numpy']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} 已安装")
        except ImportError:
            missing_modules.append(module)
            print(f"✗ {module} 未安装")
    
    if missing_modules:
        print(f"\n缺少必要模块: {', '.join(missing_modules)}")
        print("请先安装依赖包:")
        print("pip install PySide6 numpy")
        return 1
    
    # 显示可用场景
    scenes = [
        ("acid_base", "酸碱中和反应"),
        ("carbonate", "碳酸盐平衡"),
        ("redox", "氧化还原反应")
    ]
    
    print("\n可用化学反应场景:")
    for i, (key, desc) in enumerate(scenes, 1):
        print(f"  {i}. {desc} ({key})")
    
    print("\n启动选项:")
    print("  1. 直接启动GUI (默认场景)")
    print("  2. 指定场景启动")
    print("  3. 启动原始网格系统")
    print("  4. 退出")
    
    try:
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == "1":
            # 直接启动耦合仿真GUI
            print("\n启动化学-物理耦合仿真GUI...")
            subprocess.run([sys.executable, "run_modular_system.py", "--mode", "coupled"])
            
        elif choice == "2":
            # 显示场景选择
            print("\n选择化学反应场景:")
            for i, (key, desc) in enumerate(scenes, 1):
                print(f"  {i}. {desc}")
            
            try:
                scene_choice = int(input("请选择场景 (1-3): ")) - 1
                if 0 <= scene_choice < len(scenes):
                    scene_key = scenes[scene_choice][0]
                    print(f"\n启动 {scenes[scene_choice][1]} 场景...")
                    subprocess.run([sys.executable, "run_modular_system.py", "--mode", "coupled", "--scene", scene_key])
                else:
                    print("无效选择")
            except ValueError:
                print("请输入有效数字")
                
        elif choice == "3":
            # 启动原始网格系统
            print("\n启动原始网格重建系统...")
            subprocess.run([sys.executable, "run_modular_system.py", "--mode", "mesh"])
            
        elif choice == "4":
            print("再见!")
            return 0
            
        else:
            print("无效选择，请输入 1-4")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        return 0
    except Exception as e:
        print(f"启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())