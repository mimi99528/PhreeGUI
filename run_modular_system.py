#!/usr/bin/env python3
"""
运行模块化网格重建系统或化学-物理耦合仿真的入口脚本
支持多种运行模式
"""

import sys


def _force_utf8_streams():
    """见 launch_coupled_simulation.py 中同名函数：
    子进程独立继承控制台句柄，必须自己再配置一次，
    否则 GBK 控制台下中文字符串 print 会抛 UnicodeEncodeError。"""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_streams()

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QSurfaceFormat

def setup_opengl_format():
    """设置OpenGL格式"""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='化学-物理耦合仿真系统')
    parser.add_argument('--mode', choices=['mesh', 'coupled'], default='coupled',
                       help='运行模式: mesh(纯网格系统) 或 coupled(化学-物理耦合)')
    parser.add_argument('--scene', type=str, help='指定化学反应场景名称')
    
    args = parser.parse_args()
    
    print(f"启动{args.mode}模式...")
    
    # 设置OpenGL格式
    setup_opengl_format()
    
    # 创建Qt应用程序
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    try:
        if args.mode == 'coupled':
            # 运行化学-物理耦合仿真
            from coupled_chemistry_physics_gui import CoupledSimulationMainWindow
            window = CoupledSimulationMainWindow()
            window.show()
            
            # 如果指定了场景，自动选择
            if args.scene and hasattr(window, 'scenario_combo'):
                index = window.scenario_combo.findText(args.scene)
                if index >= 0:
                    window.scenario_combo.setCurrentIndex(index)
            
            print("化学-物理耦合仿真系统启动完成")
            print("功能特性:")
            print("  • 实时化学浓度可视化")
            print("  • 物理SPH流体仿真")
            print("  • 多种化学反应场景")
            print("  • 参数调节和监控")
            
        else:
            # 运行原始模块化网格系统
            from modular_mesh_system.gui.main_window import MainWindow
            window = MainWindow()
            window.show()
            
            print("模块化网格重建系统启动完成")
            print("请在GUI中测试各项功能")
        
        # 运行应用程序
        return app.exec()
        
    except Exception as e:
        print(f"程序运行出错: {e!r}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())