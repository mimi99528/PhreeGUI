#!/usr/bin/env python3
"""
模块化网格重建系统主程序
Modular Mesh Reconstruction System - Main Entry Point
"""

import sys


def _force_utf8_streams():
    """见 launch_coupled_simulation.py 中同名函数说明。
    本文件作为 python -m modular_mesh_system.main 启动时是独立进程，
    必须自行配置，否则 GBK 控制台下中文 print 会抛 UnicodeEncodeError。"""
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

from .gui.main_window import MainWindow
from .utils.config import CONFIG

def setup_opengl_format():
    """设置OpenGL格式"""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(4)  # 抗锯齿
    QSurfaceFormat.setDefaultFormat(fmt)

def main():
    """主程序入口点"""
    print("启动模块化网格重建系统...")
    print(f"Python版本: {sys.version}")
    print(f"系统配置: 粒子半径={CONFIG.PARTICLE_RADIUS}, 最大粒子数={CONFIG.MAX_PARTICLES}")
    
    # 设置OpenGL格式
    setup_opengl_format()
    
    # 创建Qt应用程序
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion样式获得更好的外观
    
    # 创建并显示主窗口
    try:
        window = MainWindow()
        window.show()
        
        print("系统启动完成，进入主循环...")
        exit_code = app.exec()
        print(f"程序退出，退出码: {exit_code}")
        return exit_code
        
    except Exception as e:
        print(f"程序启动失败: {e!r}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())