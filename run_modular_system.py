#!/usr/bin/env python3
"""
运行模块化网格重建系统的简化测试脚本
"""

import sys
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
    print("启动模块化网格重建系统测试...")
    
    # 设置OpenGL格式
    setup_opengl_format()
    
    # 创建Qt应用程序
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    try:
        # 导入并创建主窗口
        from modular_mesh_system.gui.main_window import MainWindow
        window = MainWindow()
        window.show()
        
        print("系统启动完成，显示主窗口...")
        print("请在GUI中测试各项功能")
        
        # 运行应用程序
        return app.exec()
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())