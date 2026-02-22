"""
模块化网格重建系统
Modular Mesh Reconstruction System
"""

__version__ = "1.0.0"
__author__ = "PhreeGUI Team"

# 导出主要类和函数
from .core.mesh_system import MeshSystem
from .renderer.opengl_renderer import OpenGLRenderer
from .gui.main_window import MainWindow
from .utils.config import Config
from .utils.performance_analyzer import PerformanceAnalyzer

__all__ = [
    'MeshSystem',
    'OpenGLRenderer', 
    'MainWindow',
    'Config',
    'PerformanceAnalyzer'
]