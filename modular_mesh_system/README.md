# 模块化网格重建系统

这是一个将原有的大型单一文件重构为模块化架构的网格重建系统。

## 项目结构

```
modular_mesh_system/
├── __init__.py              # 包初始化文件
├── main.py                  # 主程序入口点
├── core/                    # 核心模块
│   ├── __init__.py
│   └── mesh_system.py      # 网格系统核心逻辑
├── gui/                     # GUI界面模块
│   ├── __init__.py
│   └── main_window.py      # 主窗口和UI组件
├── renderer/                # 渲染器模块
│   ├── __init__.py
│   └── opengl_renderer.py  # OpenGL渲染器实现
└── utils/                   # 工具模块
    ├── __init__.py
    ├── config.py           # 系统配置和常量
    ├── helpers.py          # 辅助工具函数
    └── performance_analyzer.py  # 性能分析模块
```

## 模块说明

### Core模块 (`core/`)
包含系统的核心逻辑和物理仿真组件：
- `MeshSystem`: 网格重建核心系统，负责物理仿真、粒子管理和网格生成

### GUI模块 (`gui/`)
包含用户界面相关的组件：
- `MainWindow`: 主窗口类，包含所有UI控件和事件处理
- `QOpenGLWidgetWrapper`: OpenGL部件包装器

### Renderer模块 (`renderer/`)
包含图形渲染相关的实现：
- `OpenGLRenderer`: ModernGL渲染器，负责所有的OpenGL渲染逻辑

### Utils模块 (`utils/`)
包含辅助工具和配置：
- `Config`: 系统配置类，集中管理所有常量和参数
- `helpers`: 辅助函数集合，包括数学计算、性能装饰器等
- `performance_analyzer`: 性能分析和监控模块

## 使用方法

### 运行程序
```bash
python -m modular_mesh_system
```

或者直接运行主程序：
```bash
python modular_mesh_system/main.py
```

### 作为模块导入
```python
from modular_mesh_system import MeshSystem, OpenGLRenderer, MainWindow

# 创建系统实例
mesh_system = MeshSystem()
renderer = OpenGLRenderer(mesh_system)
window = MainWindow()
```

## 优势

1. **模块化设计**: 每个功能模块职责清晰，便于维护和扩展
2. **可重用性**: 核心组件可以在其他项目中重用
3. **易于测试**: 每个模块可以独立测试
4. **配置集中**: 所有配置参数集中在config模块中
5. **性能监控**: 内置详细的性能分析功能

## 依赖关系

```
main.py → gui/main_window.py → core/mesh_system.py
                             → renderer/opengl_renderer.py
                             → utils/config.py
                             → utils/performance_analyzer.py
```

## 配置说明

所有系统参数都在 `utils/config.py` 中定义，可以根据需要调整：
- 物理仿真参数
- 渲染参数  
- 性能监控设置
- SSFR参数配置

## 开发指南

1. **添加新功能**: 在相应的模块中添加，遵循现有架构
2. **性能优化**: 使用utils/helpers中的性能装饰器进行分析
3. **配置调整**: 修改utils/config.py中的相应参数
4. **UI扩展**: 在gui/main_window.py中添加新的控件和功能