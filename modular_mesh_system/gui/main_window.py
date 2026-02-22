"""
主窗口GUI模块
Main Window GUI Module
"""

import sys
import time
import numpy as np
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QCheckBox, QComboBox, QGroupBox,
    QTextEdit, QSplitter, QFrame, QTabWidget, QGridLayout
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject
from PySide6.QtGui import QFont
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..core.mesh_system import MeshSystem
from ..renderer.opengl_renderer import OpenGLRenderer
from ..utils.config import CONFIG
from ..utils.helpers import get_system_stats_formatted

class QOpenGLWidgetWrapper(QOpenGLWidget):
    """OpenGL widget wrapper for ModernGL - 修复版"""

    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer
        self.ctx = None
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def initializeGL(self):
        """Initialize OpenGL context - 修复版"""
        try:
            self.makeCurrent()
            import moderngl
            from OpenGL.GL import glGetString
            self.ctx = moderngl.create_context(standalone=False)
            print(glGetString(0x1F02))  # GL_VERSION
            print(self.ctx.version_code, self.ctx.info)
            print("✓ ModernGL context created successfully")

            if not self.renderer.initialize(self.ctx):
                print("✗ Renderer initialization failed")
                return

            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

            print("✓ OpenGL state configured successfully")
                    
        except Exception as e:
            print(f"✗ OpenGL initialization failed: {e}")

    def paintGL(self):
        """Paint OpenGL scene - 修复版"""
        if not self.ctx:
            return

        screen_fbo = self.ctx.detect_framebuffer()
        screen_fbo.use()

        self.makeCurrent()
        size = self.size()
        width, height = size.width(), size.height()

        if width > 0 and height > 0:
            self.ctx.viewport = (0, 0, width, height)
            self.ctx.clear(0.1, 0.1, 0.15, 1.0)
            self.renderer.render(width, height, screen_fbo)
        else:
            print(f"⚠ Invalid viewport size: {width}x{height}")

    def resizeGL(self, width, height):
        """Handle resize event - 修复版"""
        if self.ctx and width > 0 and height > 0:
            self.ctx.viewport = (0, 0, width, height)
            print(f"✓ Resize OpenGL viewport to {width}x{height}")

class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.mesh_system = MeshSystem()
        self.opengl_renderer = OpenGLRenderer(self.mesh_system)
        self.is_running = False
        self.is_paused = False
        self.fps_counter = 0
        self.last_fps_time = time.time()

        self.init_ui()
        self.setup_timer()

    def init_ui(self):
        """Initialize UI with separated panels"""
        self.setWindowTitle("PySide6 + ModernGL 网格重建系统")
        self.setGeometry(100, 100, 1400, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Left control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=0)

        # Center area - OpenGL widget and performance panel splitter
        center_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # OpenGL widget
        self.opengl_widget = QOpenGLWidgetWrapper(self.opengl_renderer)
        center_splitter.addWidget(self.opengl_widget)
        
        # Performance panel
        performance_panel = self.create_performance_panel()
        center_splitter.addWidget(performance_panel)
        
        # Set initial sizes
        center_splitter.setSizes([700, 300])
        
        main_layout.addWidget(center_splitter, stretch=1)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("系统就绪")

    def create_control_panel(self):
        """Create control panel"""
        panel = QFrame()
        panel.setFixedWidth(300)
        panel.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #444444;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666666;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QSlider {
                background-color: #333333;
            }
            QCheckBox {
                color: white;
            }
            QComboBox {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666666;
            }
        """)

        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("控制面板")
        title.setStyleSheet("color: #4fc3f7; font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始模拟")
        self.start_btn.clicked.connect(self.start_simulation)
        self.pause_btn = QPushButton("暂停/继续")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_simulation)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.reset_btn)
        layout.addLayout(button_layout)

        # Speed control
        layout.addWidget(QLabel("播放速度:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 30)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("1.0x")
        self.speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.speed_label)

        # Mesh toggle
        self.mesh_toggle = QCheckBox("启用网格显示")
        self.mesh_toggle.setChecked(True)
        self.mesh_toggle.stateChanged.connect(self.toggle_mesh)
        layout.addWidget(self.mesh_toggle)

        # Rendering mode toggle
        self.render_mode_toggle = QCheckBox("使用SSFR渲染(直接粒子渲染)")
        self.render_mode_toggle.setChecked(True)
        self.render_mode_toggle.stateChanged.connect(self.toggle_render_mode)
        layout.addWidget(self.render_mode_toggle)

        # Boundary handling toggle
        self.boundary_toggle = QCheckBox("启用边界碰撞")
        self.boundary_toggle.setChecked(True)
        self.boundary_toggle.stateChanged.connect(self.toggle_boundary)
        layout.addWidget(self.boundary_toggle)

        # Display mode
        layout.addWidget(QLabel("显示模式:"))
        self.display_combo = QComboBox()
        self.display_combo.addItems(["实体", "线框", "点"])
        self.display_combo.setCurrentText("实体")
        layout.addWidget(self.display_combo)

        # Camera controls
        camera_group = QGroupBox("相机控制")
        camera_layout = QVBoxLayout()

        camera_layout.addWidget(QLabel("距离:"))
        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(20, 200)
        self.distance_slider.setValue(100)
        self.distance_slider.valueChanged.connect(self.update_camera)
        camera_layout.addWidget(self.distance_slider)

        camera_layout.addWidget(QLabel("水平角度:"))
        self.angle_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_x_slider.setRange(0, 314)
        self.angle_x_slider.setValue(50)
        self.angle_x_slider.valueChanged.connect(self.update_camera)
        camera_layout.addWidget(self.angle_x_slider)

        camera_layout.addWidget(QLabel("垂直角度:"))
        self.angle_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_y_slider.setRange(-157, 157)
        self.angle_y_slider.setValue(30)
        self.angle_y_slider.valueChanged.connect(self.update_camera)
        camera_layout.addWidget(self.angle_y_slider)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # SSFR 参数调节
        ssfr_group = QGroupBox("SSFR参数调节")
        ssfr_layout = QVBoxLayout()

        ssfr_layout.addWidget(QLabel("粒子世界半径:"))
        self.ssfr_radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_radius_slider.setRange(5, 100)
        radius_slider_init = int(CONFIG.PARTICLE_RADIUS * 3.5 * 100)
        self.ssfr_radius_slider.setValue(radius_slider_init)
        self.ssfr_radius_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_radius_slider)
        self.ssfr_radius_label = QLabel(f"{CONFIG.PARTICLE_RADIUS * 3.5:.2f}")
        ssfr_layout.addWidget(self.ssfr_radius_label)

        ssfr_layout.addWidget(QLabel("模糊强度:"))
        self.ssfr_sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_sigma_slider.setRange(5, 80)
        self.ssfr_sigma_slider.setValue(50)
        self.ssfr_sigma_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_sigma_slider)
        self.ssfr_sigma_label = QLabel("5.0")
        ssfr_layout.addWidget(self.ssfr_sigma_label)

        ssfr_layout.addWidget(QLabel("深度sigma(世界单位):"))
        self.ssfr_depth_sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_depth_sigma_slider.setRange(1, 100)
        self.ssfr_depth_sigma_slider.setValue(2)
        self.ssfr_depth_sigma_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_depth_sigma_slider)
        self.ssfr_depth_sigma_label = QLabel("0.2")
        ssfr_layout.addWidget(self.ssfr_depth_sigma_label)

        ssfr_layout.addWidget(QLabel("吸收系数:"))
        self.ssfr_absorption_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_absorption_slider.setRange(5, 50)
        self.ssfr_absorption_slider.setValue(40)
        self.ssfr_absorption_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_absorption_slider)
        self.ssfr_absorption_label = QLabel("4.0")
        ssfr_layout.addWidget(self.ssfr_absorption_label)

        ssfr_layout.addWidget(QLabel("散射系数:"))
        self.ssfr_scatter_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_scatter_slider.setRange(1, 20)
        self.ssfr_scatter_slider.setValue(2)
        self.ssfr_scatter_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_scatter_slider)
        self.ssfr_scatter_label = QLabel("0.20")
        ssfr_layout.addWidget(self.ssfr_scatter_label)

        ssfr_group.setLayout(ssfr_layout)
        layout.addWidget(ssfr_group)

        # Basic statistics
        stats_group = QGroupBox("基本统计")
        stats_layout = QVBoxLayout()

        self.frame_label = QLabel("帧: 0")
        self.particle_label = QLabel("粒子数: 0")
        self.fps_label = QLabel("FPS: 0")

        stats_layout.addWidget(self.frame_label)
        stats_layout.addWidget(self.particle_label)
        stats_layout.addWidget(self.fps_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()
        return panel

    def create_performance_panel(self):
        """Create performance analysis panel"""
        panel = QFrame()
        panel.setMinimumHeight(200)
        panel.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #444444;
            }
            QLabel {
                color: white;
                font-size: 11px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666666;
                padding: 6px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QTabWidget::pane {
                border: 1px solid #444444;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3a3a3a;
                color: white;
                padding: 8px 15px;
                margin: 2px;
                border: 1px solid #555555;
            }
            QTabBar::tab:selected {
                background-color: #4a4a4a;
                border-bottom: 2px solid #4fc3f7;
            }
            QGroupBox {
                color: #4fc3f7;
                font-weight: bold;
                border: 1px solid #555555;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Panel title
        title = QLabel("性能分析面板")
        title.setStyleSheet("color: #4fc3f7; font-size: 14px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Tab widget
        tab_widget = QTabWidget()
        
        # Basic Performance Tab
        basic_tab = self.create_basic_performance_tab()
        tab_widget.addTab(basic_tab, "基础性能")
        
        # Advanced Analysis Tab
        advanced_tab = self.create_advanced_analysis_tab()
        tab_widget.addTab(advanced_tab, "高级分析")
        
        # Export/Data Tab
        export_tab = self.create_export_tab()
        tab_widget.addTab(export_tab, "数据导出")
        
        layout.addWidget(tab_widget)
        return panel

    def create_basic_performance_tab(self):
        """Create basic performance monitoring tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Performance metrics grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(8)
        
        # Row 1
        metrics_grid.addWidget(QLabel("帧时间:"), 0, 0)
        self.avg_frame_time_label = QLabel("0.00ms")
        metrics_grid.addWidget(self.avg_frame_time_label, 0, 1)
        
        metrics_grid.addWidget(QLabel("当前FPS:"), 0, 2)
        self.current_fps_label = QLabel("0")
        metrics_grid.addWidget(self.current_fps_label, 0, 3)
        
        # Row 2
        metrics_grid.addWidget(QLabel("网格重建:"), 1, 0)
        self.grid_time_label = QLabel("0ms")
        metrics_grid.addWidget(self.grid_time_label, 1, 1)
        
        metrics_grid.addWidget(QLabel("力计算:"), 1, 2)
        self.force_time_label = QLabel("0ms")
        metrics_grid.addWidget(self.force_time_label, 1, 3)
        
        # Row 3
        metrics_grid.addWidget(QLabel("网格生成:"), 2, 0)
        self.mesh_time_label = QLabel("0ms")
        metrics_grid.addWidget(self.mesh_time_label, 2, 1)
        
        metrics_grid.addWidget(QLabel("GPU上传:"), 2, 2)
        self.gpu_time_label = QLabel("0ms")
        metrics_grid.addWidget(self.gpu_time_label, 2, 3)
        
        # Row 4
        metrics_grid.addWidget(QLabel("内存使用:"), 3, 0)
        self.memory_label = QLabel("0MB")
        metrics_grid.addWidget(self.memory_label, 3, 1)
        
        metrics_grid.addWidget(QLabel("顶点数:"), 3, 2)
        self.vertex_label = QLabel("0")
        metrics_grid.addWidget(self.vertex_label, 3, 3)
        
        # Row 5
        metrics_grid.addWidget(QLabel("三角形数:"), 4, 0)
        self.triangle_label = QLabel("0")
        metrics_grid.addWidget(self.triangle_label, 4, 1)
        
        layout.addLayout(metrics_grid)
        
        # Optimization suggestions section
        suggestions_group = QGroupBox("优化建议")
        suggestions_layout = QVBoxLayout()
        
        self.optimization_label = QLabel("暂无优化建议")
        self.optimization_label.setWordWrap(True)
        self.optimization_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        suggestions_layout.addWidget(self.optimization_label)
        
        suggestions_group.setLayout(suggestions_layout)
        layout.addWidget(suggestions_group)
        
        layout.addStretch()
        return widget

    def create_advanced_analysis_tab(self):
        """Create advanced performance analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Advanced metrics
        advanced_grid = QGridLayout()
        advanced_grid.setSpacing(8)
        
        # Stability analysis
        advanced_grid.addWidget(QLabel("稳定性:"), 0, 0)
        self.stability_label = QLabel("未知")
        self.stability_label.setStyleSheet("color: #a0a0a0;")
        advanced_grid.addWidget(self.stability_label, 0, 1)
        
        # Scaling analysis
        advanced_grid.addWidget(QLabel("扩展性:"), 0, 2)
        self.scaling_label = QLabel("未知")
        self.scaling_label.setStyleSheet("color: #a0a0a0;")
        advanced_grid.addWidget(self.scaling_label, 0, 3)
        
        # Trend analysis
        advanced_grid.addWidget(QLabel("性能趋势:"), 1, 0)
        self.trend_label = QLabel("稳定")
        self.trend_label.setStyleSheet("color: green;")
        advanced_grid.addWidget(self.trend_label, 1, 1)
        
        # Bottleneck detection
        advanced_grid.addWidget(QLabel("瓶颈检测:"), 1, 2)
        self.bottleneck_label = QLabel("未检测")
        self.bottleneck_label.setStyleSheet("color: #a0a0a0;")
        advanced_grid.addWidget(self.bottleneck_label, 1, 3)
        
        layout.addLayout(advanced_grid)
        
        # Advanced recommendations
        recommendations_group = QGroupBox("高级建议")
        recommendations_layout = QVBoxLayout()
        
        self.advanced_recommendation_label = QLabel("暂无高级建议")
        self.advanced_recommendation_label.setWordWrap(True)
        self.advanced_recommendation_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        recommendations_layout.addWidget(self.advanced_recommendation_label)
        
        recommendations_group.setLayout(recommendations_layout)
        layout.addWidget(recommendations_group)
        
        # Analysis controls
        controls_layout = QHBoxLayout()
        self.refresh_analysis_btn = QPushButton("刷新分析")
        self.refresh_analysis_btn.clicked.connect(self.refresh_advanced_analysis)
        controls_layout.addWidget(self.refresh_analysis_btn)
        
        self.clear_history_btn = QPushButton("清除历史")
        self.clear_history_btn.clicked.connect(self.clear_performance_history)
        controls_layout.addWidget(self.clear_history_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        layout.addStretch()
        return widget

    def create_export_tab(self):
        """Create data export and visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Export controls
        export_group = QGroupBox("性能数据导出")
        export_layout = QVBoxLayout()
        
        # Export buttons
        buttons_layout = QHBoxLayout()
        
        self.export_perf_btn = QPushButton("导出JSON数据")
        self.export_perf_btn.clicked.connect(self.export_performance_data)
        buttons_layout.addWidget(self.export_perf_btn)
        
        self.export_csv_btn = QPushButton("导出CSV数据")
        self.export_csv_btn.clicked.connect(self.export_performance_csv)
        buttons_layout.addWidget(self.export_csv_btn)
        
        buttons_layout.addStretch()
        export_layout.addLayout(buttons_layout)
        
        # Export info
        self.export_info_label = QLabel("点击按钮导出性能数据用于进一步分析")
        self.export_info_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        export_layout.addWidget(self.export_info_label)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Data visualization placeholder
        viz_group = QGroupBox("数据可视化")
        viz_layout = QVBoxLayout()
        
        self.viz_info_label = QLabel("性能图表和趋势分析功能待实现")
        self.viz_info_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        viz_layout.addWidget(self.viz_info_label)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        layout.addStretch()
        return widget

    def setup_timer(self):
        """Setup simulation timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.setInterval(16)  # ~60 FPS

    def start_simulation(self):
        """Start simulation"""
        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.timer.start()

            particle_count = self.mesh_system.initialize_fluid_drops()

            self.start_btn.setEnabled(False)
            self.start_btn.setText("运行中...")
            self.status_bar.showMessage(f"模拟运行中... 粒子数: {particle_count}")

    def toggle_pause(self):
        """Pause/resume simulation"""
        if self.is_running:
            self.is_paused = not self.is_paused
            text = "继续" if self.is_paused else "暂停"
            self.pause_btn.setText(text)

    def reset_simulation(self):
        """Reset simulation"""
        self.is_running = False
        self.is_paused = False
        self.timer.stop()

        # Reset system
        self.mesh_system.current_frame = 0
        self.mesh_system.current_particle_num = 0
        self.mesh_system.last_grid_rebuild = -CONFIG.GRID_REBUILD_INTERVAL
        self.mesh_system.last_mesh_update = -CONFIG.MC_MESH_UPDATE_INTERVAL

        # Update UI
        self.start_btn.setEnabled(True)
        self.start_btn.setText("开始模拟")
        self.pause_btn.setText("暂停/继续")

        self.status_bar.showMessage("系统已重置")

        # Clear mesh data
        self.opengl_renderer.update_mesh_data([], [])
        self.opengl_widget.update()

    def toggle_mesh(self, state):
        """Toggle mesh display"""
        self.mesh_system.mesh_enabled = bool(state)

    def toggle_render_mode(self, state):
        """Toggle between mesh rendering and SSFR rendering"""
        self.opengl_renderer.use_ssfr = bool(state)
        CONFIG.USE_SSFR = bool(state)
        mode = "SSFR直接粒子渲染" if state else "网格生成渲染"
        if state:
            print(f"SSFR参数: radius={self.opengl_renderer.ssfr_particle_radius_world:.3f}, \
                  sigma={self.opengl_renderer.ssfr_blur_sigma:.1f}, \
                  depth_sigma={self.opengl_renderer.ssfr_depth_sigma:.3f}")
            print(f"渲染参数: sigma_a={CONFIG.SIGMA_A}, sigma_s={CONFIG.SIGMA_S}, water_color={CONFIG.WATER_COLOR}")
        print(f"渲染模式切换到: {mode}")
        self.status_bar.showMessage(f"渲染模式: {mode}")

    def toggle_boundary(self, state):
        """Toggle boundary collision handling"""
        if hasattr(self.mesh_system, 'manual_boundary_enabled'):
            self.mesh_system.manual_boundary_enabled = bool(state)
        print(f"Boundary collision handling: {'enabled' if state else 'disabled'}")

    def update_speed_label(self, value):
        """Update speed label"""
        speed = value / 10.0
        self.speed_label.setText(f"{speed:.1f}x")

    def update_camera(self):
        """Update camera parameters"""
        self.opengl_renderer.camera_distance = self.distance_slider.value() / 10.0
        self.opengl_renderer.camera_angle_x = self.angle_x_slider.value() / 100.0
        self.opengl_renderer.camera_angle_y = self.angle_y_slider.value() / 100.0
        self.opengl_widget.update()

    def update_ssfr_params(self):
        """Update SSFR rendering parameters"""
        radius = self.ssfr_radius_slider.value() / 100.0
        self.opengl_renderer.ssfr_particle_radius_world = radius
        self.ssfr_radius_label.setText(f"{radius:.2f}")

        sigma = self.ssfr_sigma_slider.value() / 10.0
        self.opengl_renderer.ssfr_blur_sigma = sigma
        self.ssfr_sigma_label.setText(f"{sigma:.1f}")

        depth_sigma = self.ssfr_depth_sigma_slider.value() / 10.0
        self.opengl_renderer.ssfr_depth_sigma = depth_sigma
        self.ssfr_depth_sigma_label.setText(f"{depth_sigma:.2f}")

        absorption = self.ssfr_absorption_slider.value() / 10.0
        if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_sigma_a' in self.opengl_renderer.composite_prog.uniforms:
            self.opengl_renderer.composite_prog['u_sigma_a'].value = absorption
        self.ssfr_absorption_label.setText(f"{absorption:.1f}")

        scatter = self.ssfr_scatter_slider.value() / 10.0
        if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_sigma_s' in self.opengl_renderer.composite_prog.uniforms:
            self.opengl_renderer.composite_prog['u_sigma_s'].value = scatter
        self.ssfr_scatter_label.setText(f"{scatter:.2f}")

        self.opengl_widget.update()

    def update_simulation(self):
        """Update simulation"""
        if not self.is_running or self.is_paused:
            return

        speed = self.speed_slider.value() / 10.0
        dt = CONFIG.DEFAULT_TIME_STEP / speed

        # Execute physics simulation
        t = time.perf_counter()
        _new_vertices, _new_triangles = self.mesh_system.step_simulation(dt)
        print(f"Simulation step time: {time.perf_counter() - t:.3f} s")
        
        # Update particles for SSFR
        t = time.perf_counter()
        self.opengl_widget.makeCurrent()
        try:
            n = self.mesh_system.current_particle_num
            positions = self.mesh_system.particle_positions[:n]
            self.opengl_renderer.update_particles(positions)
        finally:
            self.opengl_widget.doneCurrent()

        self.update_statistics()
        self.opengl_widget.update()
        print(f"Mesh update time: {time.perf_counter() - t:.3f} s")

    def export_performance_data(self):
        """导出性能数据"""
        if self.mesh_system.performance_capture:
            filename = self.mesh_system.performance_capture.export_performance_data()
            if filename:
                self.status_bar.showMessage(f"性能数据已导出到: {filename}")
            else:
                self.status_bar.showMessage("性能数据导出失败")

    def update_statistics(self):
        """Update statistics display"""
        stats = self.mesh_system.get_system_stats()
        formatted_stats = get_system_stats_formatted(stats)

        # Update basic labels
        self.frame_label.setText(f"帧: {formatted_stats['frame']}")
        self.particle_label.setText(f"粒子数: {formatted_stats['particle_count']}")
        
        # Update FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.fps_counter = 0
            self.last_fps_time = current_time

        # Update performance metrics
        perf = stats['performance']
        
        if hasattr(self, 'avg_frame_time_label'):
            self.avg_frame_time_label.setText(f"{formatted_stats['avg_frame_time_ms']:.2f}ms")
        if hasattr(self, 'current_fps_label'):
            self.current_fps_label.setText(f"{formatted_stats['fps']}")
        if hasattr(self, 'grid_time_label'):
            self.grid_time_label.setText(f"{perf['grid_rebuild_time']*1000:.1f}ms")
        if hasattr(self, 'force_time_label'):
            self.force_time_label.setText(f"{perf['force_computation_time']*1000:.1f}ms")
        if hasattr(self, 'mesh_time_label'):
            self.mesh_time_label.setText(f"{perf['mesh_generation_time']*1000:.1f}ms")
        if hasattr(self, 'gpu_time_label'):
            self.gpu_time_label.setText(f"{perf.get('gpu_upload_time', 0)*1000:.1f}ms")
        if hasattr(self, 'memory_label'):
            self.memory_label.setText(f"{formatted_stats['memory_usage_mb']}MB")
        if hasattr(self, 'vertex_label'):
            self.vertex_label.setText(f"{formatted_stats['mesh_vertices']}")
        if hasattr(self, 'triangle_label'):
            self.triangle_label.setText(f"{formatted_stats['mesh_triangles']}")
        
        # Update optimization suggestions
        if self.mesh_system.performance_capture:
            real_time_stats = self.mesh_system.performance_capture.real_time_stats
            
            if hasattr(self, 'optimization_label'):
                suggestions = self.mesh_system.performance_capture.optimization_suggestions
                if suggestions:
                    critical_suggestions = [s for s in suggestions if s['severity'] == 'high']
                    if critical_suggestions:
                        suggestion_text = f"⚠ {critical_suggestions[0]['message']}"
                        self.optimization_label.setText(suggestion_text)
                        self.optimization_label.setStyleSheet("color: red; font-weight: bold;")
                    else:
                        medium_suggestions = [s for s in suggestions if s['severity'] == 'medium']
                        if medium_suggestions:
                            suggestion_text = f"⚠ {medium_suggestions[0]['message']}"
                            self.optimization_label.setText(suggestion_text)
                            self.optimization_label.setStyleSheet("color: orange;")
                        else:
                            self.optimization_label.setText("✓ 性能良好")
                            self.optimization_label.setStyleSheet("color: green;")
                else:
                    self.optimization_label.setText("✓ 性能良好")
                    self.optimization_label.setStyleSheet("color: green;")
            
            self._update_advanced_performance_info()

    def _update_advanced_performance_info(self):
        """更新高级性能分析信息显示"""
        if not self.mesh_system.performance_capture:
            return
            
        report = self.mesh_system.performance_capture.get_performance_report()
        advanced_analysis = report.get('advanced_analysis', {})
        
        # Update stability info
        if hasattr(self, 'stability_label'):
            patterns = advanced_analysis.get('patterns', {})
            if patterns and 'stability' in patterns:
                stability_info = patterns['stability']
                unstable_metrics = [k for k, v in stability_info.items() 
                                  if v['stability_level'] in ['unstable', 'moderately_unstable']]
                if unstable_metrics:
                    self.stability_label.setText(f"⚠ 不稳定 ({len(unstable_metrics)}个组件)")
                    self.stability_label.setStyleSheet("color: orange;")
                else:
                    self.stability_label.setText("✓ 良好")
                    self.stability_label.setStyleSheet("color: green;")
            else:
                self.stability_label.setText("未知")
                self.stability_label.setStyleSheet("color: gray;")
            
        # Update scaling info
        if hasattr(self, 'scaling_label'):
            patterns = advanced_analysis.get('patterns', {})
            if patterns and 'scaling' in patterns and 'particle_scaling' in patterns['scaling']:
                scaling = patterns['scaling']['particle_scaling']
                if scaling['scaling_type'] == 'linear':
                    cost_per_particle = scaling['slope'] * 1000
                    self.scaling_label.setText(f"线性 ({cost_per_particle:.2f}ms/粒子)")
                    if cost_per_particle > 0.1:
                        self.scaling_label.setStyleSheet("color: orange;")
                    else:
                        self.scaling_label.setStyleSheet("color: green;")
                else:
                    self.scaling_label.setText(f"非线性")
                    self.scaling_label.setStyleSheet("color: yellow;")
            else:
                self.scaling_label.setText("未知")
                self.scaling_label.setStyleSheet("color: gray;")
            
        # Update trend info
        if hasattr(self, 'trend_label'):
            patterns = advanced_analysis.get('patterns', {})
            if patterns and 'trends' in patterns:
                degrading_metrics = [k for k, v in patterns['trends'].items() 
                                   if v['trend'] == 'increasing']
                improving_metrics = [k for k, v in patterns['trends'].items() 
                                   if v['trend'] == 'decreasing']
                                    
                if degrading_metrics:
                    self.trend_label.setText(f"⚠ 下降 ({len(degrading_metrics)}个指标)")
                    self.trend_label.setStyleSheet("color: red;")
                elif improving_metrics:
                    self.trend_label.setText(f"✓ 改善 ({len(improving_metrics)}个指标)")
                    self.trend_label.setStyleSheet("color: green;")
                else:
                    self.trend_label.setText("稳定")
                    self.trend_label.setStyleSheet("color: green;")
            else:
                self.trend_label.setText("未知")
                self.trend_label.setStyleSheet("color: gray;")
            
        # Update advanced recommendations
        if hasattr(self, 'advanced_recommendation_label'):
            recommendations = advanced_analysis.get('recommendations', [])
            if recommendations:
                high_priority = [r for r in recommendations if r['priority'] == 'high']
                if high_priority:
                    rec_text = f"⚠ {high_priority[0]['message'][:30]}..."
                    self.advanced_recommendation_label.setText(rec_text)
                    self.advanced_recommendation_label.setStyleSheet("color: red; font-weight: bold;")
                else:
                    medium_priority = [r for r in recommendations if r['priority'] == 'medium']
                    if medium_priority:
                        rec_text = f"⚠ {medium_priority[0]['message'][:30]}..."
                        self.advanced_recommendation_label.setText(rec_text)
                        self.advanced_recommendation_label.setStyleSheet("color: orange;")
                    else:
                        self.advanced_recommendation_label.setText("✓ 无紧急建议")
                        self.advanced_recommendation_label.setStyleSheet("color: green;")
            else:
                self.advanced_recommendation_label.setText("✓ 无高级建议")
                self.advanced_recommendation_label.setStyleSheet("color: green;")

    def refresh_advanced_analysis(self):
        """手动刷新高级性能分析"""
        if self.mesh_system.performance_capture:
            self.mesh_system.performance_capture.last_analysis_time = 0
            self._update_advanced_performance_info()
            self.status_bar.showMessage("高级性能分析已刷新")

    def clear_performance_history(self):
        """清除性能历史数据"""
        if self.mesh_system.performance_capture:
            self.mesh_system.performance_capture.clear_history()
            self.status_bar.showMessage("性能历史数据已清除")
            self._reset_performance_display()

    def _reset_performance_display(self):
        """重置性能显示为初始状态"""
        # Reset basic performance labels
        labels_to_reset = [
            ('avg_frame_time_label', "0.00ms"),
            ('current_fps_label', "0"),
            ('grid_time_label', "0ms"),
            ('force_time_label', "0ms"),
            ('mesh_time_label', "0ms"),
            ('gpu_time_label', "0ms"),
            ('memory_label', "0MB"),
            ('vertex_label', "0"),
            ('triangle_label', "0")
        ]
        
        for label_name, default_value in labels_to_reset:
            if hasattr(self, label_name):
                getattr(self, label_name).setText(default_value)
        
        # Reset optimization labels
        if hasattr(self, 'optimization_label'):
            self.optimization_label.setText("暂无优化建议")
            self.optimization_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        
        # Reset advanced analysis labels
        advanced_labels = [
            ('stability_label', "未知", "color: gray;"),
            ('scaling_label', "未知", "color: gray;"),
            ('trend_label', "稳定", "color: green;"),
            ('advanced_recommendation_label', "暂无高级建议", "color: #a0a0a0; font-style: italic;")
        ]
        
        for label_name, default_text, style in advanced_labels:
            if hasattr(self, label_name):
                getattr(self, label_name).setText(default_text)
                getattr(self, label_name).setStyleSheet(style)

    def export_performance_csv(self):
        """导出性能数据为CSV格式"""
        if self.mesh_system.performance_capture:
            try:
                import csv
                from datetime import datetime
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"performance_data_{timestamp}.csv"
                
                report = self.mesh_system.performance_capture.get_performance_report()
                raw_data = report.get('raw_data', {})
                
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    writer.writerow(['Timestamp', 'Frame Time (ms)', 'FPS'])
                    
                    timestamps = raw_data.get('timestamps', [])
                    frame_times = raw_data.get('frame_times_ms', [])
                    fps_values = raw_data.get('fps_values', [])
                    
                    max_len = max(len(timestamps), len(frame_times), len(fps_values))
                    for i in range(max_len):
                        row = [
                            timestamps[i] if i < len(timestamps) else '',
                            frame_times[i] if i < len(frame_times) else '',
                            fps_values[i] if i < len(fps_values) else ''
                        ]
                        writer.writerow(row)
                
                self.status_bar.showMessage(f"CSV数据已导出到: {filename}")
                print(f"Performance CSV exported to: {filename}")
                
            except Exception as e:
                self.status_bar.showMessage(f"CSV导出失败: {str(e)}")
                print(f"CSV export failed: {e}")