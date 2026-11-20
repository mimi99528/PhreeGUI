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

from ..core.mesh_factory import create_mesh_system
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
        self.mesh_system = create_mesh_system()
        self.opengl_renderer = OpenGLRenderer(self.mesh_system)
        self.is_running = False
        self.is_paused = False
        self.fps_counter = 0
        self.last_fps_time = time.time()

        self.init_ui()
        self.setup_timer()

    def init_ui(self):
        """Initialize UI with proper layout"""
        self.setWindowTitle("PySide6 + ModernGL 网格重建系统")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main vertical layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Top area - OpenGL widget (占主要空间)
        self.opengl_widget = QOpenGLWidgetWrapper(self.opengl_renderer)
        main_layout.addWidget(self.opengl_widget, stretch=3)  # 给OpenGL更多空间

        # Bottom area - Control panel with tabs (固定高度)
        bottom_panel = self.create_bottom_control_panel()
        main_layout.addWidget(bottom_panel, stretch=3)  # 控制面板较少空间

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("系统就绪")

    def update_material_params(self):
        """Update material parameters for SSFR rendering"""
        # Update alpha base
        if hasattr(self, 'alpha_base_slider'):
            alpha_base = self.alpha_base_slider.value() / 100.0
            CONFIG.ALPHA_BASE = alpha_base
            self.alpha_base_label.setText(f"{alpha_base:.2f}")
            if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_alpha_base' in self.opengl_renderer.composite_prog.uniforms:
                self.opengl_renderer.composite_prog['u_alpha_base'].value = alpha_base

        # Update fresnel power
        if hasattr(self, 'fresnel_power_slider'):
            fresnel_power = self.fresnel_power_slider.value() / 10.0
            CONFIG.FRESNEL_POWER = fresnel_power
            self.fresnel_power_label.setText(f"{fresnel_power:.1f}")
            if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_fresnel_power' in self.opengl_renderer.composite_prog.uniforms:
                self.opengl_renderer.composite_prog['u_fresnel_power'].value = fresnel_power

        # Update specular power
        if hasattr(self, 'spec_power_slider'):
            spec_power = float(self.spec_power_slider.value())
            CONFIG.SPEC_POWER = spec_power
            self.spec_power_label.setText(f"{spec_power:.0f}")
            if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_spec_power' in self.opengl_renderer.composite_prog.uniforms:
                self.opengl_renderer.composite_prog['u_spec_power'].value = spec_power

        # Update absorption coefficient
        if hasattr(self, 'sigma_a_slider'):
            sigma_a = self.sigma_a_slider.value() / 10.0
            CONFIG.SIGMA_A = sigma_a
            self.sigma_a_label.setText(f"{sigma_a:.1f}")
            if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_sigma_a' in self.opengl_renderer.composite_prog.uniforms:
                self.opengl_renderer.composite_prog['u_sigma_a'].value = sigma_a

        # Update scattering coefficient
        if hasattr(self, 'sigma_s_slider'):
            sigma_s = self.sigma_s_slider.value() / 100.0
            CONFIG.SIGMA_S = sigma_s
            self.sigma_s_label.setText(f"{sigma_s:.2f}")
            if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_sigma_s' in self.opengl_renderer.composite_prog.uniforms:
                self.opengl_renderer.composite_prog['u_sigma_s'].value = sigma_s

        # Update shadow strength
        if hasattr(self, 'shadow_strength_slider'):
            shadow_strength = self.shadow_strength_slider.value() / 100.0
            CONFIG.SHADOW_STRENGTH = shadow_strength
            self.shadow_strength_label.setText(f"{shadow_strength:.2f}")
            if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_shadow_strength' in self.opengl_renderer.composite_prog.uniforms:
                self.opengl_renderer.composite_prog['u_shadow_strength'].value = shadow_strength

        # Update caustic strength
        if hasattr(self, 'caustic_strength_slider'):
            caustic_strength = self.caustic_strength_slider.value() / 100.0
            CONFIG.CAUSTIC_STRENGTH = caustic_strength
            self.caustic_strength_label.setText(f"{caustic_strength:.2f}")
            if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_caustic_strength' in self.opengl_renderer.composite_prog.uniforms:
                self.opengl_renderer.composite_prog['u_caustic_strength'].value = caustic_strength

        # Update caustic power
        if hasattr(self, 'caustic_power_slider'):
            caustic_power = self.caustic_power_slider.value() / 10.0
            CONFIG.CAUSTIC_POWER = caustic_power
            self.caustic_power_label.setText(f"{caustic_power:.1f}")
            if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_caustic_power' in self.opengl_renderer.composite_prog.uniforms:
                self.opengl_renderer.composite_prog['u_caustic_power'].value = caustic_power

        # Trigger OpenGL widget update
        self.opengl_widget.update()


    def create_bottom_control_panel(self):
        """Create bottom control panel with tabbed interface"""
        panel = QFrame()
        panel.setMinimumHeight(300)  # 确保有足够的高度
        panel.setMaximumHeight(600)  # 限制最大高度
        panel.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
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
                min-width: 80px;
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
                min-width: 100px;
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
                border-bottom: none;
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
        title = QLabel("控制面板")
        title.setStyleSheet("color: #4fc3f7; font-size: 14px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Tab widget for organized controls
        tab_widget = QTabWidget()
        
        # Simulation Control Tab
        sim_tab = self.create_simulation_tab()
        tab_widget.addTab(sim_tab, "模拟控制")
        
        # Rendering Parameters Tab
        render_tab = self.create_rendering_tab()
        tab_widget.addTab(render_tab, "渲染参数")
        
        # Advanced Settings Tab
        advanced_tab = self.create_advanced_tab()
        tab_widget.addTab(advanced_tab, "高级设置")
        
        # Statistics Tab
        stats_tab = self.create_statistics_tab()
        tab_widget.addTab(stats_tab, "统计信息")
        
        layout.addWidget(tab_widget)
        return panel

    def create_simulation_tab(self):
        """Create simulation control tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

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

        # Display options
        self.mesh_toggle = QCheckBox("启用网格显示")
        self.mesh_toggle.setChecked(True)
        self.mesh_toggle.stateChanged.connect(self.toggle_mesh)
        layout.addWidget(self.mesh_toggle)

        self.render_mode_toggle = QCheckBox("使用SSFR渲染")
        self.render_mode_toggle.setChecked(True)
        self.render_mode_toggle.stateChanged.connect(self.toggle_render_mode)
        layout.addWidget(self.render_mode_toggle)

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

        layout.addStretch()
        return widget

    def create_rendering_tab(self):
        """Create rendering parameters tab with separate tabs for camera and SSFR controls"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Create tab widget for organizing controls
        tab_widget = QTabWidget()
        
        # Camera Controls Tab
        camera_tab = self.create_camera_controls_tab()
        tab_widget.addTab(camera_tab, "相机控制")
        
        # SSFR Parameters Tab
        ssfr_tab = self.create_ssfr_parameters_tab()
        tab_widget.addTab(ssfr_tab, "SSFR参数")
        
        layout.addWidget(tab_widget)
        return widget

    def create_camera_controls_tab(self):
        """Create camera controls tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Camera controls group
        camera_group = QGroupBox("相机参数")
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
        layout.addStretch()
        return widget
    def create_ssfr_parameters_tab(self):
        """Create SSFR parameters tab with balanced two-column layout"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # SSFR parameters group
        ssfr_group = QGroupBox("SSFR渲染参数")
        ssfr_layout = QGridLayout()
        ssfr_layout.setSpacing(15)

        # Column 1 - Optical Properties
        col1_layout = QVBoxLayout()
        
        # Alpha base
        col1_layout.addWidget(QLabel("基础透明度:"))
        self.alpha_base_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_base_slider.setRange(1, 50)
        self.alpha_base_slider.setValue(int(CONFIG.ALPHA_BASE * 100))
        self.alpha_base_slider.valueChanged.connect(self.update_material_params)
        col1_layout.addWidget(self.alpha_base_slider)
        self.alpha_base_label = QLabel(f"{CONFIG.ALPHA_BASE:.2f}")
        col1_layout.addWidget(self.alpha_base_label)
        
        # Fresnel power
        col1_layout.addWidget(QLabel("菲涅尔幂次:"))
        self.fresnel_power_slider = QSlider(Qt.Orientation.Horizontal)
        self.fresnel_power_slider.setRange(1, 200)
        self.fresnel_power_slider.setValue(int(CONFIG.FRESNEL_POWER * 10))
        self.fresnel_power_slider.valueChanged.connect(self.update_material_params)
        col1_layout.addWidget(self.fresnel_power_slider)
        self.fresnel_power_label = QLabel(f"{CONFIG.FRESNEL_POWER:.1f}")
        col1_layout.addWidget(self.fresnel_power_label)
        
        # Specular power
        col1_layout.addWidget(QLabel("镜面反射强度:"))
        self.spec_power_slider = QSlider(Qt.Orientation.Horizontal)
        self.spec_power_slider.setRange(10, 500)
        self.spec_power_slider.setValue(int(CONFIG.SPEC_POWER))
        self.spec_power_slider.valueChanged.connect(self.update_material_params)
        col1_layout.addWidget(self.spec_power_slider)
        self.spec_power_label = QLabel(f"{CONFIG.SPEC_POWER:.0f}")
        col1_layout.addWidget(self.spec_power_label)

        # Column 2 - Medium Properties
        col2_layout = QVBoxLayout()
        
        # Absorption coefficient
        col2_layout.addWidget(QLabel("吸收系数:"))
        self.sigma_a_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_a_slider.setRange(5, 50)
        self.sigma_a_slider.setValue(int(CONFIG.SIGMA_A * 10))
        self.sigma_a_slider.valueChanged.connect(self.update_material_params)
        col2_layout.addWidget(self.sigma_a_slider)
        self.sigma_a_label = QLabel(f"{CONFIG.SIGMA_A:.1f}")
        col2_layout.addWidget(self.sigma_a_label)
        
        # Scattering coefficient
        col2_layout.addWidget(QLabel("散射系数:"))
        self.sigma_s_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_s_slider.setRange(1, 50)
        self.sigma_s_slider.setValue(int(CONFIG.SIGMA_S * 100))
        self.sigma_s_slider.valueChanged.connect(self.update_material_params)
        col2_layout.addWidget(self.sigma_s_slider)
        self.sigma_s_label = QLabel(f"{CONFIG.SIGMA_S:.2f}")
        col2_layout.addWidget(self.sigma_s_label)
        
        # Shadow strength
        col2_layout.addWidget(QLabel("阴影强度:"))
        self.shadow_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.shadow_strength_slider.setRange(0, 100)
        self.shadow_strength_slider.setValue(int(CONFIG.SHADOW_STRENGTH * 100))
        self.shadow_strength_slider.valueChanged.connect(self.update_material_params)
        col2_layout.addWidget(self.shadow_strength_slider)
        self.shadow_strength_label = QLabel(f"{CONFIG.SHADOW_STRENGTH:.2f}")
        col2_layout.addWidget(self.shadow_strength_label)

        # Column 3 - Light Effects
        col3_layout = QVBoxLayout()
        
        # Caustic strength
        col3_layout.addWidget(QLabel("焦散强度:"))
        self.caustic_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.caustic_strength_slider.setRange(0, 100)
        self.caustic_strength_slider.setValue(int(CONFIG.CAUSTIC_STRENGTH * 100))
        self.caustic_strength_slider.valueChanged.connect(self.update_material_params)
        col3_layout.addWidget(self.caustic_strength_slider)
        self.caustic_strength_label = QLabel(f"{CONFIG.CAUSTIC_STRENGTH:.2f}")
        col3_layout.addWidget(self.caustic_strength_label)
        
        # Caustic power
        col3_layout.addWidget(QLabel("焦散幂次:"))
        self.caustic_power_slider = QSlider(Qt.Orientation.Horizontal)
        self.caustic_power_slider.setRange(1, 200)
        self.caustic_power_slider.setValue(int(CONFIG.CAUSTIC_POWER * 10))
        self.caustic_power_slider.valueChanged.connect(self.update_material_params)
        col3_layout.addWidget(self.caustic_power_slider)
        self.caustic_power_label = QLabel(f"{CONFIG.CAUSTIC_POWER:.1f}")
        col3_layout.addWidget(self.caustic_power_label)

        # Add columns to grid layout
        ssfr_layout.addLayout(col1_layout, 0, 0)
        ssfr_layout.addLayout(col2_layout, 0, 1)
        ssfr_layout.addLayout(col3_layout, 0, 2)

        ssfr_group.setLayout(ssfr_layout)
        layout.addWidget(ssfr_group)
        layout.addStretch()
        return widget



    def create_advanced_tab(self):
        """Create advanced settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Material properties
        material_group = QGroupBox("材质属性")
        material_layout = QVBoxLayout()

        material_layout.addWidget(QLabel("吸收系数:"))
        self.ssfr_absorption_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_absorption_slider.setRange(5, 50)
        self.ssfr_absorption_slider.setValue(40)
        self.ssfr_absorption_slider.valueChanged.connect(self.update_ssfr_params)
        material_layout.addWidget(self.ssfr_absorption_slider)
        self.ssfr_absorption_label = QLabel("4.0")
        material_layout.addWidget(self.ssfr_absorption_label)

        material_layout.addWidget(QLabel("散射系数:"))
        self.ssfr_scatter_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_scatter_slider.setRange(1, 20)
        self.ssfr_scatter_slider.setValue(2)
        self.ssfr_scatter_slider.valueChanged.connect(self.update_ssfr_params)
        material_layout.addWidget(self.ssfr_scatter_slider)
        self.ssfr_scatter_label = QLabel("0.20")
        material_layout.addWidget(self.ssfr_scatter_label)

        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        # ★ 新增：深度缓冲暴露控制
        depth_exposure_group = QGroupBox("深度缓冲暴露 (用于反投影)")
        depth_layout = QVBoxLayout()
        
        # 启用深度暴露复选框
        self.depth_exposure_toggle = QCheckBox("启用深度缓冲暴露")
        self.depth_exposure_toggle.stateChanged.connect(self.toggle_depth_exposure)
        depth_layout.addWidget(self.depth_exposure_toggle)
        
        # 暴露格式选择
        depth_layout.addWidget(QLabel("暴露格式:"))
        self.depth_format_combo = QComboBox()
        self.depth_format_combo.addItems(["numpy数组", "纹理对象", "两者都暴露"])
        self.depth_format_combo.setCurrentText("numpy数组")
        self.depth_format_combo.currentTextChanged.connect(self.update_depth_exposure_format)
        depth_layout.addWidget(self.depth_format_combo)
        
        # 暴露频率控制
        depth_layout.addWidget(QLabel("暴露频率:"))
        self.depth_frequency_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_frequency_slider.setRange(1, 60)  # 1-60 FPS
        self.depth_frequency_slider.setValue(30)  # 默认30 FPS
        self.depth_frequency_slider.valueChanged.connect(self.update_depth_frequency)
        depth_layout.addWidget(self.depth_frequency_slider)
        self.depth_frequency_label = QLabel("30 FPS")
        depth_layout.addWidget(self.depth_frequency_label)
        
        # 深度数据信息显示
        self.depth_info_label = QLabel("深度数据: 未启用")
        self.depth_info_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        depth_layout.addWidget(self.depth_info_label)
        
        # 测试按钮
        test_depth_btn = QPushButton("测试深度暴露")
        test_depth_btn.clicked.connect(self.test_depth_exposure)
        depth_layout.addWidget(test_depth_btn)
        
        depth_exposure_group.setLayout(depth_layout)
        layout.addWidget(depth_exposure_group)

        # Water color controls
        color_group = QGroupBox("水体颜色")
        color_layout = QGridLayout()

        color_layout.addWidget(QLabel("红色:"), 0, 0)
        self.water_r_slider = QSlider(Qt.Orientation.Horizontal)
        self.water_r_slider.setRange(0, 100)
        self.water_r_slider.setValue(int(CONFIG.WATER_COLOR[0] * 100))
        self.water_r_slider.valueChanged.connect(self.update_water_color)
        color_layout.addWidget(self.water_r_slider, 0, 1)
        self.water_r_label = QLabel(f"{CONFIG.WATER_COLOR[0]:.2f}")
        color_layout.addWidget(self.water_r_label, 0, 2)

        color_layout.addWidget(QLabel("绿色:"), 1, 0)
        self.water_g_slider = QSlider(Qt.Orientation.Horizontal)
        self.water_g_slider.setRange(0, 100)
        self.water_g_slider.setValue(int(CONFIG.WATER_COLOR[1] * 100))
        self.water_g_slider.valueChanged.connect(self.update_water_color)
        color_layout.addWidget(self.water_g_slider, 1, 1)
        self.water_g_label = QLabel(f"{CONFIG.WATER_COLOR[1]:.2f}")
        color_layout.addWidget(self.water_g_label, 1, 2)

        color_layout.addWidget(QLabel("蓝色:"), 2, 0)
        self.water_b_slider = QSlider(Qt.Orientation.Horizontal)
        self.water_b_slider.setRange(0, 100)
        self.water_b_slider.setValue(int(CONFIG.WATER_COLOR[2] * 100))
        self.water_b_slider.valueChanged.connect(self.update_water_color)
        color_layout.addWidget(self.water_b_slider, 2, 1)
        self.water_b_label = QLabel(f"{CONFIG.WATER_COLOR[2]:.2f}")
        color_layout.addWidget(self.water_b_label, 2, 2)

        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

        layout.addStretch()
        return widget

    def create_statistics_tab(self):
        """Create statistics display tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Statistics grid
        stats_grid = QGridLayout()
        stats_grid.setSpacing(8)
        
        # Row 1
        stats_grid.addWidget(QLabel("帧:"), 0, 0)
        self.frame_label = QLabel("0")
        stats_grid.addWidget(self.frame_label, 0, 1)
        
        stats_grid.addWidget(QLabel("粒子数:"), 0, 2)
        self.particle_label = QLabel("0")
        stats_grid.addWidget(self.particle_label, 0, 3)
        
        # Row 2
        stats_grid.addWidget(QLabel("FPS:"), 1, 0)
        self.fps_label = QLabel("0")
        stats_grid.addWidget(self.fps_label, 1, 1)
        
        stats_grid.addWidget(QLabel("帧时间:"), 1, 2)
        self.avg_frame_time_label = QLabel("0.00ms")
        stats_grid.addWidget(self.avg_frame_time_label, 1, 3)
        
        # Row 3
        stats_grid.addWidget(QLabel("力计算:"), 2, 2)
        self.force_time_label = QLabel("0ms")
        stats_grid.addWidget(self.force_time_label, 2, 3)
        
        # Row 4
        stats_grid.addWidget(QLabel("网格生成:"), 3, 0)
        self.mesh_time_label = QLabel("0ms")
        stats_grid.addWidget(self.mesh_time_label, 3, 1)
        
        stats_grid.addWidget(QLabel("GPU上传:"), 3, 2)
        self.gpu_time_label = QLabel("0ms")
        stats_grid.addWidget(self.gpu_time_label, 3, 3)
        
        layout.addLayout(stats_grid)
        
        # Additional info
        info_group = QGroupBox("系统信息")
        info_layout = QVBoxLayout()
        
        self.memory_label = QLabel("内存使用: 0MB")
        self.vertex_label = QLabel("顶点数: 0")
        self.triangle_label = QLabel("三角形数: 0")
        
        info_layout.addWidget(self.memory_label)
        info_layout.addWidget(self.vertex_label)
        info_layout.addWidget(self.triangle_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        return widget

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
        # Update basic SSFR parameters
        if hasattr(self, 'ssfr_radius_slider'):
            radius = self.ssfr_radius_slider.value() / 100.0
            self.opengl_renderer.ssfr_particle_radius_world = radius
            self.ssfr_radius_label.setText(f"{radius:.2f}")

        if hasattr(self, 'ssfr_sigma_slider'):
            sigma = self.ssfr_sigma_slider.value() / 10.0
            self.opengl_renderer.ssfr_blur_sigma = sigma
            self.ssfr_sigma_label.setText(f"{sigma:.1f}")

        if hasattr(self, 'ssfr_depth_sigma_slider'):
            depth_sigma = self.ssfr_depth_sigma_slider.value() / 10.0
            self.opengl_renderer.ssfr_depth_sigma = depth_sigma
            self.ssfr_depth_sigma_label.setText(f"{depth_sigma:.2f}")

        # Update material properties
        if hasattr(self, 'ssfr_absorption_slider'):
            absorption = self.ssfr_absorption_slider.value() / 10.0
            if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_sigma_a' in self.opengl_renderer.composite_prog.uniforms:
                self.opengl_renderer.composite_prog['u_sigma_a'].value = absorption
            self.ssfr_absorption_label.setText(f"{absorption:.1f}")

        if hasattr(self, 'ssfr_scatter_slider'):
            scatter = self.ssfr_scatter_slider.value() / 10.0
            if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_sigma_s' in self.opengl_renderer.composite_prog.uniforms:
                self.opengl_renderer.composite_prog['u_sigma_s'].value = scatter
            self.ssfr_scatter_label.setText(f"{scatter:.2f}")

        self.opengl_widget.update()

    def update_water_color(self):
        """Update water color parameters"""
        if hasattr(self, 'water_r_slider') and hasattr(self, 'water_g_slider') and hasattr(self, 'water_b_slider'):
            r = self.water_r_slider.value() / 100.0
            g = self.water_g_slider.value() / 100.0
            b = self.water_b_slider.value() / 100.0
            
            CONFIG.WATER_COLOR = (r, g, b)
            self.water_r_label.setText(f"{r:.2f}")
            self.water_g_label.setText(f"{g:.2f}")
            self.water_b_label.setText(f"{b:.2f}")
            
            # Update shader uniforms if available
            if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_water_color' in self.opengl_renderer.composite_prog.uniforms:
                self.opengl_renderer.composite_prog['u_water_color'].value = (r, g, b)
            
            self.opengl_widget.update()

    def update_simulation(self):
        """Update simulation"""
        if not self.is_running or self.is_paused:
            return

        # 滑块 1..30 → 速度 0.1x..3.0x。原来写的是
        #     dt = DEFAULT_TIME_STEP / speed
        # 方向是反的：滑块拉到最左（"0.1x，最慢"）反而把步长放大到 0.03 s
        # （10x），Taichi 后端拿这么大的步长直接发散（实测 1.5e4~2.7e4 m/s，
        # 水充满容器）。速度倍率的含义是"每帧推进多少物理时间"，所以只能乘。
        # 另外 Taichi 后端会把 dt 硬夹到 TAICHI_MAX_TIME_STEP（=DEFAULT_
        # TIME_STEP），所以 >1.0x 的部分实际不会生效——这和 CPU 后端一致
        # （CPU 路径本就忽略传入的 dt，恒用引擎内部步长）。要真正"更快"应该
        # 走子步（每帧多跑几次 solver.step），而不是放大单个步长。
        speed = max(0.1, min(3.0, self.speed_slider.value() / 10.0))
        dt = CONFIG.DEFAULT_TIME_STEP * speed

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

    # ★ 新增：深度缓冲暴露控制方法
    def toggle_depth_exposure(self, state):
        """切换深度缓冲暴露"""
        is_enabled = bool(state)
        
        if is_enabled:
            # 启用深度暴露
            format_map = {
                "numpy数组": "numpy",
                "纹理对象": "texture", 
                "两者都暴露": "both"
            }
            format_str = format_map.get(self.depth_format_combo.currentText(), "numpy")
            
            # 设置回调函数
            def depth_callback(depth_data):
                self._handle_depth_data(depth_data)
            
            self.opengl_renderer.enable_depth_exposure(
                callback=depth_callback,
                format=format_str
            )
            self.depth_info_label.setText("深度数据: 已启用")
            self.depth_info_label.setStyleSheet("color: green;")
            print("✓ 深度缓冲暴露已启用")
        else:
            # 禁用深度暴露
            self.opengl_renderer.disable_depth_exposure()
            self.depth_info_label.setText("深度数据: 已禁用")
            self.depth_info_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
            print("✓ 深度缓冲暴露已禁用")
        
        self.opengl_widget.update()
    
    def update_depth_exposure_format(self, format_text):
        """更新深度暴露格式"""
        if self.depth_exposure_toggle.isChecked():
            # 重新启用以应用新格式
            self.toggle_depth_exposure(False)
            self.toggle_depth_exposure(True)
    
    def update_depth_frequency(self, value):
        """更新深度暴露频率"""
        self.depth_frequency_label.setText(f"{value} FPS")
        # 这里可以实现频率控制逻辑
        print(f"深度暴露频率设置为: {value} FPS")
    
    def test_depth_exposure(self):
        """测试深度暴露功能"""
        if not self.depth_exposure_toggle.isChecked():
            print("⚠ 请先启用深度缓冲暴露")
            return
            
        # 获取当前深度数据
        depth_data = self.opengl_renderer.get_exposed_depth_data()
        if depth_data and 'data' in depth_data:
            data = depth_data['data']
            print(f"✓ 成功获取深度数据: {data.shape}, 范围 [{np.min(data):.3f}, {np.max(data):.3f}]")
            self.status_bar.showMessage(f"深度数据测试成功: {data.shape[0]}x{data.shape[1]} 像素")
        else:
            print("✗ 未能获取深度数据")
            self.status_bar.showMessage("深度数据测试失败")
    
    def _handle_depth_data(self, depth_data):
        """处理接收到的深度数据"""
        try:
            # 这里可以实现反投影逻辑
            # 示例：简单的统计信息
            if depth_data is not None and len(depth_data) > 0:
                valid_pixels = np.sum(depth_data > 0)
                total_pixels = depth_data.size
                avg_depth = np.mean(depth_data[depth_data > 0]) if valid_pixels > 0 else 0
                
                info_text = f"深度数据: {valid_pixels}/{total_pixels} 有效像素, 平均深度: {avg_depth:.3f}"
                self.depth_info_label.setText(info_text)
                
                # 可以在这里添加反投影计算逻辑
                # 例如：self.perform_back_projection(depth_data)
                
        except Exception as e:
            print(f"✗ 深度数据处理错误: {e}")