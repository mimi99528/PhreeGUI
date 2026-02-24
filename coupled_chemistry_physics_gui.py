#!/usr/bin/env python3
"""
化学-物理耦合模拟GUI应用
Coupled Chemistry-Physics Simulation GUI Application

整合模块化网格系统和化学反应模块，提供直观的GUI界面进行
化学-物理耦合仿真实验。

Features:
- 实时化学浓度可视化
- 物理SPH流体仿真
- 多种化学反应场景
- 参数调节和实时监控
- 性能分析和数据导出
"""

import sys
import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Qt相关导入
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QCheckBox, QComboBox, QGroupBox,
    QTextEdit, QSplitter, QFrame, QTabWidget, QGridLayout, QLineEdit,
    QDoubleSpinBox, QSpinBox, QProgressBar, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject, QThread
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush
from PySide6.QtOpenGLWidgets import QOpenGLWidget

# 项目模块导入
try:
    from modular_mesh_system.core.mesh_system import MeshSystem
    from modular_mesh_system.renderer.opengl_renderer import OpenGLRenderer
    from modular_mesh_system.utils.config import CONFIG
    MODULAR_MESH_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入模块化网格系统: {e}")
    MODULAR_MESH_AVAILABLE = False

try:
    from coupled_simulation import CoupledSimulation, Particle
    COUPLED_SIM_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入耦合仿真模块: {e}")
    COUPLED_SIM_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class ChemicalScenario:
    """化学反应场景配置"""
    name: str
    description: str
    initial_conditions: Dict
    parameters: Dict
    chemical_components: List[str]

class ChemistryVisualizationWidget(QWidget):
    """化学浓度可视化组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.concentration_data = np.zeros((50, 50))  # 50x50网格用于显示
        self.colorbar_width = 30
        self.colorbar_height = 200
        
    def paintEvent(self, event):
        """绘制化学浓度热力图"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.concentration_data.size == 0:
            # 绘制提示文本
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待仿真数据...")
            return
            
        # 获取widget尺寸
        width = self.width() - self.colorbar_width - 20
        height = self.height() - 40
        margin = 20
        
        # 绘制浓度热力图
        rows, cols = self.concentration_data.shape
        cell_width = width / cols
        cell_height = height / rows
        
        # 获取浓度范围
        min_val = np.min(self.concentration_data)
        max_val = np.max(self.concentration_data)
        
        for i in range(rows):
            for j in range(cols):
                # 计算颜色（从蓝到红的渐变）
                conc = self.concentration_data[i, j]
                if max_val > min_val:
                    normalized = (conc - min_val) / (max_val - min_val)
                else:
                    normalized = 0.5
                    
                # 蓝-绿-红渐变
                if normalized < 0.5:
                    # 蓝到绿
                    r = int(255 * normalized * 2)
                    g = int(255 * normalized * 2)
                    b = int(255 * (1 - normalized * 2))
                else:
                    # 绿到红
                    r = int(255 * (0.5 + (normalized - 0.5) * 2))
                    g = int(255 * (1 - (normalized - 0.5) * 2))
                    b = 0
                
                color = QColor(r, g, b)
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color))
                
                x = margin + j * cell_width
                y = margin + i * cell_height
                painter.drawRect(int(x), int(y), int(cell_width), int(cell_height))
        
        # 绘制色条
        self.draw_colorbar(painter, margin + width + 10, margin, min_val, max_val)
        
        # 绘制标题
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(margin, 15, "化学浓度分布")

    def draw_colorbar(self, painter, x, y, min_val, max_val):
        """绘制色条"""
        bar_width = self.colorbar_width
        bar_height = self.colorbar_height
        
        # 绘制渐变色条
        steps = 100
        step_height = bar_height / steps
        
        for i in range(steps):
            normalized = i / (steps - 1)
            if normalized < 0.5:
                r = int(255 * normalized * 2)
                g = int(255 * normalized * 2)
                b = int(255 * (1 - normalized * 2))
            else:
                r = int(255 * (0.5 + (normalized - 0.5) * 2))
                g = int(255 * (1 - (normalized - 0.5) * 2))
                b = 0
            
            color = QColor(r, g, b)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color))
            painter.drawRect(int(x), int(y + (steps - 1 - i) * step_height), 
                           int(bar_width), int(step_height + 1))
        
        # 绘制数值标签
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(int(x + bar_width + 5), int(y + 10), f"{max_val:.2f}")
        painter.drawText(int(x + bar_width + 5), int(y + bar_height - 5), f"{min_val:.2f}")

    def update_concentration_data(self, data: np.ndarray):
        """更新浓度数据"""
        if data is not None and data.size > 0:
            logger.debug(f"接收浓度数据，原始形状: {data.shape}, 范围: [{np.min(data):.3f}, {np.max(data):.3f}]")
            
            # 重采样到显示尺寸
            rows, cols = self.concentration_data.shape
            if len(data.shape) == 1:
                # 一维数据转二维
                data = data.reshape(-1, 1)
            
            # 简单的重采样
            if data.shape[0] >= rows and data.shape[1] >= cols:
                row_step = data.shape[0] // rows
                col_step = data.shape[1] // cols
                sampled = data[::row_step, ::col_step][:rows, :cols]
            else:
                sampled = np.pad(data, ((0, max(0, rows - data.shape[0])), 
                                      (0, max(0, cols - data.shape[1]))), 
                               mode='constant')
            
            self.concentration_data = sampled
            logger.debug(f"更新显示数据，形状: {sampled.shape}, 范围: [{np.min(sampled):.3f}, {np.max(sampled):.3f}]")
            self.update()

class SimulationWorker(QObject):
    """仿真工作线程 - 修复版：异步非阻塞执行"""
    
    # 信号定义
    simulation_updated = Signal(dict)  # 发送仿真状态
    concentration_updated = Signal(np.ndarray)  # 发送浓度数据
    ph_data_updated = Signal(np.ndarray)  # 新增：发送pH数据
    error_occurred = Signal(str)  # 发送错误信息
    simulation_initialized = Signal(bool)  # 仿真初始化完成信号
    particles_updated = Signal(np.ndarray)  # 粒子数据更新信号
    
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.is_paused = False
        self.coupled_sim = None
        self.speed_factor = 1.0
        self.frame_delay = 0.016  # 基础帧延迟(16ms ≈ 60FPS)
        
        # 获取主窗口引用以便访问OpenGL渲染器
        self.main_window = None
        
        # 异步执行相关
        self.pending_initialization = False
        self.initialization_timer = QTimer()
        self.initialization_timer.timeout.connect(self._check_initialization)
        self.initialization_timer.setSingleShot(False)
        self.initialization_timer.setInterval(100)  # 100ms检查间隔
        
        # 启动初始化检查
        self.pending_initialization = True
        self.initialization_timer.start()
    
    def _check_initialization(self):
        """检查并完成仿真初始化"""
        if self.pending_initialization:
            try:
                self.initialize_simulation()
                self.pending_initialization = False
                self.initialization_timer.stop()
                self.simulation_initialized.emit(True)
                logger.info("仿真工作线程初始化完成")
            except Exception as e:
                logger.error(f"仿真初始化失败: {e}")
                self.simulation_initialized.emit(False)
                self.initialization_timer.stop()
    
    def initialize_simulation(self):
        """初始化耦合仿真系统 - 异步版本"""
        try:
            if COUPLED_SIM_AVAILABLE:
                # 创建耦合仿真实例，传入PHREEQCRM实例
                self.coupled_sim = CoupledSimulation(phreeqc_rm=self.main_window.rm if hasattr(self.main_window, 'rm') else None)
                # 注意：这里不立即调用initialize()，避免阻塞
                logger.info("耦合仿真系统实例创建成功")
            else:
                raise ImportError("耦合仿真模块不可用")
        except Exception as e:
            logger.error(f"初始化仿真系统失败: {e}")
            raise
    
    def _setup_phreeqcrm_for_coupled_sim(self) -> None:
        """初始化PhreeqcRM实例并注入到耦合仿真系统中。

        使用与CoupledSimulation化学网格相同的单元数（chem_grid_dim^3 = 216），
        而不是core_simulation.py里的NX*NY*NZ=3375，避免尺寸不匹配。
        若phreeqcrm不可用则静默跳过。
        """
        try:
            from phreeqcrm import PhreeqcRM
            from utils import (
                load_database, get_solution_definition, NTHREADS,
                TIME_STEP, TEMPERATURE, PRESSURE, POROSITY,
                SATURATION, REPRESENTATIVE_VOLUME, REBALANCE_FRACTION,
            )
        except ImportError as e:
            logger.warning(f"PhreeqcRM不可用，化学仿真将在无化学模型模式下运行: {e}")
            return

        n_cells = self.coupled_sim.chem_field.n_cells  # 6^3 = 216
        try:
            rm = PhreeqcRM(n_cells, NTHREADS)
            # rm.SetComponentH2O(False)
            rm.SetUnitsSolution(2)
            rm.SetUnitsPPassemblage(1)
            rm.SetUnitsExchange(1)
            rm.SetPartitionUZSolids(False)
            rm.SetSpeciesSaveOn(False)
            rm.SetRebalanceFraction(REBALANCE_FRACTION)
            rm.SetErrorOn(True)
            rm.SetErrorHandlerMode(1)
            rm.SetPrintChemistryOn(False, False, False)
            rm.SetRebalanceByCell(True)

            # 加载数据库
            load_database(rm)

            # 定义初始溶液并查找化学组分
            initial_input = get_solution_definition()
            rm.RunString(True, True, True, initial_input)
            rm.FindComponents()
            n_comps = rm.GetComponentCount()
            logger.info(f"PhreeqcRM已找到 {n_comps} 个化学组分")

            # 为所有单元分配背景溶液（溶液1 = pH 7 背景水）
            for i in range(n_cells):
                rm.InitialPhreeqcCell2Module(1, [i])

            # 设置物理参数
            rm.SetTimeStep(TIME_STEP)
            rm.SetTemperature([TEMPERATURE] * n_cells)
            rm.SetPressure([PRESSURE] * n_cells)
            rm.SetPorosity([POROSITY] * n_cells)
            rm.SetSaturationUser([SATURATION] * n_cells)
            rm.SetRepresentativeVolume([REPRESENTATIVE_VOLUME] * n_cells)

            # 创建1:1映射
            rm.CreateMapping(list(range(n_cells)))

            # 运行初始平衡
            rm.RunCells()

            # 注入到耦合仿真
            self.coupled_sim.rm = rm
            logger.info(
                f"PhreeqcRM已成功初始化 ({n_cells} 个化学单元，{n_comps} 个化学组分)"
            )
        except Exception as e:
            logger.error(f"PhreeqcRM初始化失败: {e}")

    def lazy_initialize_simulation(self):
        """延迟初始化仿真系统（在第一次仿真步骤时执行）"""
        if self.coupled_sim and not hasattr(self.coupled_sim, '_initialized'):
            try:
                # 若未设置PhreeqcRM，先初始化化学模型
                if self.coupled_sim.rm is None:
                    self._setup_phreeqcrm_for_coupled_sim()

                # 将mesh_system注入到耦合仿真（由coupled_sim.step()统一管理物理步）
                if self.coupled_sim.mesh_system is None and hasattr(self.main_window, 'mesh_system'):
                    self.coupled_sim.mesh_system = self.main_window.mesh_system

                # 执行实际的初始化
                self.coupled_sim.initialize()
                self.coupled_sim._initialized = True
                logger.info("耦合仿真系统延迟初始化完成")
            except Exception as e:
                logger.error(f"延迟初始化失败: {e}")
                raise
    
    @Slot()
    def start_simulation(self):
        """开始仿真 - 异步非阻塞版本"""
        self.is_running = True
        self.is_paused = False
        
        # 启动仿真循环定时器
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self._simulation_step)
        self.simulation_timer.start(int(self.frame_delay * 1000 / self.speed_factor))
        
        logger.info("仿真开始执行")
    
    @Slot()
    def _simulation_step(self):
        """单步仿真执行 - 在Qt事件循环中异步执行"""
        if not self.is_running or self.is_paused:
            return
            
        try:
            # 确保仿真系统已初始化
            if self.coupled_sim and not hasattr(self.coupled_sim, '_initialized'):
                self.lazy_initialize_simulation()
            
            # 执行仿真步骤（物理+化学由coupled_sim.step()统一管理）
            if self.coupled_sim:
                stats = self.coupled_sim.step()
                
                # 发送粒子数据更新信号
                if hasattr(self.main_window, 'mesh_system') and self.coupled_sim:
                    # 直接从mesh_system获取最新的粒子位置，确保数据同步
                    n_particles = self.main_window.mesh_system.current_particle_num
                    if n_particles > 0:
                        particle_positions = self.main_window.mesh_system.particle_positions[:n_particles].copy()
                        if len(particle_positions) > 0:
                            self.particles_updated.emit(particle_positions)
                            logger.debug(f"发送了 {len(particle_positions)} 个粒子位置更新")
                            # 添加调试信息
                            avg_pos = np.mean(particle_positions, axis=0)
                            logger.debug(f"平均粒子位置: [{avg_pos[0]:.3f}, {avg_pos[1]:.3f}, {avg_pos[2]:.3f}]")
                
                # 获取化学浓度数据用于可视化
                if hasattr(self.coupled_sim, 'get_cell_concentrations'):
                    conc_data = self.coupled_sim.get_cell_concentrations()
                    if conc_data is not None and conc_data.size > 0:
                        # 发送浓度数据（取第一个组分）
                        first_component_data = conc_data[:, 0] if len(conc_data.shape) > 1 else conc_data
                        self.concentration_updated.emit(first_component_data)
                        logger.debug(f"发送化学浓度数据，形状: {first_component_data.shape}, 范围: [{np.min(first_component_data):.3f}, {np.max(first_component_data):.3f}]")
                
                # 获取并发送pH数据用于颜色渲染
                if hasattr(self.coupled_sim, 'get_particle_ph'):
                    ph_data = self.coupled_sim.get_particle_ph(0)  # 获取第一个组分作为pH
                    if ph_data is not None and ph_data.size > 0:
                        self.ph_data_updated.emit(ph_data)
                        logger.debug(f"发送pH数据，长度: {len(ph_data)}, 范围: [{np.min(ph_data):.3f}, {np.max(ph_data):.3f}]")
                
                # 发送仿真状态
                self.simulation_updated.emit(stats)
            
        except Exception as e:
            logger.error(f"仿真步骤执行错误: {e}")
            self.error_occurred.emit(f"仿真错误: {str(e)}")
            self.stop_simulation()
    
    @Slot()
    def pause_simulation(self):
        """暂停/继续仿真"""
        self.is_paused = not self.is_paused
        if hasattr(self, 'simulation_timer'):
            if self.is_paused:
                self.simulation_timer.stop()
            else:
                self.simulation_timer.start(int(self.frame_delay * 1000 / self.speed_factor))
    
    @Slot()
    def stop_simulation(self):
        """停止仿真"""
        self.is_running = False
        self.is_paused = False
        
        # 停止定时器
        if hasattr(self, 'simulation_timer'):
            self.simulation_timer.stop()
            del self.simulation_timer
    
    @Slot(float)
    def set_speed(self, speed: float):
        """设置仿真速度"""
        self.speed_factor = max(0.1, min(3.0, speed))
        # 更新定时器间隔
        if hasattr(self, 'simulation_timer') and self.simulation_timer.isActive():
            self.simulation_timer.setInterval(int(self.frame_delay * 1000 / self.speed_factor))
    
    def cleanup(self):
        """清理资源"""
        self.stop_simulation()
        self.initialization_timer.stop()
    

    
    @Slot()
    def pause_simulation(self):
        """暂停/继续仿真"""
        self.is_paused = not self.is_paused
    
    @Slot()
    def stop_simulation(self):
        """停止仿真"""
        self.is_running = False
    
    @Slot(float)
    def set_speed(self, speed: float):
        """设置仿真速度"""
        self.speed_factor = max(0.1, min(3.0, speed))

class CoupledSimulationMainWindow(QMainWindow):
    """化学-物理耦合仿真主窗口"""
    
    def __init__(self):
        super().__init__()
        self.simulation_worker = None
        self.simulation_thread = None
        self.is_running = False
        self.is_paused = False
        
        # 预定义化学场景
        self.scenarios = self.create_scenarios()
        self.current_scenario = None
        
        # 化学-物理耦合相关
        self.enable_ph_color_rendering = True  # 是否启用pH颜色渲染
        self.last_ph_data = None  # 缓存最后的pH数据
        
        # PHREEQCRM配置
        self.rm = None  # PHREEQCRM实例
        self.chem_initialized = False
        
        self.init_ui()
        self.setup_simulation_thread()
        self.initialize_chemistry_system()
        
    def create_scenarios(self) -> List[ChemicalScenario]:
        """创建预定义化学反应场景"""
        return [
            ChemicalScenario(
                name="酸碱中和",
                description="HCl + NaOH → NaCl + H₂O",
                initial_conditions={
                    "left_concentration": 0.01,  # 左侧H+浓度
                    "right_concentration": 0.01, # 右侧OH-浓度
                    "domain_split": 0.0          # 分割平面x坐标
                },
                parameters={
                    "reaction_rate": 1.0,
                    "diffusion_coefficient": 1e-9,
                    "temperature": 25.0
                },
                chemical_components=["H+", "OH-", "Na+", "Cl-"]
            ),
            ChemicalScenario(
                name="碳酸盐平衡",
                description="CO₂ + H₂O ⇌ H₂CO₃ ⇌ H⁺ + HCO₃⁻",
                initial_conditions={
                    "co2_concentration": 1e-3,
                    "water_ph": 7.0
                },
                parameters={
                    "equilibrium_constant": 4.3e-7,
                    "carbonic_acid_k1": 4.3e-7,
                    "bicarbonate_k2": 5.6e-11
                },
                chemical_components=["CO₂", "H₂CO₃", "H+", "HCO₃⁻", "CO₃²⁻"]
            ),
            ChemicalScenario(
                name="氧化还原反应",
                description="Fe²⁺ + Ce⁴⁺ → Fe³⁺ + Ce³⁺",
                initial_conditions={
                    "fe2_concentration": 1e-3,
                    "ce4_concentration": 1e-3
                },
                parameters={
                    "redox_potential": 1.44,
                    "reaction_order": 1.0
                },
                chemical_components=["Fe²⁺", "Fe³⁺", "Ce⁴⁺", "Ce³⁺"]
            )
        ]
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("化学-物理耦合仿真系统")
        # 增大默认窗口尺寸以更好地显示两个面板
        self.setGeometry(100, 100, 1600, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧面板 - 控制和可视化
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        splitter.setStretchFactor(0, 1)
        
        # 右侧面板 - 3D渲染（如果可用）
        if MODULAR_MESH_AVAILABLE:
            right_panel = self.create_right_panel()
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(1, 2)
            # 设置合理的初始大小比例
            splitter.setSizes([400, 800])
        else:
            # 如果没有3D渲染，隐藏右侧面板但仍保持布局
            right_panel = self.create_placeholder_panel()
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(1, 0)
            splitter.setSizes([1200, 200])
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("系统就绪 - 请选择化学反应场景")
    
    def create_left_panel(self) -> QWidget:
        """创建左侧面板"""
        panel = QFrame()
        panel.setMinimumWidth(500)
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
                padding: 8px;
                font-size: 11px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QComboBox {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666666;
                padding: 5px;
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
        layout.setSpacing(10)
        
        # 标题
        title = QLabel("化学-物理耦合仿真控制台")
        title.setStyleSheet("color: #4fc3f7; font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 标签页控件
        tab_widget = QTabWidget()
        
        # 仿真控制标签页
        control_tab = self.create_control_tab()
        tab_widget.addTab(control_tab, "仿真控制")
        
        # 场景设置标签页
        scenario_tab = self.create_scenario_tab()
        tab_widget.addTab(scenario_tab, "场景设置")
        
        # 化学参数标签页
        chemistry_tab = self.create_chemistry_tab()
        tab_widget.addTab(chemistry_tab, "化学参数")
        
        # 监控数据标签页
        monitoring_tab = self.create_monitoring_tab()
        tab_widget.addTab(monitoring_tab, "实时监控")
        
        layout.addWidget(tab_widget)
        return panel
    
    def create_control_tab(self) -> QWidget:
        """创建仿真控制标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # 控制按钮组
        button_group = QGroupBox("仿真控制")
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始仿真")
        self.start_btn.clicked.connect(self.start_simulation)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_simulation)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.stop_btn)
        button_group.setLayout(button_layout)
        layout.addWidget(button_group)
        
        # 速度控制
        speed_group = QGroupBox("仿真速度")
        speed_layout = QVBoxLayout()
        
        speed_layout.addWidget(QLabel("播放速度倍数:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 30)  # 0.1x to 3.0x
        self.speed_slider.setValue(10)  # 1.0x default
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        self.speed_label.setAlignment(Qt.AlignCenter)
        speed_layout.addWidget(self.speed_label)
        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)
        
        # 仿真状态显示
        status_group = QGroupBox("仿真状态")
        status_layout = QGridLayout()
        
        status_layout.addWidget(QLabel("帧数:"), 0, 0)
        self.frame_label = QLabel("0")
        status_layout.addWidget(self.frame_label, 0, 1)
        
        status_layout.addWidget(QLabel("粒子数:"), 0, 2)
        self.particle_label = QLabel("0")
        status_layout.addWidget(self.particle_label, 0, 3)
        
        status_layout.addWidget(QLabel("化学步:"), 1, 0)
        self.chem_step_label = QLabel("否")
        status_layout.addWidget(self.chem_step_label, 1, 1)
        
        status_layout.addWidget(QLabel("化学时间:"), 1, 2)
        self.chem_time_label = QLabel("0s")
        status_layout.addWidget(self.chem_time_label, 1, 3)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        layout.addStretch()
        return widget
    
    def create_scenario_tab(self) -> QWidget:
        """创建场景设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # 场景选择
        scenario_group = QGroupBox("化学反应场景")
        scenario_layout = QVBoxLayout()
        
        self.scenario_combo = QComboBox()
        for scenario in self.scenarios:
            self.scenario_combo.addItem(scenario.name)
        self.scenario_combo.currentTextChanged.connect(self.on_scenario_changed)
        scenario_layout.addWidget(self.scenario_combo)
        
        self.scenario_description = QTextEdit()
        self.scenario_description.setMaximumHeight(80)
        self.scenario_description.setReadOnly(True)
        scenario_layout.addWidget(self.scenario_description)
        
        scenario_group.setLayout(scenario_layout)
        layout.addWidget(scenario_group)
        
        # 场景参数设置
        self.param_group = QGroupBox("场景参数")
        self.param_layout = QGridLayout()
        self.param_group.setLayout(self.param_layout)
        layout.addWidget(self.param_group)
        
        # 初始化第一个场景
        self.on_scenario_changed(self.scenarios[0].name)
        
        layout.addStretch()
        return widget
    
    def create_chemistry_tab(self) -> QWidget:
        """创建化学参数标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # 化学步控制
        chem_step_group = QGroupBox("化学步设置")
        chem_step_layout = QGridLayout()
        
        chem_step_layout.addWidget(QLabel("化学步频率:"), 0, 0)
        self.chem_step_ratio = QSpinBox()
        self.chem_step_ratio.setRange(1, 10)
        self.chem_step_ratio.setValue(2)
        self.chem_step_ratio.setToolTip("每N帧物理步执行一次化学步")
        chem_step_layout.addWidget(self.chem_step_ratio, 0, 1)
        
        chem_step_layout.addWidget(QLabel("化学时间步长(秒):"), 1, 0)
        self.chem_dt_spinbox = QDoubleSpinBox()
        self.chem_dt_spinbox.setRange(1.0, 86400.0)
        self.chem_dt_spinbox.setValue(3600.0)  # 1小时
        self.chem_dt_spinbox.setSuffix(" s")
        chem_step_layout.addWidget(self.chem_dt_spinbox, 1, 1)
        
        chem_step_group.setLayout(chem_step_layout)
        layout.addWidget(chem_step_group)
        
        # 化学组分显示
        component_group = QGroupBox("化学组分")
        component_layout = QVBoxLayout()
        
        self.component_list = QTextEdit()
        self.component_list.setMaximumHeight(150)
        self.component_list.setReadOnly(True)
        component_layout.addWidget(self.component_list)
        
        component_group.setLayout(component_layout)
        layout.addWidget(component_group)
        
        layout.addStretch()
        return widget
    
    def create_monitoring_tab(self) -> QWidget:
        """创建实时监控标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # 化学浓度可视化
        viz_group = QGroupBox("化学浓度分布")
        viz_layout = QVBoxLayout()
        
        self.concentration_viz = ChemistryVisualizationWidget()
        viz_layout.addWidget(self.concentration_viz)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # 性能监控
        perf_group = QGroupBox("性能监控")
        perf_layout = QGridLayout()
        
        perf_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_label = QLabel("0")
        perf_layout.addWidget(self.fps_label, 0, 1)
        
        perf_layout.addWidget(QLabel("物理计算时间:"), 0, 2)
        self.physics_time_label = QLabel("0ms")
        perf_layout.addWidget(self.physics_time_label, 0, 3)
        
        perf_layout.addWidget(QLabel("化学计算时间:"), 1, 0)
        self.chemistry_time_label = QLabel("0ms")
        perf_layout.addWidget(self.chemistry_time_label, 1, 1)
        
        perf_layout.addWidget(QLabel("内存使用:"), 1, 2)
        self.memory_label = QLabel("0MB")
        perf_layout.addWidget(self.memory_label, 1, 3)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        layout.addStretch()
        return widget
    
    def create_placeholder_panel(self) -> QWidget:
        """创建占位面板（当3D渲染不可用时）"""
        panel = QFrame()
        panel.setMinimumWidth(200)
        panel.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px dashed #444444;
            }
            QLabel {
                color: #888888;
                font-size: 12px;
                font-style: italic;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 添加提示文本
        placeholder_label = QLabel("3D渲染模块不可用\n\n请安装必要的依赖:\n• moderngl\n• OpenGL 3.3+支持")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setWordWrap(True)
        layout.addWidget(placeholder_label)
        
        # 添加更多信息
        info_label = QLabel("化学浓度可视化在左侧\n实时监控标签页中显示")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #666666; font-size: 10px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self) -> QWidget:
        """创建右侧面板（3D渲染）"""
        if not MODULAR_MESH_AVAILABLE:
            # 如果模块化网格系统不可用，返回空面板
            panel = QFrame()
            layout = QVBoxLayout(panel)
            layout.addWidget(QLabel("3D渲染模块不可用"))
            return panel
            
        try:
            # 创建3D渲染面板
            panel = QFrame()
            layout = QVBoxLayout(panel)
            
            # 创建网格系统和渲染器
            self.mesh_system = MeshSystem()
            self.opengl_renderer = OpenGLRenderer(self.mesh_system)
            
            # 创建OpenGL widget包装器
            class QOpenGLWrapper(QOpenGLWidget):
                def __init__(self, renderer):
                    super().__init__()
                    self.renderer = renderer
                    self.ctx = None
                    
                def initializeGL(self):
                    try:
                        import moderngl
                        self.ctx = moderngl.create_context(standalone=False)
                        if not self.renderer.initialize(self.ctx):
                            print("渲染器初始化失败")
                        else:
                            # 启用颜色粒子渲染（但不影响主SSFR流程）
                            self.renderer.use_colored_particles = True
                            print("✓ 启用pH颜色可视化渲染")
                    except Exception as e:
                        print(f"OpenGL初始化失败: {e}")
                
                def paintGL(self):
                    if self.ctx:
                        try:
                            # 检测当前framebuffer
                            screen_fbo = self.ctx.detect_framebuffer()
                            screen_fbo.use()
                            
                            # 设置视口
                            size = self.size()
                            self.ctx.viewport = (0, 0, size.width(), size.height())
                            
                            # 清屏
                            self.ctx.clear(0.1, 0.1, 0.15, 1.0)
                            
                            # 调用渲染器
                            self.renderer.render(size.width(), size.height(), screen_fbo)
                            
                        except Exception as e:
                            print(f"渲染错误: {e}")
                            import traceback
                            traceback.print_exc()
                
                def resizeGL(self, width, height):
                    if self.ctx and width > 0 and height > 0:
                        self.ctx.viewport = (0, 0, width, height)
            
            self.opengl_widget = QOpenGLWrapper(self.opengl_renderer)
            layout.addWidget(self.opengl_widget)
            
            return panel
        except Exception as e:
            logger.error(f"创建3D渲染面板失败: {e}")
            # 返回错误面板
            panel = QFrame()
            layout = QVBoxLayout(panel)
            layout.addWidget(QLabel(f"3D渲染初始化失败: {str(e)}"))
            return panel
    
    def initialize_chemistry_system(self):
        """初始化化学系统"""
        try:
            from phreeqc_config import create_fully_initialized_phreeqcrm
            
            self.status_bar.showMessage("正在初始化化学系统...")
            QApplication.processEvents()  # 更新UI
            
            # 创建初始化的PHREEQCRM实例
            self.rm = create_fully_initialized_phreeqcrm(nx=6, ny=6, nz=6)
            
            if self.rm is not None:
                self.chem_initialized = True
                self.status_bar.showMessage("化学系统初始化完成")
                logger.info("✓ PHREEQCRM化学系统初始化成功")
            else:
                self.chem_initialized = False
                self.status_bar.showMessage("化学系统初始化失败")
                logger.error("✗ PHREEQCRM化学系统初始化失败")
                
        except Exception as e:
            self.chem_initialized = False
            self.status_bar.showMessage(f"化学系统初始化错误: {str(e)}")
            logger.error(f"化学系统初始化异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def setup_simulation_thread(self):
        """设置仿真线程 - 修复版"""
        self.simulation_thread = QThread()
        self.simulation_worker = SimulationWorker()
        # 设置主窗口引用以便粒子同步
        self.simulation_worker.main_window = self
        
        # 连接信号
        self.simulation_worker.moveToThread(self.simulation_thread)
        self.simulation_worker.simulation_updated.connect(self.on_simulation_update)
        self.simulation_worker.concentration_updated.connect(self.on_concentration_update)
        self.simulation_worker.ph_data_updated.connect(self.on_ph_data_updated)  # 新增连接
        self.simulation_worker.error_occurred.connect(self.on_simulation_error)
        self.simulation_worker.simulation_initialized.connect(self.on_simulation_initialized)
        self.simulation_worker.particles_updated.connect(self.on_particles_updated)
        
        # 启动线程
        self.simulation_thread.start()
        
        # 显示初始化状态
        self.status_bar.showMessage("正在初始化仿真系统...")
    
    @Slot(bool)
    def on_simulation_initialized(self, success: bool):
        """处理仿真初始化完成信号"""
        if success:
            status_msg = "仿真系统初始化完成"
            if self.chem_initialized:
                status_msg += " - 化学系统就绪"
            else:
                status_msg += " - 化学系统初始化失败，将使用默认模式"
            
            self.status_bar.showMessage(status_msg)
            # 启用开始按钮
            if hasattr(self, 'start_btn'):
                self.start_btn.setEnabled(True)
        else:
            self.status_bar.showMessage("仿真系统初始化失败")
            QMessageBox.warning(self, "初始化错误", "仿真系统初始化失败，请检查依赖项")
    
    def on_scenario_changed(self, scenario_name: str):
        """场景改变时的处理"""
        # 查找选中的场景
        selected_scenario = None
        for scenario in self.scenarios:
            if scenario.name == scenario_name:
                selected_scenario = scenario
                break
        
        if selected_scenario:
            self.current_scenario = selected_scenario
            # 更新描述
            self.scenario_description.setText(selected_scenario.description)
            
            # 更新化学组分显示（如果控件存在）
            if hasattr(self, 'component_list') and self.component_list:
                components_text = "\n".join([f"• {comp}" for comp in selected_scenario.chemical_components])
                self.component_list.setText(components_text)
            
            # 更新参数控制
            self.update_scenario_parameters(selected_scenario)
    
    def update_scenario_parameters(self, scenario: ChemicalScenario):
        """更新场景参数控制界面"""
        # 清除现有参数控件
        for i in reversed(range(self.param_layout.count())):
            widget = self.param_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # 添加新的参数控件
        row = 0
        for param_name, param_value in scenario.parameters.items():
            label = QLabel(f"{param_name}:")
            self.param_layout.addWidget(label, row, 0)
            
            if isinstance(param_value, (int, float)):
                spinbox = QDoubleSpinBox()
                spinbox.setRange(0.0, 1000000.0)
                spinbox.setValue(float(param_value))
                spinbox.setDecimals(6 if param_value < 1 else 2)
                self.param_layout.addWidget(spinbox, row, 1)
            
            row += 1
    
    def start_simulation(self):
        """开始仿真 - 修复版"""
        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            
            # 更新UI状态
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.reset_btn.setEnabled(False)
            
            # 如果有3D渲染，初始化流体
            if hasattr(self, 'mesh_system'):
                try:
                    particle_count = self.mesh_system.initialize_fluid_drops()
                    self.status_bar.showMessage(f"仿真开始 - 粒子数: {particle_count}")
                except Exception as e:
                    logger.error(f"初始化流体失败: {e}")
                    self.status_bar.showMessage(f"仿真开始 - 流体初始化失败: {str(e)}")
            else:
                self.status_bar.showMessage("仿真开始 - 无3D渲染")
            
            # 启动仿真线程
            if self.simulation_worker:
                self.simulation_worker.start_simulation()
    
    def toggle_pause(self):
        """暂停/继续仿真"""
        if self.simulation_worker:
            self.simulation_worker.pause_simulation()
            self.is_paused = not self.is_paused
            self.pause_btn.setText("继续" if self.is_paused else "暂停")
    
    def stop_simulation(self):
        """停止仿真"""
        if self.simulation_worker:
            self.simulation_worker.stop_simulation()
        
        self.is_running = False
        self.is_paused = False
        
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self.pause_btn.setText("暂停")
        
        self.status_bar.showMessage("仿真已停止")
    
    def reset_simulation(self):
        """重置仿真"""
        self.stop_simulation()
        
        # 重置显示数据
        self.frame_label.setText("0")
        self.particle_label.setText("0")
        self.chem_step_label.setText("否")
        self.chem_time_label.setText("0s")
        
        # 清除浓度可视化
        self.concentration_viz.update_concentration_data(np.array([]))
        
        self.status_bar.showMessage("仿真已重置")
    
    def update_speed(self, value: int):
        """更新仿真速度"""
        speed = value / 10.0
        self.speed_label.setText(f"{speed:.1f}x")
        if self.simulation_worker:
            self.simulation_worker.set_speed(speed)
    
    @Slot(dict)
    def on_simulation_update(self, stats: dict):
        """处理仿真更新"""
        # 更新状态显示
        self.frame_label.setText(str(stats.get("frame", 0)))
        self.particle_label.setText(str(stats.get("n_particles", 0)))
        self.chem_step_label.setText("是" if stats.get("chemical_step", False) else "否")
        self.chem_time_label.setText(f"{stats.get('chem_time', 0):.1f}s")
    
    @Slot(np.ndarray)
    def on_concentration_update(self, concentration_data: np.ndarray):
        """处理浓度数据更新"""
        self.concentration_viz.update_concentration_data(concentration_data)
    
    @Slot(np.ndarray)
    def on_particles_updated(self, particle_positions: np.ndarray):
        """处理粒子数据更新"""
        try:
            if hasattr(self, 'opengl_widget') and hasattr(self, 'opengl_renderer'):
                # 在GUI线程中更新OpenGL渲染器
                self.opengl_renderer.update_particles(particle_positions, ph_values=self.last_ph_data)
                
                # 如果启用了pH颜色渲染且有化学数据，则更新粒子颜色
                if (self.enable_ph_color_rendering and 
                    self.last_ph_data is not None and 
                    len(self.last_ph_data) >= len(particle_positions)):
                    
                    # 截取对应数量的pH数据
                    ph_subset = self.last_ph_data[:len(particle_positions)]
                    
                    # 转换为颜色
                    particle_colors = self.opengl_renderer.convert_ph_to_color(ph_subset)
                    
                    # 更新粒子颜色
                    self.opengl_renderer.update_particle_colors(particle_colors)
                    
                    logger.debug(f"更新了 {len(particle_colors)} 个粒子的颜色，pH范围: [{np.min(ph_subset):.2f}, {np.max(ph_subset):.2f}]")
                
                # 强制触发重绘
                self.opengl_widget.update()
                logger.debug(f"更新了 {len(particle_positions)} 个粒子到渲染器")
                
                # 调试信息：检查粒子位置是否有变化
                if len(particle_positions) > 0:
                    avg_pos = np.mean(particle_positions, axis=0)
                    logger.debug(f"平均粒子位置: [{avg_pos[0]:.3f}, {avg_pos[1]:.3f}, {avg_pos[2]:.3f}]")
                    
                    # 额外的验证：检查粒子是否在合理范围内移动
                    if hasattr(self, '_last_avg_pos'):
                        pos_change = np.linalg.norm(avg_pos - self._last_avg_pos)
                        logger.debug(f"平均位置变化量: {pos_change:.6f}")
                        if pos_change < 1e-6:
                            logger.warning("警告：粒子位置几乎没有变化，可能存在更新问题")
                    self._last_avg_pos = avg_pos.copy()
                    
        except Exception as e:
            logger.warning(f"粒子渲染更新失败: {e}")
            import traceback
            logger.warning(traceback.format_exc())

    @Slot(np.ndarray)
    def on_ph_data_updated(self, ph_data: np.ndarray):
        """处理pH数据更新"""
        try:
            self.last_ph_data = ph_data.copy() if ph_data is not None else None
            logger.debug(f"接收pH数据，长度: {len(ph_data) if ph_data is not None else 0}")
            
            # 如果已经有粒子数据，立即更新颜色
            if (hasattr(self, 'opengl_renderer') and 
                self.opengl_renderer.particle_count > 0 and 
                self.last_ph_data is not None):
                
                # 获取当前粒子数量
                current_particle_count = self.opengl_renderer.particle_count
                
                # 截取对应数量的pH数据
                if len(self.last_ph_data) >= current_particle_count:
                    ph_subset = self.last_ph_data[:current_particle_count]
                    
                    # 转换为颜色
                    particle_colors = self.opengl_renderer.convert_ph_to_color(ph_subset)
                    
                    # 更新粒子颜色
                    self.opengl_renderer.update_particle_colors(particle_colors)
                    
                    logger.debug(f"实时更新粒子颜色，基于 {len(ph_subset)} 个pH值")
                    
                    # 触发重绘
                    if hasattr(self, 'opengl_widget'):
                        self.opengl_widget.update()
                        
        except Exception as e:
            logger.warning(f"pH数据处理失败: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    @Slot(str)
    def on_simulation_error(self, error_msg: str):
        """处理仿真错误"""
        logger.error(f"仿真错误: {error_msg}")
        QMessageBox.critical(self, "仿真错误", error_msg)
        self.stop_simulation()
    
    def closeEvent(self, event):
        """窗口关闭事件 - 修复版"""
        # 停止仿真线程
        if self.simulation_worker:
            self.simulation_worker.cleanup()
            self.simulation_worker.deleteLater()
        
        if self.simulation_thread:
            self.simulation_thread.quit()
            # 等待线程结束，但设置超时防止无限等待
            if not self.simulation_thread.wait(3000):  # 3秒超时
                logger.warning("仿真线程未能正常退出")
            
        event.accept()

def main():
    """主函数"""
    print("启动化学-物理耦合仿真GUI应用...")
    
    # 设置OpenGL格式
    from PySide6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 创建主窗口
    try:
        window = CoupledSimulationMainWindow()
        window.show()
        print("应用启动成功")
        return app.exec()
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        QMessageBox.critical(None, "启动错误", f"应用启动失败: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())