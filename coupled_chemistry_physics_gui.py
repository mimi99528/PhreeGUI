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
import math
import traceback
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


def _force_utf8_streams():
    """见 launch_coupled_simulation.py 中同名函数说明。
    本文件可直接作为入口运行（见文件末尾 __main__），
    因此必须自己把 stdout/stderr 切到 UTF-8，否则 GBK 控制台下
    中文/对勾 print 会抛 UnicodeEncodeError 并把启动过程打崩。"""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_streams()

# Qt相关导入
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QCheckBox, QComboBox, QGroupBox,
    QTextEdit, QSplitter, QFrame, QTabWidget, QGridLayout, QLineEdit,
    QDoubleSpinBox, QSpinBox, QProgressBar, QFileDialog, QMessageBox,
    QDockWidget, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject, QThread, QRect
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QImage
from PySide6.QtOpenGLWidgets import QOpenGLWidget

# 项目模块导入
try:
    from modular_mesh_system.core.mesh_factory import create_mesh_system
    from modular_mesh_system.renderer.opengl_renderer import OpenGLRenderer
    from modular_mesh_system.utils.config import CONFIG
    from modular_mesh_system.utils.helpers import calculate_camera_position
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

try:
    from phreeqc_config import FLUID_TYPE_NAMES, WATER_COMPOSITIONS
except ImportError as e:
    print(f"警告: 无法导入 phreeqc_config: {e}")
    FLUID_TYPE_NAMES = {1: "水", 2: "酸性溶液 (HCl)", 3: "碱性溶液 (NaOH)"}
    WATER_COMPOSITIONS = {1: None, 2: None, 3: None}

# 配置日志：默认 INFO；诊断时用 PHREEGUI_LOG_LEVEL=DEBUG 启动
logging.basicConfig(
    level=getattr(logging, os.environ.get("PHREEGUI_LOG_LEVEL", "INFO").upper(),
                  logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class ChemicalScenario:
    """化学反应场景配置。

    fluid_types 决定这个场景下 Ctrl+滚轮能切到哪些液体（只影响**新生成**的
    粒子，已有粒子的化学状态保持不动）；field_kind 决定分布图显示哪个量。
    """
    name: str
    description: str
    initial_conditions: Dict
    parameters: Dict
    chemical_components: List[str]
    fluid_types: List[int] = field(default_factory=lambda: [1, 2, 3])
    default_fluid_type: int = 1
    field_kind: str = "solute"

class ChemistryVisualizationWidget(QWidget):
    """化学浓度可视化组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.concentration_data = np.zeros((50, 50))  # 50x50网格用于显示
        self.colorbar_width = 30
        self.colorbar_height = 200
        self._image = None          # 预渲染的热力图
        self._image_rgb = None      # 保持 QImage 底层缓冲有效
        self._colorbar_img = None   # 预渲染的色条
        self._min_val = 0.0
        self._max_val = 0.0
        # 标题里的指标名：场景决定看「溶质浓度 / pH / 溶解态铁 …」，
        # 不能让三个场景都用同一句「化学场分布」，否则看不出区别。
        self._field_label = "化学场"

    _LUT = None  # 256 级颜色查找表（首次使用时构建）

    @classmethod
    def _colormap(cls, normalized: np.ndarray) -> np.ndarray:
        """蓝->绿->红渐变，返回 (..., 3) uint8。

        原来每个像素都要算 6 次浮点运算+3 个临时数组，50x50 时耗时约 0.23 ms，
        数据形状被 pad 到 1000x50 时涨到 1.3 ms。改为 256 级 LUT 查表后
        只做一次量化+索引，耗时与像素数近乎线性但常数小一个量级。"""
        if cls._LUT is None:
            n = np.linspace(0.0, 1.0, 256, dtype=np.float32)
            low = n < 0.5
            r = np.where(low, 510.0 * n, 255.0 * (2.0 * n - 0.5))
            g = np.where(low, 510.0 * n, 255.0 * (2.0 - 2.0 * n))
            b = np.where(low, 255.0 * (1.0 - 2.0 * n), 0.0)
            cls._LUT = np.clip(np.stack([r, g, b], axis=-1), 0.0, 255.0).astype(np.uint8)

        idx = np.clip(normalized * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        return cls._LUT[idx]

    def paintEvent(self, event):
        """绘制化学浓度热力图（只 drawImage，不再逐格画矩形）"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        if self._image is None:
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待仿真数据...")
            return

        width = self.width() - self.colorbar_width - 20
        height = self.height() - 40
        margin = 20

        painter.drawImage(QRect(margin, margin, max(1, width), max(1, height)),
                          self._image)
        self.draw_colorbar(painter, margin + width + 10, margin,
                           self._min_val, self._max_val)

        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(margin, 15, f"{self._field_label}（X–Z 平面）")
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QPen(QColor(180, 180, 180)))
        painter.drawText(margin, margin + height + 14, "X →")
        painter.drawText(margin + width - 24, margin + height + 14, "Z ↓")

    def draw_colorbar(self, painter, x, y, min_val, max_val):
        """绘制色条（渐变图只生成一次）"""
        bar_width = self.colorbar_width
        bar_height = self.colorbar_height

        if self._colorbar_img is None:
            # 顶部 = 最大值，底部 = 最小值
            norm = np.linspace(1.0, 0.0, 100, dtype=np.float64).reshape(100, 1)
            rgb = self._colormap(norm)  # (100, 1, 3)
            self._colorbar_img = QImage(np.ascontiguousarray(rgb).data, 1, 100, 3,
                                        QImage.Format_RGB888).copy()

        painter.drawImage(QRect(int(x), int(y), int(bar_width), int(bar_height)),
                          self._colorbar_img)

        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(int(x + bar_width + 5), int(y + 10), f"{max_val:.2f}")
        painter.drawText(int(x + bar_width + 5), int(y + bar_height - 5), f"{min_val:.2f}")

    def clear(self):
        """清空热力图（重置仿真时用）"""
        self.concentration_data = np.zeros((50, 50))
        self._image = None
        self._image_rgb = None
        self.update()

    def set_field_label(self, name: str):
        """设置分布图标题里的指标名（场景切换时由主窗口调用）。"""
        name = (name or "").strip()
        if not name or name == self._field_label:
            return
        self._field_label = name
        self.update()

    # 无数据单元（NaN）用中性灰，和「蓝→绿→红」色带明显区分
    _NO_DATA_RGB = np.array([58, 62, 72], dtype=np.uint8)

    def update_concentration_data(self, data: np.ndarray):
        """更新浓度数据（重采样成显示网格 + 预渲染成 QImage）。

        输入是 get_2d_field() 的 (n_dim, n_dim) 空间场（行 = z、列 = x），
        也可能是旧的一维数组（沿最后兼容）。含 NaN 的单元是「该竖直柱
        本步没有化学数据」，画成中性灰，不再因为一个 NaN 就整帧丢弃。
        """
        if data is None or data.size == 0:
            return

        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        rows, cols = self.concentration_data.shape

        # 抽稀/复制到显示网格。10x10 的化学场会被平滑放大，视觉上就是
        # 连续场；不要在源数据上做 stride 抽样后再 pad（会把真实分布打乱）。
        src_r, src_c = arr.shape
        row_idx = np.clip((np.arange(rows) * src_r / rows).astype(np.int64), 0, src_r - 1)
        col_idx = np.clip((np.arange(cols) * src_c / cols).astype(np.int64), 0, src_c - 1)
        sampled = arr[np.ix_(row_idx, col_idx)]

        self.concentration_data = sampled
        finite = np.isfinite(sampled)
        if np.any(finite):
            vmin = float(np.min(sampled[finite]))
            vmax = float(np.max(sampled[finite]))
        else:
            vmin = vmax = 0.0
        self._min_val, self._max_val = vmin, vmax

        if vmax > vmin + 1e-12:
            normalized = (sampled - vmin) / (vmax - vmin)
        else:
            normalized = np.where(finite, 0.5, np.nan)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

        if (self._image is None or self._image_rgb is None
                or self._image_rgb.shape != sampled.shape + (3,)):
            # QImage 直接包住这块常驻缓冲，之后只改内容不重建对象，
            # 避免每帧 ascontiguousarray + QImage.copy() 两次额外分配。
            self._image_rgb = np.zeros(sampled.shape + (3,), dtype=np.uint8)
            h, w = sampled.shape
            self._image = QImage(self._image_rgb.data, w, h, w * 3,
                                 QImage.Format_RGB888)

        rgb = self._colormap(normalized)
        rgb[~finite] = self._NO_DATA_RGB
        self._image_rgb[...] = rgb
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
    status_message = Signal(str)  # worker 线程 -> 主线程状态栏（跨线程只能走信号）
    
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.is_paused = False
        self.coupled_sim = None
        self.speed_factor = 1.0
        self.frame_delay = 0.016  # 基础帧延迟(16ms ≈ 60FPS)
        # 每个 UI 帧跑几次物理步。dt≈0.003s、60FPS 时 1 子步 ≈ 0.18× 实时，
        # 水面看起来像慢动作糖浆；默认 3 子步 ≈ 0.54× 实时，可在调试面板实时调。
        self.physics_substeps = 3
        self.simulation_timer = None  # 在 worker 线程的 start_simulation 里创建

        # 分布图指标：由场景决定（'solute' / 'ph' / 'Fe' / 'C' 或组分名）。
        # 默认为 'solute'，主窗口切换场景时通过 worker_field_kind_requested 改。
        self.field_kind = "solute"

        # 【决策 D9】pH 账目。闭合式：
        #   frames_emitted + frames_no_data_at_all + fetch_failed
        #       == 尝试取 pH 的帧数
        # values_ok / values_no_data 是"按粒子"口径：无数据的粒子照常发送，
        # 由渲染器画「无数据」色（不再有"整帧丢弃"这一项，它恒为 0）。
        self.ph_accounting = {
            "frames_emitted": 0,        # 真的发出 pH 数组的帧数
            "frames_no_data_at_all": 0,  # get_particle_ph 返回 None/空（还没点粒子）
            "fetch_failed": 0,           # 取数抛异常
            "values_ok": 0,              # 有 pH 值的粒子个数（累计）
            "values_no_data": 0,         # 本步没参与化学反应 → NaN 的粒子个数（累计）
            "inf_as_no_data": 0,         # ±inf 被并入"无数据"的个数（累计）
        }
        
        # 获取主窗口引用以便访问OpenGL渲染器
        self.main_window = None
        
        # 异步执行相关
        # 【线程归属】QTimer 只能在其所属线程里 start/stop。本对象随后会被
        # moveToThread 到仿真线程，所以定时器不能在 __init__（主线程）里建，
        # 必须在 worker 线程里创建（见 _start_initialization_timer）。
        self.pending_initialization = True
        self.initialization_timer = None

    @Slot()
    def _start_initialization_timer(self):
        """由 QThread.started 触发，在本对象所属线程里创建初始化检查定时器。"""
        if self.initialization_timer is not None:
            return
        self.initialization_timer = QTimer(self)
        self.initialization_timer.timeout.connect(self._check_initialization)
        self.initialization_timer.setSingleShot(False)
        self.initialization_timer.setInterval(100)  # 100ms检查间隔
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
                # 【顺序修复】rm 缺失必须硬失败，不能把 None 传给 CoupledSimulation：
                # 那样会留下一个"看起来建好了、实际没有化学"的耦合系统，
                # 之后既无法播种粒子，又会在点开始时再走一次初始化路径。
                rm = getattr(self.main_window, 'rm', None)
                if rm is None:
                    raise RuntimeError(
                        "化学系统尚未就绪（MainWindow.rm 为 None）。"
                        "拒绝创建没有 PhreeqcRM 的耦合仿真系统。")
                # 创建耦合仿真实例，传入PHREEQCRM实例
                self.coupled_sim = CoupledSimulation(phreeqc_rm=rm)
                # 注意：这里不立即调用initialize()，避免阻塞
                logger.info("耦合仿真系统实例创建成功")
            else:
                raise ImportError("耦合仿真模块不可用")
        except Exception as e:
            logger.error(f"初始化仿真系统失败: {e}")
            raise
    
    def _setup_phreeqcrm_for_coupled_sim(self) -> None:
        """复用主窗口已初始化的 PhreeqcRM，绝不在此处新建第二个实例。

        【决策 D7 / 顺序修复】本函数原先自己 new 一个 PhreeqcRM，是第二条
        化学初始化路径。它只在"worker 拿到的 rm 是 None"时被调用，而那正是
        主线程还没完成化学初始化的竞态窗口：于是既造出第二个实例被守卫拦下
        （点"开始仿真"弹窗），又让已生成的粒子无法播种化学。
        现在收敛为唯一正确行为：
          - 主窗口 rm 可用 -> 直接复用；
          - 主窗口 rm 不可用 -> 明确硬失败，不再偷偷造实例。
        """
        rm = getattr(self.main_window, 'rm', None)
        if rm is None:
            raise RuntimeError(
                "主窗口化学系统未就绪（rm 为 None）。本进程只允许一条化学初始化"
                "路径（决策 D7），worker 线程不会再创建第二个 PhreeqcRM。"
                "请检查 MainWindow.initialize_chemistry_system() 的日志。")
        if self.coupled_sim is None:
            raise RuntimeError("耦合仿真系统未创建，无法注入 PhreeqcRM")

        # 与 CoupledSimulation.__init__ 传入 rm 时保持一致的文件输出配置。
        rm.SetFilePrefix("coupled_simulation")
        rm.SetPrintChemistryOn(True, True, True)
        rm.OpenFiles()
        self.coupled_sim.rm = rm
        logger.info("worker 复用主窗口已初始化的 PhreeqcRM（未新建实例）")

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
        """开始仿真 - 异步非阻塞版本。

        【线程归属】本槽由主窗口通过排队信号调用，执行在 worker 线程；
        因此这里创建的 QTimer 也归 worker 线程（tick 直接在本线程调用
        _simulation_step），pause/stop/set_speed 同样排队过来改它，
        不再出现"主线程停不掉 worker 的定时器"。
        """
        if self.simulation_timer is None:
            self.simulation_timer = QTimer(self)
            self.simulation_timer.timeout.connect(self._simulation_step)

        self.is_running = True
        self.is_paused = False
        if not self.simulation_timer.isActive():
            self.simulation_timer.start(int(self.frame_delay * 1000 / self.speed_factor))

        # 流体初始化也要在 worker 线程做：它读 C++ 粒子缓冲，放主线程会和
        # 本线程正在跑的步骤抢同一个 MeshSystem。
        if self.main_window is not None and hasattr(self.main_window, 'mesh_system'):
            try:
                particle_count = self.main_window.mesh_system.initialize_fluid_drops()
                self.status_message.emit(f"仿真开始 - 粒子数: {particle_count}")
            except Exception as e:
                logger.error(f"初始化流体失败: {e}")
                self.status_message.emit(f"仿真开始 - 流体初始化失败: {str(e)}")
        else:
            self.status_message.emit("仿真开始 - 无3D渲染")

        logger.info("仿真开始执行")
    
    @Slot()
    def _simulation_step(self):
        """单步仿真执行 - 在Qt事件循环中异步执行 - 增强安全性版本"""
        if not self.is_running or self.is_paused:
            return
            
        try:
            # 确保仿真系统已初始化
            if self.coupled_sim and not hasattr(self.coupled_sim, '_initialized'):
                self.lazy_initialize_simulation()
            
            # 执行仿真步骤（物理+化学由coupled_sim.step()统一管理）
            if self.coupled_sim:
                substeps = max(1, min(8, int(getattr(self, 'physics_substeps', 1))))
                for _ in range(substeps):
                    stats = self.coupled_sim.step()
                    if not self.is_running or self.is_paused:
                        break
                
                # 发送粒子数据更新信号
                if hasattr(self.main_window, 'mesh_system') and self.coupled_sim:
                    # 直接从mesh_system获取最新的粒子位置，确保数据同步
                    n_particles = self.main_window.mesh_system.current_particle_num
                    if n_particles > 0:
                        # 安全地复制粒子位置，避免数组索引错误
                        try:
                            particle_positions = self.main_window.mesh_system.particle_positions[:n_particles].copy()
                            if len(particle_positions) > 0:
                                # 验证粒子位置数据的有效性
                                # isfinite 一次扫完 NaN 和 ±inf；原来 isnan+isinf
                                # 是两趟全数组扫描，10M 粒子时每帧多读 240 MB。
                                if np.all(np.isfinite(particle_positions)):
                                    self.particles_updated.emit(particle_positions)
                                else:
                                    logger.warning("粒子位置包含无效值，跳过更新")
                                    self.particles_updated.emit(np.zeros((0, 3), dtype=np.float32))
                        except Exception as e:
                            logger.error(f"复制粒子位置失败: {e}")
                            # 发送空数组作为回退
                            self.particles_updated.emit(np.zeros((0, 3), dtype=np.float32))
                
                # 获取化学浓度数据用于可视化
                # 【分布图修复】旧代码取 conc_data[:, 0] —— 那是组分 0 = H（水本身
                # 的元素总量，约 110.68 mol/L 的常数），一维数组再被 pad 成 50x50，
                # 所以永远是"一片均匀色、不像真实分布"。
                # 现在改用 get_2d_field()：
                #   - 'solute' = 真实溶质(Cl/Na/…)在 X-Z 平面上沿 Y 的柱积分；
                #   - 'ph'     = 按柱取最偏离 7 的 pH（酸碱中和 / 碳酸盐场景）；
                #   - 'Fe+2' / 'C' = SelectedOutput 里该物种 / 组分的柱积分
                #                    （氧化还原场景看亚铁被氧化掉）。
                # 具体看哪个量由场景决定（self.field_kind），主窗口切换场景时改。
                # 只有 'solute' 这一种"自动挑溶质"的指标在主场景没溶质时退到
                # 'ph'；显式指定的指标即使整场接近 0 也照实画（那是真实结果）。
                if hasattr(self.coupled_sim, 'get_2d_field'):
                    try:
                        kind = getattr(self, 'field_kind', 'solute') or 'solute'
                        field = self.coupled_sim.get_2d_field(kind)
                        has_finite = (field is not None and field.size > 0
                                      and np.any(np.isfinite(field)))
                        if kind == 'solute':
                            # 纯水场景没有溶质：整场 0 画出来是"一片假热区"，
                            # 这时才退到 pH。显式指定的指标（ph / Fe+2 / C …）
                            # 即使数值很小也是真实结果，不许被替换掉标题。
                            usable = (has_finite
                                      and float(np.nanmax(np.abs(field))) > 1e-9)
                            if not usable:
                                logger.debug("分布场 solute 本帧无溶质，退到 pH 场")
                                field = self.coupled_sim.get_2d_field('ph')
                                usable = (field is not None and field.size > 0
                                          and np.any(np.isfinite(field)))
                        else:
                            usable = has_finite
                        if usable:
                            self.concentration_updated.emit(
                                np.asarray(field, dtype=np.float64))
                    except Exception as e:
                        logger.error(f"获取化学浓度分布失败: {e}")
                        logger.error(traceback.format_exc())
                
                # 获取并发送pH数据用于颜色渲染
                if hasattr(self.coupled_sim, 'get_particle_ph'):
                    try:
                        # 【修复 b6】get_particle_ph() 不再接受"组分列号"参数：
                        # pH 只能来自 PHREEQC 的 SelectedOutput，不是某一列浓度。
                        # 拿不到时返回 None，界面按"无 pH 数据"处理，
                        # 绝不再用 -log10(元素H总量) 编一个数出来。
                        #
                        # 【决策 D9 —— 用户裁决：按粒子画「无数据」，其余照画】
                        # 原逻辑是"数组里只要有**任何一个** NaN 就整帧不发"，
                        # 并打 logger.warning("pH数据包含无效值，跳过发送")。
                        # run10 实测 575 帧里 297 帧（52%）因此完全没有 pH 更新，
                        # 而这些 NaN 恰恰是 get_particle_ph 按 _chem_mapped_mask
                        # 诚实标出的"该粒子本步没参与化学反应"（见
                        # coupled_simulation.py :1125）。也就是说：数据没错，
                        # 是显示层把"有一部分没数据"误判成"整帧不可用"。
                        # 现在：NaN = 该粒子无数据 → 原样发给渲染器画「无数据」色；
                        # ±inf 不是合法 pH，统一并入"无数据"（NaN），绝不留在数组里
                        # 污染 min/max；整帧丢弃路径删除。
                        ph_raw = self.coupled_sim.get_particle_ph()
                        if ph_raw is None or ph_raw.size == 0:
                            # 没有粒子 / SelectedOutput 不可用：如实记账，不发帧。
                            # 注意这不是"丢弃无效数据"，是"根本没有数据"。
                            self.ph_accounting["frames_no_data_at_all"] += 1
                            logger.debug("本帧无 pH 数据（还没有粒子，或 pH 列不可用）")
                        else:
                            ph_data = np.array(ph_raw, dtype=np.float64, copy=True)
                            inf_mask = np.isinf(ph_data)
                            n_inf = int(np.count_nonzero(inf_mask))
                            if n_inf:
                                ph_data[inf_mask] = np.nan
                                self.ph_accounting["inf_as_no_data"] += n_inf
                            n_no_data = int(np.count_nonzero(np.isnan(ph_data)))
                            n_ok = int(ph_data.size) - n_no_data
                            self.ph_accounting["frames_emitted"] += 1
                            self.ph_accounting["values_ok"] += n_ok
                            self.ph_accounting["values_no_data"] += n_no_data
                            self.ph_data_updated.emit(ph_data)
                            logger.debug(
                                "发送pH数据，长度=%d, 有值=%d, 无数据=%d",
                                ph_data.size, n_ok, n_no_data)
                    except Exception as e:
                        self.ph_accounting["fetch_failed"] += 1
                        logger.error(f"获取pH数据失败: {e}")
                
                # 检查数据同步状态
                if hasattr(self.main_window, 'mesh_system'):
                    mesh_particles = self.main_window.mesh_system.current_particle_num
                    # 如果数据不同步，以物理层为准
                    if mesh_particles != stats['n_particles']:
                        stats['n_particles'] = mesh_particles
                
                # 每帧一条：端到端验收驱动按此行统计"仿真状态帧"，用于
                # 校验 D9 的 pH 账目闭合。每帧仅 1 行（约 25 行/秒），
                # 不是原先每帧上百条的诊断洪泛。
                print("准备发送仿真状态")
                try:
                    self.simulation_updated.emit(stats)
                except Exception as e:
                    print(f"仿真状态发送失败: {e}")
                    # 【收尾 4 —— 本轮实测驱动】这里原来有一句函数内
                    # `import traceback`。Python 把函数体内任何位置的赋值/导入
                    # 都当作整个函数的局部名，于是外层 except 里
                    # `print(traceback.format_exc())`（本函数 :579 之后那句）
                    # 在局部 import 之前执行时会抛
                    # UnboundLocalError: cannot access local variable 'traceback'，
                    # **把真正的异常（本例：粒子身份与位置长度不一致）整条盖掉**。
                    # 模块级 :23 早已 `import traceback`，删掉这句局部 import 即可。
                    print(traceback.format_exc())
                    # 尝试发送一个简化的状态
                    safe_stats = {
                        "frame": int(stats.get("frame", 0)),
                        "chemical_step": bool(stats.get("chemical_step", False)),
                        "n_particles": int(stats.get("n_particles", 0)),
                        "chem_time": float(stats.get("chem_time", 0.0))
                    }
                    print(f"尝试发送安全状态: {safe_stats}")
                    try:
                        self.simulation_updated.emit(safe_stats)
                        print("安全状态发送成功")
                    except Exception as e2:
                        print(f"安全状态也发送失败: {e2}")
                        # 最后尝试发送空字典
                        self.simulation_updated.emit({})
            
        except Exception as e:
            print(traceback.format_exc())
            logger.error(f"仿真步骤执行错误: {e}")
            self.error_occurred.emit(f"仿真错误: {str(e)}")
            self.stop_simulation()
    
    @Slot()
    def pause_simulation(self):
        """暂停/继续仿真（在 worker 线程执行，定时器也归本线程）"""
        self.is_paused = not self.is_paused
        if self.simulation_timer is not None:
            if self.is_paused:
                self.simulation_timer.stop()
            else:
                self.simulation_timer.start(int(self.frame_delay * 1000 / self.speed_factor))

    @Slot()
    def stop_simulation(self):
        """停止仿真（在 worker 线程执行）"""
        self.is_running = False
        self.is_paused = False

        # 停止定时器。不能 del：本方法可能由 _simulation_step 的 except 分支
        # 直接调用，删掉属性后 start_simulation 还得重建；保留对象更简单。
        if self.simulation_timer is not None:
            self.simulation_timer.stop()

    @Slot(float)
    def set_speed(self, speed: float):
        """设置仿真速度（在 worker 线程执行，改的是本线程定时器的间隔）"""
        self.speed_factor = max(0.1, min(3.0, speed))
        if self.simulation_timer is not None and self.simulation_timer.isActive():
            self.simulation_timer.setInterval(int(self.frame_delay * 1000 / self.speed_factor))

    @Slot(int)
    def set_physics_substeps(self, n: int):
        """设置每 UI 帧的物理子步数（worker 线程执行，下一 tick 即生效）。"""
        self.physics_substeps = max(1, min(8, int(n)))

    @Slot(str)
    def set_field_kind(self, kind: str):
        """设置分布图取哪个量（worker 线程执行，下一 tick 即生效）。

        切换场景只改"之后画什么"，不动化学状态、不重建 PhreeqcRM。
        """
        kind = (kind or "").strip()
        if not kind:
            return
        self.field_kind = kind

    def cleanup(self):
        """清理资源。必须在本对象所属线程调用（主窗口用 BlockingQueuedConnection）。"""
        self.stop_simulation()
        if self.initialization_timer is not None:
            self.initialization_timer.stop()

    @Slot(object, int, int)
    def add_particles_request(self, position, fluid_type: int, count: int):
        """【竞态修复】在主线程（鼠标事件）只算世界坐标，真正的粒子生成在这里做。

        物理步（_simulation_step）与化学步都在 worker 线程；此槽同样由
        worker 线程的排队事件调用，因此和物理/化学串行执行，不会出现
        "读 ids(n) 之后、读 positions(n+8) 之前被加上一批粒子"的撕裂。
        """
        try:
            if self.main_window is None or not hasattr(self.main_window, 'mesh_system'):
                raise RuntimeError("网格系统未初始化，无法添加粒子")
            mesh_system = self.main_window.mesh_system

            # 决策 D4：化学系统必须先行就绪，不允许"只出粒子不出化学"
            if self.coupled_sim is None:
                raise RuntimeError("化学耦合系统未初始化，拒绝只生成没有化学的粒子")

            if int(fluid_type) not in WATER_COMPOSITIONS:
                raise RuntimeError(
                    f"未知的流体类型: {fluid_type!r}（合法值 "
                    f"{sorted(WATER_COMPOSITIONS)}，见 phreeqc_config.WATER_COMPOSITIONS）")

            count = int(count)
            if count < 1:
                raise RuntimeError(f"粒子数量必须 >= 1（收到 {count}）")
            if count > self.SPAWN_MAX_COUNT:
                raise RuntimeError(
                    f"单次最多生成 {self.SPAWN_MAX_COUNT} 个粒子（收到 {count}）；"
                    "超过这个量实测会把水打飞，需要更多请分多次点击")
            capacity = int(mesh_system.capacity())
            active = int(mesh_system.current_particle_num)
            if active + count > capacity:
                raise RuntimeError(
                    f"引擎容量不足：现有 {active} + 本次 {count} > 上限 {capacity}")

            anchor = np.asarray(position, dtype=np.float64)

            # 大批量分块"灌注"：一次把上万个粒子以静止实心块塞进水里，
            # 等价于瞬移一块水，DFSPH 只能靠压力把它推开。实测 50000 个
            # 一次加：max|v|=16.5 m/s，水柱冲到 1.66 m。现在按"逐层浇在
            # 料堆表面"投递：每块不超过约 SPAWN_CHUNK_TARGET 个粒子且共用
            # 同一足迹，块与块之间跑几帧物理让它落定。总量不变。
            n_xy, _ = self._lattice_plan(count, mesh_system)
            cells = n_xy * n_xy
            layers_per_chunk = max(
                1, int(np.ceil(self.SPAWN_CHUNK_TARGET / float(cells))))
            chunk = min(count, layers_per_chunk * cells)

            particle_count_before = int(mesh_system.current_particle_num)
            total_added = 0
            total_seeded = 0
            try:
                remaining = count
                while remaining > 0:
                    n_now = min(chunk, remaining)
                    positions_array = self._build_lattice_particles(
                        mesh_system, anchor, n_now, n_xy=n_xy)
                    if positions_array.shape[0] != n_now:
                        raise RuntimeError(
                            f"点阵生成数量不符：请求 {n_now}，"
                            f"实际 {positions_array.shape[0]}")
                    if (np.any(np.isnan(positions_array))
                            or np.any(np.isinf(positions_array))):
                        raise RuntimeError("粒子位置包含 NaN/Inf")

                    # 先入物理引擎、拿回稳定 particleId，再按 id 播种化学（决策 D3/D7）
                    new_ids = mesh_system.add_particles(positions_array, fluid_type)
                    if not new_ids:
                        raise RuntimeError("add_particles 没有返回任何 particleId")
                    total_seeded += self.coupled_sim.seed_particle_chemistry(
                        new_ids, fluid_type)
                    total_added += len(new_ids)
                    remaining -= n_now

                    # 【可操作性修复】未开始仿真时也要能看见刚生成的粒子：
                    # 渲染器只认主线程 on_particles_updated() 推送的位置。
                    # 每块推一次，用户能看到"正在灌注"而不是界面一动不动。
                    n_active = int(mesh_system.current_particle_num)
                    if n_active > 0:
                        self.particles_updated.emit(
                            mesh_system.particle_positions[:n_active].copy())

                    if remaining > 0:
                        # add_particles 设了 _skip_physics_count=1：
                        # 第一次 step 只做数据同步，实际物理步 = 次数 - 1。
                        for _ in range(self.SPAWN_SETTLE_STEPS):
                            mesh_system.step_simulation()
                        self.status_message.emit(
                            f"正在灌注 {count - remaining}/{count} 个粒子…")
            except Exception:
                if total_added:
                    # 部分成功必须说清楚：引擎里已经有这些粒子了
                    self.status_message.emit(
                        f"灌注中断：已加入 {total_added}/{count} 个粒子")
                raise

            particle_count_after = int(mesh_system.current_particle_num)
            print(f"[OK] 流体类型 {fluid_type}: 添加 {total_added} 个粒子，"
                  f"粒子数 {particle_count_before} -> {particle_count_after}，"
                  f"化学播种 {total_seeded} 个，分块 {chunk} 个/批")
            self.status_message.emit(
                f"已生成 {total_added} 个粒子（总粒子数 {particle_count_after}）")
        except Exception as e:
            msg = f"生成粒子失败：{e}"
            print(f"[FAIL] {msg}")
            traceback.print_exc()
            self.status_message.emit(msg)

    # ------------------------------------------------------------------
    # 粒子生成：指定数量的近立方点阵 + 就地堆叠
    # ------------------------------------------------------------------
    # 间距 = PARTICLE_SPACING = 2r（supportRadius 的一半，与 FluidBlock 采样
    # 密度一致），N 个粒子按最接近立方体的方式排布，例如 8 -> 2x2x2、
    # 1000 -> 10x10x10。空场第一次落点离地板 SPAWN_CLEARANCE*r。
    #
    # 为什么不是贴地 1r：Bender2019 的边界体积图只在 0 < SDF < h 的壳层里
    # 给密度贡献体积，1r 出生时首帧密度虚高数倍，DFSPH 压力把整片水打成
    # 喷泉。实测 13500 粒子、底面贴地出生：max|v|=16.5 m/s、maxY 顶到 2.39 m
    # 的盒盖、3000 帧里 2294 帧超速；底面抬到 h=4r 后 maxY 0.19 m。
    SPAWN_CLEARANCE = 4.0    # 出生底面离地板的最小高度（以半径为单位的 h）

    # 侧墙安全间距。实测（Taichi/Akinci2012 边界粒子，+x 墙层在 x=0.78，
    # 静止密度 1000、容器中心孤立粒子 254）出生密度随"到墙粒子层的距离"
    # 变化：1r→2768、2r→1273、3r→592、4r→510。旧 clip 值 bmax-radius 恰好
    # 是 1r，而空场点击的落点就在前墙面上，于是点阵外沿压在墙粒子层上：
    # 实测 8 个粒子 spawn rho_max=3747、第一步 max|v|=738 m/s，整团飞出容器。
    # Taichi 墙块厚 2r、墙粒子层在 AABB 面内缩 2r，所以从面算起要留 2r+3r。
    SPAWN_WALL_CLEARANCE = 3.0   # 到墙粒子层的最小距离（以半径为单位）
    SPAWN_WALL_INSET = 2.0       # 墙粒子层相对容器 AABB 面的内缩（以半径为单位）

    # 每块灌注的粒子数上限（按整层取整，实际值可能略大）。
    # 单块越小越温和；块间跑 SPAWN_SETTLE_STEPS 帧物理让它落定。
    SPAWN_CHUNK_TARGET = 2000
    SPAWN_SETTLE_STEPS = 20
    # 单次请求的粒子数上限。实测（真实 MeshSystem + 真 add_particles 路径）：
    #   20000 个一次灌注 = 1~2 s，水面最高被顶到 0.4 m，没有粒子出界；
    #   30000~40000 个仍有局部 3~4×密度0 的瞬态；
    #   50000 个会在静置几百帧后失控（max|v|=55 m/s、粒子顶到 2.4 m 盒盖、
    #   有粒子跑出容器）。这个上限取在实测安全区的上沿，再多请分多次点击。
    SPAWN_MAX_COUNT = 20000

    def _spawn_wall_margin(self) -> float:
        """粒子中心到容器 AABB 面的最小距离（米），点阵足迹整体不得越界。

        按 Taichi 墙粒子层的真实位置取（面内缩 2r）再加 3r 安全距离。
        CPU 后端的墙面就在 AABB 面上，这个值只是更保守一些。
        """
        radius = float(CONFIG.PARTICLE_RADIUS)
        return (self.SPAWN_WALL_INSET + self.SPAWN_WALL_CLEARANCE) * radius

    def _lattice_plan(self, count: int, mesh_system) -> Tuple[int, float]:
        """返回正方点阵的每层边长 n_xy 与半宽 half（米）。

        放不进容器时先摊平（层数换面积），再放不进就明确失败。
        大批量灌注需要整批共用的足迹，所以单独抽出来。
        """
        spacing = float(getattr(mesh_system, 'particle_spacing',
                                CONFIG.PARTICLE_SPACING))
        margin = self._spawn_wall_margin()
        bmin = np.asarray(mesh_system.boundary_min, dtype=np.float64)
        bmax = np.asarray(mesh_system.boundary_max, dtype=np.float64)
        count = int(count)

        n_xy = max(1, int(np.ceil(count ** (1.0 / 3.0))))
        while n_xy ** 3 < count:
            n_xy += 1
        half = 0.5 * (n_xy - 1) * spacing

        max_half_x = 0.5 * (bmax[0] - bmin[0]) - margin
        max_half_z = 0.5 * (bmax[2] - bmin[2]) - margin
        if half > max_half_x or half > max_half_z:
            n_xy = int(np.ceil(np.sqrt(count)))
            half = 0.5 * (n_xy - 1) * spacing
            if half > max_half_x or half > max_half_z:
                raise RuntimeError(
                    f"{count} 个粒子放不进容器：单层需要 "
                    f"{2 * half:.2f}×{2 * half:.2f} m，扣除侧墙安全间距后"
                    f"只有 {2 * max_half_x:.2f}×{2 * max_half_z:.2f} m"
                    f"（容器 {bmax[0] - bmin[0]:.2f}×"
                    f"{bmax[2] - bmin[2]:.2f} m）")
        return n_xy, half

    def _build_lattice_particles(self, mesh_system, anchor: np.ndarray,
                                 count: int,
                                 n_xy: Optional[int] = None) -> np.ndarray:
        """在 anchor 处生成恰好 count 个粒子（间距 2r 的点阵），返回坐标。

        落点规则：
        - 底面高度不低于 bmin + 4r（见 SPAWN_CLEARANCE）。
        - x/z 足迹整体离侧墙至少 SPAWN_WALL_INSET+SPAWN_WALL_CLEARANCE 个半径
          （见 _spawn_wall_margin）：空场点击的落点就在墙面上，贴墙出生会与
          边界粒子重叠，首帧密度暴涨后被压力弹飞。
        - 只认"本批足迹内"的料堆顶。以前回看半宽 + 1 个间距，会把刚生成、
          还在下落的那一批也算成料堆：连续快速点击时出生点一次比一次高
          （实测 25 次点击爬到 0.46 m），最后几批从高处砸进水里，水面
          连续喷出粒子。
        - 出生前检查与现有粒子的最小间距，必要时整批上抬。完全重合的两个
          粒子会让核函数除零（NaN），邻域搜索随后卡死。

        n_xy 由调用方给出时（大批量灌注），本批沿用整批的足迹，
        只改变层数，这样每块都是一张薄板而不是一个独立的立方体。
        """
        spacing = float(getattr(mesh_system, 'particle_spacing',
                                CONFIG.PARTICLE_SPACING))
        radius = float(CONFIG.PARTICLE_RADIUS)
        bmin = np.asarray(mesh_system.boundary_min, dtype=np.float64)
        bmax = np.asarray(mesh_system.boundary_max, dtype=np.float64)
        count = int(count)

        if n_xy is None:
            n_xy, _ = self._lattice_plan(count, mesh_system)
        n_layers = int(np.ceil(count / float(n_xy * n_xy)))
        half = 0.5 * (n_xy - 1) * spacing
        margin = self._spawn_wall_margin()

        cx = float(np.clip(anchor[0], bmin[0] + margin + half,
                           bmax[0] - margin - half))
        cz = float(np.clip(anchor[2], bmin[2] + margin + half,
                           bmax[2] - margin - half))

        base_y = float(bmin[1] + self.SPAWN_CLEARANCE * radius)
        max_base = float(bmax[1] - radius - (n_layers - 1) * spacing)
        if base_y > max_base:
            raise RuntimeError(
                f"{count} 个粒子叠起来 {n_layers} 层，超出容器高度")

        n = int(getattr(mesh_system, 'current_particle_num', 0))
        pos = mesh_system.particle_positions[:n] if n > 0 else None
        if pos is not None:
            # 引擎缓冲是 float32；只取 x/z 做足迹筛选、y 做最大值，
            # 不整块转 float64（1M 粒子时要多分配 24 MB）。
            px = pos[:, 0].astype(np.float64)
            pz = pos[:, 2].astype(np.float64)
            py = pos[:, 1].astype(np.float64)
            near = np.abs(px - cx) <= half + radius
            near &= np.abs(pz - cz) <= half + radius
            near &= py > bmin[1] - 0.5
            if np.any(near):
                base_y = max(base_y, float(py[near].max()) + spacing)
                if base_y > max_base:
                    raise RuntimeError(
                        f"料堆顶已经到 {base_y:.2f} m，这批 {count} 个粒子"
                        f"（{n_layers} 层）放不下")

            # 与现有粒子的重叠规避：本批体积内出现任何粒子就把整批上抬。
            for _ in range(12):
                win = np.abs(px - cx) <= half + spacing
                win &= np.abs(pz - cz) <= half + spacing
                win &= py >= base_y - 0.75 * spacing
                win &= py <= base_y + (n_layers - 1) * spacing + 0.75 * spacing
                if not np.any(win):
                    break
                base_y += 0.5 * spacing
                if base_y > max_base:
                    raise RuntimeError(
                        f"{count} 个粒子在该位置与已有粒子重叠，"
                        "上抬后超出容器高度")

        # y 轴最慢 → 截断 count 时先填满每层，最后一层留残缺。
        offsets = np.arange(n_xy) * spacing - half
        gy, gx, gz = np.meshgrid(
            np.arange(n_layers) * spacing, offsets, offsets, indexing='ij')
        lattice = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)[:count]
        lattice[:, 0] += cx
        lattice[:, 1] += base_y
        lattice[:, 2] += cz

        # 兜底：上面的 cx/cz 已保证整块足迹在安全区内，这里只防浮点溢出。
        # 阈值必须与生成边界一致，否则会把粒子推回墙粒子层上。
        lattice[:, 0] = np.clip(lattice[:, 0], bmin[0] + margin, bmax[0] - margin)
        lattice[:, 1] = np.clip(lattice[:, 1], bmin[1] + radius, bmax[1] - radius)
        lattice[:, 2] = np.clip(lattice[:, 2], bmin[2] + margin, bmax[2] - margin)
        return lattice.astype(np.float32)

class CoupledSimulationMainWindow(QMainWindow):
    """化学-物理耦合仿真主窗口"""

    # 【竞态修复】worker 的定时器/粒子生成只在 worker 线程执行，
    # 主线程一律通过排队信号请求，不再直接调 worker 方法。
    worker_start_requested = Signal()
    worker_pause_requested = Signal()
    worker_stop_requested = Signal()
    worker_speed_requested = Signal(float)
    worker_substeps_requested = Signal(int)
    worker_field_kind_requested = Signal(str)
    worker_cleanup_requested = Signal()

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
        self._physics_substeps = 3
        self._debug_defaults = []
        
        # PHREEQCRM配置
        self.rm = None  # PHREEQCRM实例
        self.chem_initialized = False
        
        self.init_ui()
        # 【顺序修复】化学系统必须先于仿真线程就绪。
        # setup_simulation_thread() 会立刻启动 worker 线程，worker 的初始化
        # 定时器 100ms 后就会读 self.rm 去构造 CoupledSimulation。
        # 若化学初始化更慢（磁盘/杀软/低配机器），worker 读到的 rm 还是 None，
        # CoupledSimulation 就被建成"无 rm"状态：左键生成粒子在播种化学时
        # 直接失败，点"开始仿真"又会走 lazy 路径去造第二个 PhreeqcRM 被守卫拦下。
        # 先把 rm 建好，worker 线程启动时读到的必然是可用的实例。
        self.initialize_chemistry_system()
        self.setup_simulation_thread()

        # 【性能】渲染与物理解耦。原来 paintGL 只由 worker 的 particles_updated
        # 信号触发，而 10k 粒子时一个 worker tick 实测 ~160ms（化学 ~95ms 串行
        # + 物理 3×~20ms），界面帧率被仿真帧率绑架到 ~6fps。这里用独立的 60fps
        # 定时器驱动重绘，worker 信号只负责更新 VBO 数据；仿真慢时界面依然流畅。
        # 注意：粒子运动的平滑度仍受物理帧率限制，本改动解决的是 UI/相机卡顿。
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(16)
        self.render_timer.timeout.connect(self._render_tick)
        
    def create_scenarios(self) -> List[ChemicalScenario]:
        """创建预定义化学反应场景。

        每个场景都绑定到 phreeqc_config.WATER_COMPOSITIONS 里的真实液体类型，
        数值来自 _probe_chem_species10 的实测（生产路径 CreateMapping+RunCells）：
          1 中性水   pH 6.997 / pe 13.94
          2 酸性 HCl pH 3.014 / pe 17.93
          3 碱性 NaOH pH 10.98 / pe 9.96
          4 碳酸水   pH 4.680，碳形态 CO2 9.8e-4（占 98%）
          5 碳酸氢钠 pH 8.269，碳形态 HCO3- 9.8e-4（占 98%）
          6 亚铁水   pH 6.291 / pe -1.96，Fe 全部是 Fe2+（无沉淀）
          7 曝气水   pH 6.997 / pe 13.95，O2 5.5e-3
        6+7 混合实测 pH 3.61 / pe 17.26：Fe2+ 被氧化成 Fe3+，溶解态铁
        从 5e-4 降到 ~1e-5（其余水解成 Fe(OH)3 沉淀）。
        """
        return [
            ChemicalScenario(
                name="酸碱中和",
                description="HCl + NaOH → NaCl + H₂O\n"
                            "Ctrl+滚轮在 水/酸性/碱性 之间切换；"
                            "两种水在同一化学单元里相遇即按电荷平衡重新计算 pH。",
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
                chemical_components=["H+", "OH-", "Na+", "Cl-"],
                fluid_types=[1, 2, 3],
                default_fluid_type=2,
                field_kind="ph",
            ),
            ChemicalScenario(
                name="碳酸盐平衡",
                description="CO₂ + H₂O ⇌ H₂CO₃ ⇌ H⁺ + HCO₃⁻ ⇌ 2H⁺ + CO₃²⁻\n"
                            "碳酸水(4)：pH≈4.68，98% 碳以 CO₂ 存在；\n"
                            "碳酸氢钠水(5)：pH≈8.27，98% 碳以 HCO₃⁻ 存在。\n"
                            "两者等量混合实测 pH≈6.34，碳形态按平衡重新分配成\n"
                            "CO₂:HCO₃⁻ ≈ 50:50。分布图画 pH：酸区与碱区之间出现过渡带。",
                initial_conditions={
                    "co2_concentration": 1e-3,
                    "water_ph": 7.0
                },
                parameters={
                    "equilibrium_constant": 4.3e-7,
                    "carbonic_acid_k1": 4.3e-7,
                    "bicarbonate_k2": 5.6e-11
                },
                chemical_components=["CO₂", "H₂CO₃", "H+", "HCO₃⁻", "CO₃²⁻"],
                fluid_types=[4, 5],
                default_fluid_type=4,
                field_kind="ph",
            ),
            ChemicalScenario(
                name="氧化还原反应",
                description="4Fe²⁺ + O₂ + 10H₂O → 4Fe(OH)₃↓ + 8H⁺\n"
                            "亚铁水(6)：pH≈6.29 / pe≈-2.12，Fe 全为 Fe²⁺；\n"
                            "曝气水(7)：pH≈7.00 / pe≈+14.0，含溶解氧 O₂≈1.0e-2。\n"
                            "混合后实测 pH≈3.61 / pe≈+17.3，Fe²⁺ 从 1e-3 降到\n"
                            "≈4e-10（氧化成 Fe³⁺ 后水解沉淀）。\n"
                            "分布图画「溶解态 Fe²⁺」：亚铁水那侧有颜色，被氧化处归零。\n"
                            "注意组分总量 Fe 不会降（沉淀仍计入总量），别拿它当指标。",
                initial_conditions={
                    "fe2_concentration": 1e-3,
                    "o2_concentration": 1.0e-2
                },
                parameters={
                    "redox_potential": 0.77,   # Fe3+/Fe2+ 标准电极电位 (V)
                    "reaction_order": 1.0
                },
                chemical_components=["Fe²⁺", "Fe³⁺", "O₂", "Fe(OH)₃"],
                fluid_types=[6, 7],
                default_fluid_type=6,
                field_kind="Fe+2",
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
        
        # 实时调试面板（可拖动/关闭，不影响主布局）
        if MODULAR_MESH_AVAILABLE:
            self.create_debug_dock()

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("系统就绪 - 请选择化学反应场景")

        # 【顺序修复】create_scenario_tab 里已经调过一次 on_scenario_changed，
        # 但那时右侧面板（opengl_widget / fluid_desc_layout）还没建，
        # 液体白名单和分布图标题都没落到位。这里在两侧面板都建好之后
        # 按当前场景再应用一次（幂等：只改"之后生成什么水/画什么量"）。
        combo = getattr(self, 'scenario_combo', None)
        if combo is not None:
            self.on_scenario_changed(combo.currentText())
    
    def create_debug_dock(self):
        """实时调试面板：滑块即改即生效，用来现场微调水的外观与流速。

        做成 QDockWidget 而不是 tab：调参要一边看水一边拖，dock 能浮在
        3D 视口旁边，关掉也不影响主界面。渲染参数下一帧生效；物理子步
        通过排队信号发给 worker，下一 tick 生效。
        """
        dock = QDockWidget("实时调试（粒子 / 水面）", self)
        dock.setObjectName("realtime_debug_dock")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hint = QLabel("拖动即生效；「恢复默认」一键回到初始值。"
                      "觉得水太稠：调小「表面平滑」和「粒子显示半径」，"
                      "调大「物理子步/帧」。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9e9e9e; font-size:10px;")
        layout.addWidget(hint)

        render_group = QGroupBox("渲染 / 水面")
        rg = QVBoxLayout(render_group)
        rg.setSpacing(4)
        self._debug_slider(
            rg, "粒子显示半径 (m)", 0.001, 0.20,
            CONFIG.SSFR_PARTICLE_RADIUS_WORLD, 0.001,
            self._render_setter("ssfr_particle_radius_world"))
        self._debug_slider(
            rg, "表面平滑 σxy", 0.5, 16.0, CONFIG.SSFR_BLUR_SIGMA, 0.1,
            self._render_setter("ssfr_blur_sigma"))
        self._debug_slider(
            rg, "深度平滑 σz (m)", 0.005, 0.60, CONFIG.SSFR_DEPTH_SIGMA, 0.005,
            self._render_setter("ssfr_depth_sigma"))
        self._debug_slider(
            rg, "平滑迭代次数", 1, 8, len(CONFIG.SSFR_FILTER_INTERVALS), 1,
            self._filter_iterations_setter(), "{:.0f}")
        self._debug_slider(
            rg, "水面基础透明度", 0.0, 0.30, CONFIG.ALPHA_BASE, 0.005,
            self._config_setter("ALPHA_BASE"))
        self._debug_slider(
            rg, "pH 染色强度", 0.0, 1.0, 0.35, 0.01,
            self._render_setter("particle_tint"))
        self._debug_slider(
            rg, "焦散强度", 0.0, 1.0, CONFIG.CAUSTIC_STRENGTH, 0.01,
            self._config_setter("CAUSTIC_STRENGTH"))
        self._debug_slider(
            rg, "高光锐度", 20, 300, CONFIG.SPEC_POWER, 1,
            self._config_setter("SPEC_POWER"), "{:.0f}")
        self._debug_slider(
            rg, "菲涅尔幂次", 1.0, 16.0, CONFIG.FRESNEL_POWER, 0.1,
            self._config_setter("FRESNEL_POWER"))
        layout.addWidget(render_group)

        physics_group = QGroupBox("物理 / 流速")
        pg = QVBoxLayout(physics_group)
        pg.setSpacing(4)
        self._debug_slider(
            pg, "物理子步/帧", 1, 8, self._physics_substeps, 1,
            self._set_physics_substeps, "{:.0f}")
        layout.addWidget(physics_group)

        reset_btn = QPushButton("恢复默认")
        reset_btn.clicked.connect(self._reset_debug_sliders)
        layout.addWidget(reset_btn)
        layout.addStretch(1)

        scroll.setWidget(inner)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._debug_dock = dock
        return dock

    def _debug_slider(self, layout, title, minimum, maximum, default, step,
                      on_change, fmt="{:.3f}"):
        """一行调试滑块：标题 + 当前值 + 滑块，valueChanged 直接回调。"""
        row = QWidget()
        vbox = QVBoxLayout(row)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(1)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        name = QLabel(title)
        name.setStyleSheet("color:#dddddd; font-size:10px;")
        value_label = QLabel(fmt.format(default))
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setStyleSheet("color:#4fc3f7; font-size:10px;")
        head.addWidget(name)
        head.addStretch(1)
        head.addWidget(value_label)

        scale = max(1, int(round(1.0 / step)))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(round(minimum * scale)), int(round(maximum * scale)))
        slider.setPageStep(1)
        slider.setValue(int(round(default * scale)))

        def _changed(iv):
            v = iv / float(scale)
            value_label.setText(fmt.format(v))
            try:
                on_change(v)
            except Exception as e:
                logger.warning(f"调试滑块「{title}」应用失败: {e}")

        slider.valueChanged.connect(_changed)
        vbox.addLayout(head)
        vbox.addWidget(slider)
        layout.addWidget(row)
        self._debug_defaults.append((slider, int(round(default * scale))))
        return slider

    def _render_setter(self, attr):
        def setter(v):
            if getattr(self, 'opengl_renderer', None) is not None:
                setattr(self.opengl_renderer, attr, v)
                if getattr(self, 'opengl_widget', None) is not None:
                    self.opengl_widget.update()
        return setter

    def _config_setter(self, attr):
        def setter(v):
            setattr(CONFIG, attr, v)
            if getattr(self, 'opengl_widget', None) is not None:
                self.opengl_widget.update()
        return setter

    def _filter_iterations_setter(self):
        def setter(v):
            n = max(1, min(8, int(round(v))))
            intervals = [1, 1, 2, 2, 4, 4, 8, 8][:n]
            if getattr(self, 'opengl_renderer', None) is not None:
                self.opengl_renderer.ssfr_filter_intervals = intervals
                if getattr(self, 'opengl_widget', None) is not None:
                    self.opengl_widget.update()
        return setter

    def _set_physics_substeps(self, v):
        self._physics_substeps = max(1, min(8, int(round(v))))
        if self.simulation_worker is not None:
            self.worker_substeps_requested.emit(self._physics_substeps)

    def _reset_debug_sliders(self):
        for slider, default in self._debug_defaults:
            slider.setValue(default)

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
        # 仿真系统初始化完成（on_simulation_initialized(True)）后才允许开始，
        # 避免在 rm / CoupledSimulation 尚未就绪时点击进入半初始化状态。
        self.start_btn.setEnabled(False)
        
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
        # 值 ÷ 10 = 播放倍数：1 → 0.1x（最左），10 → 1.0x，30 → 3.0x。
        # 单步/翻页都设成 1 格 = 0.1x，否则默认 pageStep=10 时，
        # 点滑槽或按 PageUp/PageDown 只会跳到 1.0x/2.0x，永远落不到 0.1x。
        self.speed_slider.setRange(1, 30)
        self.speed_slider.setSingleStep(1)
        self.speed_slider.setPageStep(1)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(5)
        self.speed_slider.setValue(10)  # 1.0x default
        self.speed_slider.setToolTip("0.1x（最左）～ 3.0x（最右），每格 0.1x")
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
        
        # 摄像机控制
        camera_group = QGroupBox("摄像机控制")
        camera_layout = QGridLayout()
        
        camera_layout.addWidget(QLabel("水平角度:"), 0, 0)
        self.camera_angle_x_label = QLabel("0.00")
        camera_layout.addWidget(self.camera_angle_x_label, 0, 1)
        
        camera_layout.addWidget(QLabel("垂直角度:"), 0, 2)
        self.camera_angle_y_label = QLabel("0.00")
        camera_layout.addWidget(self.camera_angle_y_label, 0, 3)
        
        camera_layout.addWidget(QLabel("距离:"), 1, 0)
        self.camera_distance_label = QLabel("1.00")
        camera_layout.addWidget(self.camera_distance_label, 1, 1)
        
        # 重置摄像机按钮
        self.reset_camera_btn = QPushButton("重置摄像机")
        self.reset_camera_btn.clicked.connect(self.reset_camera_view)
        camera_layout.addWidget(self.reset_camera_btn, 1, 2, 1, 2)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # 粒子生成控制
        particle_group = QGroupBox("粒子生成控制")
        particle_layout = QGridLayout()
        
        particle_layout.addWidget(QLabel("当前液体类型:"), 0, 0)
        self.fluid_type_label = QLabel("1 (水)")
        self.fluid_type_label.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        particle_layout.addWidget(self.fluid_type_label, 0, 1)
        
        # 液体类型说明
        type_info = QLabel("鼠标左键: 生成粒子\nCtrl+滚轮: 切换液体类型\n右键拖拽: 旋转视角")
        type_info.setStyleSheet("color: #aaaaaa; font-size: 9px;")
        particle_layout.addWidget(type_info, 1, 0, 1, 2)
        
        # 液体类型详细说明 —— 内容随场景白名单重建（见 _rebuild_fluid_desc_labels）。
        # 用独立的 QVBoxLayout 承载说明行，切换场景时整块替换。
        fluid_desc_host = QWidget()
        self.fluid_desc_layout = QVBoxLayout(fluid_desc_host)
        self.fluid_desc_layout.setContentsMargins(0, 0, 0, 0)
        self.fluid_desc_layout.setSpacing(2)
        particle_layout.addWidget(fluid_desc_host, 2, 0, 1, 2)
        self.fluid_desc_labels = []
        
        # 每次点击放入的粒子数量（间距 2r 的近立方点阵）
        particle_layout.addWidget(QLabel("每次生成粒子数:"), 6, 0)
        self.particle_count_spin = QSpinBox()
        self.particle_count_spin.setRange(1, 20000)
        self.particle_count_spin.setValue(8)      # 默认与旧的 2x2x2 一致
        self.particle_count_spin.setSingleStep(8)
        self.particle_count_spin.setToolTip(
            "鼠标左键单击/按住时，每次放入的粒子个数。\n"
            "粒子按间距 2r 的点阵排布，例如 8=2x2x2、1000=10x10x10。\n"
            "超过约 2000 个会分块逐层灌注（每块间自动跑几帧物理），\n"
            "这样大批量不会像瞬移一块水那样炸开。\n"
            "单次上限 20000：再大的量实测会把水打飞，请分多次点击。")
        particle_layout.addWidget(self.particle_count_spin, 6, 1)

        particle_layout.addWidget(
            QLabel("按住左键会每 50ms 追加一批；大批量自动分块灌注，单次上限 20000。"),
            7, 0, 1, 2)
        
        particle_group.setLayout(particle_layout)
        layout.addWidget(particle_group)
        
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
            self.mesh_system = create_mesh_system()
            self.opengl_renderer = OpenGLRenderer(self.mesh_system)
            
            # 创建OpenGL widget包装器
            class QOpenGLWrapper(QOpenGLWidget):
                # 鼠标线程只负责把"在世界坐标哪里、生成哪种流体、多少个"
                # 投递给 worker 线程。
                particles_requested = Signal(object, int, int)

                def __init__(self, renderer, mesh_system):
                    super().__init__()
                    self.renderer = renderer
                    self.mesh_system = mesh_system  # 直接传入网格系统引用
                    self.ctx = None
                    
                    # 鼠标控制相关属性
                    self.last_pos = None
                    self.is_dragging = False
                    self.drag_button = None
                    
                    # 粒子生成相关属性
                    self.current_fluid_type = 1  # 当前液体类型 (1..7)
                    self.allowed_fluid_types = [1, 2, 3]  # 当前场景允许的液体
                    self.particles_per_click = 8  # 每次生成的粒子数（左侧面板可调）
                    self.is_generating_particles = False
                    self.particle_generation_timer = QTimer()
                    self.particle_generation_timer.timeout.connect(self.continue_particle_generation)
                    self.particle_generation_timer.setInterval(50)  # 50ms间隔持续生成
                    
                    # 设置鼠标跟踪
                    self.setMouseTracking(True)
                
                def set_fluid_types(self, allowed, default=None):
                    """限定本场景可生成哪些液体（场景切换时由主窗口调用）。

                    只影响"之后点击生成什么水"：已有粒子的化学状态保持不动。
                    当前类型不在白名单里时立刻切到默认类型，避免下一次点击
                    生成本场景之外的液体。
                    """
                    allowed = [int(t) for t in allowed if t]
                    if not allowed:
                        return
                    self.allowed_fluid_types = allowed
                    if default is not None:
                        default = int(default)
                        self.current_fluid_type = (
                            default if default in allowed else allowed[0])
                    elif self.current_fluid_type not in allowed:
                        self.current_fluid_type = allowed[0]
                    self.update()

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
                
                def mousePressEvent(self, event):
                    """鼠标按下事件 - 开始拖拽或粒子生成"""
                    if event.button() == Qt.LeftButton:
                        # 左键：开始粒子生成
                        self.is_generating_particles = True
                        self.generate_particles_at_mouse(event.pos())
                        self.particle_generation_timer.start()
                        event.accept()
                    elif event.button() == Qt.RightButton:
                        # 右键：开始摄像机拖拽
                        self.is_dragging = True
                        self.drag_button = event.button()
                        self.last_pos = event.pos()
                        event.accept()
                
                def mouseReleaseEvent(self, event):
                    """鼠标释放事件 - 结束拖拽或粒子生成"""
                    if event.button() == Qt.LeftButton:
                        # 停止粒子生成
                        self.is_generating_particles = False
                        self.particle_generation_timer.stop()
                        event.accept()
                    elif event.button() == self.drag_button:
                        # 结束摄像机拖拽
                        self.is_dragging = False
                        self.drag_button = None
                        self.last_pos = None
                        event.accept()
                
                def mouseMoveEvent(self, event):
                    """鼠标移动事件 - 处理摄像机旋转和粒子生成"""
                    if self.is_dragging and self.last_pos is not None:
                        delta = event.pos() - self.last_pos
                        
                        # 根据拖拽方向调整摄像机角度
                        sensitivity = 0.01  # 敏感度调节
                        
                        if self.drag_button == Qt.RightButton:
                            # 右键拖拽：围绕注视点旋转摄像机
                            self.renderer.camera_angle_x += delta.x() * sensitivity
                            self.renderer.camera_angle_y -= delta.y() * sensitivity  # Y轴反向更符合直觉
                            
                            # 限制垂直角度范围，避免万向节锁
                            self.renderer.camera_angle_y = max(-math.pi/2 + 0.1, 
                                                              min(math.pi/2 - 0.1, 
                                                                  self.renderer.camera_angle_y))
                            
                            # 触发重绘
                            self.update()
                        
                        self.last_pos = event.pos()
                        event.accept()
                    elif self.is_generating_particles:
                        # 持续生成粒子
                        self.generate_particles_at_mouse(event.pos())
                        event.accept()
                    else:
                        super().mouseMoveEvent(event)
                
                def paintGL(self):
                    if self.ctx:
                        try:
                            # 检查OpenGL上下文状态（使用更兼容的方式）
                            try:
                                # 尝试访问上下文的基本属性来验证其有效性
                                _ = self.ctx.viewport  # 简单的验证访问
                            except:
                                print("OpenGL上下文无效，跳过渲染")
                                return
                            
                            # 检测当前framebuffer
                            screen_fbo = self.ctx.detect_framebuffer()
                            if screen_fbo is None:
                                print("无法检测到framebuffer，跳过渲染")
                                return
                            
                            screen_fbo.use()
                            
                            # 设置视口
                            size = self.size()
                            if size.width() <= 0 or size.height() <= 0:
                                print("无效的视口尺寸，跳过渲染")
                                return
                            
                            self.ctx.viewport = (0, 0, size.width(), size.height())
                            
                            # 清屏
                            self.ctx.clear(0.1, 0.1, 0.15, 1.0)
                            
                            # 调用渲染器
                            self.renderer.render(size.width(), size.height(), screen_fbo)
                            
                        except Exception as e:
                            print(f"渲染错误: {e}")
                            import traceback
                            traceback.print_exc()
                
                def wheelEvent(self, event):
                    """鼠标滚轮事件 - 缩放摄像机距离和切换液体类型"""
                    delta = event.angleDelta().y()
                    
                    if event.modifiers() & Qt.ControlModifier:
                        # Ctrl+滚轮：在当前场景允许的液体类型里循环切换
                        allowed = list(getattr(self, 'allowed_fluid_types', [1, 2, 3]))
                        try:
                            i = allowed.index(self.current_fluid_type)
                        except ValueError:
                            i = 0
                        if delta > 0:
                            i = (i + 1) % len(allowed)
                        else:
                            i = (i - 1) % len(allowed)
                        self.current_fluid_type = allowed[i]
                        print(f"切换到液体类型 {self.current_fluid_type}")
                    else:
                        # 普通滚轮：缩放摄像机距离
                        zoom_sensitivity = 0.1
                        
                        # 调整摄像机距离
                        if delta > 0:
                            self.renderer.camera_distance *= (1.0 - zoom_sensitivity)
                        else:
                            self.renderer.camera_distance *= (1.0 + zoom_sensitivity)
                        
                        # 限制距离范围：容器 1.6 m 宽、高 2.4 m，
                        # 能看全整个杯子大致需要 2.5~8 m。
                        self.renderer.camera_distance = max(1.2, min(9.0, self.renderer.camera_distance))
                    
                    # 触发重绘
                    self.update()
                    event.accept()
                
                def resizeGL(self, width, height):
                    if self.ctx and width > 0 and height > 0:
                        self.ctx.viewport = (0, 0, width, height)
                
                def generate_particles_at_mouse(self, mouse_pos):
                    """在鼠标位置生成粒子 - 修复版"""
                    try:
                        # 获取OpenGL坐标
                        size = self.size()
                        x = mouse_pos.x()
                        y = mouse_pos.y()  # 不再需要Y轴翻转，screen_to_world中已经处理
                        
                        # 创建投影和视图矩阵
                        projection = self.create_projection_matrix(size.width(), size.height())
                        view = self.create_view_matrix()
                        
                        # 计算世界坐标
                        world_pos = self.screen_to_world(x, y, size.width(), size.height(), 
                                                        projection, view)
                        
                        # 限制生成位置在容器内部
                        # 【修复】原来这里硬写 [-2,0.5,-2] ~ [2,3,2]，与 4x6x4 的
                        # 大箱体配套；容器已改为 1.6x2.4x1.6，必须直接用
                        # mesh_system 的边界，否则点击点会被推到容器外面。
                        bounds_min = np.asarray(self.mesh_system.boundary_min)
                        bounds_max = np.asarray(self.mesh_system.boundary_max)
                        radius = float(CONFIG.PARTICLE_RADIUS)
                        for axis in range(3):
                            world_pos[axis] = np.clip(world_pos[axis],
                                                      bounds_min[axis] + radius,
                                                      bounds_max[axis] - radius)

                        # 生成粒子
                        self.add_fluid_particles(world_pos, self.current_fluid_type)
                        
                    except Exception as e:
                        print(f"粒子生成错误: {e}")
                        import traceback
                        traceback.print_exc()
                
                def continue_particle_generation(self):
                    """持续生成粒子（当鼠标按下时）"""
                    if self.is_generating_particles:
                        cursor_pos = self.mapFromGlobal(self.cursor().pos())
                        if self.rect().contains(cursor_pos):
                            self.generate_particles_at_mouse(cursor_pos)
                
                def create_projection_matrix(self, width, height):
                    """创建投影矩阵 - 修复版"""
                    import numpy as np
                    fov = 60.0
                    aspect = width / height if height > 0 else 1.0
                    near = 0.1
                    far = 100.0
                    
                    f = 1.0 / math.tan(math.radians(fov) / 2.0)
                    nf = 1.0 / (near - far)
                    
                    proj = np.zeros((4, 4), dtype=np.float32)
                    proj[0, 0] = f / aspect
                    proj[1, 1] = f
                    proj[2, 2] = (far + near) * nf
                    proj[3, 2] = -1.0
                    proj[2, 3] = 2.0 * far * near * nf
                    proj[3, 3] = 0.0
                    return proj
                
                def create_view_matrix(self):
                    """创建视图矩阵 - 修复版"""
                    import numpy as np
                    
                    # 计算摄像机位置
                    cam_x = self.renderer.camera_target[0] + self.renderer.camera_distance * math.sin(self.renderer.camera_angle_x) * math.cos(self.renderer.camera_angle_y)
                    cam_y = self.renderer.camera_target[1] + self.renderer.camera_distance * math.sin(self.renderer.camera_angle_y)
                    cam_z = self.renderer.camera_target[2] + self.renderer.camera_distance * math.cos(self.renderer.camera_angle_x) * math.cos(self.renderer.camera_angle_y)
                    # 计算视图矩阵 (LookAt矩阵)
                    eye = np.array([cam_x, cam_y, cam_z])
                    target = np.array(self.renderer.camera_target)
                    up = np.array([0.0, 1.0, 0.0])
                    
                    # 前向向量
                    f = target - eye
                    f = f / np.linalg.norm(f)
                    
                    # 右向量
                    s = np.cross(f, up)
                    s = s / np.linalg.norm(s)
                    
                    # 上向量
                    u = np.cross(s, f)
                    
                    # 构造视图矩阵
                    view = np.eye(4, dtype=np.float32)
                    view[0, :3] = s
                    view[1, :3] = u
                    view[2, :3] = -f
                    view[0, 3] = -np.dot(s, eye)
                    view[1, 3] = -np.dot(u, eye)
                    view[2, 3] = np.dot(f, eye)
                    return view

                def screen_to_world(self, x, y, width, height, projection, view):
                    """屏幕坐标转世界坐标 - 射线法（推荐）"""
                    import numpy as np

                    # 1. 构造射线（从相机出发，穿过屏幕像素）
                    ndc_x = (2.0 * x / width) - 1.0
                    ndc_y = 1.0 - (2.0 * y / height)

                    # 射线起点（近裁剪面）
                    near_point = np.array([ndc_x, ndc_y, -1.0, 1.0])
                    # 射线方向点（远裁剪面）
                    far_point = np.array([ndc_x, ndc_y, 1.0, 1.0])

                    vp_matrix = projection @ view
                    inv_vp = np.linalg.inv(vp_matrix)

                    # 转换到世界空间
                    near_world = (inv_vp @ near_point)
                    far_world = (inv_vp @ far_point)
                    near_world /= near_world[3]
                    far_world /= far_world[3]

                    ray_origin = near_world[:3]
                    ray_dir = far_world[:3] - ray_origin
                    ray_dir = ray_dir / np.linalg.norm(ray_dir)

                    # 2. 有水时直接按鼠标像素处的线性深度反投影到世界点。
                    #    注意 depth 是视图空间深度(-z)，不是射线参数 t：
                    #    t = depth / dot(ray_dir, cam_forward)，否则点击会落偏。
                    depth = self._get_depth_from_texture(x, y, width, height)
                    if depth is not None and 0.0 < depth < 100.0:
                        cam_pos = np.array(calculate_camera_position(
                            self.renderer.camera_distance, self.renderer.camera_angle_x,
                            self.renderer.camera_angle_y, self.renderer.camera_target),
                            dtype=np.float64)
                        cam_fwd = np.asarray(self.renderer.camera_target, dtype=np.float64) - cam_pos
                        cam_fwd /= (np.linalg.norm(cam_fwd) + 1e-12)
                        denom = float(np.dot(ray_dir, cam_fwd))
                        if denom > 1e-6:
                            return ray_origin + ray_dir * (depth / denom)

                    # 3. 空场/没点在水面上：射线与容器内部求交。
                    #    旧逻辑回退成"沿射线走 10 m"，射线早已飞出容器，
                    #    所有点击都被 clip 到同一圈边缘高度 —— 这就是
                    #    "永远是一个平面"的另一半原因。
                    hit = self._ray_box_intersection(
                        ray_origin, ray_dir,
                        np.asarray(self.mesh_system.boundary_min, dtype=np.float64),
                        np.asarray(self.mesh_system.boundary_max, dtype=np.float64))
                    if hit is not None:
                        return hit

                    # 4. 极端视角下射线不与容器相交：退到地板平面
                    if abs(ray_dir[1]) > 1e-6:
                        t = (self.mesh_system.boundary_min[1] - ray_origin[1]) / ray_dir[1]
                        if t > 0.0:
                            return ray_origin + ray_dir * t
                    return ray_origin + ray_dir * 4.0

                def _ray_box_intersection(self, origin, direction, bmin, bmax):
                    """射线 × 轴对齐盒：返回射线进入盒内那一点（无交返回 None）。"""
                    import numpy as np
                    inv = np.where(np.abs(direction) < 1e-9, 1e9, 1.0 / direction)
                    t0 = (bmin - origin) * inv
                    t1 = (bmax - origin) * inv
                    t_near = np.max(np.minimum(t0, t1))
                    t_far = np.min(np.maximum(t0, t1))
                    if t_far < max(t_near, 0.0):
                        return None
                    t = t_near if t_near > 0.0 else t_far
                    return origin + direction * t
                
                def _get_depth_from_texture(self, x, y, width, height):
                    """从深度纹理获取指定像素的深度值"""
                    try:
                        # 确保渲染器已启用深度暴露，并请求本帧回读一次深度
                        if not hasattr(self.renderer, 'expose_depth_buffer') or not self.renderer.expose_depth_buffer:
                            self.renderer.enable_depth_exposure(format='numpy')
                            print("已启用深度缓冲暴露")
                        self.renderer.request_depth_exposure()

                        # 获取最近一次暴露的深度数据
                        depth_data = self.renderer.get_exposed_depth_data()
                        if depth_data and 'data' in depth_data:
                            depth_array = depth_data['data']
                            # 转换屏幕坐标到纹理坐标
                            tex_x = int(x)
                            tex_y = int(y)
                            # 边界检查
                            if (0 <= tex_x < depth_array.shape[1] and 
                                0 <= tex_y < depth_array.shape[0]):
                                depth_value = depth_array[tex_y, tex_x]
                                # 只返回有效的深度值（排除背景）
                                if depth_value > 0:
                                    return depth_value
                        
                        return None
                    except Exception as e:
                        print(f"获取深度值失败: {e}")
                        return None
                
                def set_particle_count(self, n):
                    """左侧面板"每次生成粒子数"的值（主线程内读写）。"""
                    self.particles_per_click = max(1, int(n))

                def add_fluid_particles(self, position, fluid_type):
                    """把"在世界坐标 position 生成一批 fluid_type 粒子"投递给 worker 线程。

                    【竞态修复】以前这里直接在鼠标事件（主线程）里调
                    mesh_system.add_particles() + seed_particle_chemistry()。
                    物理步/化学步跑在 worker 线程，两边无锁读写同一个 MeshSystem：
                    _update_particle_mapping 先读 ids(n)、再读 positions 时，
                    中间落进主线程一批 8 个粒子 → ids/positions 长度不一致，
                    守卫抛错后整场仿真停摆（交接文档第 4 节，中性水约 1/3 概率）。

                    现在本方法只做参数校验 + 发请求；真正的 add/seed 在
                    SimulationWorker.add_particles_request() 里执行 —— 它由
                    worker 线程的排队事件调用，天然与物理/化学步串行。
                    决策 D3（先加后播种、按引擎稳定 id）在 worker 侧保持不变。
                    """
                    if not hasattr(self, 'mesh_system') or self.mesh_system is None:
                        raise RuntimeError("网格系统未初始化，无法添加粒子")
                    if int(fluid_type) not in WATER_COMPOSITIONS:
                        raise RuntimeError(
                            f"未知的流体类型: {fluid_type!r}（合法值 "
                            f"{sorted(WATER_COMPOSITIONS)}，"
                            "见 phreeqc_config.WATER_COMPOSITIONS）")

                    count = int(self.particles_per_click)
                    if count < 1:
                        raise RuntimeError(f"每次生成数量必须 >= 1（收到 {count}）")

                    pos = np.asarray(position, dtype=np.float32).copy()
                    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                        raise RuntimeError("粒子位置包含 NaN/Inf")

                    print(f"投递粒子生成请求: 位置={pos.tolist()} 类型={fluid_type} "
                          f"数量={count}")
                    self.particles_requested.emit(pos, int(fluid_type), count)
                    self.update()

                # 【修复 b1/b4/b5/c3 + 决策 D6/D7】此处原有三个方法：
                #   _initialize_neutral_water_chemistry / _initialize_acidic_chemistry
                #   _initialize_alkaline_chemistry
                # 三份互相拷贝的实现各自硬写一张表，键是"物种名"
                #   {'H+':1.006e-07, 'OH-':1.006e-07, 'H':1.433e-25, 'O':0, 'Charge':0.0}
                # 而 PhreeqcRM 的真实组分名是 H/O/Charge/Cl/Na（rm.GetComponents() 实测）：
                #   - 'H+'/'OH-'/'Cl-'/'Na+' 四个键永远查不中 -> .get(name, 1e-6) 兜底
                #   - 'H' 被写成 1.433e-25（全仓库无出处），'O' 被写成 0
                # 结果喂给 RunCells 的浓度里没有水，PHREEQC 报
                #   "Oxygen Mass of oxygen has not converged" 后进程 exit(127)。
                # 而且它们用 start_pid=end_pid 之前的"行号"当粒子号，还带
                #   if pid < len(...):  # 越界就静默跳过
                # 的假成功分支。现在唯一入口是
                #   CoupledSimulation.seed_particle_chemistry(ids, fluid_type)
                # 数值表统一在 phreeqc_config.WATER_COMPOSITIONS。
            self.opengl_widget = QOpenGLWrapper(self.opengl_renderer, self.mesh_system)
            # 左侧面板在右侧面板之前创建，spinbox 一定已存在。
            # 两边都在主线程：valueChanged 与鼠标事件串行。
            self.opengl_widget.set_particle_count(self.particle_count_spin.value())
            self.particle_count_spin.valueChanged.connect(
                self.opengl_widget.set_particle_count)
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
        """初始化化学系统

        【决策 D4】原来这里 `except Exception: 记一条 error 就完了`，
        self.rm 保持 None、chem_initialized=False，界面照常起来，用户以为在跑
        化学耦合仿真，其实一层化学都没算；更糟的是 worker 随后会走第二条
        路径再造一个 PhreeqcRM（实测半初始化的 rm 上读浓度是 exit 139 段错误）。
        现在失败就显式报错并上抛，状态栏同步给出人话。
        """
        try:
            from phreeqc_config import create_fully_initialized_phreeqcrm

            self.status_bar.showMessage("正在初始化化学系统...")
            QApplication.processEvents()  # 更新UI

            # 创建初始化的PHREEQCRM实例
            # nx/ny/nz=10 与 coupled_simulation.CHEM_GRID_DIM=10 对齐（10^3=1000）
            self.rm = create_fully_initialized_phreeqcrm(nx=10, ny=10, nz=10)

            if self.rm is None:
                # create_fully_initialized_phreeqcrm 已改为失败即上抛；
                # 真返回 None 只可能是 phreeqcrm 不可用这条分支。
                raise RuntimeError(
                    "create_fully_initialized_phreeqcrm 返回 None —— "
                    "PHREEQC 不可用（PHREEQC_AVAILABLE=%r）"
                    % (getattr(__import__('phreeqc_config'),
                               'PHREEQC_AVAILABLE', None),))

            self.chem_initialized = True
            self.status_bar.showMessage("化学系统初始化完成")
            logger.info("PHREEQCRM化学系统初始化成功")
        except Exception as e:
            self.chem_initialized = False
            self.rm = None
            msg = f"化学系统初始化失败，仿真不能启动: {e}"
            self.status_bar.showMessage(msg, 10000)
            logger.error(msg)
            import traceback
            logger.error(traceback.format_exc())
            raise
    
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
        self.simulation_worker.status_message.connect(self.on_worker_status)

        # 控制类调用全部排队到 worker 线程（显式 QueuedConnection，不依赖默认推断）。
        self.worker_start_requested.connect(
            self.simulation_worker.start_simulation, Qt.QueuedConnection)
        self.worker_pause_requested.connect(
            self.simulation_worker.pause_simulation, Qt.QueuedConnection)
        self.worker_stop_requested.connect(
            self.simulation_worker.stop_simulation, Qt.QueuedConnection)
        self.worker_speed_requested.connect(
            self.simulation_worker.set_speed, Qt.QueuedConnection)
        self.worker_substeps_requested.connect(
            self.simulation_worker.set_physics_substeps, Qt.QueuedConnection)
        self.worker_field_kind_requested.connect(
            self.simulation_worker.set_field_kind, Qt.QueuedConnection)
        # dock 在 worker 之前创建，把调试面板里的当前值补发一次
        self.worker_substeps_requested.emit(self._physics_substeps)
        # init_ui 阶段 worker 还不存在，场景里选的分布图指标没发出去；这里补发。
        if self.current_scenario is not None:
            self.worker_field_kind_requested.emit(self.current_scenario.field_kind)
        # cleanup 停的是 worker 线程的定时器，必须在 worker 线程里执行完再退线程。
        self.worker_cleanup_requested.connect(
            self.simulation_worker.cleanup, Qt.BlockingQueuedConnection)

        # 初始化检查定时器也必须在 worker 线程里创建（QTimer 有线程归属）。
        self.simulation_thread.started.connect(
            self.simulation_worker._start_initialization_timer)

        # 【竞态修复】鼠标只投递位置，add_particles 在 worker 线程执行。
        if hasattr(self, 'opengl_widget'):
            self.opengl_widget.particles_requested.connect(
                self.simulation_worker.add_particles_request, Qt.QueuedConnection)
        
        # 创建摄像机参数更新定时器
        self.camera_update_timer = QTimer()
        self.camera_update_timer.timeout.connect(self.update_camera_labels)
        self.camera_update_timer.start(100)  # 每100ms更新一次
        
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

            # 场景只决定"之后点击生成什么水"：切换液体类型白名单 + 默认类型，
            # 不重建 PhreeqcRM、不清场、不冻结已有粒子（用户裁决）。
            allowed = list(selected_scenario.fluid_types)
            if hasattr(self, 'opengl_widget') and self.opengl_widget is not None:
                self.opengl_widget.set_fluid_types(
                    allowed, selected_scenario.default_fluid_type)
            if hasattr(self, 'fluid_desc_layout') and self.fluid_desc_layout is not None:
                self._rebuild_fluid_desc_labels(allowed)
            if hasattr(self, 'fluid_type_label') and self.fluid_type_label is not None:
                ft = selected_scenario.default_fluid_type
                self.fluid_type_label.setText(
                    f"{ft} ({FLUID_TYPE_NAMES.get(ft, '?')})")
            self._update_field_label(selected_scenario.field_kind)
            if self.simulation_worker is not None:
                self.worker_field_kind_requested.emit(selected_scenario.field_kind)

    def _update_field_label(self, kind: str):
        """分布图标题：把 'Fe' / 'ph' / 'solute' 翻译成用户能读懂的指标名。"""
        labels = {
            "solute": "溶质浓度",
            "ph": "pH",
            "Fe": "溶解态铁（Fe 总量）",
            "Fe+2": "溶解态 Fe²⁺（亚铁）",
            "Fe+3": "溶解态 Fe³⁺（三价铁）",
            "C": "溶解态碳（C 总量）",
        }
        if hasattr(self, 'concentration_viz') and self.concentration_viz is not None:
            self.concentration_viz.set_field_label(labels.get(kind, kind))

    def _rebuild_fluid_desc_labels(self, allowed: List[int]):
        """按场景的液体白名单重建右侧"液体类型"说明行。"""
        for label in list(getattr(self, 'fluid_desc_labels', [])):
            label.setParent(None)
        self.fluid_desc_labels = []
        default_ft = None
        if getattr(self, 'current_scenario', None) is not None:
            default_ft = self.current_scenario.default_fluid_type
        for ft in allowed:
            label = QLabel(f"{ft}号液体: {FLUID_TYPE_NAMES.get(ft, '?')}")
            if ft == default_ft:
                label.setStyleSheet(
                    "color: #4fc3f7; font-weight: bold; font-size: 10px;")
            else:
                label.setStyleSheet("color: #888888; font-size: 10px;")
            self.fluid_desc_layout.addWidget(label)
            self.fluid_desc_labels.append(label)
    
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
    
    def _render_tick(self):
        """独立 60fps 重绘驱动（GUI 线程）。只触发 repaint，不碰仿真数据。"""
        if getattr(self, 'opengl_widget', None) is not None:
            self.opengl_widget.update()

    def start_simulation(self):
        """开始仿真 - 修复版

        流体初始化 + 定时器启动都交给 worker 线程（排队信号），
        主线程只改按钮状态。见 setup_simulation_thread 的连接说明。
        """
        if not self.is_running:
            self.is_running = True
            self.is_paused = False

            # 更新UI状态
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.reset_btn.setEnabled(False)

            self.worker_start_requested.emit()
            self.render_timer.start()

    def toggle_pause(self):
        """暂停/继续仿真"""
        if self.simulation_worker:
            self.worker_pause_requested.emit()
            self.is_paused = not self.is_paused
            self.pause_btn.setText("继续" if self.is_paused else "暂停")

    def stop_simulation(self):
        """停止仿真"""
        if self.simulation_worker:
            self.worker_stop_requested.emit()

        self.is_running = False
        self.is_paused = False
        if getattr(self, 'render_timer', None) is not None:
            self.render_timer.stop()

        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self.pause_btn.setText("暂停")

        self.status_bar.showMessage("仿真已停止")

    @Slot(str)
    def on_worker_status(self, message: str):
        """worker 线程的状态/失败信息统一走信号回主线程显示。"""
        self.status_bar.showMessage(message)
    
    def reset_simulation(self):
        """重置仿真"""
        self.stop_simulation()
        
        # 重置显示数据
        self.frame_label.setText("0")
        self.particle_label.setText("0")
        self.chem_step_label.setText("否")
        self.chem_time_label.setText("0s")
        
        # 清除浓度可视化
        self.concentration_viz.clear()
        
        self.status_bar.showMessage("仿真已重置")

    def reset_camera_view(self):
        """重置摄像机视图到默认位置"""
        if hasattr(self, 'opengl_renderer'):
            # 用 CONFIG 里的默认值。原来这里硬写 distance=1.0，
            # 相当于把镜头怼进杯子里：粒子被投影成几十像素的大色块，
            # 整片水看起来是一团糊，完全看不出水面。
            self.opengl_renderer.camera_target = np.asarray(CONFIG.CAMERA_TARGET,
                                                            dtype=np.float64).copy()
            self.opengl_renderer.camera_angle_x = CONFIG.CAMERA_ANGLE_X
            self.opengl_renderer.camera_angle_y = CONFIG.CAMERA_ANGLE_Y
            self.opengl_renderer.camera_distance = CONFIG.CAMERA_DISTANCE

            # 更新显示标签
            self.update_camera_labels()

            # 触发重绘
            if hasattr(self, 'opengl_widget'):
                self.opengl_widget.update()

            self.status_bar.showMessage("摄像机视图已重置")
    
    def update_camera_labels(self):
        """更新摄像机参数显示标签"""
        if hasattr(self, 'opengl_renderer'):
            # 更新角度显示（转换为度数便于理解）
            angle_x_deg = math.degrees(self.opengl_renderer.camera_angle_x)
            angle_y_deg = math.degrees(self.opengl_renderer.camera_angle_y)
            
            self.camera_angle_x_label.setText(f"{angle_x_deg:.1f}°")
            self.camera_angle_y_label.setText(f"{angle_y_deg:.1f}°")
            self.camera_distance_label.setText(f"{self.opengl_renderer.camera_distance:.2f}")
            
            # 更新液体类型显示
            if hasattr(self, 'opengl_widget') and hasattr(self.opengl_widget, 'current_fluid_type'):
                fluid_type = self.opengl_widget.current_fluid_type
                self.fluid_type_label.setText(
                    f"{fluid_type} ({FLUID_TYPE_NAMES.get(fluid_type, '?')})")

                # 更新液体类型描述颜色（只在类型变化时改样式表：
                # setStyleSheet 每 100 ms 重解析一次 CSS 并触发 polish/重绘）
                if fluid_type != getattr(self, '_last_fluid_type', None):
                    self._last_fluid_type = fluid_type
                    allowed = getattr(self.opengl_widget, 'allowed_fluid_types',
                                      [1, 2, 3])
                    for i, label in enumerate(self.fluid_desc_labels):
                        # 说明行按 allowed 顺序排列，第 i 行对应 allowed[i]
                        if i < len(allowed) and allowed[i] == fluid_type:
                            label.setStyleSheet("color: #4fc3f7; font-weight: bold; font-size: 10px;")
                        else:
                            label.setStyleSheet("color: #888888; font-size: 10px;")
    
    def update_speed(self, value: int):
        """更新仿真速度"""
        speed = value / 10.0
        self.speed_label.setText(f"{speed:.1f}x")
        if self.simulation_worker:
            self.worker_speed_requested.emit(speed)
    
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
        """处理粒子数据更新（位置；颜色由 on_ph_data_updated 单独刷新）"""
        try:
            if hasattr(self, 'opengl_widget') and hasattr(self, 'opengl_renderer'):
                renderer = self.opengl_renderer
                # 在GUI线程中更新OpenGL渲染器。
                # 这里不再传 ph_values：那会让 convert_ph_to_color 每帧跑第二遍
                # （on_ph_data_updated 已经跑过一遍），10M 粒子时是每帧多花数秒，
                # 而且用的是上一帧的旧 pH。位置更新只管位置。
                renderer.update_particles(particle_positions)

                # 新增粒子 / 颜色 VBO 刚被重建时，尾部还没有颜色。只补新增那段，
                # 不重算整帧（决策 D3：老粒子行号不变，颜色仍然有效）。
                n = renderer.particle_count
                colored = renderer.colored_particle_count
                if n > colored:
                    renderer.fill_new_particle_colors(colored, n)

                # 强制触发重绘（同一帧内的多次 update() 会被 Qt 合并）
                self.opengl_widget.update()
        except Exception as e:
            logger.warning(f"粒子渲染更新失败: {e}")
            logger.warning(traceback.format_exc())

    @Slot(np.ndarray)
    def on_ph_data_updated(self, ph_data: np.ndarray):
        """处理pH数据更新 - 修复版：增强线程安全和错误处理"""
        try:
            # 确保在GUI线程中执行
            if not hasattr(self, 'opengl_widget') or not self.opengl_widget.isValid():
                logger.warning("OpenGL上下文无效，跳过pH颜色更新")
                return
            
            self.last_ph_data = ph_data.copy() if ph_data is not None else None
            
            # 如果已经有粒子数据，立即更新颜色
            if (hasattr(self, 'opengl_renderer') and 
                hasattr(self.opengl_renderer, 'particle_count') and 
                self.last_ph_data is not None):
                
                # 获取当前粒子数量
                current_particle_count = self.opengl_renderer.particle_count
                
                # 安全检查：确保粒子数量有效
                if current_particle_count <= 0:
                    return
                
                # 截取对应数量的pH数据
                if len(self.last_ph_data) >= current_particle_count:
                    ph_subset = self.last_ph_data[:current_particle_count]
                    
                    # 【决策 D9】NaN 原样交给 convert_ph_to_color，逐粒子画
                    # 「无数据」色；有值的粒子不受影响，绝不整帧替换成 7.0。
                    try:
                        particle_colors = self.opengl_renderer.convert_ph_to_color(ph_subset)
                    except Exception as e:
                        logger.error(f"pH数据转换颜色失败: {e}")
                        logger.error(traceback.format_exc())
                        particle_colors = np.full((len(ph_subset), 3), [0.5, 0.5, 0.5])
                    
                    # 本槽本身就在 GUI 线程，直接更新；原来每帧再投一个
                    # QTimer.singleShot(0, lambda) 会额外堆一条事件和一份闭包。
                    try:
                        self.safe_update_particle_colors(particle_colors)
                    except Exception as e:
                        logger.error(f"粒子颜色更新失败: {e}")
                        logger.error(traceback.format_exc())
                        return
                    
                    # 触发OpenGL重绘
                    if hasattr(self, 'opengl_widget'):
                        self.opengl_widget.update()
                        
        except Exception as e:
            logger.error(f"pH数据处理失败: {e}")
            logger.error(traceback.format_exc())
    
    def safe_update_particle_colors(self, particle_colors: np.ndarray):
        """安全的粒子颜色更新方法，防止OpenGL上下文冲突"""
        try:
            # 检查OpenGL上下文是否有效
            if not hasattr(self, 'opengl_renderer') or not hasattr(self, 'opengl_widget'):
                logger.warning("OpenGL渲染器或部件不可用")
                return
            
            if not self.opengl_widget.isValid():
                logger.warning("OpenGL上下文无效，跳过颜色更新")
                return
            
            # 确保在正确的线程中执行
            if not self.opengl_widget.context().isValid():
                logger.warning("OpenGL上下文无效")
                return
            
            # 检查颜色数据有效性
            if particle_colors is None or len(particle_colors) == 0:
                logger.warning("无效的颜色数据")
                return
            
            # 确保颜色数据格式正确
            if len(particle_colors.shape) != 2 or particle_colors.shape[1] != 3:
                logger.error(f"颜色数据格式错误: {particle_colors.shape}")
                return
            
            # 检查NaN和Inf值（一趟 isfinite 扫完，原来 isnan+isinf 是两趟）
            if not np.all(np.isfinite(particle_colors)):
                logger.error("颜色数据包含无效值，使用默认颜色")
                particle_colors = np.full((len(particle_colors), 3), [0.5, 0.5, 0.5])
            
            # 执行颜色更新
            self.opengl_renderer.update_particle_colors(particle_colors)
            logger.debug(f"安全更新了 {len(particle_colors)} 个粒子的颜色")
            
        except Exception as e:
            logger.error(f"安全颜色更新失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    @Slot(str)
    def on_simulation_error(self, error_msg: str):
        """处理仿真错误"""
        logger.error(f"仿真错误: {error_msg}")
        QMessageBox.critical(self, "仿真错误", error_msg)
        self.stop_simulation()
    
    def closeEvent(self, event):
        """窗口关闭事件 - 修复版"""
        # 停止摄像机更新定时器
        if hasattr(self, 'camera_update_timer'):
            self.camera_update_timer.stop()
        if getattr(self, 'render_timer', None) is not None:
            self.render_timer.stop()
            
        # 停止仿真线程
        # cleanup 停的是 worker 线程里的定时器，必须在 worker 线程执行；
        # 用 BlockingQueuedConnection 保证它跑完再退线程。
        # 线程已退出时不能再发阻塞排队信号（会永远等不到接收方）。
        if self.simulation_worker:
            if self.simulation_thread is not None and self.simulation_thread.isRunning():
                self.worker_cleanup_requested.emit()
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