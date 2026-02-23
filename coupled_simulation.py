"""
化学-物理耦合模块
Chemistry-Physics Coupling Prototype Module

将 modular_mesh_system（SPH物理系统）与 core_simulation.py（PHREEQCRM化学系统）整合，
暂时只做液相。

核心设计：
  - 粒子→反应单元映射：粗网格哈希（6×6×6=216 cells）
  - 化学时间步 = 物理步的 N 倍（N=2，可配置）
  - 若 PHREEQCRM 不收敛，自动减半 dt_chem 重试，最多 3 次
  - 浓度同步：体积加权平均（cell→粒子广播）
  - 负浓度自动截断并标记，写入 chemical_warnings.log
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ==================== 日志配置 ====================
_chem_logger = logging.getLogger("chemical_warnings")
if not _chem_logger.handlers:
    _chem_logger.setLevel(logging.WARNING)
    _fh = logging.FileHandler("chemical_warnings.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    _chem_logger.addHandler(_fh)

# ==================== 默认配置 ====================
CHEM_GRID_DIM: int = 6          # 化学单元网格边长（6×6×6=216）
CHEM_STEP_RATIO: int = 2        # 每 N 帧物理步执行一次化学步
CHEM_MAX_RETRIES: int = 3       # PHREEQCRM.RunCells() 失败时最大重试次数
DEFAULT_PARTICLE_RADIUS: float = 0.05          # 来自 Config.PARTICLE_RADIUS
DEFAULT_PARTICLE_VOLUME: float = (4.0 / 3.0) * np.pi * DEFAULT_PARTICLE_RADIUS ** 3


# ==================== 数据结构 ====================

@dataclass
class Particle:
    """
    粒子数据结构（含化学属性扩展）

    Attributes:
        pid:      粒子唯一 ID
        position: 三维坐标 (x, y, z)，float32
        velocity: 三维速度 (vx, vy, vz)，float32
        solution: 液相化学组成 {element: molality}
        cell_id:  所属化学反应单元 ID（-1 表示未分配）
    """
    pid: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    solution: Dict[str, float] = field(default_factory=dict)
    cell_id: int = -1

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)


class ChemicalField:
    """
    全局化学状态管理

    Attributes:
        n_cells:        反应单元总数 (n_chem_cells_per_dim^3)
        cell_particles: 每个单元包含的粒子 ID 列表
        cell_volume:    单元体积（粒子数 × 粒子体积）
        phreeqc_conc:   PHREEQCRM 输入/输出缓冲区（n_cells × n_comps，展平）
        reaction_dt:    化学时间步累计（秒）
    """

    def __init__(
        self,
        n_chem_cells_per_dim: int = CHEM_GRID_DIM,
        particle_volume: float = DEFAULT_PARTICLE_VOLUME,
    ):
        self.n_chem_cells_per_dim: int = n_chem_cells_per_dim
        self.n_cells: int = n_chem_cells_per_dim ** 3
        self.particle_volume: float = particle_volume

        self.cell_particles: List[List[int]] = [[] for _ in range(self.n_cells)]
        self.cell_volume: np.ndarray = np.zeros(self.n_cells, dtype=np.float64)
        self.phreeqc_conc: Optional[np.ndarray] = None  # 由外部初始化后填充
        self.reaction_dt: float = 0.0

    def reset_cell_assignments(self) -> None:
        """清空粒子→单元分配，每次重新建立映射前调用。"""
        for cell_list in self.cell_particles:
            cell_list.clear()
        self.cell_volume[:] = 0.0

    def assign_particle(self, pid: int, cell_id: int) -> None:
        """将粒子注册到指定化学单元，并累加单元体积。"""
        self.cell_particles[cell_id].append(pid)
        self.cell_volume[cell_id] += self.particle_volume


# ==================== 耦合仿真主类 ====================

class CoupledSimulation:
    """
    化学-物理耦合仿真系统（液相原型）

    执行流程（每帧）：
        1. 物理步：调用 MeshSystem.step_simulation()
        2. 更新粒子→单元映射（基于新粒子位置）
        3. 化学步（每 chem_step_ratio 帧执行一次）：
           a. 聚合单元浓度（体积加权平均）
           b. 调用 PHREEQCRM.RunCells()（含重试）
           c. 将新浓度广播回单元内所有粒子
           d. 累计化学时间步

    Args:
        mesh_system:      modular_mesh_system.core.MeshSystem 实例（可为 None）
        phreeqc_rm:       phreeqcrm.PhreeqcRM 实例（可为 None，用于测试）
        domain_min:       仿真域最小坐标
        domain_max:       仿真域最大坐标
        chem_step_ratio:  化学步频率（每 N 帧物理步执行一次）
        chem_grid_dim:    化学粗网格边长（默认 6，即 6×6×6=216 cells）
        particle_volume:  单个粒子体积（m³）
        chem_dt:          化学时间步长（秒，默认 86400 s = 1 day）
    """

    def __init__(
        self,
        mesh_system=None,
        phreeqc_rm=None,
        domain_min: Optional[np.ndarray] = None,
        domain_max: Optional[np.ndarray] = None,
        chem_step_ratio: int = CHEM_STEP_RATIO,
        chem_grid_dim: int = CHEM_GRID_DIM,
        particle_volume: float = DEFAULT_PARTICLE_VOLUME,
        chem_dt: float = 86400.0,
    ):
        self.mesh_system = mesh_system
        self.rm = phreeqc_rm

        # 仿真域边界
        if domain_min is None:
            domain_min = np.array([-0.8, -0.8, -0.8], dtype=np.float64)
        if domain_max is None:
            domain_max = np.array([0.8, 0.8, 0.8], dtype=np.float64)
        self.domain_min: np.ndarray = np.asarray(domain_min, dtype=np.float64)
        self.domain_max: np.ndarray = np.asarray(domain_max, dtype=np.float64)
        self.domain_size: np.ndarray = self.domain_max - self.domain_min

        # 化学步配置
        self.chem_step_ratio: int = chem_step_ratio
        self.chem_dt: float = chem_dt
        self.accumulated_chem_dt: float = 0.0

        # 化学场
        self.chem_field = ChemicalField(chem_grid_dim, particle_volume)

        # 化学组分信息（由 initialize() 填充）
        self.n_comps: int = 0
        self.comp_names: List[str] = []

        # 粒子化学数据（按粒子 ID 索引的数组）
        # _particle_solutions shape: (n_particles, n_comps)
        self._particle_solutions: Optional[np.ndarray] = None
        # _particle_cell_ids shape: (n_particles,)
        self._particle_cell_ids: Optional[np.ndarray] = None

        self.frame_count: int = 0

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        初始化化学系统并建立粒子→单元映射。

        必须在 PHREEQCRM 完成 FindComponents() 后调用。
        """
        if self.rm is not None:
            self.n_comps = self.rm.GetComponentCount()
            self.comp_names = list(self.rm.GetComponents())
            n_cells = self.chem_field.n_cells
            self.chem_field.phreeqc_conc = np.zeros(
                n_cells * self.n_comps, dtype=np.float64
            )

        self._update_particle_mapping()

    # ------------------------------------------------------------------
    # 粒子位置获取
    # ------------------------------------------------------------------

    def _get_particle_positions(self) -> np.ndarray:
        """
        从 MeshSystem 获取当前粒子位置。

        Returns:
            positions: (N, 3) float32 数组；若无粒子则返回 (0, 3) 空数组。
        """
        if self.mesh_system is not None:
            n = self.mesh_system.current_particle_num
            if n > 0:
                return self.mesh_system.particle_positions[:n]
        return np.zeros((0, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # 粒子→单元映射
    # ------------------------------------------------------------------

    def _position_to_cell_id(self, positions: np.ndarray) -> np.ndarray:
        """
        将粒子位置批量映射到化学单元 ID（粗网格哈希）。

        采用类欧拉视角的固定化学网格：将仿真域等分为
        n_chem_cells_per_dim^3 个单元，按三维线性索引编号。

        Args:
            positions: (N, 3) 粒子坐标数组

        Returns:
            cell_ids: (N,) int32 化学单元 ID 数组
        """
        n_dim = self.chem_field.n_chem_cells_per_dim

        # 归一化坐标到 [0, n_dim)，避免越界
        norm = (positions.astype(np.float64) - self.domain_min) / self.domain_size
        norm = np.clip(norm, 0.0, 1.0 - 1e-9)

        grid_coords = (norm * n_dim).astype(np.int32)
        grid_coords = np.clip(grid_coords, 0, n_dim - 1)

        # 三维线性化索引：ix + iy*n + iz*n^2
        cell_ids = (
            grid_coords[:, 0]
            + grid_coords[:, 1] * n_dim
            + grid_coords[:, 2] * n_dim * n_dim
        )
        return cell_ids.astype(np.int32)

    def _update_particle_mapping(self) -> None:
        """
        根据当前粒子位置重新分配粒子到化学单元。

        同时维护 _particle_cell_ids 和 _particle_solutions 数组。
        """
        positions = self._get_particle_positions()
        n = len(positions)

        # 重置单元分配
        self.chem_field.reset_cell_assignments()

        if n == 0:
            return

        # 按需分配/扩展粒子数据数组
        if self._particle_cell_ids is None or len(self._particle_cell_ids) < n:
            self._particle_cell_ids = np.full(n, -1, dtype=np.int32)
            if self.n_comps > 0:
                self._particle_solutions = np.zeros(
                    (n, self.n_comps), dtype=np.float64
                )

        # 计算并存储单元 ID
        cell_ids = self._position_to_cell_id(positions)
        self._particle_cell_ids[:n] = cell_ids

        # 注册粒子到化学单元
        for pid in range(n):
            self.chem_field.assign_particle(int(pid), int(cell_ids[pid]))

    # ------------------------------------------------------------------
    # 浓度聚合（粒子→单元）
    # ------------------------------------------------------------------

    def _aggregate_cell_concentrations(self) -> np.ndarray:
        """
        体积加权平均：将粒子浓度聚合到各化学单元。

        由于每个粒子体积相同，体积加权平均等于算术平均。

        Returns:
            cell_conc: (n_cells, n_comps) float64 每个单元的平均浓度
        """
        n_cells = self.chem_field.n_cells
        cell_conc = np.zeros((n_cells, self.n_comps), dtype=np.float64)

        if self._particle_solutions is None or self.n_comps == 0:
            return cell_conc

        n = (
            self.mesh_system.current_particle_num
            if self.mesh_system is not None
            else (
                len(self._particle_cell_ids)
                if self._particle_cell_ids is not None
                else 0
            )
        )
        if n == 0:
            return cell_conc

        for cell_id in range(n_cells):
            particles = self.chem_field.cell_particles[cell_id]
            if not particles:
                continue
            # 等体积粒子：直接算术平均
            cell_conc[cell_id] = self._particle_solutions[
                np.array(particles, dtype=np.int32)
            ].mean(axis=0)

        return cell_conc

    # ------------------------------------------------------------------
    # 化学反应步（PHREEQCRM）
    # ------------------------------------------------------------------

    def _run_chemical_step(
        self, cell_conc: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        调用 PHREEQCRM 执行化学反应，含自动减半 dt 重试机制。

        若 PHREEQCRM 不收敛（RunCells 返回非零状态或抛出异常），
        自动将 dt 减半并重试，最多重试 CHEM_MAX_RETRIES 次。

        Args:
            cell_conc: (n_cells, n_comps) 输入浓度矩阵
            dt:        化学时间步（秒）

        Returns:
            new_conc: (n_cells, n_comps) 更新后的浓度矩阵
                      负浓度已被截断为 0，并记录到 chemical_warnings.log。
        """
        if self.rm is None:
            return cell_conc.copy()

        n_cells = self.chem_field.n_cells

        # 设置浓度到 PHREEQCRM（展平为 n_cells*n_comps 向量）
        conc_flat = cell_conc.flatten()
        self.rm.SetConcentrations(conc_flat)
        self.rm.SetTime(self.accumulated_chem_dt)

        # 重试逻辑
        current_dt = dt
        converged = False
        for retry in range(CHEM_MAX_RETRIES):
            self.rm.SetTimeStep(current_dt)
            try:
                status = self.rm.RunCells()
                if status == 0:
                    converged = True
                    break
                else:
                    current_dt *= 0.5
                    _chem_logger.warning(
                        f"[CoupledSim] RunCells status={status}, "
                        f"retry {retry + 1}/{CHEM_MAX_RETRIES} with dt={current_dt:.2f}s"
                    )
            except Exception as e:
                current_dt *= 0.5
                _chem_logger.warning(
                    f"[CoupledSim] RunCells error: {e}, "
                    f"retry {retry + 1}/{CHEM_MAX_RETRIES} with dt={current_dt:.2f}s"
                )

        if not converged:
            _chem_logger.warning(
                f"Frame {self.frame_count}: RunCells did not converge after "
                f"{CHEM_MAX_RETRIES} retries; returning pre-reaction concentrations."
            )
            return cell_conc.copy()

        # 读取更新后的浓度
        new_conc_flat = np.array(self.rm.GetConcentrations(), dtype=np.float64)
        new_conc = new_conc_flat.reshape((n_cells, self.n_comps))

        # 负浓度截断 + 告警日志
        neg_mask = new_conc < 0.0
        if neg_mask.any():
            n_neg = int(neg_mask.sum())
            _chem_logger.warning(
                f"Frame {self.frame_count}: {n_neg} negative concentration value(s) "
                f"detected after RunCells; clipping to 0."
            )
            np.clip(new_conc, 0.0, None, out=new_conc)

        return new_conc

    # ------------------------------------------------------------------
    # 浓度广播（单元→粒子）
    # ------------------------------------------------------------------

    def _broadcast_to_particles(self, cell_conc: np.ndarray) -> None:
        """
        将化学单元浓度广播回单元内所有粒子。

        Args:
            cell_conc: (n_cells, n_comps) 更新后的单元浓度
        """
        if self._particle_solutions is None or self.n_comps == 0:
            return

        for cell_id in range(self.chem_field.n_cells):
            particles = self.chem_field.cell_particles[cell_id]
            if not particles:
                continue
            conc = cell_conc[cell_id]
            for pid in particles:
                self._particle_solutions[pid] = conc

    # ------------------------------------------------------------------
    # 粒子化学状态访问
    # ------------------------------------------------------------------

    def get_particle_ph(self, ph_component_index: int = 0) -> Optional[np.ndarray]:
        """
        获取每个粒子的 pH（或指定组分浓度）。

        Args:
            ph_component_index: 对应 pH 的组分索引（默认 0）

        Returns:
            ph_array: (n_active_particles,) float64 数组，或 None
        """
        if self._particle_solutions is None or self.n_comps == 0:
            return None
        n = (
            self.mesh_system.current_particle_num
            if self.mesh_system is not None
            else len(self._particle_cell_ids or [])
        )
        if n == 0:
            return None
        return self._particle_solutions[:n, ph_component_index].copy()

    def get_cell_concentrations(self) -> np.ndarray:
        """
        获取当前化学单元浓度矩阵（聚合但未执行化学步）。

        Returns:
            cell_conc: (n_cells, n_comps) float64
        """
        return self._aggregate_cell_concentrations()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def step(self) -> Dict:
        """
        执行一帧耦合仿真。

        Returns:
            stats: 帧统计字典，包含：
                - frame: 帧号
                - chemical_step: 本帧是否执行了化学步
                - chem_time: 化学时间累计（秒，仅化学步帧）
                - n_particles: 当前活跃粒子数
        """
        self.frame_count += 1
        stats: Dict = {
            "frame": self.frame_count,
            "chemical_step": False,
            "n_particles": 0,
        }

        # ── 物理步（每帧执行）──
        if self.mesh_system is not None:
            self.mesh_system.step_simulation()

        # ── 更新粒子→单元映射 ──
        self._update_particle_mapping()

        # 记录活跃粒子数
        n_active = (
            self.mesh_system.current_particle_num
            if self.mesh_system is not None
            else (
                len(self._particle_cell_ids)
                if self._particle_cell_ids is not None
                else 0
            )
        )
        stats["n_particles"] = n_active

        # ── 化学步（每 chem_step_ratio 帧执行）──
        if self.frame_count % self.chem_step_ratio == 0:
            # 1. 聚合单元浓度（体积加权平均）
            cell_conc = self._aggregate_cell_concentrations()

            # 2. 调用 PHREEQCRM.RunCells()
            new_conc = self._run_chemical_step(cell_conc, self.chem_dt)

            # 3. 广播回单元内所有粒子
            self._broadcast_to_particles(new_conc)

            # 4. 累计化学时间步
            self.accumulated_chem_dt += self.chem_dt
            self.chem_field.reaction_dt += self.chem_dt

            stats["chemical_step"] = True
            stats["chem_time"] = self.accumulated_chem_dt

        return stats

    def run(self, n_frames: int) -> List[Dict]:
        """
        运行 n_frames 帧耦合仿真。

        Args:
            n_frames: 仿真帧数

        Returns:
            history: 每帧统计信息列表
        """
        history = []
        for _ in range(n_frames):
            stats = self.step()
            history.append(stats)
        return history
