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
import phreeqcrm

# ==================== 日志配置 ====================
_chem_logger = logging.getLogger("chemical_warnings")
if not _chem_logger.handlers:
    _chem_logger.setLevel(logging.DEBUG)
    _fh = logging.FileHandler("chemical_warnings.log", encoding='utf-8')
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    _chem_logger.addHandler(_fh)

# ==================== 默认配置 ====================
CHEM_GRID_DIM: int = 6          # 化学单元网格边长（6×6×6=216）
CHEM_STEP_RATIO: int = 1        # 每 N 帧物理步执行一次化学步
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
        self.rm: Optional[phreeqcrm.PhreeqcRM] = phreeqc_rm

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

        # 初始平衡浓度缓存（用于空单元填充和回退）
        self._initial_equilibrium_conc: Optional[np.ndarray] = None
        
    def _initialize_particle_chemistry(self) -> None:
        """
        初始化粒子的化学状态，使用PHREEQCRM计算好的溶液而不是手动设置浓度
        """
        _chem_logger.info("开始初始化粒子化学状态（使用PHREEQCRM计算溶液）")
        
        # 即使没有mesh_system，我们也需要初始化化学状态用于测试
        if self.mesh_system is None:
            _chem_logger.info("mesh_system为None，使用默认粒子数进行初始化")
            n_particles = 100  # 使用默认粒子数进行测试
        else:
            n_particles = self.mesh_system.current_particle_num
            _chem_logger.info(f"检测到 {n_particles} 个粒子")
        
        if n_particles == 0:
            _chem_logger.warning("没有粒子，跳过化学状态初始化")
            return
            
        # 确保组分信息已正确设置
        _chem_logger.info(f"当前组分信息: n_comps={self.n_comps}, comp_names={self.comp_names}")
        if self.n_comps == 0 or not self.comp_names:
            _chem_logger.warning("化学组分信息未设置，使用默认组分")
            self.comp_names = ['H+', 'OH-', 'Na+', 'Cl-']
            self.n_comps = len(self.comp_names)
            
        # 如果粒子溶液数组尚未初始化，则创建它
        _chem_logger.info(f"检查粒子溶液数组: 当前形状={self._particle_solutions.shape if self._particle_solutions is not None else 'None'}")
        if self._particle_solutions is None or self._particle_solutions.shape[0] < n_particles or self._particle_solutions.shape[1] != self.n_comps:
            self._particle_solutions = np.zeros((n_particles, self.n_comps), dtype=np.float64)
            _chem_logger.info(f"创建了新的粒子溶液数组，形状: {self._particle_solutions.shape}")
        
        # 使用PHREEQCRM计算初始溶液状态
        # 这里模仿phreeGUI.py中的逻辑，让PHREEQCRM计算平衡状态
        if self.rm is not None:
            try:
                _chem_logger.info("使用PHREEQCRM计算初始化学平衡状态...")
                
                # 设置初始时间
                self.rm.SetTime(0.0)
                self.rm.SetTimeStep(0.0)  # 瞬时平衡计算
                
                # 运行初始平衡计算
                status = self.rm.RunCells()
                if status == 0:
                    # 成功计算平衡状态，获取浓度
                    equilibrium_conc = np.array(self.rm.GetConcentrations(), dtype=np.float64)
                    n_cells = self.chem_field.n_cells
                    if len(equilibrium_conc) == n_cells * self.n_comps:
                        # 将平衡浓度广播到所有粒子
                        cell_conc = equilibrium_conc.reshape((n_cells, self.n_comps))
                        # 缓存初始平衡浓度，用于后续空单元填充
                        self._initial_equilibrium_conc = cell_conc.copy()
                        self._broadcast_to_particles(cell_conc)
                        _chem_logger.info("成功使用PHREEQCRM计算初始平衡状态")
                    else:
                        _chem_logger.warning(f"浓度数组长度不匹配: 期望{n_cells * self.n_comps}, 实际{len(equilibrium_conc)}")
                        self._set_default_initial_concentrations(n_particles)
                else:
                    _chem_logger.warning(f"PHREEQCRM初始平衡计算失败 (status={status})，使用默认浓度")
                    self._set_default_initial_concentrations(n_particles)
                    
            except Exception as e:
                _chem_logger.warning(f"PHREEQCRM初始化失败: {e}，使用默认浓度")
                self._set_default_initial_concentrations(n_particles)
        else:
            _chem_logger.warning("PHREEQCRM实例不可用，使用默认浓度")
            self._set_default_initial_concentrations(n_particles)
            
        # 验证设置结果
        if self._particle_solutions is not None and n_particles > 0:
            min_conc = np.min(self._particle_solutions[:n_particles])
            max_conc = np.max(self._particle_solutions[:n_particles])
            mean_conc = np.mean(self._particle_solutions[:n_particles])
            _chem_logger.info(f"最终浓度设置完成 - 范围: [{min_conc:.2e}, {max_conc:.2e}], 平均: {mean_conc:.2e}")
        
        _chem_logger.info(f"初始化了 {n_particles} 个粒子的化学状态，组分: {self.comp_names}")

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        初始化化学系统并建立粒子→单元映射。

        必须在 PHREEQCRM 完成 FindComponents() 后调用。
        """
        if self.rm is not None:
            try:
                self.n_comps = self.rm.GetComponentCount()
                self.comp_names = list(self.rm.GetComponents())
                
                # 验证组件数量
                if self.n_comps == 0:
                    _chem_logger.warning("警告: 检测到0个化学组分，可能需要检查数据库加载和初始条件设置")
                    # 设置默认组分以防万一
                    self.comp_names = ['H+', 'OH-', 'Na+', 'Cl-']
                    self.n_comps = len(self.comp_names)
                
                n_cells = self.chem_field.n_cells
                self.chem_field.phreeqc_conc = np.zeros(
                    n_cells * self.n_comps, dtype=np.float64
                )
                
                _chem_logger.info(f"化学系统初始化完成: {self.n_comps} 个组分 - {self.comp_names}")
                
            except Exception as e:
                _chem_logger.error(f"化学系统初始化失败: {e}")
                # 使用默认配置继续运行
                self.comp_names = ['H+', 'OH-', 'Na+', 'Cl-']
                self.n_comps = len(self.comp_names)
                n_cells = self.chem_field.n_cells
                self.chem_field.phreeqc_conc = np.zeros(
                    n_cells * self.n_comps, dtype=np.float64
                )

        # 初始化粒子化学状态
        self._initialize_particle_chemistry()
        
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
        # 注意：避免覆盖已有的化学浓度数据
        need_initialize_solutions = False
        
        if self._particle_cell_ids is None or len(self._particle_cell_ids) < n:
            self._particle_cell_ids = np.full(n, -1, dtype=np.int32)
            need_initialize_solutions = True
        elif self.n_comps > 0 and self._particle_solutions is None:
            # n_comps 在首次分配 _particle_cell_ids 之后才变为非零时，
            # 需要补充初始化 _particle_solutions
            need_initialize_solutions = True
        
        # 只有在确实需要初始化且之前没有初始化过的情况下才创建新的溶液数组
        if need_initialize_solutions and self.n_comps > 0:
            # 保存现有的浓度数据（如果存在）
            old_solutions = self._particle_solutions
            self._particle_solutions = np.zeros((n, self.n_comps), dtype=np.float64)
            
            # 如果有旧数据，复制过去（在数组大小允许的范围内）
            if old_solutions is not None and old_solutions.shape[0] > 0:
                copy_rows = min(old_solutions.shape[0], n)
                copy_cols = min(old_solutions.shape[1], self.n_comps)
                self._particle_solutions[:copy_rows, :copy_cols] = old_solutions[:copy_rows, :copy_cols]
                _chem_logger.info(f"保留了 {copy_rows} 行 {copy_cols} 列的原有浓度数据")

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
        空单元格设置为特殊标记值，避免传递无物理意义的0浓度。

        Returns:
            cell_conc: (n_cells, n_comps) float64 每个单元的平均浓度
        """
        n_cells = self.chem_field.n_cells
        cell_conc = np.full((n_cells, self.n_comps), -1.0, dtype=np.float64)  # 使用-1标记空单元格

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

    def _set_default_initial_concentrations(self, n_particles: int) -> None:
        """
        设置默认的初始浓度（当PHREEQCRM计算失败时的备用方案）
        """
        _chem_logger.info("设置默认初始浓度...")
        
        # 使用合理的初始浓度（mol/L）
        initial_concentrations = {
            'H+': 1.0e-7,      # 中性pH
            'OH-': 1.0e-7,     # 中性pH
            'H2O': 55.5,       # 纯水
            'Na+': 1.0e-3,     # 微量钠离子
            'Cl-': 1.0e-3,     # 微量氯离子
            'HCl': 1.0e-12,    # 极微量
            'H2': 1.0e-15,     # 极微量
            'O2': 1.0e-15      # 极微量
        }
        
        # 将初始浓度映射到对应的组分索引
        for i, comp_name in enumerate(self.comp_names):
            conc = initial_concentrations.get(comp_name, 1e-6)  # 默认1e-6 mol/L
            self._particle_solutions[:n_particles, i] = conc
            _chem_logger.debug(f"  组分 {comp_name} (索引 {i}): 设置浓度 {conc:.2e}")
    
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

        # ── 识别并排除空单元（无粒子）──
        # _aggregate_cell_concentrations 使用 -1.0 标记空单元。
        # 对这些单元调用 RunCells 会导致 "A(H2O) Activity of water has not
        # converged" 错误，因为全零/极低浓度不构成有效的化学溶液。
        # 通过 PhreeqcRM.CreateMapping 将空单元映射为 -1（非活跃），
        # RunCells 将跳过这些单元，避免无意义的收敛尝试。
        empty_cell_mask = np.any(cell_conc < 0, axis=1)
        n_active = int(np.sum(~empty_cell_mask))

        if n_active == 0:
            _chem_logger.warning(
                f"Frame {self.frame_count}: 所有化学单元均为空，跳过化学计算"
            )
            return cell_conc.copy()

        if np.any(empty_cell_mask):
            # 使用连续的化学单元索引，确保 GetChemistryCellCount 正确减少。
            # 旧的 identity 映射 [i if active else -1] 在活跃单元分散时会导致
            # 大量未映射的化学单元仍被 RunCells 计算，可能因陈旧数据而收敛失败。
            mapping = [-1] * n_cells
            chem_id = 0
            for i in range(n_cells):
                if not empty_cell_mask[i]:
                    mapping[i] = chem_id
                    chem_id += 1
            try:
                self.rm.CreateMapping(mapping)
                _chem_logger.debug(
                    f"Frame {self.frame_count}: 排除 {int(np.sum(empty_cell_mask))}/{n_cells} 个空单元"
                )
            except Exception as e:
                _chem_logger.warning(f"Frame {self.frame_count}: CreateMapping 更新失败: {e}")
        else:
            # 所有单元均有粒子，恢复完整映射
            try:
                self.rm.CreateMapping(list(range(n_cells)))
            except Exception as e:
                _chem_logger.debug(f"Frame {self.frame_count}: 恢复完整映射失败: {e}")

        # 检查输入浓度是否有效
        if np.allclose(cell_conc, 0.0):
            _chem_logger.warning(
                f"Frame {self.frame_count}: 检测到所有浓度为0，跳过化学计算"
            )
            return cell_conc.copy()

        # 设置浓度到 PHREEQCRM（展平为 n_cells*n_comps 向量）
        conc_flat = cell_conc.flatten().copy()  # 创建副本避免修改原始数据
        
        # 修正空单元和负浓度：
        # 空单元使用初始平衡浓度（若可用），否则使用背景值。
        # 这确保 SetConcentrations 传入的数据对所有单元都是物理有效的。
        background = 1e-12
        if self._initial_equilibrium_conc is not None and np.any(empty_cell_mask):
            for i in np.where(empty_cell_mask)[0]:
                start = i * self.n_comps
                end = start + self.n_comps
                conc_flat[start:end] = self._initial_equilibrium_conc[i]
        neg_mask = conc_flat < 0
        if np.any(neg_mask):
            conc_flat[neg_mask] = background
            
        # 特别处理水浓度：确保不低于1e-6 mol/L
        if self.n_comps > 0 and 'H2O' in self.comp_names:
            h2o_index = self.comp_names.index('H2O')
            min_h2o_conc = 1e-6
            low_h2o_mask = conc_flat[h2o_index::self.n_comps] < min_h2o_conc
            if low_h2o_mask.any():
                n_low = np.sum(low_h2o_mask)
                # 提升水浓度
                conc_flat[h2o_index::self.n_comps][low_h2o_mask] = min_h2o_conc
                # 清空同单元格的其他组分
                for i in range(self.n_comps):
                    if i != h2o_index:
                        conc_flat[i::self.n_comps][low_h2o_mask] = 0.0
                _chem_logger.debug(f"Frame {self.frame_count}: 提升 {n_low} 个低水浓度至 {min_h2o_conc:.1e} mol/L，并清空其他组分")

        # 最终验证：确保数组长度正确且无无效值
        expected_length = self.chem_field.n_cells * self.n_comps
        if len(conc_flat) != expected_length:
            _chem_logger.error(
                f"Frame {self.frame_count}: 浓度数组长度错误！期望 {expected_length}，实际 {len(conc_flat)}"
            )
            return cell_conc.copy()
        
        # 调试信息
        _chem_logger.debug(f"设置浓度到PHREEQCRM: 总浓度数={len(conc_flat)}, min={np.min(conc_flat):.2e}, max={np.max(conc_flat):.2e}")
        
        # 设置完整长度的浓度数组到PHREEQCRM
        self.rm.SetConcentrations(conc_flat.tolist())
        self.rm.SetTime(self.accumulated_chem_dt)

        # 重试逻辑
        current_dt = dt
        converged = False
        last_error_msg = ""
        
        for retry in range(CHEM_MAX_RETRIES):
            self.rm.SetTimeStep(current_dt)
            try:
                status = self.rm.RunCells()
                if status == 0:
                    converged = True
                    break
                else:
                    current_dt *= 0.5
                    last_error_msg = f"RunCells status={status}"
                    _chem_logger.warning(
                        f"[CoupledSim] Frame {self.frame_count} RunCells status={status}, "
                        f"retry {retry + 1}/{CHEM_MAX_RETRIES} with dt={current_dt:.2f}s"
                    )
            except Exception as e:
                current_dt *= 0.5
                last_error_msg = str(e)
                _chem_logger.warning(
                    f"[CoupledSim] Frame {self.frame_count} RunCells error: {e}, "
                    f"retry {retry + 1}/{CHEM_MAX_RETRIES} with dt={current_dt:.2f}s"
                )

        if not converged:
            _chem_logger.error(
                f"Frame {self.frame_count}: RunCells 未收敛，错误信息: {last_error_msg}"
            )
            _chem_logger.error(
                f"输入浓度统计: min={np.min(cell_conc):.2e}, max={np.max(cell_conc):.2e}, "
                f"mean={np.mean(cell_conc):.2e}"
            )
            return cell_conc.copy()

        # 读取更新后的浓度
        try:
            new_conc_flat = np.array(self.rm.GetConcentrations(), dtype=np.float64)
            new_conc = new_conc_flat.reshape((n_cells, self.n_comps))
            
            # 验证输出浓度的有效性
            if np.any(np.isnan(new_conc)) or np.any(np.isinf(new_conc)):
                _chem_logger.warning(
                    f"Frame {self.frame_count}: 检测到输出浓度包含NaN或Inf值，使用输入浓度"
                )
                return cell_conc.copy()
            
            # 确保输出浓度为正值
            new_conc = np.maximum(new_conc, 0.0)
                
        except Exception as e:
            _chem_logger.error(
                f"Frame {self.frame_count}: 读取浓度失败: {e}"
            )
            return cell_conc.copy()

        # 负浓度截断 + 告警日志
        neg_mask = new_conc < 0.0
        if neg_mask.any():
            n_neg = int(neg_mask.sum())
            _chem_logger.warning(
                f"Frame {self.frame_count}: {n_neg} 个负浓度值检测到，截断为0"
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
        # 调试信息
        if self._particle_solutions is not None:
            _chem_logger.debug(f"粒子溶液数据形状: {self._particle_solutions.shape}, 组分数: {self.n_comps}")
        
        if self._particle_solutions is None or self.n_comps == 0:
            _chem_logger.warning("粒子溶液数据为空或组分数为0")
            return None
            
        n = (
            self.mesh_system.current_particle_num
            if self.mesh_system is not None
            else len(self._particle_cell_ids or [])
        )
        if n == 0:
            _chem_logger.warning("没有活跃粒子")
            return None
            
        if ph_component_index >= self.n_comps:
            _chem_logger.warning(f"请求的组分索引 {ph_component_index} 超出范围 (0-{self.n_comps-1})")
            ph_component_index = 0  # 回退到第一个组分
            
        ph_data = self._particle_solutions[:n, ph_component_index].copy()
        
        # 验证pH数据的有效性
        if np.allclose(ph_data, 0.0):
            _chem_logger.warning(f"检测到所有pH值为0，可能存在问题")
        elif np.any(np.isnan(ph_data)) or np.any(np.isinf(ph_data)):
            _chem_logger.warning(f"检测到无效pH值 (NaN/Inf)，返回默认值")
            ph_data = np.full_like(ph_data, 7.0)  # 默认中性pH
        
        _chem_logger.debug(f"返回pH数据，范围: [{np.min(ph_data):.3f}, {np.max(ph_data):.3f}]，长度: {len(ph_data)}")
        return ph_data

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
            "chem_time": 0.0
        }
        
        # ── 物理步（每帧执行）──
        if self.mesh_system is not None:
            try:
                self.mesh_system.step_simulation()
            except Exception as e:
                _chem_logger.error(f"物理步执行失败: {e}")

        # ── 更新粒子→单元映射 ──
        try:
            self._update_particle_mapping()
        except Exception as e:
            _chem_logger.error(f"粒子映射更新失败: {e}")

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
        
        # 调试信息
        _chem_logger.debug(f"Frame {self.frame_count}: 活跃粒子数 = {n_active}")

        # ── 化学步（每 chem_step_ratio 帧执行）──
        if self.frame_count % self.chem_step_ratio == 0 and self.rm is not None:
            try:
                # 1. 聚合单元浓度（体积加权平均）
                cell_conc = self._aggregate_cell_concentrations()
                
                # 调试信息
                if cell_conc.size > 0:
                    _chem_logger.debug(f"聚合浓度统计: min={np.min(cell_conc):.2e}, max={np.max(cell_conc):.2e}, mean={np.mean(cell_conc):.2e}")
                
                # 2. 调用 PHREEQCRM.RunCells()
                new_conc = self._run_chemical_step(cell_conc, self.chem_dt)
                
                # 3. 广播回单元内所有粒子
                self._broadcast_to_particles(new_conc)

                # 4. 累计化学时间步
                self.accumulated_chem_dt += self.chem_dt
                self.chem_field.reaction_dt += self.chem_dt
                
                # 获取pH数据用于调试
                ph_data = self.get_particle_ph()
                if ph_data is not None:
                    _chem_logger.debug(f"pH数据范围: [{np.min(ph_data):.3f}, {np.max(ph_data):.3f}]，长度: {len(ph_data)}")
                
                stats["chemical_step"] = True
                stats["chem_time"] = self.accumulated_chem_dt
                
            except Exception as e:
                _chem_logger.error(f"化学步执行失败 (Frame {self.frame_count}): {e}")
                import traceback
                _chem_logger.error(traceback.format_exc())

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
