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
from utils import build_consecutive_mapping

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
        self.rm.SetFilePrefix("coupled_simulation")
        self.rm.SetPrintChemistryOn( True,  True,  True)
        self.rm.OpenFiles()
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
                    # print(list(equilibrium_conc))
                    n_cells = self.chem_field.n_cells
                    if len(equilibrium_conc) == n_cells * self.n_comps:
                        # 将平衡浓度广播到所有粒子
                        # PHREEQCRM返回的是component-major布局: [comp0_cell0, comp0_cell1, ..., comp1_cell0, comp1_cell1, ...]
                        cell_conc = equilibrium_conc.reshape((self.n_comps, n_cells)).T
                        # 缓存初始平衡浓度，用于后续空单元填充
                        self._initial_equilibrium_conc = cell_conc.copy()
                                        
                        # 识别并标记纯水单元
                        h2o_comp_index = None
                        # 识别纯水单元：除了H, O, Charge（前三个元素）外都为0
                        if self.n_comps >= 3:
                            # 前三个组分索引（H, O, Charge）
                            major_components = [0, 1, 2]
                            # 其他组分索引
                            other_comps_indices = [i for i in range(self.n_comps) if i not in major_components]

                            # 检查其他组分是否都为0
                            if other_comps_indices:
                                other_comps_zero = np.all(cell_conc[:, other_comps_indices] == 0, axis=1)
                                self.pure_water_cells =  other_comps_zero
                                n_pure_water = int(np.sum(self.pure_water_cells))
                                if n_pure_water > 0:
                                    _chem_logger.info(f"初始化时识别到 {n_pure_water} 个纯水单元")
                        
                        # 确保粒子映射是最新的
                        self._update_particle_mapping()
                        
                        # 广播浓度到粒子
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
        _chem_logger.info("开始初始化化学系统...")
        
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

        # 先更新粒子映射（确保cell_particles有正确的粒子分配）
        _chem_logger.info("更新粒子映射...")
        self._update_particle_mapping()
        
        # 验证粒子映射结果
        total_assigned_particles = sum(len(particles) for particles in self.chem_field.cell_particles)
        _chem_logger.info(f"粒子映射完成: 分配了 {total_assigned_particles} 个粒子到化学单元")
        
        # 再初始化粒子化学状态
        _chem_logger.info("初始化粒子化学状态...")
        self._initialize_particle_chemistry()
        
        _chem_logger.info("化学系统初始化完毕")

    # ------------------------------------------------------------------
    # 粒子位置获取
    # ------------------------------------------------------------------

    def _get_particle_positions(self) -> np.ndarray:
        """
        从 MeshSystem 获取当前粒子位置。

        Returns:
            positions: (N, 3) float32 数组；若无粒子则返回 (0, 3) 空数组。
        """
        _chem_logger.debug(f"[_get_particle_positions] mesh_system is None: {self.mesh_system is None}")
        
        if self.mesh_system is not None:
            n = self.mesh_system.current_particle_num
            _chem_logger.debug(f"[_get_particle_positions] current_particle_num: {n}")
            
            if n > 0:
                positions = self.mesh_system.particle_positions[:n]
                _chem_logger.debug(f"[_get_particle_positions] 返回 {n} 个粒子位置，形状: {positions.shape}")
                return positions
            else:
                _chem_logger.debug("[_get_particle_positions] 粒子数量为0")
        else:
            _chem_logger.debug("[_get_particle_positions] mesh_system 为 None")
            
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
        _chem_logger.debug("[_update_particle_mapping] 开始更新粒子映射")
        
        positions = self._get_particle_positions()
        n = len(positions)
        
        _chem_logger.debug(f"[_update_particle_mapping] 获取到 {n} 个粒子位置")

        # 重置单元分配
        self.chem_field.reset_cell_assignments()
        _chem_logger.debug("[_update_particle_mapping] 已重置化学单元分配")

        if n == 0:
            _chem_logger.debug("[_update_particle_mapping] 没有粒子，跳过映射更新")
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
        
        _chem_logger.debug(f"[_update_particle_mapping] 计算得到 {len(cell_ids)} 个单元ID")

        # 注册粒子到化学单元
        assigned_count = 0
        for pid in range(n):
            self.chem_field.assign_particle(int(pid), int(cell_ids[pid]))
            assigned_count += 1
            
        _chem_logger.debug(f"[_update_particle_mapping] 成功分配 {assigned_count} 个粒子到化学单元")
        
        # 验证分配结果
        total_particles_in_cells = sum(len(particles) for particles in self.chem_field.cell_particles)
        _chem_logger.debug(f"[_update_particle_mapping] 化学场中总粒子数: {total_particles_in_cells}")

    # ------------------------------------------------------------------
    # 浓度聚合（粒子→单元）
    # ------------------------------------------------------------------

    def _aggregate_cell_concentrations(self) -> np.ndarray:
        """
        体积加权平均：将粒子浓度聚合到各化学单元。

        由于每个粒子体积相同，体积加权平均等于算术平均。
        空单元格使用背景浓度值，避免传递负数浓度给PHREEQCRM。

        Returns:
            cell_conc: (n_cells, n_comps) float64 每个单元的平均浓度
        """
        n_cells = self.chem_field.n_cells
        # 使用非常小的正数作为背景浓度，而不是负数
        background_conc = 0.0000
        cell_conc = np.full((n_cells, self.n_comps), background_conc, dtype=np.float64)

        if self._particle_solutions is None or self.n_comps == 0:
            _chem_logger.debug("[_aggregate_cell_concentrations] 粒子溶液数据未初始化或组分数量为0")
            return cell_conc

        # 更准确地获取粒子数量
        n = 0
        if self.mesh_system is not None:
            n = self.mesh_system.current_particle_num
            _chem_logger.debug(f"[_aggregate_cell_concentrations] 从mesh_system获取粒子数: {n}")
        elif self._particle_cell_ids is not None:
            n = len(self._particle_cell_ids)
            _chem_logger.debug(f"[_aggregate_cell_concentrations] 从_particle_cell_ids获取粒子数: {n}")
        else:
            _chem_logger.debug("[_aggregate_cell_concentrations] 无法获取粒子数量")
            
        if n == 0:
            _chem_logger.debug("[_aggregate_cell_concentrations] 没有活跃粒子，返回背景浓度矩阵")
            return cell_conc

        # 验证粒子溶液数组大小
        if self._particle_solutions.shape[0] < n:
            _chem_logger.warning(f"[_aggregate_cell_concentrations] 粒子溶液数组大小不足: {self._particle_solutions.shape[0]} < {n}")
            n = self._particle_solutions.shape[0]

        # 聚合浓度
        assigned_particles = 0
        assigned_cells = 0
        for cell_id in range(n_cells):
            particles = self.chem_field.cell_particles[cell_id]
            if not particles:
                continue
            # 验证粒子ID有效性
            valid_particles = [pid for pid in particles if 0 <= pid < n]
            if not valid_particles:
                continue
            # 等体积粒子：直接算术平均
            cell_conc[cell_id] = self._particle_solutions[
                np.array(valid_particles, dtype=np.int32)
            ].mean(axis=0)
            assigned_particles += len(valid_particles)
            assigned_cells += 1
            
        _chem_logger.debug(f"[_aggregate_cell_concentrations] 成功聚合 {assigned_particles} 个粒子到 {assigned_cells} 个化学单元")
        _chem_logger.debug(f"[_aggregate_cell_concentrations] 浓度范围: [{np.min(cell_conc):.2e}, {np.max(cell_conc):.2e}]")
        print("cell_conc:", cell_conc)
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
            'H+': 1.006e-07,      # 氢离子 molality
            'OH-': 1.006e-07,     # 氢氧根离子 molality
            'H': 1.433e-25,       # 氢原子 molality
            'H2': 7.166e-26,      # 氢气 molality
            'O': 0.000e+00,       # 氧原子 molality
            'O2': 0.000e+00,      # 氧气 molality
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
    
        修改说明：
        1. 始终构建完整的 nxyz × ncomps 浓度数组传给 SetConcentrations
        2. 用 CreateMappingSWIG 标记非活跃单元为 -1
        3. 非活跃单元填入合理的背景值（纯水浓度），避免 H/O 为零
        4. 从 GetConcentrations 解析返回值时，按完整 nxyz × ncomps 处理
    
        Args:
            cell_conc: (n_cells, n_comps) 输入浓度矩阵
            dt:        化学时间步（秒）
    
        Returns:
            new_conc: (n_cells, n_comps) 更新后的完整网格浓度矩阵
                      负浓度已被截断为 0，并记录到 chemical_warnings.log。
        """
        if self.rm is None:
            return cell_conc.copy()

        n_cells = self.chem_field.n_cells
    
        # ── 识别活跃单元 ──
        major_components = [0, 1, 2]  # H, O, Charge
        other_comps_indices = [i for i in range(self.n_comps) if i not in major_components]
        
        # 检查哪些单元有粒子且不是纯水
        if other_comps_indices:
            other_comps_zero = np.all(cell_conc[:, other_comps_indices] == 0.0, axis=1)
            has_particles = np.array([len(self.chem_field.cell_particles[i]) > 0 for i in range(n_cells)])
            significant_cells_mask = has_particles & ~other_comps_zero
        else:
            significant_cells_mask = np.any(cell_conc > 1e-9, axis=1)
            
        n_significant = int(np.sum(significant_cells_mask))
        if n_significant == 0:
            _chem_logger.info(f"Frame {self.frame_count}: 没有显著浓度的化学单元，跳过化学计算")
            return cell_conc.copy()
    
        # ── 创建映射 ──
        swig_mapping = [-1] * n_cells
        c = 0
        for i in range(n_cells):
            if significant_cells_mask[i]:
                swig_mapping[i] = c
                c += 1
        
        try:
            self.rm.CreateMappingSWIG(swig_mapping)
        except Exception as e:
            _chem_logger.warning(f"Frame {self.frame_count}: CreateMappingSWIG 失败: {e}")
            return cell_conc.copy()

        # ──【关键修复】构建完整的 nxyz × ncomps 浓度数组 ──
        # 非活跃单元填入缓存的初始平衡浓度（纯水值），避免 H/O 为零
        full_conc = cell_conc.copy()  # (n_cells, n_comps)

        # 对没有粒子的空单元，填入合理的纯水背景浓度
        if self._initial_equilibrium_conc is not None:
            # 用初始化时缓存的平衡浓度中的"纯水"行作为背景
            # 找一个纯水单元的浓度作为模板
            pure_water_row = None
            if hasattr(self, 'pure_water_cells') and np.any(self.pure_water_cells):
                first_pw_idx = np.where(self.pure_water_cells)[0][0]
                pure_water_row = self._initial_equilibrium_conc[first_pw_idx]

            if pure_water_row is not None:
                for i in range(n_cells):
                    if not significant_cells_mask[i]:
                        full_conc[i] = pure_water_row
        else:
            # 没有缓存，手动设置纯水浓度
            for i in range(n_cells):
                if not significant_cells_mask[i]:
                    # H ≈ 110.68, O ≈ 55.35, Charge ≈ 0, 其他 = 0
                    if self.n_comps >= 3:
                        full_conc[i, 0] = 110.68  # H
                        full_conc[i, 1] = 55.35   # O
                        full_conc[i, 2] = 0.0     # Charge

        # SetConcentrations 要求 component-major 布局:
        # [comp0_cell0, comp0_cell1, ..., comp0_cellN, comp1_cell0, ...]
        # 即 (n_comps, n_cells) 展平
        conc_flat_full = full_conc.T.flatten().tolist()  # 长度 = n_comps * n_cells
        
        # ── 设置时间参数和物理参数 ──
        self.rm.SetTime(self.accumulated_chem_dt)
        self.rm.SetSaturationUser([1.0] * n_cells)
        self.rm.SetRepresentativeVolume([1.0] * n_cells)
        self.rm.SetPorosity([0.2] * n_cells)
        self.rm.SetTemperature([25.0] * n_cells)
        self.rm.SetPressure([1.0] * n_cells)
    
        # ── 重试逻辑 ──
        current_dt = dt
        converged = False
        last_error_msg = ""
            
        for retry in range(CHEM_MAX_RETRIES):
            self.rm.SetTimeStep(current_dt)
            try:
                # 【关键】传入完整数组！
                self.rm.SetConcentrations(conc_flat_full)
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
    
        # ── 解析返回值（也是完整 nxyz × ncomps） ──
        result_flat = np.array(self.rm.GetConcentrations(), dtype=np.float64)
        expected_len = n_cells * self.n_comps
        if len(result_flat) != expected_len:
            _chem_logger.warning(f"Frame {self.frame_count}: 返回长度不匹配: 期望 {expected_len}, 实际 {len(result_flat)}")
            return cell_conc.copy()

        # GetConcentrations 也是 component-major 布局
        new_conc = result_flat.reshape((self.n_comps, n_cells)).T  # (n_cells, n_comps)

        # 非活跃单元的值被 PHREEQCRM 设为 1e30，还原为原始值
        for i in range(n_cells):
            if not significant_cells_mask[i]:
                new_conc[i] = cell_conc[i]

        # 负浓度截断
        np.clip(new_conc, 0.0, None, out=new_conc)
        
        _chem_logger.debug(f"Frame {self.frame_count}: 化学步执行完成，输出浓度范围: [{np.min(new_conc):.2e}, {np.max(new_conc):.2e}]")
        
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
        # 抽一个粒子出来看看数据对不对
        if len(self._particle_solutions) > 0:
            print(f"[CoupledSim] 第0个粒子的浓度: {self._particle_solutions[0]}")
        else:
            print("[CoupledSim] 粒子溶液数组为空")

    def _calculate_ph_from_concentrations(self, h_index: int = 0) -> Optional[np.ndarray]:
        """
        从粒子浓度数据计算pH值（回退方案）
        
        Args:
            h_index: H+组分的索引
            
        Returns:
            ph_array: (n_active_particles,) float64 数组，或None
        """
        if self._particle_solutions is None:
            return None
            
        n_active_particles = (
            self.mesh_system.current_particle_num
            if self.mesh_system is not None
            else (
                len(self._particle_cell_ids)
                if self._particle_cell_ids is not None
                else 0
            )
        )
        
        if n_active_particles == 0:
            return None
            
        # 确保H+索引有效
        if h_index >= self.n_comps:
            # 尝试查找H+组分
            if 'H+' in self.comp_names:
                h_index = self.comp_names.index('H+')
            else:
                h_index = 0  # 默认使用第一个组分
        
        # 从粒子溶液中提取H+浓度并计算pH
        h_concentrations = self._particle_solutions[:n_active_particles, h_index]
        # pH = -log10[H+]，注意浓度单位转换
        ph_values = -np.log10(np.maximum(h_concentrations, 1e-14))  # 避免log(0)
        
        _chem_logger.debug(f"通过浓度计算得到 {n_active_particles} 个粒子的pH值，范围: [{np.min(ph_values):.2f}, {np.max(ph_values):.2f}]")
        return ph_values


    # ------------------------------------------------------------------
    # 粒子化学状态访问
    # ------------------------------------------------------------------

    def get_particle_ph(self, ph_component_index: int = 0) -> Optional[np.ndarray]:
        """
        使用PHREEQCRM的SelectedOutput获取每个粒子的pH值。

        Args:
            ph_component_index: 对应pH的组分索引（默认0，通常是H+）

        Returns:
            ph_array: (n_active_particles,) float64 数组，或None
        """
        if self.rm is None or self._particle_solutions is None:
            return None
            
        try:
            # 使用SelectedOutput获取pH值
            # PHREEQCRM的SelectedOutput返回的是cell级别的数据（numpy数组格式）
            selected_output = self.rm.GetSelectedOutput()
            
            if selected_output is not None and selected_output.size > 0:
                # selected_output已经是numpy数组格式
                if not isinstance(selected_output, np.ndarray):
                    selected_output = np.array(selected_output, dtype=np.float64)
                
                # 验证SelectedOutput的长度是否匹配化学单元数
                expected_length = self.chem_field.n_cells
                if len(selected_output) != expected_length:
                    _chem_logger.warning(f"SelectedOutput长度({len(selected_output)})与化学单元数({expected_length})不匹配")
                    # 尝试截断或填充到正确长度
                    if len(selected_output) > expected_length:
                        selected_output = selected_output[:expected_length]
                    else:
                        padded = np.full(expected_length, 7.0, dtype=np.float64)
                        padded[:len(selected_output)] = selected_output
                        selected_output = padded
                
                # 获取当前活跃粒子数
                n_active_particles = (
                    self.mesh_system.current_particle_num
                    if self.mesh_system is not None
                    else (
                        len(self._particle_cell_ids)
                        if self._particle_cell_ids is not None
                        else 0
                    )
                )
                
                if n_active_particles > 0:
                    # 创建pH数组
                    ph_values = np.zeros(n_active_particles, dtype=np.float64)
                    
                    # 根据粒子所在的化学单元获取对应的pH值
                    for i in range(n_active_particles):
                        if self._particle_cell_ids is not None and i < len(self._particle_cell_ids):
                            cell_id = self._particle_cell_ids[i]
                            if 0 <= cell_id < len(selected_output):
                                # 从SelectedOutput获取该单元的pH值
                                ph_values[i] = selected_output[cell_id]
                            else:
                                # 如果无法获取，使用默认pH值7.0
                                ph_values[i] = 7.0
                        else:
                            # 默认pH值
                            ph_values[i] = 7.0
                    
                    _chem_logger.debug(f"成功获取 {n_active_particles} 个粒子的pH值，范围: [{np.min(ph_values):.2f}, {np.max(ph_values):.2f}]")
                    return ph_values
            
            # 如果SelectedOutput不可用，回退到使用粒子溶液数据计算pH
            return self._calculate_ph_from_concentrations(ph_component_index)
            
        except Exception as e:
            _chem_logger.warning(f"获取pH值失败: {e}")
            # 回退方案：使用粒子溶液数据计算pH
            return self._calculate_ph_from_concentrations(ph_component_index)

    def get_cell_concentrations(self) -> np.ndarray:
        """
        获取当前化学单元浓度矩阵（聚合但未执行化学步）。

        Returns:
            cell_conc: (n_cells, n_comps) float64
        """
        return self._aggregate_cell_concentrations()
    
    def force_update_mapping(self) -> None:
        """
        强制重新更新粒子映射（用于调试和故障排除）
        """
        _chem_logger.info("强制更新粒子映射...")
        self._update_particle_mapping()
        total_assigned_particles = sum(len(particles) for particles in self.chem_field.cell_particles)
        _chem_logger.info(f"强制更新完成: 分配了 {total_assigned_particles} 个粒子到化学单元")

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
                _chem_logger.info(f"Frame {self.frame_count}: 开始执行化学步")
                
                # 1. 聚合单元浓度（体积加权平均）
                cell_conc = self._aggregate_cell_concentrations()
                
                # 详细调试信息
                if cell_conc.size > 0:
                    valid_cells = ~np.all(cell_conc == -1.0, axis=1)
                    n_valid_cells = np.sum(valid_cells)
                    if n_valid_cells > 0:
                        valid_conc = cell_conc[valid_cells]
                        _chem_logger.info(f"Frame {self.frame_count}: 有效化学单元数: {n_valid_cells}/{self.chem_field.n_cells}")
                        _chem_logger.debug(f"聚合浓度统计: min={np.min(valid_conc):.2e}, max={np.max(valid_conc):.2e}, mean={np.mean(valid_conc):.2e}")
                    else:
                        _chem_logger.warning(f"Frame {self.frame_count}: 没有有效的化学单元！")
                        _chem_logger.debug(f"完整浓度矩阵形状: {cell_conc.shape}, 全部标记为-1")
                
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
                
                _chem_logger.info(f"Frame {self.frame_count}: 化学步执行完成")
                
            except Exception as e:
                _chem_logger.error(f"化学步执行失败 (Frame {self.frame_count}): {e}")
                import traceback
                _chem_logger.error(traceback.format_exc())
        else:
            _chem_logger.debug(f"Frame {self.frame_count}: 不执行化学步")
            # # 直接获取初始浓度值作为浓度（注意：PHREEQCRM返回component-major布局）
            # concentrations = np.array(self.rm.GetConcentrations(), dtype=np.float64)
            # print(f"原始浓度形状: {concentrations.shape}")
            # print(f"目标形状: ({self.chem_field.n_cells}, {self.n_comps})")
            #
            # # PHREEQCRM返回的是component-major布局：[comp0_cell0, comp0_cell1, ..., comp1_cell0, comp1_cell1, ...]
            # # 需要reshape为(n_comps, n_cells)然后转置得到(n_cells, n_comps)
            # cell_conc = concentrations.reshape((self.n_comps, self.chem_field.n_cells)).T
            # print(f"重塑后形状: {cell_conc.shape}")
            #
            # self._broadcast_to_particles(cell_conc)
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
        # self.rm.SetComponentH2O(False)
        for _ in range(n_frames):
            stats = self.step()
            history.append(stats)
        return history
