"""
化学-物理耦合模块
Chemistry-Physics Coupling Prototype Module

将 modular_mesh_system（SPH物理系统）与 core_simulation.py（PHREEQCRM化学系统）整合，
暂时只做液相。

核心设计：
  - 粒子→反应单元映射：粗网格哈希（CHEM_GRID_DIM^3 个单元，当前 10^3=1000）
  - 化学时间步 = 物理步的 N 倍（N=2，可配置）
  - 若 PHREEQCRM 不收敛，自动减半 dt_chem 重试，最多 3 次
  - 浓度同步：体积加权平均（cell→粒子广播）
  - 负浓度自动截断并标记，写入 chemical_warnings.log

粒子身份约定（决策 D3）：化学状态一律以引擎的稳定 particleId 为主键，
_particle_solutions 的第 i 行属于"id == i 的那个粒子"，而不是"当前排在第 i
行的那个粒子"。物理层的行序由 MeshSystem.particle_ids 逐帧给出映射。
"""

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import phreeqcrm
from utils import build_consecutive_mapping
from phreeqc_config import (
    WATER_COMPOSITIONS,
    build_component_major_concentrations,
    composition_values,
    mark_rm_mapped,
    neutral_water_values,
    require_components,
    require_rm_mapped,
)

# ==================== 日志配置 ====================
# 原实现每帧写数十条 DEBUG 诊断，实测 26 s 追加 3.4 MB，历史文件已到 100 MB。
# 现在 DEBUG 一律不落盘（需要时 PHREEGUI_LOG_LEVEL=DEBUG 显式打开），
# 默认 INFO 只保留每帧 2 条决策级证据：
#   "本步下发反应的化学单元: n/1000"（D7 判据）
#   "化学步执行完成"
# 控制台默认只出 WARNING/ERROR。
_LOG_LEVEL_NAME = os.environ.get("PHREEGUI_LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.WARNING)
_chem_logger = logging.getLogger("chemical_warnings")
if not _chem_logger.handlers:
    _chem_logger.setLevel(logging.DEBUG)
    _fh = logging.FileHandler("chemical_warnings.log", encoding='utf-8')
    _fh.setLevel(_LOG_LEVEL)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    _chem_logger.addHandler(_fh)
    # 根 logger 在 GUI 里也挂了控制台 handler；不关传播会把每条记录再写一遍控制台。
    _chem_logger.propagate = False
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.WARNING)
    _ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    _chem_logger.addHandler(_ch)

# ==================== 默认配置 ====================
CHEM_GRID_DIM: int = 10          # 化学单元网格边长（10×10×10 = 1000 个单元）
CHEM_STEP_RATIO: int = 3        # 每 N 帧物理步执行一次化学步
CHEM_MAX_RETRIES: int = 3       # PHREEQCRM.RunCells() 失败时最大重试次数
DEFAULT_PARTICLE_RADIUS: float = 0.002        # 来自 Config.PARTICLE_RADIUS
DEFAULT_PARTICLE_VOLUME: float = (4.0 / 3.0) * np.pi * DEFAULT_PARTICLE_RADIUS ** 3

last_data = None

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
        cell_particles: 每个单元包含的粒子 ID（批量登记时是 numpy 视图，
                        单粒子登记时是 Python list；两者都支持 len / in / 迭代）
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

        # 两种登记方式共用一个出口（cell_particles 属性）：
        #   单粒子 assign_particle -> _cell_lists（Python list，测试/小场景用）
        #   批量 assign_particles_bulk -> CSR（_bulk_offsets + _bulk_ids）
        # 批量路径绝不物化 Python int：10M 粒子时 .tolist() 要造 10M 个
        # PyObject（约 360 MB）并逐个写入 list，实测占整帧 glue 的 72%。
        # 生产代码只读 len()/占用与否，numpy 视图完全够用。
        self._cell_lists: List[List[int]] = [[] for _ in range(self.n_cells)]
        self._bulk_ids: Optional[np.ndarray] = None
        self._bulk_offsets: Optional[np.ndarray] = None
        self.cell_volume: np.ndarray = np.zeros(self.n_cells, dtype=np.float64)
        self.phreeqc_conc: Optional[np.ndarray] = None  # 由外部初始化后填充
        self.reaction_dt: float = 0.0

    @property
    def cell_particles(self):
        """每个单元的粒子 ID 序列（长度 n_cells）。"""
        if self._bulk_ids is not None:
            off = self._bulk_offsets
            ids = self._bulk_ids
            return [ids[off[c]:off[c + 1]] for c in range(self.n_cells)]
        return self._cell_lists

    @property
    def cell_counts(self) -> np.ndarray:
        """(n_cells,) int64，每个单元的粒子数。"""
        if self._bulk_offsets is not None:
            return np.diff(self._bulk_offsets)
        return np.fromiter((len(c) for c in self._cell_lists),
                           dtype=np.int64, count=self.n_cells)

    def reset_cell_assignments(self) -> None:
        """清空粒子→单元分配，每次重新建立映射前调用。"""
        self._bulk_ids = None
        self._bulk_offsets = None
        for cell_list in self._cell_lists:
            cell_list.clear()
        self.cell_volume[:] = 0.0

    def assign_particle(self, pid: int, cell_id: int) -> None:
        """将粒子注册到指定化学单元，并累加单元体积。"""
        if self._bulk_ids is not None:
            # 单粒子追加只出现在测试/小场景。先把批量结果落成 list 再追加，
            # 保证两种登记方式混用时 cell_particles 仍然是同一份真相。
            self._cell_lists = [list(v) for v in self.cell_particles]
            self._bulk_ids = None
            self._bulk_offsets = None
        self._cell_lists[cell_id].append(int(pid))
        self.cell_volume[cell_id] += self.particle_volume

    def assign_particles_bulk(self, ids: np.ndarray, cell_ids: np.ndarray) -> None:
        """批量登记粒子→单元，结果与逐个 assign_particle 完全一致。

        逐粒子 append 是 O(N) 次 Python 调用，外加 O(N) 次 numpy 标量读改写
        （cell_volume[c] += v 每次都要进出 numpy）。这里改成一次排序分组。

        【为什么不用 argsort(kind="stable")】numpy 没有基数 argsort，stable
        argsort 退化成归并排序：实测 10M 粒子要 3231 ms，是当时整帧 glue
        （5808 ms）里最大的单项。
        改为把 (cell, 行号) 打包成一个**唯一**的 int64 键再 np.sort：
          - 键唯一 ⇒ 任何正确的排序都给出同一个顺序，稳定性不再需要，
            于是可以用最快的 introsort（实测 10M 只要 158 ms，快 20×）；
          - 低 shift 位就是行号，掩码取回即得与 stable argsort 逐位相同的 order
            （已用 100K/1M/10M 三档对照验证）。
        单元内粒子顺序仍按传入顺序（与 append 顺序一致）。
        """
        ids = np.asarray(ids)
        n = int(ids.size)
        counts = np.bincount(cell_ids, minlength=self.n_cells)
        self.cell_volume = counts * self.particle_volume
        self._bulk_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
        if n == 0:
            self._bulk_ids = np.zeros(0, dtype=ids.dtype)
            return

        shift = max(1, n.bit_length())
        mask = np.int64((1 << shift) - 1)
        # 原地构造键，少两块 N*8 的瞬时分配
        key = np.asarray(cell_ids, dtype=np.int64).copy()
        key <<= shift
        key |= np.arange(n, dtype=np.int64)
        key.sort()
        self._bulk_ids = ids[key & mask]


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
        chem_grid_dim:    化学粗网格边长（默认 CHEM_GRID_DIM=10，即 10×10×10=1000 cells）
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
        chem_dt: float = 0.017,
    ):
        self.mesh_system = mesh_system
        self.rm: Optional[phreeqcrm.PhreeqcRM] = phreeqc_rm
        if self.rm is not None:
            self.rm.SetFilePrefix("coupled_simulation")
            self.rm.SetPrintChemistryOn(True, True, True)
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

        # 粒子化学数据（【决策 D3】按稳定 particleId 索引，不是按引擎行号）
        # _particle_solutions shape: (max_particle_id + 1, n_comps)，第 i 行属于 id==i 的粒子
        self._particle_solutions: Optional[np.ndarray] = None
        # _particle_solution_ready[i]：id==i 的粒子是否已被播种过（避免重复初始化覆盖结果）
        self._particle_solution_ready: Optional[np.ndarray] = None
        # _particle_cell_ids: 同上，按 particleId 索引，值 = 所属化学单元
        self._particle_cell_ids: Optional[np.ndarray] = None
        # _row_ids: 本帧"引擎行号 -> particleId"的映射（每帧从 MeshSystem 刷新）
        self._row_ids: Optional[np.ndarray] = None

        self.frame_count: int = 0

        # 初始平衡浓度缓存（用于空单元填充和回退）
        self._initial_equilibrium_conc: Optional[np.ndarray] = None
        # (n_cells,) bool：该单元的初始平衡场里没有任何次要组分 = 纯水
        self.pure_water_cells: np.ndarray = np.zeros(0, dtype=bool)

        # 【实测驱动的新增，决策 D4】上一次"收敛成功"的 RunCells 里，究竟哪些
        # 单元真的参与了化学计算。独立探针（D:\tmp_pg_verify\impl\
        # selout_probe_utf8.log，exit 0）实测：CreateMappingSWIG 置 -1 的单元
        # 在 GetConcentrations() 里是 1e30（生产代码已在 :878 还原），但在
        # GetSelectedOutput() 里是**恒 0.0**，且 997/997 全部如此。
        # 也就是说 SelectedOutput 的 1000 行里，"本步没算的单元"给的是 0 这个
        # 具体数字，不是 NaN —— 直接取用就等于把"没算"画成"pH=0 的强酸"。
        #  None = 本次运行还没有任何一次收敛的 RunCells，此时 1000 行全是
        #  initialize_grid_distribution 的平衡值。
        #  【决策 D8 后重测，run17/run18 两次 e2e 的"点击前"探针】
        #  旧的实测区间 min=2.0404 / max=11.9493 属于** acid/base 预铺时代 **：
        #  那是 180 个酸性单元 + 80 个碱性单元被预先铺进化学场的结果，
        #  D8 之后启动布局只有背景溶液，1000 行 pH 恒等于 6.997394
        #  （min=max=mean=median，Cl/Na 全 0，全零行 0 个），
        #  见 D:\tmp_pg_verify\impl\e2e2\run17_neutral_final.txt 的阶段0(D8) 行。
        self._chem_mapped_mask: Optional[np.ndarray] = None
        self._chem_mapped_frame: int = -1
        # 【收尾 1 —— 领队 seq 26】本帧 _run_chemical_step 是否**真的**跑了一次
        # 收敛的 RunCells。原来 step() 无条件打印"化学步执行完成"并把
        # stats["chemical_step"] 置 True，于是"跳过化学计算"的帧在日志和界面上
        # 都显示为"化学步已执行"（决策 D4：假成功比失败更危险）。
        # （上面 :207-208 引用的 min=2.0404/max=11.9493 是**决策 D8 之前**酸/碱
        #  预铺时的实测；D8 改为全中性背景后这个区间不再成立，本轮重测并记录。）
        self._chem_step_executed: bool = False

    # ------------------------------------------------------------------
    # 粒子身份（决策 D3）：稳定 particleId 是唯一主键
    # ------------------------------------------------------------------

    def _active_ids(self) -> np.ndarray:
        """本帧所有活跃粒子的稳定 particleId（int64，按引擎当前行序）。

        【修复 b3】旧代码把"引擎行号"当"粒子号"用。引擎每 stepsPerZSort 帧会做一次
        z-sort，把 position/velocity/.../particleId 一起重排（见
        SPlisHSPlasH/SPlisHSPlasH/FluidModel.cpp:330 performNeighborhoodSearchSort），
        行号与粒子就不再是一回事。这里显式向 MeshSystem 要 id，绝不假设 row==id。

        实测说明：本应用运行时 sim.zSortEnabled() == False，所以今天 row 恰好
        等于 id；但那是配置巧合，不是契约，代码不再依赖它。
        """
        if self.mesh_system is None:
            return np.zeros(0, dtype=np.int64)

        n = int(self.mesh_system.current_particle_num)
        if n <= 0:
            return np.zeros(0, dtype=np.int64)

        ids = getattr(self.mesh_system, "particle_ids", None)
        if ids is None:
            raise RuntimeError(
                "MeshSystem 未暴露 particle_ids，化学层无法建立稳定的粒子身份。"
                "（决策 D4：禁止退回'行号当粒子号'的旧假设。）")

        ids = np.asarray(ids[:n], dtype=np.int64)
        if ids.size != n:
            raise RuntimeError(
                f"particle_ids 长度不足: 需要 {n}，只有 {ids.size}")
        if np.any(ids < 0):
            raise RuntimeError(
                f"particle_ids 含负值（最小 {int(ids.min())}），引擎身份缓冲区异常")
        return ids

    def _ensure_solution_rows(self, n_rows: int) -> None:
        """把按 id 索引的化学数组扩到至少 n_rows 行（行号 == particleId）。"""
        if self.n_comps == 0 or not self.comp_names:
            self._sync_components()

        n_rows = int(n_rows)
        comps = self.n_comps

        if self._particle_solutions is None:
            self._particle_solutions = np.zeros((n_rows, comps), dtype=np.float64)
            self._particle_solution_ready = np.zeros(n_rows, dtype=bool)
            _chem_logger.debug("创建粒子溶液数组: (%d, %d)", n_rows, comps)
            return

        n_have = int(self._particle_solutions.shape[0])

        # 【不变量 I1，决策 D4】_particle_solutions 与 _particle_solution_ready
        # 必须同生同长。二者在正常链路里只会被本方法 :244-246 成对创建、
        # 被 seed_particle_chemistry :319-321 成对使用。一旦只有一边可信，
        # "未播种掩码"就不可信，两种猜法都是投毒：
        #   当成"全未播种" -> _initialize_new_particles(:681) 用中性水覆盖
        #                     已经跑过反应的老粒子（b4/b5 明令禁止的行为）；
        #   当成"全已播种" -> 漏播种的粒子永远保持零浓度喂给 PHREEQC。
        # 所以这里只能硬失败，不能兜底重建。
        ready = self._particle_solution_ready
        if ready is None or int(ready.shape[0]) != n_have:
            raise RuntimeError(
                "化学层内部状态不一致：_particle_solutions 有 %d 行，"
                "_particle_solution_ready 却是 %s。未播种掩码不可信，"
                "继续运行必然在\"中性水覆盖已反应粒子\"和\"零浓度喂 PHREEQC\""
                "之间二选一，两者都是静默降级。请检查是否有代码绕过 "
                "seed_particle_chemistry()/_ensure_solution_rows() 单独赋值了"
                "浓度数组。（决策 D4：禁止兜底重建。）"
                % (n_have, "None（从未创建）" if ready is None
                   else "只有 %d 行" % int(ready.shape[0])))

        if self._particle_solutions.shape[1] != comps:
            raise RuntimeError(
                f"粒子浓度数组列数 {self._particle_solutions.shape[1]} != 组分数 {comps}："
                "组分定义在运行中发生了变化，旧数据无法安全沿用。"
                "（决策 D4：禁止静默重建并丢弃已有化学状态。）")

        if n_have < n_rows:
            grow_to = max(n_rows, n_have * 2, 8)
            old = self._particle_solutions
            grown = np.zeros((grow_to, comps), dtype=np.float64)
            grown[:n_have, :] = old
            self._particle_solutions = grown

            # 上面已断言 ready 与 old 等长，老粒子的"已播种"标记原样搬过去，
            # 新扩出来的行一律 False（= 未播种），由 _initialize_new_particles 兜底。
            grown_ready = np.zeros(grow_to, dtype=bool)
            grown_ready[:n_have] = ready
            self._particle_solution_ready = grown_ready
            _chem_logger.debug("扩展粒子溶液数组: %s -> (%d, %d)",
                               old.shape, grow_to, comps)
            ready = grown_ready

        if self._particle_cell_ids is None or self._particle_cell_ids.shape[0] < n_rows:
            old_ids = self._particle_cell_ids
            cell_ids = np.full(n_rows, -1, dtype=np.int32)
            if old_ids is not None:
                keep = min(old_ids.shape[0], n_rows)
                cell_ids[:keep] = old_ids[:keep]
            self._particle_cell_ids = cell_ids

    def seed_particle_chemistry(self, particle_ids: Iterable[int],
                                fluid_type: int) -> int:
        """【修复 b1/b4/b5 + 决策 D6/D7】全仓库唯一的"给新粒子设定初始水化学"入口。

        Args:
            particle_ids: 新增粒子的稳定 particleId（必须由物理层真实返回）
            fluid_type:   1=中性水 / 2=酸性 / 3=碱性，对应 WATER_COMPOSITIONS

        Returns:
            实际播种的粒子数。

        这里刻意不做任何"兜底"：组分表拿不到、类型未知、值为全零，一律抛异常，
        由调用方给用户看（决策 D4）。
        """
        ids = np.asarray(list(particle_ids), dtype=np.int64) if not isinstance(
            particle_ids, np.ndarray) else np.asarray(particle_ids, dtype=np.int64)
        if ids.size == 0:
            return 0

        table = WATER_COMPOSITIONS.get(int(fluid_type))
        if table is None:
            raise RuntimeError(
                f"未知流体类型 {fluid_type!r}；phreeqc_config.WATER_COMPOSITIONS "
                f"只登记了 {sorted(WATER_COMPOSITIONS)}")
        if ids.min() < 0:
            raise RuntimeError(f"particle_ids 含负值: {ids[:8].tolist()}")

        if self.n_comps == 0 or not self.comp_names:
            self._sync_components()

        values = np.asarray(
            composition_values(self.comp_names, table, _chem_logger),
            dtype=np.float64)
        if not np.any(values > 0.0):
            raise RuntimeError(
                f"流体类型 {fluid_type} 的组分表对真实组分 {self.comp_names} "
                "一个都没给出正浓度——等于'没有水'，喂给 PHREEQC 必然不收敛。"
                "请补全 phreeqc_config 里的组分表。")

        self._ensure_solution_rows(int(ids.max()) + 1)
        self._particle_solutions[ids, :] = values
        self._particle_solution_ready[ids] = True

        _chem_logger.info(
            "按 particleId 播种 %d 个粒子（流体类型 %s）: %s",
            ids.size, fluid_type,
            ", ".join("%s=%.6g" % (n, v) for n, v in zip(self.comp_names, values)))
        return int(ids.size)

    def _other_comp_indices(self) -> List[int]:
        """按组分名（不是列号）挑出"次要组分"列下标。

        原来三处硬写 major_components = [0, 1, 2]  # H, O, Charge ——
        那是把当次数据库恰好的前三列当成了语义。现在按名字取，
        H/O/Charge 不在组分表里时返回空列表，由调用方走"任意正浓度"分支。
        """
        major = ("H", "O", "Charge")
        return [i for i, name in enumerate(self.comp_names) if name not in major]

        
    def _initialize_particle_chemistry(self) -> None:
        """计算初始平衡场（供空单元背景使用），并为已存在的粒子播种化学状态。

        【修复 b7 / 决策 D1】原实现第一件事就是"没粒子就直接 return"。
        在"点一下才生成粒子"的新形态下，启动时永远是 0 个粒子，于是
        _initial_equilibrium_conc 永远是 None，_run_chemical_step 只能走
        那条手写死值（1.433e-25）的兜底分支——第一批粒子必然把 PHREEQC 打死。
        现在：单元场的平衡计算无条件做（它只依赖单元浓度，与粒子无关），
        只有"把结果广播给粒子"这一步才按粒子存在与否跳过。
        """
        ids = self._active_ids()
        n_particles = int(ids.size)

        _chem_logger.info(
            "初始化化学状态：当前活跃粒子 %d 个，ids=%s",
            n_particles, ids[:16].tolist())

        if self.n_comps == 0 or not self.comp_names:
            self._sync_components()

        if self.rm is None:
            raise RuntimeError(
                "PhreeqcRM 实例不可用，无法计算初始平衡场。"
                "（决策 D4：禁止退回到手写死浓度继续跑。）")

        # ── 1) 单元场：无条件算一次瞬时平衡，缓存成"纯水背景" ──
        # 【发现 1 修复】读写浓度/跑反应之前必须先确认 rm 已经 CreateMapping。
        # 实测在未映射的 rm 上调 GetConcentrations() 是进程级段错误
        # （exit 139，D:\tmp_pg_verify\impl\raw_raw.log），Python 层拦不住，
        # 只能在这种合法调用序列的入口先拦下来。
        require_rm_mapped(self.rm, "CoupledSimulation._initialize_particle_chemistry")
        self.rm.SetTime(0.0)
        self.rm.SetTimeStep(0.0)          # 瞬时平衡
        self.rm.SetConcentrations(self._create_initial_concentration_array_for_phreeqcrm())

        status = self.rm.RunCells()
        if status != 0:
            raise RuntimeError(
                f"初始平衡计算 RunCells 失败 (status={status})。"
                "此时后续每一步化学都没有合法的纯水背景可用，禁止继续。")

        equilibrium_conc = np.array(self.rm.GetConcentrations(), dtype=np.float64)
        n_cells = self.chem_field.n_cells
        if equilibrium_conc.size != n_cells * self.n_comps:
            raise RuntimeError(
                f"初始平衡返回长度不匹配: 期望 {n_cells * self.n_comps}, "
                f"实际 {equilibrium_conc.size}")

        # GetConcentrations 与 SetConcentrations 同为 component-major
        cell_conc = equilibrium_conc.reshape((self.n_comps, n_cells)).T
        self._initial_equilibrium_conc = cell_conc.copy()

        other_idx = self._other_comp_indices()
        if other_idx:
            self.pure_water_cells = np.all(cell_conc[:, other_idx] == 0.0, axis=1)
            _chem_logger.info("初始化时识别到 %d 个纯水单元",
                              int(np.sum(self.pure_water_cells)))

        # ── 2) 粒子：只给"已存在且尚未播种"的粒子按稳定 id 补中性水初值 ──
        if n_particles > 0:
            self._update_particle_mapping()
            # 先保证按 id 索引的化学数组与"已播种"掩码存在且等长，
            # 才能安全读取未播种掩码（不变量 I1）。
            self._ensure_solution_rows(int(ids.max()) + 1)
            # 【顺序修复】用户可能在点"开始仿真"之前就左键生成了粒子并按
            # fluid_type 播种过化学（酸/碱）。initialize() 的背景平衡值只能
            # 补给未播种的 id，不能整片回写，否则刚播好的流体类型会被洗成中性。
            unsown_mask = ~self._particle_solution_ready[ids]
            unsown_ids = ids[unsown_mask]
            self._initialize_new_particles()
            if unsown_ids.size > 0:
                self._broadcast_to_particles(cell_conc, particle_ids=unsown_ids)
            seeded = self._particle_solutions[ids]
            _chem_logger.info(
                "%d 个粒子化学状态就绪，范围 [%.3e, %.3e]",
                n_particles, float(seeded.min()), float(seeded.max()))
            if np.allclose(seeded, 0.0):
                raise RuntimeError(
                    "粒子化学状态全零：真实组分 "
                    f"{self.comp_names} 没有任何一列被中性水组分表覆盖。")
        else:
            _chem_logger.info("启动时无粒子（决策 D1），化学状态将在点击生成时按 id 播种")

    def _create_initial_concentration_array_for_phreeqcrm(self) -> list:
        """创建用于 PHREEQCRM 的初始浓度数组（component-major 布局）。

        【修复 b1/b2/b4/b5】原实现有两处致命约定错误：
        1) 键写的是"物种名" H+/OH-/Na+/Cl-，而真实组分名是
           H/O/Charge/Cl/Na（rm.GetComponents() 实测）→ 每个 .get() 都落到
           默认分支 → 传进 SetConcentrations 的是全零数组（连水都没有）→
           PhreeqcRM 报 "Oxygen Mass of oxygen has not converged"、
           RunCells 失败并让整进程 exit(127)。
        2) 注释写着 component-major，循环却是"单元在外层"（cell-major）。
           PhreeqcRM 的约定是 component-major，与 _run_chemical_step 里
           full_conc.T.flatten() 的写法一致 —— 本轮统一到 component-major。
           （旧写法"看不出错"只是因为当时初始值全零/对称，转置不可见。）
        """
        n_cells = self.chem_field.n_cells

        flat = build_component_major_concentrations(
            self.comp_names, n_cells, _chem_logger)

        for comp_name, conc in zip(self.comp_names,
                                   neutral_water_values(self.comp_names, _chem_logger)):
            _chem_logger.debug(f"  组分 {comp_name}: 初始浓度 {conc:.2e} mol/L")

        _chem_logger.info(
            f"创建了PHREEQCRM初始浓度数组，长度: {len(flat)}, "
            f"布局: component-major ({self.n_comps}, {n_cells})")
        return flat.tolist()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        初始化化学系统并建立粒子→单元映射。

        必须在 PHREEQCRM 完成 FindComponents() 后调用。
        """
        _chem_logger.info("开始初始化化学系统...")

        # 化学域必须与 SPH 容器的真实内部尺寸一致。
        # 之前一直用 __init__ 的默认 ±0.8，而容器是 4x6x4、粒子活动范围是
        # x,z∈[-2,2]、y∈[0,6] —— 绝大多数粒子落在化学域之外，被 clip 到
        # 边界单元里，浓度分布图自然只剩边缘一条竖线。现在从 mesh_system 取。
        self._sync_domain_with_mesh()

        if self.rm is not None:
            # 【修复 b1/D7】组分名唯一来源 = rm.GetComponentCount() + rm.GetComponents()。
            # 拿不到就硬失败：猜一份 ['H+','OH-','Na+','Cl-'] 会让后面所有
            # dict.get(真实组分名) 落到默认分支，把"没有水"的零浓度喂给 PhreeqcRM，
            # 表现为 RunCells 失败 + 进程 exit(127)，而不是明确的组分不匹配错误。
            self.comp_names = require_components(self.rm)
            self.n_comps = len(self.comp_names)

            n_cells = self.chem_field.n_cells
            self.chem_field.phreeqc_conc = np.zeros(
                n_cells * self.n_comps, dtype=np.float64
            )

            _chem_logger.info(f"化学系统初始化完成: {self.n_comps} 个组分 - {self.comp_names}")
        else:
            raise RuntimeError(
                "ChemistryPhysicsCoupling.initialize(): rm 为 None，"
                "无法确定化学组分。组分必须来自 PhreeqcRM，不允许降级到硬编码列表。")

        # 先更新粒子映射（确保cell_particles有正确的粒子分配）
        _chem_logger.info("更新粒子映射...")
        self._update_particle_mapping()

        # 验证粒子映射结果
        total_assigned_particles = int(self.chem_field.cell_counts.sum())
        _chem_logger.info(f"粒子映射完成: 分配了 {total_assigned_particles} 个粒子到化学单元")

        # 【修复 b7/D1】无条件初始化：单元场的瞬时平衡与"有没有粒子"无关，
        # 它算出的 _initial_equilibrium_conc 是后续每一步空单元的化学背景。
        # 原来用 total_assigned_particles > 0 当门槛，在"点一下才有粒子"的形态下
        # 永远为假 → 背景永远是 None → 首个化学步走手写死值的兜底分支。
        _chem_logger.info("初始化化学平衡场与粒子化学状态...")
        self._initialize_particle_chemistry()

        _chem_logger.info("化学系统初始化完毕")

    # ------------------------------------------------------------------
    # 粒子位置获取
    # ------------------------------------------------------------------

    def _sync_domain_with_mesh(self) -> None:
        """把化学域对齐到 SPH 容器的真实内部边界（唯一真源 = MeshSystem）。

        为什么必须有这一步：_position_to_cell_id 会把域外坐标 clip 到最近单元，
        域配错不会报错，只会让分布图静默变成一条边。容器 4x6x4 时代化学域是
        ±0.8 立方体，等于把 90% 以上的粒子塞进网格边缘，看起来"浓度分布不像
        真实分布"。现在域跟着容器走，并补一点点余量吸收数值越界。
        """
        if self.mesh_system is None:
            _chem_logger.debug(
                "[_sync_domain_with_mesh] mesh_system 为 None，沿用默认化学域 "
                "x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]",
                self.domain_min[0], self.domain_max[0],
                self.domain_min[1], self.domain_max[1],
                self.domain_min[2], self.domain_max[2])
            return

        bmin = getattr(self.mesh_system, "boundary_min", None)
        bmax = getattr(self.mesh_system, "boundary_max", None)
        if bmin is None or bmax is None:
            raise RuntimeError(
                "MeshSystem 未暴露 boundary_min/boundary_max，化学域无法与容器对齐。"
                "（决策 D4：宁可硬失败，也不沿用与容器不一致的默认域。）")
        self.domain_min = np.asarray(bmin, dtype=np.float64).copy()
        self.domain_max = np.asarray(bmax, dtype=np.float64).copy()
        self.domain_size = self.domain_max - self.domain_min
        if np.any(self.domain_size <= 0.0):
            raise RuntimeError(
                f"化学域尺寸非法: min={self.domain_min.tolist()} "
                f"max={self.domain_max.tolist()}")

        _chem_logger.info(
            "化学域已对齐容器: x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]",
            self.domain_min[0], self.domain_max[0],
            self.domain_min[1], self.domain_max[1],
            self.domain_min[2], self.domain_max[2])

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
        n = int(positions.shape[0])
        cell_ids = np.zeros(n, dtype=np.int32)
        if n == 0:
            return cell_ids

        # 逐轴算，不物化 (N,3) float64。原写法是
        #   positions.astype(f64) -> 减 -> 除 -> clip -> 乘 -> astype(i32) -> clip
        # 每一步都新分配一整块 N*3*8 字节：10M 粒子时约 1.5 GB 瞬时分配，
        # 分配器/缺页开销让这一项比排序本身还贵。逐轴后峰值只有 ~2*N*8。
        # 运算顺序与标量版逐字一致（先减后除、clip 到 1-1e-9、再乘 n_dim、
        # 截断成 int32、再 clip 到 n_dim-1），所以结果逐位相同。
        pos64 = np.empty(n, dtype=np.float64)
        for ax, stride in enumerate((1, n_dim, n_dim * n_dim)):
            np.subtract(positions[:, ax], self.domain_min[ax],
                        out=pos64, dtype=np.float64, casting="unsafe")
            pos64 /= self.domain_size[ax]
            np.clip(pos64, 0.0, 1.0 - 1e-9, out=pos64)
            pos64 *= n_dim
            grid = pos64.astype(np.int32)
            np.clip(grid, 0, n_dim - 1, out=grid)
            cell_ids += grid * np.int32(stride)
        return cell_ids

    def _update_particle_mapping(self) -> None:
        """按当前粒子位置重算"粒子 → 化学单元"归属。

        【修复 b3 / 决策 D3】chem_field 里登记的键、以及 _particle_cell_ids /
        _particle_solutions 的行号，一律是引擎的稳定 particleId，不再是行号。
        每帧从 MeshSystem.particle_ids 取"行号 → id"映射后再登记。
        """
        _chem_logger.debug("[_update_particle_mapping] 开始更新粒子映射")

        ids = self._active_ids()
        positions = self._get_particle_positions()
        n = int(ids.size)

        if positions.shape[0] != n:
            raise RuntimeError(
                f"粒子身份与位置长度不一致: ids={n}, positions={positions.shape[0]}。"
                "MeshSystem 在本帧内被改过（计数与缓冲区不同步），继续算会张冠李戴。")

        self._row_ids = ids

        # 重置单元分配
        self.chem_field.reset_cell_assignments()

        if n == 0:
            # 【修复 b3】不再把 _particle_cell_ids 截成空数组：那是按 id 索引的
            # 状态，截断等于把已存在粒子的归属抹掉。粒子暂时为 0 不代表将来为 0。
            _chem_logger.debug("[_update_particle_mapping] 当前无活跃粒子，保留既有 id 状态")
            return

        if self.n_comps == 0 or not self.comp_names:
            self._sync_components()
        self._ensure_solution_rows(int(ids.max()) + 1)

        # 计算并存储单元 ID（按 id 落位，而不是按行号）
        cell_ids = self._position_to_cell_id(positions)
        self._particle_cell_ids[ids] = cell_ids

        # 注册粒子到化学单元（键 = 稳定 particleId）
        self.chem_field.assign_particles_bulk(ids, cell_ids)

        _chem_logger.debug(
            "[_update_particle_mapping] %d 个粒子登记到化学单元，id 范围 [%d, %d]",
            n, int(ids.min()), int(ids.max()))

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

        # 【修复 b3/D3】cell_particles 里存的已经是稳定 particleId，
        # 合法性只能拿"化学数组的行数（= max_id+1）"来判，不能再拿粒子个数判。
        n_rows = int(self._particle_solutions.shape[0])
        if n_rows == 0:
            _chem_logger.debug("[_aggregate_cell_concentrations] 化学数组为空，返回背景浓度矩阵")
            return cell_conc

        # 聚合浓度：等体积粒子的体积加权平均 == 算术平均，用 bincount 一次算完。
        # 数据源与 cell_particles 同源：两者都由 _update_particle_mapping 建立，
        # ids = 本帧活跃粒子，cells = 它们各自的单元号。
        # 原实现是"遍历 1000 个单元 × 逐粒子列表推导 + mean"，O(N) 次 Python 操作。
        ids = self._row_ids
        if ids is None or ids.size == 0:
            _chem_logger.debug("[_aggregate_cell_concentrations] 本帧无活跃粒子")
            return cell_conc

        cells = self._particle_cell_ids[ids]
        valid = (ids >= 0) & (ids < n_rows) & (cells >= 0) & (cells < n_cells)
        dropped = int(np.count_nonzero(~valid))
        if dropped:
            _chem_logger.error(
                "[_aggregate_cell_concentrations] 有 %d 个粒子的 id 超出化学数组行数 %d，"
                "它们的浓度没参与聚合——播种路径漏了，请查 seed_particle_chemistry 调用点",
                dropped, n_rows)
        if not np.any(valid):
            return cell_conc

        vid = ids[valid]
        vcell = cells[valid]
        counts = np.bincount(vcell, minlength=n_cells)
        occupied = counts > 0
        inv_n = np.zeros(n_cells, dtype=np.float64)
        inv_n[occupied] = 1.0 / counts[occupied]
        for c in range(self.n_comps):
            sums = np.bincount(vcell, weights=self._particle_solutions[vid, c],
                               minlength=n_cells)
            cell_conc[:, c] = sums * inv_n

        _chem_logger.debug(
            "[_aggregate_cell_concentrations] 成功聚合 %d 个粒子到 %d 个化学单元",
            int(vid.size), int(np.count_nonzero(occupied)))
        return cell_conc

    # ------------------------------------------------------------------
    def _initialize_new_particles(self) -> None:
        """给"引擎里已存在、但化学层还没播种过"的粒子补一份中性水初值。

        【修复 b3/D3】不再用"粒子数量的增量"猜哪些是新粒子：那假设了行号==粒子号，
        而且一旦有粒子被移除/重排就会把老粒子的浓度覆盖掉。现在按
        _particle_solution_ready 掩码精确挑出未播种的 id。

        正常路径下 GUI 在生成粒子时就调 seed_particle_chemistry() 播过种了，
        这里只会兜住"物理层凭空多出来的粒子"（例如引擎 Emitter）。
        """
        ids = self._active_ids()
        if ids.size == 0:
            return

        self._ensure_solution_rows(int(ids.max()) + 1)
        ready = self._particle_solution_ready
        unseeded = ids[~ready[ids]]
        if unseeded.size == 0:
            return

        _chem_logger.warning(
            "%d 个粒子（ids=%s...）没有经过播种就出现在物理层，按中性水兜底",
            unseeded.size, unseeded[:8].tolist())
        self.seed_particle_chemistry(unseeded, 1)

    def _sync_components(self) -> List[str]:
        """【修复 b1 / 决策 D7】组分名唯一入口；拿不到就硬失败。

        原先这里有 5 处各写一遍 ['H+','OH-','Na+','Cl-'] 的"兜底"。
        那不是兜底而是投毒：物种名 ≠ 组分名，真实组分是 H/O/Charge/Cl/Na，
        于是所有按组分名查表的代码全部落到默认分支，浓度变成全零。
        """
        names = require_components(self.rm)
        self.comp_names = names
        self.n_comps = len(names)
        _chem_logger.info("组分取自 PhreeqcRM: %d 个 - %s", self.n_comps, self.comp_names)
        return names

    def _set_default_initial_concentrations(self, particle_ids: Iterable[int]) -> int:
        """中性水兜底播种：只给指定 id 播种，绝不按行号整片覆盖。

        【修复 b4/b5 / 缺陷 9-3】原实现按"列号 0..3"硬塞
        [1.006e-07, 1.006e-07, 0.0, 0.0]，而真实列序是 H,O,Charge,Cl,Na
        —— 结果 O 列被写成 1e-7（应是 55.35），Cl 列拿到本想给 Na 的值，
        Na 列永远是 0。现在按组分名取值，并且只碰传进来的那批 id：
        以前"发现零浓度就把前 n 行全刷成默认值"会连带抹掉已经算过化学反应
        的老粒子状态，属于用掩盖代替修复。
        """
        return self.seed_particle_chemistry(particle_ids, 1)
    
    def _significant_cells_mask(self, cell_conc: np.ndarray) -> np.ndarray:
        """本步要交给 PHREEQC 求平衡的单元 —— 唯一判据，禁止在别处再写一遍。

        【决策 D7 用户裁决】判据是 **"含粒子"**（不再要求次要组分非 0，
        否则中性水永远不进化学步）。唯一保留的下限是"该行有组成"：
        整行 0 意味着没有水，PHREEQC 必然不收敛（见 :348-352 的同一条不变量）。

        以前 `_run_chemical_step` 与 `_identify_inactive_cells` 各写了一份
        判据，改一处就会让另一处变成过期真相；现在两者共用本函数。

        Returns:
            (n_cells,) bool，True = 本步下发给 PHREEQC。
        """
        has_particles = self.chem_field.cell_counts > 0
        has_composition = np.any(cell_conc > 0.0, axis=1)
        return has_particles & has_composition

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
            self._chem_step_executed = False
            return cell_conc.copy()

        # 【收尾 1】每次进入先清零：只有走到"RunCells 收敛且输出解析成功"
        # 这一条路径末尾才会置 True。下面的每一个提前 return 都代表
        # "本帧确实没有发生化学反应"，调用方必须照此诚实报告。
        self._chem_step_executed = False

        n_cells = self.chem_field.n_cells

        # ── 识别本步要交给 PHREEQC 的单元 ──
        # 【决策 D7 —— 用户裁决：中性水也必须进化学步】
        # 原判据是"含粒子 且 次要组分(Cls/Na 之类)非 0"。中性水的 Cl/Na 按
        # D6 就是 0，于是默认点击路径（fluid_type=1）下上一轮实测 502 帧
        # **100% 命中"没有显著浓度的化学单元，跳过化学计算"**，一次 RunCells
        # 都没跑 —— "仿真在跑"其实是假的，pH 显示的也只是初始平衡值。
        # 现在判据改成"含粒子就求平衡"，纯水也交给 PHREEQC。
        # 仍然只对**含粒子**的单元跑，不退化成每帧算全部 1000 个单元（性能）。
        # 唯一的例外：浓度行整行为 0 的单元。那不是"纯水"，是"没有水"——
        # 本文件 :348-352 对同一条不变量已有硬失败措辞（"喂给 PHREEQC 必然不
        # 收敛"），所以这种单元仍不下发，并把数量打进日志（决策 D4：不许静默）。
        significant_cells_mask = self._significant_cells_mask(cell_conc)

        n_significant = int(np.sum(significant_cells_mask))
        if n_significant == 0:
            _chem_logger.info(
                f"Frame {self.frame_count}: 本步没有任何含粒子且有组成的单元，"
                "化学步确实未执行（不是'跳过'当装饰）")
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
            _chem_logger.error(f"Frame {self.frame_count}: CreateMappingSWIG 失败: {e}")
            return cell_conc.copy()
        # 【实测更正】原注释说"SelectedOutput 的行数会随映射变化"——端到端实测
        # 证伪了它：行数恒等于 chem_field.n_cells（run9 562 次采样全部 =1000，
        # 同期显著单元数在 2~97 之间）。真正的后果不是行数，而是**被置 -1 的
        # 单元那一行恒为 0.0**（独立探针 selout_probe_utf8.log，exit 0）。
        # 所以这里记下本步真实参与化学计算的单元，pH 只信这些行。
        mark_rm_mapped(self.rm, "CoupledSimulation._run_chemical_step")

        # ── 构建完整的 nxyz × ncomps 浓度数组 ──
        # 非活跃单元填入缓存的初始平衡浓度（纯水值），避免 H/O 为零
        full_conc = cell_conc.copy()  # (n_cells, n_comps)

        if self._initial_equilibrium_conc is None:
            raise RuntimeError(
                "本步要用纯水背景填充非活跃单元，但 _initial_equilibrium_conc 是 None："
                "initialize() 的初始平衡场没能算出来。"
                "（决策 D4：禁止退回手写死浓度——原来这里写的是 full_conc[i,0]=1.433e-25，"
                "一个全仓库查不到出处的数，注释却写着 'H ≈ 110.68'。）")

        # 用初始化时缓存的平衡浓度中的"纯水"行作为背景
        pure_water_row = None
        if np.any(self.pure_water_cells):
            first_pw_idx = int(np.where(self.pure_water_cells)[0][0])
            pure_water_row = self._initial_equilibrium_conc[first_pw_idx]

        if pure_water_row is None:
            # 一个纯水单元都没有（全被溶质污染）时，用初始平衡场的逐单元值兜底
            pure_water_row = self._initial_equilibrium_conc[0]
            _chem_logger.warning(
                "Frame %d: 初始平衡场里没有纯水单元，空单元背景退回 0 号单元值",
                self.frame_count)

        full_conc[~significant_cells_mask] = pure_water_row

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

        # 本步真实参与化学计算的单元到此确定。SelectedOutput 只有这些行的值是
        # 算出来的，其余行恒为 0.0（实测，见 __init__ 里 _chem_mapped_mask 的注）；
        # 放在收敛且长度校验之后，保证掩码和当前 GetSelectedOutput() 同源。
        self._chem_mapped_mask = significant_cells_mask.copy()
        self._chem_mapped_frame = self.frame_count
        # 【收尾 1】到这里才叫"化学步真的执行了"：RunCells 返回 0，
        # 且 GetConcentrations 长度校验通过。
        self._chem_step_executed = True

        # 负浓度截断
        np.clip(new_conc, 0.0, None, out=new_conc)
        
        _chem_logger.debug(f"Frame {self.frame_count}: 化学步执行完成，输出浓度范围: [{np.min(new_conc):.2e}, {np.max(new_conc):.2e}]")
        
        return new_conc

    # ------------------------------------------------------------------
    # 浓度广播（单元→粒子）
    # ------------------------------------------------------------------

    def _broadcast_to_particles(self, cell_conc: np.ndarray,
                                particle_ids: Optional[Iterable[int]] = None) -> None:
        """将化学单元浓度广播回单元内粒子（可按 id 限定范围）。

        【修复 b3/D3】cell_particles 里的键是稳定 particleId，
        _particle_solutions 的行号也是 particleId —— 两边天然对齐，
        不需要（也绝不能）再假设"第几行就是第几个粒子"。

        Args:
            cell_conc: (n_cells, n_comps) 更新后的单元浓度
            particle_ids: 只回写这些 id；None = 回写全部粒子。
                initialize() 用它把"背景平衡值"只补给尚未播种的粒子，
                不再覆盖用户点击时按 fluid_type 播种的酸/碱化学状态。
        """
        if self._particle_solutions is None or self.n_comps == 0:
            return
        ids = self._row_ids
        if ids is None or ids.size == 0:
            return
        n_rows = int(self._particle_solutions.shape[0])
        n_cells = self.chem_field.n_cells

        cells = self._particle_cell_ids[ids]
        valid = (ids >= 0) & (ids < n_rows) & (cells >= 0) & (cells < n_cells)
        skipped = int(np.count_nonzero((ids >= n_rows) | (ids < 0)))
        if particle_ids is not None:
            allowed = np.asarray(list(particle_ids), dtype=np.int64)
            valid &= np.isin(ids, allowed)

        if skipped:
            _chem_logger.error(
                "[_broadcast_to_particles] %d 个粒子的 id 超出化学数组行数 %d，"
                "反应结果没有回写到它们身上", skipped, n_rows)

        if not np.any(valid):
            _chem_logger.debug("[_broadcast_to_particles] 回写 0 个粒子")
            return

        vid = ids[valid]
        vcell = cells[valid]
        # 逐列散射：避免一次性物化 (N, n_comps) 的中间数组
        for c in range(self.n_comps):
            self._particle_solutions[vid, c] = cell_conc[vcell, c]
        self._particle_solution_ready[vid] = True

        _chem_logger.debug("[_broadcast_to_particles] 回写 %d 个粒子", int(vid.size))

    def _identify_inactive_cells(self, cell_conc: np.ndarray) -> np.ndarray:
        """
        识别不活跃的化学单元（复用_run_chemical_step中的逻辑）

        Args:
            cell_conc: (n_cells, n_comps) 单元浓度矩阵

        Returns:
            inactive_mask: (n_cells,) bool数组，True表示不活跃单元
        """
        # 【决策 D7】判据唯一来源是 _significant_cells_mask，这里只做取反。
        # 原来本函数自己又写了一遍"含粒子且次要组分非 0"，与化学步那份一起
        # 构成了"两处真相"；任何一处改动都会让另一处变成过期结论。
        return ~self._significant_cells_mask(cell_conc)
        
    # ------------------------------------------------------------------
    # 粒子化学状态访问
    # ------------------------------------------------------------------

    def get_particle_ph(self) -> Optional[np.ndarray]:
        """按稳定 particleId 返回每个活跃粒子的 pH；拿不到就返回 None。

        【修复 b6 / 决策 D4，并按探针 b7A 实测更正】原来这条路径是"编造"而不是
        "降级"：它走 `_calculate_ph_from_concentrations`，算的是
        -log10(元素 H 的总量)。H 总量按 D6 是 110.68 mol/L，于是界面画出来的
        "中性水 pH ≈ -2.04" 是个假数；而且它把 H 列当成 'H+'（组分名 vs 物种名）。
        该方法已删除，现在只返回 PHREEQC 真正产出的 SelectedOutput。

        实测更正（D:\\tmp_pg_verify\\impl\\b7a.log，独立子进程，exit 0）：
          GetSelectedOutputCount() == 1，GetSelectedOutputHeadings() == ['pH']，
          GetSelectedOutput() 返回 ndarray，形状 (单元数, 1)。
        当时写的"RunCells 后 pH 取值 2.0404 ~ 11.9493、无 NaN"是**不完整的**：
        那次探针映射了全部单元，所以看不到另一半真相。后续探针
        （D:\\tmp_pg_verify\\impl\\selout_probe_utf8.log，独立子进程，exit 0）
        只映射 3 个单元做 RunCells，结果 1000 行里 997 行恒等于 0.0 —— 未参与
        本步化学的单元，PHREEQC 给的是常量 0 而不是"没数"。所以本函数现在会
        按 _chem_mapped_mask 把这些行改成 NaN，见下面的 Returns。
        也就是说 pH 走的是 setup_default_chemical_system 里 RunString 注入的
        `SELECTED_OUTPUT 1 / -pH / END`，不需要改 phreeqc.dat，也不需要
        SetSelectedOutputOn —— 之前注释里写的"生产路径永远取不到 pH"是错的。

        Returns:
            ph_array: (n_active_particles,) float64。
                      **顺序契约**：第 k 个元素对应"引擎当前第 k 行"的粒子，
                      即 self._active_ids()[k]（= mesh_system.particle_ids[k]）。
                      不是按 particleId 升序，也不是按单元号，调用方不得再排序；
                      需要 id 时请与 _active_ids() 成对使用。
                      **取值契约**：NaN = 该粒子所在单元本步没有参与化学计算，
                      不等于 pH=0（决策 D4：宁可不给，不可给假数）。调用方若因
                      子存在 NaN 而整帧丢弃 pH，是显示层的取舍，不得改成"把 NaN
                      当 0 画"。
                      完全不可用时返回 None。
        """
        if self.rm is None or self._particle_solutions is None:
            return None

        ids = self._active_ids()
        if ids.size == 0:
            return None

        try:
            n_sel = int(self.rm.GetSelectedOutputCount())
        except Exception as e:
            _chem_logger.error("GetSelectedOutputCount 失败，pH 不可用: %r", e)
            return None

        if n_sel <= 0:
            self._warn_no_ph_once()
            return None

        try:
            headings = [str(h) for h in self.rm.GetSelectedOutputHeadings()]
        except Exception as e:
            _chem_logger.error("GetSelectedOutputHeadings 失败，pH 不可用: %r", e)
            return None

        if 'pH' not in headings:
            _chem_logger.error(
                "SelectedOutput 里没有 pH 列（现有列: %s）——"
                "需要在化学输入里加 SELECTED_OUTPUT 定义，"
                "而不是由程序猜一个 pH。", headings)
            return None
        ph_col = headings.index('pH')

        selected_output = np.asarray(self.rm.GetSelectedOutput(), dtype=np.float64)
        if selected_output.ndim == 1:
            selected_output = selected_output.reshape((-1, 1))
        if selected_output.ndim != 2 or selected_output.shape[1] <= ph_col:
            _chem_logger.error(
                "SelectedOutput 形状异常: %s，无法按列 %r 取 pH",
                selected_output.shape, 'pH')
            return None

        # SelectedOutput 的行序约定 —— 已实测定死，不保留"两条都接受"的猜测分支。
        # 证据一（端到端 run9：真实 GUI 对象图、酸性流体、256 粒子、562 次采样，
        # D:\tmp_pg_verify\impl\e2e2\run9_acid_post_harden.txt）：
        #   行数恒 == 1000 == chem_field.n_cells，而同步"显著单元数"在 2~97 之间
        #   变化 —— CreateMappingSWIG 把非显著单元置 -1 **不会**改变行数。
        #   所以原先的 (b) 分支（行数==显著单元数）从来没成立过；更糟的是万一某步
        #   恰好 n_significant == n_cells，它会把整场 pH 张冠李戴且毫不报错。
        # 证据二（独立子进程探针，exit 0，D:\tmp_pg_verify\impl\selout_probe_utf8.log）：
        #   只映射 3 个单元跑 RunCells 之后，1000 行里被置 -1 的 997 行**全部恒等于
        #   0.0**；同样这些单元在 GetConcentrations() 里给的是 1e30（生产已在
        #   _run_chemical_step 里逐单元还原）。也就是说"行数对得上"并不等于"值可用"：
        #   整场取用等于把"这格本步没算"画成"pH=0 的强酸"，run9 里 GUI 收到的
        #   [0.000, 3.014] 就是它。
        # 处置（决策 D4：宁可不给，不可给假数）：只信上一次收敛的 RunCells 真正算过
        #   的行（_chem_mapped_mask），其余置 NaN。掩码为 None 说明本次运行还没跑过
        #   任何收敛的化学步，此时 1000 行全是 initialize_grid_distribution 的平衡值；
        #   决策 D8 之后启动布局只有背景溶液，那一批值实测恒为 6.997394（run17 探针），
        #   预铺时代的 2.0404/11.9493 已不存在，可以整体取用。
        n_rows = selected_output.shape[0]
        if n_rows != self.chem_field.n_cells:
            _chem_logger.error(
                "SelectedOutput 行数(%d) != 单元总数(%d)。端到端实测的约定是"
                "\"每个单元出一行、与 CreateMapping 的 -1 排除无关\"，现在这个数"
                "对不上，说明 PhreeqcRM 版本/输入定义发生了变化，pH 无法安全对回"
                "单元，本次不提供 pH。（决策 D4：禁止退回按显著单元数对齐的旧猜测。）",
                n_rows, self.chem_field.n_cells)
            return None

        ph_by_cell = selected_output[:, ph_col].copy()
        mask = self._chem_mapped_mask
        if mask is None:
            _chem_logger.debug(
                "SelectedOutput 行数实测=%d == 单元总数 %d；本次运行尚无收敛的 "
                "RunCells，全部行按初始化平衡值取用；n_sel=%d headings=%s",
                n_rows, self.chem_field.n_cells, n_sel, headings)
        elif mask.size != self.chem_field.n_cells:
            _chem_logger.error(
                "化学步参与掩码长度(%d) != 单元总数(%d)：无法判断 SelectedOutput 的"
                "哪些行是真算出来的，本次不提供 pH。（决策 D4）",
                mask.size, self.chem_field.n_cells)
            return None
        else:
            n_unmapped = int((~mask).sum())
            ph_by_cell[~mask] = np.nan
            _chem_logger.debug(
                "SelectedOutput 行数实测=%d == 单元总数 %d；其中第 %d 帧收敛的 "
                "RunCells 真正算过的单元=%d，其余 %d 行是 PHREEQC 给未映射单元的"
                "常量 0.0，已按 D4 置 NaN（不画成 pH=0）；n_sel=%d headings=%s",
                n_rows, self.chem_field.n_cells, self._chem_mapped_frame,
                int(mask.sum()), n_unmapped, n_sel, headings)

        # 粒子 -> 单元 -> pH（_particle_cell_ids 按 particleId 索引）
        # 【待办 3 / 返回顺序契约】ids 来自 _active_ids()，它按引擎行序取
        # mesh_system.particle_ids[:n] —— 第 k 个元素就是"引擎第 k 行"的粒子号。
        # 所以 ph_values[k] 天然对应引擎行 k，调用方（pH 着色）可以直接按行号用，
        # 不需要也不能再排一次序。z-sort 关闭时行序恒等于 id 序（D3' 前提锁死），
        # 两种情况下本函数的返回顺序都是"当前引擎行序"。
        cells = self._particle_cell_ids[ids] if self._particle_cell_ids is not None \
            else np.full(ids.shape, -1, dtype=np.int32)
        ph_values = np.full(ids.shape, np.nan, dtype=np.float64)
        ok = (cells >= 0) & (cells < self.chem_field.n_cells)
        ph_values[ok] = ph_by_cell[cells[ok]]

        n_missing = int(np.isnan(ph_values).sum())
        if n_missing:
            _chem_logger.debug("%d/%d 个粒子所在单元未参与化学计算，pH 记为 NaN",
                               n_missing, ids.size)

        # 返回顺序 == 当前引擎行序（见上），长度 == 活跃粒子数
        return ph_values

    def _warn_no_ph_once(self) -> None:
        if getattr(self, '_ph_missing_warned', False):
            return
        self._ph_missing_warned = True
        _chem_logger.error(
            "PHREEQC 本轮没有产出任何 SelectedOutput，pH 图层将显示为'无数据'。"
            "实测可用路径：phreeqc_config.setup_default_chemical_system 的 RunString "
            "里带了 `SELECTED_OUTPUT 1 / -pH / END`，独立探针下 "
            "GetSelectedOutputCount()==1（见 D:\\tmp_pg_verify\\impl\\b7a.log）。"
            "所以这里为 0 说明走的是没注入该定义的化学初始化分支，"
            "不是 pH 本身不可求。补法：在自定义化学输入里加 selected_output 定义，"
            "不要改 phreeqc.dat 的化学内容。")

    def get_cell_concentrations(self) -> np.ndarray:
        """
        获取当前化学单元浓度矩阵（聚合但未执行化学步）。

        Returns:
            cell_conc: (n_cells, n_comps) float64
        """
        return self._aggregate_cell_concentrations()

    # 溶质组分：H/O 是水本身的元素总量（110.68 / 55.35 mol/L），
    # 把它们画进"浓度分布"只会得到一整片饱和色，看不出任何反应。
    # 真正的溶质是电荷载体 Cl/Na 这类组分。
    _SOLUTE_COMPONENT_NAMES = ('Cl', 'Na', 'S', 'C', 'Ca', 'Mg', 'K', 'N')

    def get_solute_component_index(self) -> int:
        """返回"最值得画"的溶质组分列号；没有溶质时返回 0。

        H/O 不是溶质，所以优先在 _SOLUTE_COMPONENT_NAMES 里挑；
        挑了也没有任何非零值时，说明当前只有纯水，返回 -1 让调用方
        画"无溶质"而不是画一片假热区。
        """
        if not self.comp_names:
            return -1
        silent = {'H', 'O', 'Charge'}
        for name in self._SOLUTE_COMPONENT_NAMES:
            if name in self.comp_names:
                return self.comp_names.index(name)
        for idx, name in enumerate(self.comp_names):
            if name not in silent:
                return idx
        return -1

    def get_2d_field(self, kind: str = "solute") -> Optional[np.ndarray]:
        """把 3D 化学网格在 Y 方向积分（求和）成 2D 场，供分布图直接显示。

        为什么是"沿 Y 求和"：如果把 1000 个单元按线性索引抽稀成 50x50，
        X 轴会变成"x + y*10 + z*100"的大杂烩，同一个像素上叠着不同高度、
        不同深度的单元，看起来就是一团不像分布的噪点。
        沿 Y 求和后 X/Z 平面才是真正的空间分布：某个 (x,z) 位置有多少溶质，
        就是该竖直柱子里所有单元的溶质总量。

        Args:
            kind: 'solute' = 溶质浓度柱积分；'ph' = 该柱子的 pH 极值。
                  pH 的极性必须按"最偏离中性"取（酸取最小、碱取最大），
                  否则一杯 pH 3 的酸被柱子里中性单元平均一下就变成 pH 6。

        Returns:
            field2d: (n_dim, n_dim) float64，行 = iz、列 = ix；无数据时返回 None。
        """
        if self.n_comps == 0 or not self.comp_names:
            return None

        cell_conc = self._aggregate_cell_concentrations()
        n_cells = self.chem_field.n_cells
        if cell_conc.shape[0] != n_cells:
            return None
        n_dim = self.chem_field.n_chem_cells_per_dim

        if kind == "ph":
            ph_flat = self._cell_ph_map()
            if ph_flat is None:
                return None
            # 线性索引 ix + iy*n + iz*n^2 对应reshape后的 (iz, iy, ix)
            grid = ph_flat.reshape((n_dim, n_dim, n_dim))
            empty_col = np.all(~np.isfinite(grid), axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # 整柱 NaN 是正常状态
                acid = np.nanmin(grid, axis=1)
                base = np.nanmax(grid, axis=1)
            field2d = np.where(np.abs(acid - 7.0) >= np.abs(base - 7.0), acid, base)
            field2d[empty_col] = np.nan  # 整个竖直柱都没数据
            return field2d  # (iz, ix)

        idx = self.get_solute_component_index()
        if idx < 0:
            return None
        grid = cell_conc[:, idx].reshape((n_dim, n_dim, n_dim))
        # 沿 Y（轴 1）求和 => (iz, ix)：正好是"俯视 X-Z 平面"的显示方向
        return grid.sum(axis=1)

    def _cell_ph_map(self) -> Optional[np.ndarray]:
        """(n_cells,) 的单元 pH 场；未参与本步化学的单元是 NaN。

        直接复用 get_particle_ph 的同一份 SelectedOutput + 掩码逻辑，
        但按"单元"口径返回，而不是按粒子重排——分布图要的是场，不是粒子。
        """
        if self.rm is None:
            return None
        try:
            n_sel = int(self.rm.GetSelectedOutputCount())
            if n_sel <= 0:
                return None
            headings = [str(h) for h in self.rm.GetSelectedOutputHeadings()]
        except Exception as e:
            _chem_logger.error("读取 SelectedOutput 表头失败，pH 场不可用: %r", e)
            return None
        if 'pH' not in headings:
            return None
        ph_col = headings.index('pH')

        try:
            selected_output = np.asarray(self.rm.GetSelectedOutput(), dtype=np.float64)
        except Exception as e:
            _chem_logger.error("读取 SelectedOutput 失败，pH 场不可用: %r", e)
            return None
        if selected_output.ndim == 1:
            selected_output = selected_output.reshape((-1, 1))

        n_cells = self.chem_field.n_cells
        if selected_output.shape[0] < n_cells or selected_output.shape[1] <= ph_col:
            return None

        ph = np.array(selected_output[:n_cells, ph_col], dtype=np.float64, copy=True)
        mask = self._chem_mapped_mask
        if mask is not None and mask.shape[0] == n_cells:
            ph[~mask] = np.nan
        else:
            ph[~(ph > 0.0)] = np.nan
        ph[np.isinf(ph)] = np.nan
        return ph


    def force_update_mapping(self) -> None:
        """
        强制重新更新粒子映射（用于调试和故障排除）
        """
        _chem_logger.info("强制更新粒子映射...")
        self._update_particle_mapping()
        total_assigned_particles = int(self.chem_field.cell_counts.sum())
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
        _chem_logger = logging.getLogger("chemical_warnings")
        global last_data
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
        # 【修复 b3/D3】不再用"粒子数量增量"识别新粒子：那假设了行号==粒子号。
        # 每帧重新取一次稳定 id，再按 _particle_solution_ready 掩码兜住漏播种的粒子。
        self._update_particle_mapping()
        self._initialize_new_particles()

        # 记录活跃粒子数（口径唯一：就是本帧的稳定 id 集合）
        ids = self._row_ids if self._row_ids is not None else np.zeros(0, dtype=np.int64)
        n_active = int(ids.size)
        stats["n_particles"] = n_active

        # 一致性检查：物理层每个活跃粒子都必须有一行"已播种"的化学状态
        if n_active > 0:
            if self._particle_solution_ready is None or \
                    not bool(self._particle_solution_ready[ids].all()):
                raise RuntimeError(
                    f"Frame {self.frame_count}: 有活跃粒子没有已播种的化学行，"
                    "化学结果将不完整（决策 D4：不做静默补零）")

        # 调试信息
        _chem_logger.debug(f"Frame {self.frame_count}: 活跃粒子数 = {n_active}")

        # 【修复 b4】删掉"每帧抽查前 3 行，发现零浓度就把前 n 行全刷成默认值"的
        # 所谓紧急修复。它有两重罪：一是抽查行号（行≠粒子，抽查结果本身就不代表
        # 任何具体粒子），二是整片覆盖会把已经算过反应的老粒子打回初始水。
        # 现在由 _initialize_new_particles() 精确兜底"未播种的那几个 id"，
        # 播种缺失会打 warning，而不是被静默改写。

        # ── 化学步（每 chem_step_ratio 帧执行）──
        if self.frame_count % self.chem_step_ratio == 0 and self.rm is not None:
            try:
                _chem_logger.debug(f"Frame {self.frame_count}: 开始执行化学步")
                
                # 1. 聚合单元浓度（体积加权平均）
                cell_conc = self._aggregate_cell_concentrations()
                
                # 详细调试信息
                # 【收尾 1 的同一条诚实原则延伸 —— 领队 seq 26 未点名，主动报备】
                # 老代码 "有效化学单元数 = ~(整行都是 -1)" 是 -1 占位符时代的遗产：
                # _aggregate_cell_concentrations 早就改成用 0.0 作背景浓度
                # （见 :660 的 background_conc），-1 再也不会出现，于是这句
                # **每个化学帧都报 1000/1000**，读日志的人会以为"全场都在算化学"。
                # 实际参与反应的单元远小于 1000（本轮实测：run22 728 个粒子时
                # 每帧 18~130 个；run19 256 个粒子时 2~97 个，1000 单元），
                # 也就是说
                # 决策 D7"只退化到含粒子单元、不每帧算满 1000 单元"这条约束本身
                # 是成立的，但旧日志把它表达成了反面的样子。
                # 这是和"化学步执行完成"同一类的假成功信号，故一并改成直接引用
                # D7 的唯一判据 _significant_cells_mask。只改日志，不改任何算法。
                if cell_conc.size > 0:
                    _sig = self._significant_cells_mask(cell_conc)
                    n_sig = int(np.sum(_sig))
                    _chem_logger.info(
                        f"Frame {self.frame_count}: 本步下发反应的化学单元: "
                        f"{n_sig}/{self.chem_field.n_cells}"
                        f"（决策 D7 判据 = 含粒子且有组成）")
                    _chem_logger.debug(
                        f"聚合浓度统计: min={np.min(cell_conc):.2e}, "
                        f"max={np.max(cell_conc):.2e}, "
                        f"mean={np.mean(cell_conc):.2e}")
                
                # 2. 调用 PHREEQCRM.RunCells()
                new_conc = self._run_chemical_step(cell_conc, self.chem_dt)
                # 【收尾 1 —— 领队 seq 26 二.1】_run_chemical_step 有多条提前
                # return 的路径（rm=None / 本步无可算单元 / CreateMappingSWIG 失败 /
                # RunCells 不收敛 / 返回长度不匹配）。原代码不区分，一律把
                # stats["chemical_step"] 置 True 并打印"化学步执行完成"——
                # run11 实测 502 帧全部"跳过化学计算"却帧帧报"执行完成"，
                # 日志在说谎。现在照实际执行结果说话。
                if not self._chem_step_executed:
                    _chem_logger.debug(
                        f"Frame {self.frame_count}: 化学步**未执行**"
                        "（本帧没有任何一次收敛的 RunCells，原因见上面的逐条日志）")
                    stats["chem_time"] = float(self.accumulated_chem_dt)
                else:
                    # 3. 广播回单元内所有粒子
                    self._broadcast_to_particles(new_conc)

                    # 4. 累计化学时间步
                    self.accumulated_chem_dt += self.chem_dt
                    self.chem_field.reaction_dt += self.chem_dt

                    # 【收尾 3 —— 领队 seq 26 二.3】原来这里为了一条 debug 日志
                    # 额外调用 get_particle_ph()：它要遍历全部粒子、查 SelectedOutput
                    # 掩码、按 id 重排，每个化学帧纯浪费一次。
                    # pH 的真实消费方是 GUI worker（coupled_chemistry_physics_gui.py
                    # :482，每帧都要发给渲染器），那一处保留；这里删除。

                    stats["chemical_step"] = True
                    stats["chem_time"] = float(self.accumulated_chem_dt)  # 转换为Python float类型

                    _chem_logger.info(f"Frame {self.frame_count}: 化学步执行完成")
                
            except Exception as e:
                _chem_logger.error(f"化学步执行失败 (Frame {self.frame_count}): {e}")
                import traceback
                _chem_logger.error(traceback.format_exc())
        else:
            _chem_logger.debug(f"Frame {self.frame_count}: 不执行化学步")
            # 即使不执行化学步，也要保持累计的化学时间
            stats["chem_time"] = float(self.accumulated_chem_dt)  # 转换为Python float类型
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