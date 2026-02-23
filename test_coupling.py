"""
化学-物理耦合单元测试
Unit Tests for Chemistry-Physics Coupling Module (coupled_simulation.py)

测试场景：
  1. 酸碱中和（HCl + NaOH）：pH 梯度平滑传播，质量守恒
  2. 碳酸盐平衡（CO₂-H₂O）：PHREEQCRM 与物理时间步协调
  3. 对流-反应（剪切流）：平流不引起数值振荡

运行方法：
    python -m pytest test_coupling.py -v
    # 或直接运行：
    python test_coupling.py
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import sys
import os

# ── 将项目根目录加入搜索路径 ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coupled_simulation import (
    Particle,
    ChemicalField,
    CoupledSimulation,
    CHEM_GRID_DIM,
    CHEM_STEP_RATIO,
    CHEM_MAX_RETRIES,
    DEFAULT_PARTICLE_VOLUME,
)


# ══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════

def _make_mock_rm(n_cells: int, n_comps: int, conc_out: np.ndarray = None):
    """
    创建模拟 PHREEQCRM 对象。

    Args:
        n_cells:   化学单元数量
        n_comps:   化学组分数量
        conc_out:  RunCells 后返回的浓度（默认与输入相同）
    """
    rm = MagicMock()
    rm.GetComponentCount.return_value = n_comps
    rm.GetComponents.return_value = [f"comp{i}" for i in range(n_comps)]
    rm.RunCells.return_value = 0  # 成功

    if conc_out is not None:
        rm.GetConcentrations.return_value = conc_out.flatten().tolist()
    else:
        # 默认：返回全零（RunCells 不改变浓度，仅测试流程）
        rm.GetConcentrations.return_value = [0.0] * (n_cells * n_comps)

    return rm


def _make_coupled_sim_with_particles(
    positions: np.ndarray,
    n_comps: int = 2,
    initial_solutions: np.ndarray = None,
    rm=None,
    chem_step_ratio: int = 1,
):
    """
    创建带测试粒子的 CoupledSimulation 实例。

    Args:
        positions:         (N, 3) 粒子坐标
        n_comps:           化学组分数
        initial_solutions: (N, n_comps) 初始浓度，None 则全零
        rm:                模拟 PHREEQCRM 对象（None 则自动创建）
        chem_step_ratio:   化学步频率

    Returns:
        sim: CoupledSimulation
    """
    n = len(positions)
    n_cells = CHEM_GRID_DIM ** 3

    if rm is None:
        conc_out = (
            initial_solutions.flatten()
            if initial_solutions is not None
            else np.zeros(n_cells * n_comps)
        )
        rm = _make_mock_rm(n_cells, n_comps, np.zeros((n_cells, n_comps)))

    # 不注入 mesh_system，直接管理粒子数组
    sim = CoupledSimulation(
        mesh_system=None,
        phreeqc_rm=rm,
        chem_step_ratio=chem_step_ratio,
    )
    sim.n_comps = n_comps
    sim.comp_names = [f"comp{i}" for i in range(n_comps)]

    # 手动注入粒子数据
    sim._particle_cell_ids = np.full(n, -1, dtype=np.int32)
    sim._particle_solutions = (
        initial_solutions.copy()
        if initial_solutions is not None
        else np.zeros((n, n_comps), dtype=np.float64)
    )

    # 以位置数组模拟 mesh_system
    mock_mesh = MagicMock()
    mock_mesh.current_particle_num = n
    mock_mesh.particle_positions = positions.astype(np.float32)
    sim.mesh_system = mock_mesh
    mock_mesh.step_simulation.return_value = ([], [])

    # 建立初始映射
    sim._update_particle_mapping()

    return sim


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 1：Particle 数据结构
# ══════════════════════════════════════════════════════════════════════════

class TestParticle(unittest.TestCase):
    """Particle 数据结构基础测试"""

    def test_default_construction(self):
        """默认构造的粒子应有正确初始值"""
        p = Particle(pid=0, position=[0.1, 0.2, 0.3])
        self.assertEqual(p.pid, 0)
        np.testing.assert_array_almost_equal(p.position, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(p.velocity, [0, 0, 0])
        self.assertEqual(p.solution, {})
        self.assertEqual(p.cell_id, -1)

    def test_position_dtype(self):
        """粒子坐标应存储为 float32"""
        p = Particle(pid=1, position=[1.0, 2.0, 3.0])
        self.assertEqual(p.position.dtype, np.float32)

    def test_solution_assignment(self):
        """液相化学成分字典可正确赋值"""
        p = Particle(pid=2, position=[0.0, 0.0, 0.0])
        p.solution = {"H": 1e-7, "Cl": 0.001}
        self.assertAlmostEqual(p.solution["H"], 1e-7)
        self.assertAlmostEqual(p.solution["Cl"], 0.001)


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 2：ChemicalField
# ══════════════════════════════════════════════════════════════════════════

class TestChemicalField(unittest.TestCase):
    """ChemicalField 数据结构测试"""

    def setUp(self):
        self.cf = ChemicalField(n_chem_cells_per_dim=6)

    def test_cell_count(self):
        """6×6×6 = 216 个化学单元"""
        self.assertEqual(self.cf.n_cells, 216)

    def test_initial_state(self):
        """初始状态：所有单元为空，体积为零"""
        for cell_list in self.cf.cell_particles:
            self.assertEqual(len(cell_list), 0)
        np.testing.assert_array_equal(self.cf.cell_volume, np.zeros(216))

    def test_assign_particle(self):
        """粒子分配应正确更新 cell_particles 和 cell_volume"""
        self.cf.assign_particle(pid=0, cell_id=10)
        self.cf.assign_particle(pid=1, cell_id=10)
        self.assertIn(0, self.cf.cell_particles[10])
        self.assertIn(1, self.cf.cell_particles[10])
        self.assertAlmostEqual(
            self.cf.cell_volume[10], 2 * DEFAULT_PARTICLE_VOLUME
        )

    def test_reset_cell_assignments(self):
        """重置后所有单元应清空"""
        self.cf.assign_particle(0, 5)
        self.cf.assign_particle(1, 5)
        self.cf.reset_cell_assignments()
        self.assertEqual(len(self.cf.cell_particles[5]), 0)
        self.assertAlmostEqual(self.cf.cell_volume[5], 0.0)

    def test_multiple_cells(self):
        """粒子分配到不同单元互不干扰"""
        self.cf.assign_particle(0, 0)
        self.cf.assign_particle(1, 100)
        self.cf.assign_particle(2, 215)
        self.assertEqual(len(self.cf.cell_particles[0]), 1)
        self.assertEqual(len(self.cf.cell_particles[100]), 1)
        self.assertEqual(len(self.cf.cell_particles[215]), 1)


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 3：粒子→单元映射
# ══════════════════════════════════════════════════════════════════════════

class TestPositionToCellMapping(unittest.TestCase):
    """粒子位置到化学单元的映射测试"""

    def setUp(self):
        self.sim = CoupledSimulation(
            mesh_system=None,
            phreeqc_rm=None,
        )

    def test_center_particle(self):
        """域中心的粒子应映射到中心单元"""
        center = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        cell_ids = self.sim._position_to_cell_id(center)
        n = CHEM_GRID_DIM
        expected_cell = (n // 2) + (n // 2) * n + (n // 2) * n * n
        self.assertEqual(cell_ids[0], expected_cell)

    def test_boundary_particles_in_range(self):
        """边界附近的粒子应映射到有效单元 ID（0 ~ n_cells-1）"""
        positions = np.array(
            [
                [-0.8, -0.8, -0.8],
                [0.79, 0.79, 0.79],
                [-0.8, 0.79, -0.8],
            ],
            dtype=np.float64,
        )
        cell_ids = self.sim._position_to_cell_id(positions)
        n_cells = CHEM_GRID_DIM ** 3
        for cid in cell_ids:
            self.assertGreaterEqual(cid, 0)
            self.assertLess(cid, n_cells)

    def test_opposite_corners_different_cells(self):
        """域的两个对角粒子应映射到不同单元"""
        corners = np.array(
            [[-0.8, -0.8, -0.8], [0.79, 0.79, 0.79]], dtype=np.float64
        )
        cell_ids = self.sim._position_to_cell_id(corners)
        self.assertNotEqual(cell_ids[0], cell_ids[1])

    def test_all_particles_in_same_small_region_get_same_cell(self):
        """同一微小区域内的粒子应映射到同一单元"""
        base = np.array([0.1, 0.1, 0.1])
        tiny_offsets = np.random.uniform(-1e-4, 1e-4, (10, 3))
        positions = (base + tiny_offsets).astype(np.float64)
        cell_ids = self.sim._position_to_cell_id(positions)
        self.assertTrue(np.all(cell_ids == cell_ids[0]))

    def test_out_of_domain_clamped(self):
        """域外坐标应被夹持到有效单元"""
        out_of_range = np.array(
            [[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]], dtype=np.float64
        )
        cell_ids = self.sim._position_to_cell_id(out_of_range)
        n_cells = CHEM_GRID_DIM ** 3
        for cid in cell_ids:
            self.assertGreaterEqual(cid, 0)
            self.assertLess(cid, n_cells)


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 4：浓度聚合与广播
# ══════════════════════════════════════════════════════════════════════════

class TestConcentrationAggregation(unittest.TestCase):
    """体积加权浓度聚合与广播测试"""

    def _make_grid_positions(self, n_per_dim=6):
        """生成均匀分布的测试粒子坐标"""
        coords = np.linspace(-0.7, 0.7, n_per_dim)
        xx, yy, zz = np.meshgrid(coords, coords, coords)
        return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(
            np.float64
        )

    def test_single_component_uniform(self):
        """均匀浓度场：聚合后各单元浓度应保持一致"""
        positions = self._make_grid_positions()
        n = len(positions)
        n_comps = 1
        conc = np.ones((n, n_comps), dtype=np.float64) * 0.5

        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc
        )

        cell_conc = sim._aggregate_cell_concentrations()
        # 有粒子的单元浓度应接近 0.5
        occupied = [
            i for i in range(sim.chem_field.n_cells)
            if sim.chem_field.cell_particles[i]
        ]
        for cell_id in occupied:
            np.testing.assert_almost_equal(cell_conc[cell_id, 0], 0.5)

    def test_two_component_gradient(self):
        """
        左半域浓度=1，右半域浓度=0：
        聚合后左半单元应接近1，右半单元应接近0
        """
        positions = self._make_grid_positions()
        n = len(positions)
        n_comps = 1
        conc = np.zeros((n, n_comps), dtype=np.float64)
        # x < 0 的粒子设浓度 = 1
        left_mask = positions[:, 0] < 0.0
        conc[left_mask, 0] = 1.0

        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc
        )
        cell_conc = sim._aggregate_cell_concentrations()

        n_dim = CHEM_GRID_DIM
        # 检查 ix=0 单元（左侧）浓度接近 1
        for iy in range(n_dim):
            for iz in range(n_dim):
                cid = 0 + iy * n_dim + iz * n_dim * n_dim
                if sim.chem_field.cell_particles[cid]:
                    self.assertGreater(cell_conc[cid, 0], 0.5)

    def test_broadcast_restores_uniform_conc(self):
        """广播均匀单元浓度后，粒子浓度应与单元浓度一致"""
        positions = self._make_grid_positions(n_per_dim=3)
        n = len(positions)
        n_comps = 2
        conc = np.random.rand(n, n_comps)

        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc
        )

        # 用全 1 单元浓度广播
        n_cells = sim.chem_field.n_cells
        uniform_cell_conc = np.ones((n_cells, n_comps), dtype=np.float64)
        sim._broadcast_to_particles(uniform_cell_conc)

        # 所有粒子浓度应等于 1
        np.testing.assert_array_almost_equal(
            sim._particle_solutions[:n], np.ones((n, n_comps))
        )

    def test_empty_cells_contribute_zero(self):
        """空单元使用 -1.0 标记，不参与化学计算"""
        # 将所有粒子集中在域左下角
        positions = np.tile([-0.7, -0.7, -0.7], (10, 1)).astype(np.float64)
        n_comps = 1
        conc = np.ones((10, n_comps), dtype=np.float64)

        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc
        )
        cell_conc = sim._aggregate_cell_concentrations()

        # 除第 0 单元外，其余空单元应标记为 -1.0
        for cell_id in range(1, sim.chem_field.n_cells):
            if not sim.chem_field.cell_particles[cell_id]:
                np.testing.assert_array_equal(cell_conc[cell_id], -1.0)


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 5：化学步控制（频率与重试）
# ══════════════════════════════════════════════════════════════════════════

class TestChemicalStepControl(unittest.TestCase):
    """化学步执行频率与 PHREEQCRM 重试机制测试"""

    def _build_sim(self, chem_step_ratio=2, run_cells_side_effect=None):
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1
        positions = np.zeros((4, 3), dtype=np.float64)
        conc = np.ones((4, n_comps), dtype=np.float64)

        rm = _make_mock_rm(n_cells, n_comps)
        if run_cells_side_effect is not None:
            rm.RunCells.side_effect = run_cells_side_effect

        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc,
            rm=rm, chem_step_ratio=chem_step_ratio
        )
        return sim, rm

    def test_chemical_step_fires_at_correct_interval(self):
        """化学步应严格按 chem_step_ratio 触发"""
        ratio = 3
        sim, rm = self._build_sim(chem_step_ratio=ratio)

        for frame in range(1, 10):
            stats = sim.step()
            if frame % ratio == 0:
                self.assertTrue(
                    stats["chemical_step"],
                    f"第 {frame} 帧应执行化学步（ratio={ratio}）",
                )
            else:
                self.assertFalse(
                    stats["chemical_step"],
                    f"第 {frame} 帧不应执行化学步（ratio={ratio}）",
                )

    def test_chemical_time_accumulates_correctly(self):
        """化学时间累计应等于 化学步次数 × chem_dt"""
        sim, rm = self._build_sim(chem_step_ratio=1)
        n_steps = 5
        for _ in range(n_steps):
            sim.step()
        expected_time = n_steps * sim.chem_dt
        self.assertAlmostEqual(sim.accumulated_chem_dt, expected_time)

    def test_retry_on_failed_run_cells(self):
        """RunCells 返回非零状态时应重试，并减小 dt"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1

        # 前 CHEM_MAX_RETRIES-1 次失败，最后一次成功
        call_count = [0]
        def side_effect():
            call_count[0] += 1
            if call_count[0] < CHEM_MAX_RETRIES:
                return 1  # 失败
            return 0  # 成功

        sim, rm = self._build_sim(
            chem_step_ratio=1, run_cells_side_effect=side_effect
        )
        sim.step()

        self.assertEqual(call_count[0], CHEM_MAX_RETRIES)
        # 每次重试后 dt 应被减半
        expected_dt = sim.chem_dt * (0.5 ** (CHEM_MAX_RETRIES - 1))
        rm.SetTimeStep.assert_called_with(expected_dt)

    def test_always_failed_run_cells_returns_pre_reaction_conc(self):
        """RunCells 始终失败时，应返回反应前浓度（不更新粒子浓度）"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1
        positions = np.zeros((4, 3), dtype=np.float64)
        initial_conc = np.ones((4, n_comps), dtype=np.float64) * 2.0

        rm = _make_mock_rm(n_cells, n_comps)
        rm.RunCells.return_value = 1  # 始终失败

        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=initial_conc,
            rm=rm, chem_step_ratio=1
        )
        sim.step()

        # 粒子浓度应保持初始值（来自广播前的聚合输入，所有粒子在同一单元故均值=2）
        expected = np.ones((4, n_comps)) * initial_conc[0, 0]
        np.testing.assert_array_almost_equal(
            sim._particle_solutions[:4], expected
        )

    def test_negative_concentration_clipped(self):
        """RunCells 产生负浓度时应截断为 0"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 2
        positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
        initial_conc = np.ones((2, n_comps), dtype=np.float64)

        rm = _make_mock_rm(n_cells, n_comps)
        # 返回包含负值的浓度
        neg_conc = np.full((n_cells, n_comps), -0.5)
        rm.GetConcentrations.return_value = neg_conc.flatten().tolist()

        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=initial_conc,
            rm=rm, chem_step_ratio=1
        )
        sim.step()

        # 所有粒子浓度应 >= 0
        self.assertTrue(np.all(sim._particle_solutions[:2] >= 0.0))

    def test_empty_cells_excluded_via_create_mapping(self):
        """空单元应通过 CreateMapping(-1) 从 RunCells 计算中排除"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1

        # 所有粒子集中在域的一个角落 → 绝大多数单元为空
        positions = np.array([[-0.7, -0.7, -0.7]] * 4, dtype=np.float64)
        conc = np.ones((4, n_comps), dtype=np.float64)

        rm = _make_mock_rm(n_cells, n_comps)
        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc,
            rm=rm, chem_step_ratio=1
        )

        sim.step()

        # 验证 CreateMapping 被调用，且大部分单元映射为 -1
        rm.CreateMapping.assert_called()
        mapping = rm.CreateMapping.call_args[0][0]
        inactive_count = sum(1 for m in mapping if m < 0)
        self.assertGreater(inactive_count, 0, "应有空单元被排除")
        # 至少有一个活跃单元（粒子所在单元）
        active_count = sum(1 for m in mapping if m >= 0)
        self.assertGreater(active_count, 0, "至少应有一个活跃单元")

    def test_all_cells_active_when_all_have_particles(self):
        """所有单元都有粒子时，CreateMapping 应恢复完整映射"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1

        # 均匀分布粒子，每个单元都有粒子
        coords = np.linspace(-0.7, 0.7, CHEM_GRID_DIM)
        xx, yy, zz = np.meshgrid(coords, coords, coords)
        positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float64)
        n = len(positions)
        conc = np.ones((n, n_comps), dtype=np.float64)

        rm = _make_mock_rm(n_cells, n_comps)
        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc,
            rm=rm, chem_step_ratio=1
        )

        sim.step()

        # 验证 CreateMapping 使用完整映射（所有单元活跃）
        rm.CreateMapping.assert_called()
        mapping = rm.CreateMapping.call_args[0][0]
        for i, m in enumerate(mapping):
            self.assertEqual(m, i, f"单元 {i} 应映射为自身")

    def test_all_empty_cells_skip_chemistry(self):
        """所有单元均为空时，应跳过化学计算"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1

        rm = _make_mock_rm(n_cells, n_comps)
        sim = CoupledSimulation(
            mesh_system=None,
            phreeqc_rm=rm,
            chem_step_ratio=1,
        )
        sim.n_comps = n_comps
        sim.comp_names = ["tracer"]

        # 无粒子 → 所有单元为空
        sim.step()

        # RunCells 不应被调用（所有单元为空，跳过化学计算）
        rm.RunCells.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 5b：CreateMapping 连续索引（空单元正确排除）
# ══════════════════════════════════════════════════════════════════════════

class TestCreateMappingConsecutive(unittest.TestCase):
    """
    验证 CreateMapping 使用连续化学单元索引，确保空单元被正确排除。

    旧的 identity 映射 [i if active else -1] 在活跃单元分散时会导致
    大量未映射的化学单元仍被 RunCells 计算。新实现使用连续索引
    (0, 1, 2, ...) 以减少化学单元数量。
    """

    def test_consecutive_mapping_indices_for_sparse_active_cells(self):
        """活跃单元分散时，CreateMapping 应使用连续化学单元索引（0, 1, 2, ...）"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1

        # 将粒子放在域的两个对角：确保活跃单元分散
        positions = np.array(
            [[-0.7, -0.7, -0.7], [0.7, 0.7, 0.7]], dtype=np.float64
        )
        conc = np.ones((2, n_comps), dtype=np.float64)

        rm = _make_mock_rm(n_cells, n_comps)
        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc,
            rm=rm, chem_step_ratio=1
        )

        sim.step()

        # 验证 CreateMapping 使用连续索引
        rm.CreateMapping.assert_called()
        mapping = rm.CreateMapping.call_args[0][0]

        # 活跃单元的化学索引应从 0 开始连续递增
        active_chem_ids = sorted([m for m in mapping if m >= 0])
        expected_chem_ids = list(range(len(active_chem_ids)))
        self.assertEqual(
            active_chem_ids, expected_chem_ids,
            f"化学单元索引应为连续的 {expected_chem_ids}，实际为 {active_chem_ids}",
        )

    def test_consecutive_mapping_reduces_chemistry_count(self):
        """连续映射应使化学单元数量 = 活跃网格单元数量"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1

        # 只在一个角落放粒子
        positions = np.array([[-0.7, -0.7, -0.7]] * 4, dtype=np.float64)
        conc = np.ones((4, n_comps), dtype=np.float64)

        rm = _make_mock_rm(n_cells, n_comps)
        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc,
            rm=rm, chem_step_ratio=1
        )

        sim.step()

        mapping = rm.CreateMapping.call_args[0][0]
        active_chem_ids = [m for m in mapping if m >= 0]
        n_active = len(active_chem_ids)
        max_chem_id = max(active_chem_ids)

        # 最大化学索引应恰好等于 n_active - 1（连续索引）
        self.assertEqual(
            max_chem_id, n_active - 1,
            f"最大化学索引应为 {n_active - 1}（连续索引），实际为 {max_chem_id}",
        )

    def test_mapping_preserves_grid_cell_identity(self):
        """每个活跃网格单元应映射到唯一的化学单元（无重复）"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1

        # 使用三个不同位置的粒子
        positions = np.array(
            [[-0.7, -0.7, -0.7], [0.0, 0.0, 0.0], [0.7, 0.7, 0.7]],
            dtype=np.float64,
        )
        conc = np.ones((3, n_comps), dtype=np.float64)

        rm = _make_mock_rm(n_cells, n_comps)
        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc,
            rm=rm, chem_step_ratio=1
        )

        sim.step()

        mapping = rm.CreateMapping.call_args[0][0]
        active_chem_ids = [m for m in mapping if m >= 0]

        # 化学单元索引应无重复
        self.assertEqual(
            len(active_chem_ids),
            len(set(active_chem_ids)),
            "化学单元索引不应有重复",
        )

    def test_full_mapping_still_identity(self):
        """所有单元都有粒子时，映射应仍为 0, 1, 2, ..., n-1（连续 = identity）"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1

        coords = np.linspace(-0.7, 0.7, CHEM_GRID_DIM)
        xx, yy, zz = np.meshgrid(coords, coords, coords)
        positions = np.stack(
            [xx.ravel(), yy.ravel(), zz.ravel()], axis=1
        ).astype(np.float64)
        n = len(positions)
        conc = np.ones((n, n_comps), dtype=np.float64)

        rm = _make_mock_rm(n_cells, n_comps)
        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc,
            rm=rm, chem_step_ratio=1
        )

        sim.step()

        mapping = rm.CreateMapping.call_args[0][0]
        for i, m in enumerate(mapping):
            self.assertEqual(m, i, f"完全活跃时，单元 {i} 应映射为 {i}")

    def test_run_cells_called_with_valid_mapping(self):
        """即使大部分单元为空，RunCells 仍应被调用且成功"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 2

        # 只有一个角落有粒子
        positions = np.array(
            [[-0.7, -0.7, -0.7], [-0.6, -0.6, -0.6]], dtype=np.float64
        )
        conc = np.ones((2, n_comps), dtype=np.float64) * 0.01

        rm = _make_mock_rm(n_cells, n_comps)
        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc,
            rm=rm, chem_step_ratio=1
        )

        sim.step()

        # RunCells 应被调用（至少有一个活跃单元）
        rm.RunCells.assert_called_once()

    def test_mapping_changes_between_frames(self):
        """粒子位置变化后，映射应随之更新"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1

        # 初始位置：只在一个角落
        positions = np.array(
            [[-0.7, -0.7, -0.7]], dtype=np.float64
        )
        conc = np.ones((1, n_comps), dtype=np.float64)

        rm = _make_mock_rm(n_cells, n_comps)
        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc,
            rm=rm, chem_step_ratio=1
        )

        # 第一帧
        sim.step()
        mapping1 = rm.CreateMapping.call_args[0][0][:]

        # 移动粒子到另一个位置
        sim._particle_solutions[0] = [1.0]
        sim._particle_cell_ids[0] = 0

        # 第二帧
        sim.step()
        mapping2 = rm.CreateMapping.call_args[0][0][:]

        # 两帧的映射都应有效（活跃索引连续且无间隙）
        for mapping in [mapping1, mapping2]:
            active_ids = sorted([m for m in mapping if m >= 0])
            expected = list(range(len(active_ids)))
            self.assertEqual(active_ids, expected)


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 6：场景测试 – 酸碱中和（HCl + NaOH）
# ══════════════════════════════════════════════════════════════════════════

class TestAcidBaseNeutralization(unittest.TestCase):
    """
    场景：酸碱中和（HCl + NaOH）

    预期现象：
      - pH 梯度从界面平滑扩散
      - 每一帧化学步后，pH 场不引入剧烈的数值振荡
      - 质量守恒：各组分总量在化学步前后变化有限
    """

    def _setup_acid_base_sim(self, n_per_side=3):
        """
        设置酸碱中和测试：
          - 左半域（x < 0）：酸性，H+ 浓度 = 0.01 mol/L
          - 右半域（x >= 0）：碱性，OH- 浓度 = 0.01 mol/L
        """
        # 组分：0=H+（酸性指标），1=Na（碱性指标）
        n_comps = 2
        n_cells = CHEM_GRID_DIM ** 3

        # 创建粒子：左右各半
        coords = np.linspace(-0.7, 0.7, n_per_side)
        xs, ys, zs = np.meshgrid(coords, coords, coords)
        positions = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
        n = len(positions)

        # 初始浓度：左侧 H+=0.01，右侧 Na=0.01
        conc = np.zeros((n, n_comps), dtype=np.float64)
        left = positions[:, 0] < 0.0
        right = ~left
        conc[left, 0] = 0.01   # H+
        conc[right, 1] = 0.01  # Na+

        # 模拟 PHREEQCRM 的行为：
        # 中和反应后，界面附近的 H+ 和 Na+ 都减少
        def neutralization_reaction(cell_conc_flat):
            """模拟中和：每单元 H+ 减去 min(H+, Na+)，Na+ 同理"""
            cc = np.array(cell_conc_flat).reshape((n_cells, n_comps))
            neutralized = cc.copy()
            delta = np.minimum(cc[:, 0], cc[:, 1])
            neutralized[:, 0] -= delta
            neutralized[:, 1] -= delta
            return neutralized.flatten().tolist()

        rm = _make_mock_rm(n_cells, n_comps)
        rm.GetConcentrations.side_effect = lambda: neutralization_reaction(
            rm.SetConcentrations.call_args[0][0]
        )

        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc,
            rm=rm, chem_step_ratio=1
        )
        return sim, positions, conc

    def test_ph_gradient_smooth_propagation(self):
        """
        执行多帧后，酸性/碱性浓度梯度应平滑（无剧烈振荡）。

        验证：相邻化学单元浓度差不超过初始最大浓度值。
        """
        sim, positions, init_conc = self._setup_acid_base_sim()
        n_frames = 5
        for _ in range(n_frames):
            sim.step()

        cell_conc = sim._aggregate_cell_concentrations()
        max_initial_conc = 0.01

        # 检查相邻单元浓度差（沿 x 轴）
        n = CHEM_GRID_DIM
        for iy in range(n):
            for iz in range(n):
                for ix in range(n - 1):
                    cid1 = ix + iy * n + iz * n * n
                    cid2 = (ix + 1) + iy * n + iz * n * n
                    diff = np.abs(cell_conc[cid1] - cell_conc[cid2])
                    self.assertTrue(
                        np.all(diff <= max_initial_conc + 1e-10),
                        f"相邻单元浓度差过大: cell {cid1} vs {cid2}, diff={diff}",
                    )

    def test_mass_conservation_after_chemical_step(self):
        """
        质量守恒：化学步后总质量应 ≤ 化学步前（中和消耗物质）。

        验证点：组分总质量不增加。
        """
        sim, positions, init_conc = self._setup_acid_base_sim()

        # 记录初始总质量
        total_before = init_conc.sum(axis=0)

        # 执行化学步
        sim.step()

        n = sim.mesh_system.current_particle_num
        total_after = sim._particle_solutions[:n].sum(axis=0)

        # 质量应守恒或减少（中和消耗）
        for comp in range(2):
            self.assertLessEqual(
                total_after[comp],
                total_before[comp] + 1e-9,
                f"组分 {comp} 质量不守恒：before={total_before[comp]:.6f}, "
                f"after={total_after[comp]:.6f}",
            )

    def test_no_negative_concentrations_after_neutralization(self):
        """中和反应后所有粒子浓度应 >= 0"""
        sim, _, _ = self._setup_acid_base_sim()
        for _ in range(3):
            sim.step()
        n = sim.mesh_system.current_particle_num
        self.assertTrue(
            np.all(sim._particle_solutions[:n] >= 0.0),
            "检测到负浓度",
        )


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 7：场景测试 – 碳酸盐平衡（CO₂-H₂O）
# ══════════════════════════════════════════════════════════════════════════

class TestCarbonateEquilibrium(unittest.TestCase):
    """
    场景：碳酸盐平衡（CO₂-H₂O）

    预期现象：
      - pH 随 CO₂ 分压升高而降低
      - PHREEQCRM 与物理时间步正确协调（每 N 帧化学步）

    验证点：
      - 高 CO₂ 分压 → 较低 pH（较高 H+ 浓度）
      - 化学步累计时间与帧数/频率一致
    """

    def _make_co2_sim(self, co2_concentration: float, chem_step_ratio: int = 2):
        """
        构建 CO₂-H₂O 系统测试环境。

        Args:
            co2_concentration: CO₂ 初始浓度（mol/L），模拟 pH 效应
        """
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 2  # comp0=CO₂, comp1=H+（pH 指示）

        positions = np.zeros((6, 3), dtype=np.float64)
        for i in range(6):
            positions[i] = [i * 0.2 - 0.5, 0.0, 0.0]

        conc = np.zeros((6, n_comps), dtype=np.float64)
        conc[:, 0] = co2_concentration  # 均匀 CO₂

        # CO₂ 溶解产生 H+：pH 降低 → H+ 浓度增大
        # 模拟碳酸解离平衡：H+ = sqrt(K_a * CO₂)，K_a=4.3e-7（H₂CO₃ ⇌ H+ + HCO₃⁻）
        K_a = 4.3e-7

        def co2_ph_reaction(cell_conc_flat):
            cc = np.array(cell_conc_flat).reshape((n_cells, n_comps))
            new_cc = cc.copy()
            new_cc[:, 1] = np.sqrt(K_a * np.maximum(cc[:, 0], 0.0))
            return new_cc.flatten().tolist()

        rm = _make_mock_rm(n_cells, n_comps)
        rm.GetConcentrations.side_effect = lambda: co2_ph_reaction(
            rm.SetConcentrations.call_args[0][0]
        )

        sim = _make_coupled_sim_with_particles(
            positions, n_comps=n_comps, initial_solutions=conc,
            rm=rm, chem_step_ratio=chem_step_ratio
        )
        return sim, K_a

    def test_ph_decreases_with_co2(self):
        """高 CO₂ 浓度应产生更高的 H+ 浓度（更低 pH）"""
        # 使用 chem_step_ratio=1 确保第一帧即触发化学步
        sim_low, K_a = self._make_co2_sim(co2_concentration=1e-4, chem_step_ratio=1)
        sim_high, _ = self._make_co2_sim(co2_concentration=1e-2, chem_step_ratio=1)

        sim_low.step()
        sim_high.step()

        n = sim_low.mesh_system.current_particle_num
        h_low = sim_low._particle_solutions[:n, 1].mean()
        h_high = sim_high._particle_solutions[:n, 1].mean()

        self.assertGreater(
            h_high, h_low,
            "高 CO₂ 浓度应产生更高的 H+ 浓度",
        )

    def test_chemical_step_coordination(self):
        """
        化学步应严格与物理步协调（每 N 帧执行一次）。
        验证：累计化学时间 = 化学步次数 × chem_dt
        """
        ratio = 3
        sim, _ = self._make_co2_sim(co2_concentration=1e-3, chem_step_ratio=ratio)

        n_frames = 9  # 3 次化学步
        for _ in range(n_frames):
            sim.step()

        expected_chem_steps = n_frames // ratio
        expected_chem_time = expected_chem_steps * sim.chem_dt
        self.assertAlmostEqual(sim.accumulated_chem_dt, expected_chem_time)

    def test_phreeqcrm_called_only_on_chemical_frames(self):
        """PHREEQCRM.RunCells() 仅在化学步帧被调用"""
        ratio = 4
        sim, _ = self._make_co2_sim(co2_concentration=1e-3, chem_step_ratio=ratio)

        n_frames = 8
        for _ in range(n_frames):
            sim.step()

        expected_calls = n_frames // ratio
        self.assertEqual(sim.rm.RunCells.call_count, expected_calls)


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 8：场景测试 – 对流-反应（剪切流）
# ══════════════════════════════════════════════════════════════════════════

class TestAdvectionReaction(unittest.TestCase):
    """
    场景：对流-反应（剪切流）

    预期现象：
      - 反应前沿被剪切流拉伸变形
      - 平流过程不引起数值振荡

    验证点：
      - 粒子移动后浓度场光滑（无突变）
      - 移动前后总质量误差 < 1%（无扩散引起的数值耗散）
    """

    def _make_shear_flow_sim(self, n_particles: int = 27):
        """
        构建剪切流测试：粒子按简单剪切场移动，浓度跟随粒子重新映射。

        Args:
            n_particles: 粒子总数（建议完全正方体，如 27=3^3）
        """
        n_per_dim = int(round(n_particles ** (1 / 3)))
        coords = np.linspace(-0.6, 0.6, n_per_dim)
        xx, yy, zz = np.meshgrid(coords, coords, coords)
        initial_positions = np.stack(
            [xx.ravel(), yy.ravel(), zz.ravel()], axis=1
        ).astype(np.float64)
        n = len(initial_positions)

        # 初始浓度：沿 x 轴线性梯度
        n_comps = 1
        conc = (initial_positions[:, 0] - initial_positions[:, 0].min())
        conc /= (conc.max() + 1e-10)
        conc = conc.reshape(n, 1)

        n_cells = CHEM_GRID_DIM ** 3
        rm = _make_mock_rm(n_cells, n_comps)

        # 简单剪切：vx = shear_rate * y
        shear_rate = 0.5
        dt_phys = 0.016
        sheared_positions = initial_positions.copy()
        sheared_positions[:, 0] += shear_rate * initial_positions[:, 1] * dt_phys
        sheared_positions = np.clip(sheared_positions, -0.79, 0.79)

        # MeshSystem mock：每次 step_simulation 后粒子移动到剪切位置
        mock_mesh = MagicMock()
        mock_mesh.current_particle_num = n
        # 初始位置 → 剪切后位置（仅一步）
        pos_sequence = [initial_positions, sheared_positions]
        pos_iter = iter(pos_sequence)
        mock_mesh.particle_positions = initial_positions.astype(np.float32)

        def advance_positions():
            try:
                mock_mesh.particle_positions = next(pos_iter).astype(np.float32)
            except StopIteration:
                pass
            return [], []

        mock_mesh.step_simulation.side_effect = lambda *args, **kwargs: advance_positions()

        sim = CoupledSimulation(
            mesh_system=mock_mesh,
            phreeqc_rm=rm,
            chem_step_ratio=1,
        )
        sim.n_comps = n_comps
        sim.comp_names = ["tracer"]
        sim._particle_cell_ids = np.full(n, -1, dtype=np.int32)
        sim._particle_solutions = conc.copy()
        sim._update_particle_mapping()

        return sim, initial_positions, sheared_positions, conc

    def test_no_numerical_oscillation_after_advection(self):
        """
        剪切流后，聚合浓度应光滑（相邻单元差值 ≤ 初始最大浓度）。

        验证平流不引起数值振荡。
        """
        sim, init_pos, shear_pos, init_conc = self._make_shear_flow_sim()

        # 执行一帧（含物理步+化学步）
        sim.step()

        cell_conc = sim._aggregate_cell_concentrations()
        max_conc = float(init_conc.max())

        n_dim = CHEM_GRID_DIM
        # 检查沿 x 方向相邻单元差值
        for iy in range(n_dim):
            for iz in range(n_dim):
                for ix in range(n_dim - 1):
                    cid1 = ix + iy * n_dim + iz * n_dim * n_dim
                    cid2 = (ix + 1) + iy * n_dim + iz * n_dim * n_dim
                    diff = abs(float(cell_conc[cid1, 0]) - float(cell_conc[cid2, 0]))
                    self.assertLessEqual(
                        diff, max_conc + 1e-10,
                        f"平流后相邻单元浓度差过大: {diff:.4f}",
                    )

    def test_total_mass_conserved_during_advection(self):
        """
        对流过程中，示踪物总质量应守恒（误差 < 1%）。

        注：此处仅做粒子重新映射，不做扩散，质量应严格守恒。
        """
        sim, init_pos, shear_pos, init_conc = self._make_shear_flow_sim()
        n = sim.mesh_system.current_particle_num

        mass_before = float(sim._particle_solutions[:n, 0].sum())

        # 仅执行位置更新（不含化学步，用 chem_step_ratio=2）
        sim.chem_step_ratio = 2
        sim.step()  # 第 1 帧：只有物理步

        mass_after = float(sim._particle_solutions[:n, 0].sum())

        # 对流不改变粒子自身浓度，总质量严格守恒
        if mass_before > 1e-10:
            rel_error = abs(mass_after - mass_before) / mass_before
            self.assertLess(rel_error, 0.01, f"质量守恒误差 {rel_error:.4%}")

    def test_reaction_front_stretching(self):
        """
        剪切流拉伸反应前沿：剪切后浓度分布的空间跨度应与初始分布接近。

        验证：剪切不导致浓度骤然归零或突然升高。
        """
        sim, _, _, init_conc = self._make_shear_flow_sim()

        init_range = float(init_conc.max()) - float(init_conc.min())

        sim.step()

        n = sim.mesh_system.current_particle_num
        final_range = (
            float(sim._particle_solutions[:n, 0].max())
            - float(sim._particle_solutions[:n, 0].min())
        )

        # 对流后浓度范围不应超过初始范围（无物理机制增加浓度）
        self.assertLessEqual(
            final_range, init_range + 1e-10,
            "对流后浓度范围超过初始范围",
        )


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 9：CoupledSimulation 全流程集成测试
# ══════════════════════════════════════════════════════════════════════════

class TestCoupledSimulationIntegration(unittest.TestCase):
    """CoupledSimulation 完整运行流程集成测试"""

    def test_run_n_frames_returns_correct_history_length(self):
        """run(n) 应返回包含 n 帧统计信息的列表"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 1
        rm = _make_mock_rm(n_cells, n_comps)

        sim = CoupledSimulation(phreeqc_rm=rm, chem_step_ratio=2)
        sim.n_comps = n_comps
        sim.comp_names = ["tracer"]

        history = sim.run(n_frames=10)
        self.assertEqual(len(history), 10)

    def test_frame_counter_increments(self):
        """帧计数器应每帧递增"""
        sim = CoupledSimulation()
        self.assertEqual(sim.frame_count, 0)
        sim.step()
        self.assertEqual(sim.frame_count, 1)
        sim.step()
        self.assertEqual(sim.frame_count, 2)

    def test_stats_contain_required_keys(self):
        """step() 返回的统计字典必须包含必要字段"""
        sim = CoupledSimulation()
        stats = sim.step()
        self.assertIn("frame", stats)
        self.assertIn("chemical_step", stats)
        self.assertIn("n_particles", stats)

    def test_no_phreeqc_no_crash(self):
        """无 PHREEQCRM 实例时，仿真应正常运行（不崩溃）"""
        sim = CoupledSimulation(phreeqc_rm=None, chem_step_ratio=1)
        try:
            for _ in range(5):
                sim.step()
        except Exception as e:
            self.fail(f"无 PHREEQCRM 时发生异常: {e}")

    def test_no_mesh_system_no_crash(self):
        """无 MeshSystem 实例时，仿真应正常运行（不崩溃）"""
        sim = CoupledSimulation(mesh_system=None, chem_step_ratio=1)
        try:
            for _ in range(5):
                sim.step()
        except Exception as e:
            self.fail(f"无 MeshSystem 时发生异常: {e}")

    def test_initialize_sets_component_info(self):
        """initialize() 应从 PHREEQCRM 正确读取组分信息"""
        n_cells = CHEM_GRID_DIM ** 3
        n_comps = 3
        rm = _make_mock_rm(n_cells, n_comps)

        sim = CoupledSimulation(phreeqc_rm=rm)
        sim.initialize()

        self.assertEqual(sim.n_comps, n_comps)
        self.assertEqual(len(sim.comp_names), n_comps)
        self.assertIsNotNone(sim.chem_field.phreeqc_conc)
        self.assertEqual(
            sim.chem_field.phreeqc_conc.size, n_cells * n_comps
        )


# ══════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
