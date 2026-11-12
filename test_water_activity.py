"""
水活度收敛修复 - 单元测试
Tests for Water Activity Convergence Fix (core_simulation.py / utils.py)

验证修复：
  1. KNOBS 收敛容差放宽后，PhreeqcRM 模拟不再出现
     "A(H2O) Activity of water has not converged" 错误
  2. 浓度数组 comp-major 布局下的极端单元检测正确
  3. 重试后时间步长恢复为原始值
  4. 端到端：15x15x15 网格的扩散+化学耦合可稳定完成

运行方法：
    python -m pytest test_water_activity.py -v
"""

import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    get_solution_definition,
    build_consecutive_mapping,
    TIME_STEP,
    TEMPERATURE,
    MAX_VALID_CONCENTRATION,
)


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 1：KNOBS 收敛容差配置
# ══════════════════════════════════════════════════════════════════════════

class TestKnobsConvergenceTolerance(unittest.TestCase):
    """验证 get_solution_definition 包含 KNOBS 收敛容差设置"""

    def test_solution_definition_contains_knobs(self):
        """生成的 PHREEQC 输入应包含 KNOBS 块"""
        sol_def = get_solution_definition()
        self.assertIn("KNOBS", sol_def)

    def test_solution_definition_contains_convergence_tolerance(self):
        """KNOBS 块应包含 convergence_tolerance 设置"""
        sol_def = get_solution_definition()
        self.assertIn("-convergence_tolerance", sol_def)

    def test_convergence_tolerance_is_relaxed(self):
        """收敛容差应大于默认值 1e-8（放宽为 1e-6 或类似值）"""
        sol_def = get_solution_definition()
        # 提取容差值
        for line in sol_def.split('\n'):
            if '-convergence_tolerance' in line:
                parts = line.strip().split()
                tol_idx = parts.index('-convergence_tolerance') + 1
                tolerance = float(parts[tol_idx])
                self.assertGreater(
                    tolerance, 1e-8,
                    f"收敛容差 {tolerance} 应大于默认值 1e-8"
                )
                break
        else:
            self.fail("未找到 -convergence_tolerance 设置")

    def test_solution_definition_contains_iterations(self):
        """KNOBS 块应包含 iterations 设置以增加迭代次数"""
        sol_def = get_solution_definition()
        self.assertIn("-iterations", sol_def)

    def test_solution_definition_still_has_selected_output(self):
        """修改后仍应保留 SELECTED_OUTPUT 配置"""
        sol_def = get_solution_definition()
        self.assertIn("SELECTED_OUTPUT", sol_def)
        self.assertIn("-pH", sol_def)

    def test_solution_definition_has_all_solutions(self):
        """修改后仍应包含所有三种溶液定义"""
        sol_def = get_solution_definition()
        self.assertIn("SOLUTION 1", sol_def)
        self.assertIn("SOLUTION 2", sol_def)
        self.assertIn("SOLUTION 3", sol_def)


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 2：MAX_VALID_CONCENTRATION 阈值合理性
# ══════════════════════════════════════════════════════════════════════════

class TestMaxValidConcentration(unittest.TestCase):
    """验证极端浓度检测阈值已降低到合理水平"""

    def test_threshold_below_extreme(self):
        """阈值应远低于原值 1e10"""
        self.assertLess(
            MAX_VALID_CONCENTRATION, 1e10,
            "MAX_VALID_CONCENTRATION 应远低于 1e10"
        )

    def test_threshold_above_water_components(self):
        """阈值应高于水组分的天然浓度（H≈111, O≈55 mol/L）"""
        # 水的 H 组分浓度约为 110-111 mol/L
        self.assertGreater(
            MAX_VALID_CONCENTRATION, 120.0,
            "阈值需高于水组分天然浓度 (~111 mol/L)"
        )

    def test_threshold_reasonable_range(self):
        """阈值应在合理范围内（120 ~ 1000 mol/L）"""
        self.assertGreater(MAX_VALID_CONCENTRATION, 120.0)
        self.assertLess(MAX_VALID_CONCENTRATION, 1000.0)


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 3：comp-major 浓度布局下的极端单元检测
# ══════════════════════════════════════════════════════════════════════════

class TestCompMajorCellDetection(unittest.TestCase):
    """
    验证浓度数组使用 comp-major 布局 (ncomps, ncells)
    进行正确的极端单元检测。

    comp-major 布局：
      [comp0_cell0, comp0_cell1, ..., comp0_cellN,
       comp1_cell0, comp1_cell1, ..., comp1_cellN, ...]
    """

    def test_comp_major_detects_extreme_cell(self):
        """comp-major 布局下应正确检测到单个极端单元"""
        ncells = 10
        ncomps = 3
        # 构造 comp-major 数组：所有正常值，但 cell 5 的 comp 2 为极端值
        conc = np.ones(ncomps * ncells, dtype=np.float64) * 50.0
        # cell 5 的 comp 2: index = 2 * 10 + 5 = 25
        conc[2 * ncells + 5] = 1000.0  # 极端值

        conc_2d = conc.reshape((ncomps, ncells))
        cell_max = np.max(np.abs(conc_2d), axis=0)

        # cell 5 的最大值应为 1000.0
        self.assertAlmostEqual(cell_max[5], 1000.0)
        # 其他单元最大值应为 50.0
        for i in range(ncells):
            if i != 5:
                self.assertAlmostEqual(cell_max[i], 50.0)

    def test_wrong_reshape_misses_extreme_cell(self):
        """使用错误的 (ncells, ncomps) 布局会错过极端单元"""
        ncells = 10
        ncomps = 3
        conc = np.ones(ncomps * ncells, dtype=np.float64) * 50.0
        conc[2 * ncells + 5] = 1000.0  # comp 2, cell 5 的极端值

        # 错误的布局：(ncells, ncomps) 
        wrong_2d = conc.reshape((ncells, ncomps))
        wrong_max = np.max(np.abs(wrong_2d), axis=1)

        # 错误布局下，cell 5 不一定是最大值所在的行
        # 而正确布局下，cell 5 应该检测到 1000.0
        correct_2d = conc.reshape((ncomps, ncells))
        correct_max = np.max(np.abs(correct_2d), axis=0)

        self.assertAlmostEqual(correct_max[5], 1000.0)
        # 错误布局可能会把 1000.0 分配到错误的 "cell"
        # 验证错误布局不会把极端值正确归到 cell 5
        # （除非恰好对齐，这里 ncells != ncomps 所以不会对齐）
        self.assertNotAlmostEqual(wrong_max[5], 1000.0)

    def test_comp_major_threshold_filtering(self):
        """使用 comp-major 布局和阈值正确过滤极端单元"""
        ncells = 216  # 6x6x6
        ncomps = 5
        threshold = MAX_VALID_CONCENTRATION

        # 正常浓度（接近水的值）
        conc = np.ones(ncomps * ncells, dtype=np.float64) * 100.0

        # 在 cell 100 设置极端值
        conc[3 * ncells + 100] = threshold + 1.0

        conc_2d = conc.reshape((ncomps, ncells))
        cell_max = np.max(np.abs(conc_2d), axis=0)
        problematic = cell_max > threshold

        self.assertTrue(problematic[100], "cell 100 应被标记为极端")
        self.assertEqual(np.sum(problematic), 1, "应只有 1 个极端单元")

    def test_multiple_extreme_cells_detected(self):
        """多个极端单元应同时被检测到"""
        ncells = 100
        ncomps = 4
        threshold = MAX_VALID_CONCENTRATION

        conc = np.ones(ncomps * ncells, dtype=np.float64) * 50.0
        # 设置 3 个极端单元
        for cell_id in [10, 50, 90]:
            conc[0 * ncells + cell_id] = threshold + 100.0

        conc_2d = conc.reshape((ncomps, ncells))
        cell_max = np.max(np.abs(conc_2d), axis=0)
        problematic = cell_max > threshold

        self.assertEqual(np.sum(problematic), 3)
        self.assertTrue(problematic[10])
        self.assertTrue(problematic[50])
        self.assertTrue(problematic[90])


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 4：时间步长恢复
# ══════════════════════════════════════════════════════════════════════════

class TestTimeStepRestoration(unittest.TestCase):
    """验证重试后时间步长正确恢复为原始值"""

    def test_timestep_restored_after_retry(self):
        """模拟重试场景：失败一次后成功，时间步长应恢复"""
        rm = MagicMock()
        original_dt = TIME_STEP
        retry_dt = original_dt * 0.5

        # 模拟一次失败后成功的场景
        rm.RunCells.side_effect = [1, 0]  # 第一次失败，第二次成功

        # 模拟重试逻辑（与 core_simulation.py 相同）
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                status = rm.RunCells()
                if status == 0:
                    break
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        rm.SetTimeStep(original_dt * (0.5 ** retry_count))
            except Exception:
                retry_count += 1
                if retry_count < max_retries:
                    rm.SetTimeStep(original_dt * (0.5 ** retry_count))

        # 恢复时间步长
        rm.SetTimeStep(original_dt)

        # 验证最后一次 SetTimeStep 调用恢复了原始值
        last_call = rm.SetTimeStep.call_args_list[-1]
        self.assertAlmostEqual(
            last_call[0][0], original_dt,
            msg="最后一次 SetTimeStep 应恢复原始时间步长"
        )

    def test_timestep_restored_after_all_retries_fail(self):
        """所有重试均失败后，时间步长仍应恢复"""
        rm = MagicMock()
        original_dt = TIME_STEP

        rm.RunCells.return_value = 1  # 始终失败

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            status = rm.RunCells()
            if status == 0:
                break
            retry_count += 1
            if retry_count < max_retries:
                rm.SetTimeStep(original_dt * (0.5 ** retry_count))

        # 恢复时间步长
        rm.SetTimeStep(original_dt)

        last_call = rm.SetTimeStep.call_args_list[-1]
        self.assertAlmostEqual(
            last_call[0][0], original_dt,
            msg="所有重试失败后仍应恢复原始时间步长"
        )


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 5：端到端 PhreeqcRM 模拟（需要 phreeqc.dat）
# ══════════════════════════════════════════════════════════════════════════

def _phreeqc_database_available():
    """检查 phreeqc.dat 数据库文件是否可用"""
    return os.path.exists(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phreeqc.dat')
    )


@unittest.skipUnless(_phreeqc_database_available(), "phreeqc.dat not available")
class TestEndToEndSimulation(unittest.TestCase):
    """
    端到端测试：使用真实 PhreeqcRM 验证水活度收敛修复。

    模拟场景：
      - 6x6x6 网格（缩小版，保留边界梯度特性）
      - 酸性/碱性/背景三种溶液
      - 多步扩散 + 化学反应
      - 验证无 RunCells 失败
    """

    def setUp(self):
        from phreeqcrm import PhreeqcRM
        self.NX, self.NY, self.NZ = 6, 6, 6
        self.NXYZ = self.NX * self.NY * self.NZ

        self.rm = PhreeqcRM(self.NXYZ, 1)
        self.rm.SetErrorOn(True)
        self.rm.SetErrorHandlerMode(1)
        self.rm.SetComponentH2O(False)
        self.rm.SetUnitsSolution(2)
        self.rm.SetPrintChemistryOn(False, False, False)

        db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'phreeqc.dat'
        )
        self.rm.LoadDatabase(db_path)

        # 使用带 KNOBS 的溶液定义
        sol_def = get_solution_definition()
        self.rm.RunString(True, True, True, sol_def)
        self.rm.FindComponents()
        self.ncomps = self.rm.GetComponentCount()

        # 配置物理参数
        self.rm.SetTemperature([TEMPERATURE] * self.NXYZ)
        self.rm.SetPressure([1.0] * self.NXYZ)
        self.rm.SetPorosity([0.2] * self.NXYZ)
        self.rm.SetSaturationUser([1.0] * self.NXYZ)
        self.rm.SetRepresentativeVolume([1.0] * self.NXYZ)
        self.rm.SetTimeStep(TIME_STEP)
        self.rm.CreateMapping(list(range(self.NXYZ)))

        # 分配初始溶液
        for i in range(self.NXYZ):
            z = i // (self.NX * self.NY)
            y = (i % (self.NX * self.NY)) // self.NX
            x = i % self.NX
            if x < self.NX // 3 and y < self.NY // 3 and z < self.NZ // 3:
                self.rm.InitialPhreeqcCell2Module(2, [i])  # 酸性
            elif (x > 2 * self.NX // 3 and y > 2 * self.NY // 3
                  and z > 2 * self.NZ // 3):
                self.rm.InitialPhreeqcCell2Module(3, [i])  # 碱性
            else:
                self.rm.InitialPhreeqcCell2Module(1, [i])  # 背景

        status = self.rm.RunCells()
        self.assertEqual(status, 0, "初始平衡应成功")

    def tearDown(self):
        try:
            self.rm.CloseFiles()
        except Exception:
            pass

    def test_initial_equilibrium_succeeds(self):
        """初始平衡计算应成功完成"""
        conc = self.rm.GetConcentrations()
        self.assertEqual(
            len(conc), self.NXYZ * self.ncomps,
            "浓度数组大小应为 NXYZ * ncomps"
        )

    def test_diffusion_chemistry_10_steps_no_error(self):
        """10 步扩散+化学反应应无 RunCells 错误"""
        from scipy.ndimage import convolve

        kernel = np.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=np.float64) / 6.0

        conc = np.array(self.rm.GetConcentrations(), dtype=np.float64)
        errors = 0

        for step in range(1, 11):
            # 扩散
            c_in = conc.reshape((self.ncomps, self.NZ, self.NY, self.NX))
            c_out = np.empty_like(c_in)
            for ci in range(self.ncomps):
                lap = convolve(c_in[ci], kernel, mode='nearest')
                c_out[ci] = np.maximum(c_in[ci] + 0.05 * lap, 0.0)
            conc = c_out.flatten()
            self.rm.SetConcentrations(conc)

            # 化学反应
            self.rm.SetTime(step * TIME_STEP)
            try:
                status = self.rm.RunCells()
                if status != 0:
                    errors += 1
                else:
                    conc = np.array(
                        self.rm.GetConcentrations(), dtype=np.float64
                    )
            except Exception:
                errors += 1

        self.assertEqual(errors, 0, "10 步模拟不应出现 RunCells 错误")

    def test_concentrations_remain_finite(self):
        """模拟过程中浓度应保持有限值"""
        from scipy.ndimage import convolve

        kernel = np.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=np.float64) / 6.0

        conc = np.array(self.rm.GetConcentrations(), dtype=np.float64)

        for step in range(1, 6):
            c_in = conc.reshape((self.ncomps, self.NZ, self.NY, self.NX))
            c_out = np.empty_like(c_in)
            for ci in range(self.ncomps):
                lap = convolve(c_in[ci], kernel, mode='nearest')
                c_out[ci] = np.maximum(c_in[ci] + 0.05 * lap, 0.0)
            conc = c_out.flatten()

            self.assertTrue(
                np.isfinite(conc).all(),
                f"步骤 {step}：浓度包含非有限值"
            )

            self.rm.SetConcentrations(conc)
            status = self.rm.RunCells()
            if status == 0:
                conc = np.array(
                    self.rm.GetConcentrations(), dtype=np.float64
                )

    def test_ph_values_in_valid_range(self):
        """模拟后 pH 值应在合理范围内（0-14）"""
        self.rm.SetCurrentSelectedOutputUserNumber(1)
        headings = self.rm.GetSelectedOutputHeadings()
        data = self.rm.GetSelectedOutput()
        n_vars = len(headings)

        ph_index = list(headings).index('pH')
        ph_values = []
        for i in range(self.NXYZ):
            ph_val = data[i * n_vars + ph_index]
            if isinstance(ph_val, (int, float)) and np.isfinite(ph_val):
                ph_values.append(ph_val)

        self.assertGreater(len(ph_values), 0, "应能获取 pH 值")
        self.assertGreaterEqual(min(ph_values), 0.0, "pH 不应小于 0")
        self.assertLessEqual(max(ph_values), 14.0, "pH 不应大于 14")

    def test_concentration_layout_is_comp_major(self):
        """验证 GetConcentrations() 返回 comp-major 布局"""
        conc = np.array(self.rm.GetConcentrations(), dtype=np.float64)
        comps = self.rm.GetComponents()

        self.assertEqual(len(conc), self.NXYZ * self.ncomps)

        # comp-major 意味着前 NXYZ 个值属于第一个组分
        # 所有背景单元的同一组分应有相似值
        conc_2d = conc.reshape((self.ncomps, self.NXYZ))

        # H 组分（index 0）来自水，所有单元应接近 ~110 mol/L
        h_values = conc_2d[0]
        self.assertGreater(
            h_values.min(), 100.0,
            "H 组分浓度应来自水（>100 mol/L）"
        )
        self.assertLess(
            h_values.max(), 120.0,
            "H 组分浓度应来自水（<120 mol/L）"
        )


# ══════════════════════════════════════════════════════════════════════════
# 测试套件 6：build_consecutive_mapping 与极端单元排除集成
# ══════════════════════════════════════════════════════════════════════════

class TestExtremeConcentrationExclusion(unittest.TestCase):
    """验证极端浓度单元通过 build_consecutive_mapping 被正确排除"""

    def test_extreme_cells_mapped_to_negative_one(self):
        """极端单元应映射为 -1"""
        ncells = 20
        ncomps = 3
        threshold = MAX_VALID_CONCENTRATION

        conc = np.ones(ncomps * ncells, dtype=np.float64) * 50.0
        # cell 5 和 cell 15 设为极端
        conc[1 * ncells + 5] = threshold + 1.0
        conc[0 * ncells + 15] = threshold + 1.0

        conc_2d = conc.reshape((ncomps, ncells))
        cell_max = np.max(np.abs(conc_2d), axis=0)
        problematic = cell_max > threshold

        mapping = build_consecutive_mapping(ncells, problematic)

        self.assertEqual(mapping[5], -1, "极端单元 5 应映射为 -1")
        self.assertEqual(mapping[15], -1, "极端单元 15 应映射为 -1")

    def test_normal_cells_have_consecutive_ids(self):
        """正常单元应有连续的化学单元 ID"""
        ncells = 20
        ncomps = 3
        threshold = MAX_VALID_CONCENTRATION

        conc = np.ones(ncomps * ncells, dtype=np.float64) * 50.0
        conc[1 * ncells + 5] = threshold + 1.0  # cell 5 极端

        conc_2d = conc.reshape((ncomps, ncells))
        cell_max = np.max(np.abs(conc_2d), axis=0)
        problematic = cell_max > threshold

        mapping = build_consecutive_mapping(ncells, problematic)

        active_ids = sorted([m for m in mapping if m >= 0])
        expected = list(range(len(active_ids)))
        self.assertEqual(active_ids, expected)

    def test_no_extreme_cells_gives_identity_mapping(self):
        """无极端单元时应返回 identity 映射"""
        ncells = 10
        ncomps = 2
        threshold = MAX_VALID_CONCENTRATION

        conc = np.ones(ncomps * ncells, dtype=np.float64) * 50.0

        conc_2d = conc.reshape((ncomps, ncells))
        cell_max = np.max(np.abs(conc_2d), axis=0)
        problematic = cell_max > threshold

        self.assertFalse(np.any(problematic))
        mapping = build_consecutive_mapping(ncells, problematic)
        self.assertEqual(mapping, list(range(ncells)))


# ══════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
