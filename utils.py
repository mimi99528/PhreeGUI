"""
PhreeqcRM 模拟工具模块
包含通用工具函数、常量定义和基础配置
"""

import numpy as np
import phreeqcrm
from phreeqcrm import PhreeqcRM
import time

# ==================== 基础配置常量 ====================
# 网格尺寸配置
NX, NY, NZ = 15, 15, 15  # 三维网格大小
NXYZ = NX * NY * NZ      # 总网格数
NTHREADS = 8             # 并行线程数

# 物理参数配置
TIME_STEP = 86400.0      # 时间步长(秒)
TEMPERATURE = 25.0       # 温度(摄氏度)
PRESSURE = 1.0           # 压力(atm)
POROSITY = 0.2           # 孔隙度
SATURATION = 1.0         # 饱和度
REPRESENTATIVE_VOLUME = 1.0  # 代表性体积

# 数值计算参数
DIFFUSION_COEFFICIENT = 0.05  # 扩散系数
CHEMICAL_SYNC_INTERVAL = 2    # 化学同步间隔
RECORD_INTERVAL = 10          # 数据记录间隔
REBALANCE_FRACTION = 0.1      # 负载平衡分数
# 原定义于 core_simulation.py，因 DPG 时代模块归档而迁至此共享叶子模块，数值保持不变
MAX_VALID_CONCENTRATION = 500.0  # 单元浓度有效性上限(mol/L)，超过视为数值异常

# ==================== 数据库路径配置 ====================
# 【修复 b6】这里原来另有一份候选列表（含两条裸 "phreeqc.dat"，按 cwd 解析），
# 与 phreeqc_config.DATABASE_PATHS 内容不同 —— 同一件事有两个真相，
# 导致"运行时用系统库、pytest 用仓库库"这种双源结果。
# 现在唯一真源在 phreeqc_config（决策 D2：仓库根优先，系统路径只作 fallback，
# 且解析一律 __file__ 基，不再随 cwd 漂移）。本名字仅为兼容旧导入保留。
from phreeqc_config import (  # noqa: E402
    REPO_DATABASE_PATH,
    SYSTEM_DATABASE_PATHS,
    find_phreeqc_database,
)

DATABASE_PATHS = [REPO_DATABASE_PATH] + SYSTEM_DATABASE_PATHS

# ==================== 溶液定义 ====================
INITIAL_SOLUTIONS = {
    'background': {
        'number': 1,
        'name': 'Background Water',
        'pH': 7.0,
        'components': {}
    },
    'acidic': {
        'number': 2,
        'name': 'Acidic Source',
        'pH': 3.0,
        'components': {'Cl': [1000.0, "HCl"]}  # 作为HCl
    },
    'alkaline': {
        'number': 3,
        'name': 'Alkaline Source',
        'pH': 11.0,
        'components': {'Na': [1000.0, "NaOH"]} # 作为NaOH，单位mg/L
    }
}

# ==================== 工具函数 ====================

def initialize_phreeqcrm():
    """
    初始化PhreeqcRM实例

    【收尾 2 —— 领队 seq 26 二.2】接入 rm 台账。
    本函数目前**没有生产调用方**（全仓库只有 phreeqc_config:606 把同名符号
    指向 create_fully_initialized_phreeqcrm，真正的 rm 创建走
    phreeqc_config.create_robust_phreeqcrm_instance /
    coupled_chemistry_physics_gui._setup_phreeqcrm_for_coupled_sim，两者都已登记）。
    领队裁决：这条路径**不删**，但必须过台账 —— 否则将来任何一处调用它，
    就会绕过"同进程只允许一个存活 rm"的守卫（决策 D7），而守卫是 exit 139
    的唯一 Python 层防线（require_rm_mapped 的说明里记着实测）。
    这里用函数内延迟导入：utils 是叶子模块，phreeqc_config 依赖它，
    顶层反向 import 会成环。

    Returns:
        PhreeqcRM: 配置好的PhreeqcRM实例

    Raises:
        RuntimeError: 本进程已有存活的 PhreeqcRM（台账硬失败，不静默造第二个）
    """
    from phreeqc_config import _register_rm

    rm = PhreeqcRM(NXYZ, NTHREADS)
    _register_rm(rm, "utils.initialize_phreeqcrm", nxyz=NXYZ, nthreads=NTHREADS)
    
    # 设置基本配置
    rm.SetErrorOn(True)
    rm.SetErrorHandlerMode(1)
    rm.SetPrintChemistryOn(False, False, False)
    # 安全性优化：注释掉OpenFiles避免权限问题
    # rm.OpenFiles()
    
    # 优化并行计算设置
    rm.SetRebalanceByCell(True)
    rm.SetRebalanceFraction(REBALANCE_FRACTION)
    
    return rm

def load_database(rm):
    """
    加载 Phreeqc 数据库

    【决策 D2 / 修复 b6】原来这里是"遍历 DATABASE_PATHS 逐个 try，全失败就
    打印'将尝试使用InitialPhreeqc定义组件'然后 return False"。那有两重问题：
      1) 候选表与 phreeqc_config 的那份是两份，且按 cwd 解析，命中哪个取决于
         从哪里启动 —— 这正是"仓库版 vs 系统版 phreeqc.dat 双真源"的来源；
      2) 失败后继续跑，让 FindComponents 拿到一个组分集合不完整/为空的 rm，
         后面 RunCells 直接以 exit 127 崩掉。
    现在只问 phreeqc_config.find_phreeqc_database() 要那一个路径（仓库根优先，
    系统安装路径仅作 fallback），并把失败原样回报给调用方，由调用方硬失败。

    Args:
        rm (PhreeqcRM): PhreeqcRM实例

    Returns:
        bool: 是否成功加载数据库
    """
    from phreeqc_config import find_phreeqc_database

    db_path = find_phreeqc_database()
    if not db_path:
        print("   错误: 找不到 phreeqc.dat（仓库根与系统安装位置都试过）")
        return False
    try:
        rm.LoadDatabase(db_path)
        print(f"   成功加载数据库: {db_path}")
        return True
    except Exception as e:
        print(f"   错误: 加载 {db_path} 失败: {e!r}")
        return False

def configure_physical_parameters(rm):
    """
    配置物理参数和单位
    
    Args:
        rm (PhreeqcRM): PhreeqcRM实例
    """
    # 设置时间步长和基本物理参数
    rm.SetTimeStep(TIME_STEP)
    rm.SetTemperature([TEMPERATURE] * NXYZ)
    rm.SetPressure([PRESSURE] * NXYZ)
    rm.SetPorosity([POROSITY] * NXYZ)
    rm.SetSaturationUser([SATURATION] * NXYZ)
    
    # 设置浓度单位为 mol/L
    rm.SetUnitsSolution(2)
    
    # 设置代表性体积
    rm.SetRepresentativeVolume([REPRESENTATIVE_VOLUME] * NXYZ)
    
    # 优化负载平衡设置
    rm.SetRebalanceFraction(REBALANCE_FRACTION)
    rm.SetRebalanceByCell(True)
    
    # 减少化学输出以提高性能
    rm.SetPrintChemistryOn(False, False, False)

def get_solution_definition():
    """
    获取初始溶液定义字符串
    
    Returns:
        str: Phreeqc输入字符串
    """
    solution_input = ""
    
    # `定义三种溶液`
    for sol_key, sol_config in INITIAL_SOLUTIONS.items():
        solution_input += f"SOLUTION {sol_config['number']} {sol_config['name']}\n"
        solution_input += f"    units mg/L; temp {TEMPERATURE}; pH {sol_config['pH']}; water 1.0\n"
        # 添加组件（如果有）
        for component, concentration in sol_config.get('components', {}).items():
            solution_input += f"    {component} {concentration[0]} as {concentration[1]} charge\n"
        solution_input += "END\n"
    
    # 放宽 PHREEQC 收敛容差，避免 "A(H2O) Activity of water has not converged"
    # 默认容差约 1e-8，扩散后边界单元残差可能略高于此阈值
    solution_input += """
KNOBS
    -convergence_tolerance 1e-6
    -iterations 200
END
"""

    # 添加SELECTED_OUTPUT配置
    solution_input += """
# 正确配置SELECTED_OUTPUT，只获取pH值
SELECTED_OUTPUT 1
    -reset false
    -pH
END
"""
    return solution_input

def create_mapping(rm):
    """
    创建固定的网格到化学单元映射
    
    Args:
        rm (PhreeqcRM): PhreeqcRM实例
    """
    grid2chem = list(range(NXYZ))  # 1:1映射，避免动态重分配
    rm.CreateMapping(grid2chem)
    print("   固定负载分配完成")

def build_consecutive_mapping(n_cells, exclude_mask):
    """
    构建连续化学单元索引映射，排除指定单元。

    活跃单元按顺序映射到化学单元 0, 1, 2, ...，被排除的单元映射为 -1。
    这确保 PhreeqcRM.GetChemistryCellCount() 只反映活跃单元数量，
    RunCells 不会处理被排除的单元。

    Args:
        n_cells (int):          网格单元总数
        exclude_mask (iterable): 布尔数组或列表，True 表示排除该单元

    Returns:
        list[int]: 长度为 n_cells 的映射列表
    """
    mapping = [-1] * n_cells
    chem_id = 0
    for i in range(n_cells):
        if not exclude_mask[i]:
            mapping[i] = chem_id
            chem_id += 1
    return mapping

def get_elapsed_time(start_time):
    """
    计算经过的时间
    
    Args:
        start_time (float): 开始时间戳
        
    Returns:
        float: 经过的时间（秒）
    """
    return time.time() - start_time

def print_simulation_header(title="PhreeqcRM 空间扩散模拟"):
    """
    打印模拟标题
    
    Args:
        title (str): 模拟标题
    """
    print("=" * 60)
    print(title)
    print("=" * 60)