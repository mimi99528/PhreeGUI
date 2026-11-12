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
DATABASE_PATHS = [
    r"C:\phreeqc\database\phreeqc.dat",
    r"C:\Program Files\USGS\phreeqc\database\phreeqc.dat",
    r"phreeqc.dat",
    "phreeqc.dat"
]

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
    
    Returns:
        PhreeqcRM: 配置好的PhreeqcRM实例
    """
    rm = PhreeqcRM(NXYZ, NTHREADS)
    
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
    尝试加载Phreeqc数据库
    
    Args:
        rm (PhreeqcRM): PhreeqcRM实例
        
    Returns:
        bool: 是否成功加载数据库
    """
    for db_path in DATABASE_PATHS:
        try:
            rm.LoadDatabase(db_path)
            print(f"   成功加载数据库: {db_path}")
            return True
        except Exception as e:
            continue
    
    print("   警告: 无法加载标准数据库文件")
    print("   将尝试使用InitialPhreeqc定义组件...")
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

def assign_initial_solutions(rm):
    """
    为网格分配初始溶液
    
    Args:
        rm (PhreeqcRM): PhreeqcRM实例
    """
    for i in range(NXYZ):
        z_idx = i // (NX * NY)
        remaining = i % (NX * NY)
        y_idx = remaining // NX
        x_idx = remaining % NX
        
        # 左上角前层区域 (酸性源)
        if x_idx < NX//3 and y_idx < NY//3 and z_idx < NZ//3:
            rm.InitialPhreeqcCell2Module(INITIAL_SOLUTIONS['acidic']['number'], [i])
        # 右下角后层区域 (碱性源)
        elif x_idx > 2*NX//3 and y_idx > 2*NY//3 and z_idx > 2*NZ//3:
            rm.InitialPhreeqcCell2Module(INITIAL_SOLUTIONS['alkaline']['number'], [i])
        else:  # 其他位置使用背景溶液
            rm.InitialPhreeqcCell2Module(INITIAL_SOLUTIONS['background']['number'], [i])
    
    print("   初始溶液已分配: 左上角 = 酸性源, 右下角 = 碱性源, 其余 = 背景")

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