"""
PHREEQCRM 配置和初始化模块
提供更稳健的化学系统初始化和错误处理
"""

import os
import logging
from typing import List, Dict, Optional
import numpy as np

try:
    from phreeqcrm import PhreeqcRM
    PHREEQC_AVAILABLE = True
except ImportError:
    PHREEQC_AVAILABLE = False
    print("警告: phreeqcrm 模块未安装")

# 配置日志
_chem_logger = logging.getLogger("phreeqc_config")

# 默认配置参数
DEFAULT_NX = 6
DEFAULT_NY = 6  
DEFAULT_NZ = 6
DEFAULT_NXYZ = DEFAULT_NX * DEFAULT_NY * DEFAULT_NZ
DEFAULT_NTHREADS = 1

# 数据库路径候选列表
DATABASE_PATHS = [
    # "PITZER.DAT",
    r"C:\phreeqc\database\phreeqc.dat",
    r"C:\Program Files\USGS\phreeqc\database\phreeqc.dat",
    r"/usr/local/share/phreeqc/phreeqc.dat",
    r"/usr/share/phreeqc/phreeqc.dat",
    "../phreeqc.dat"
]

def find_phreeqc_database() -> Optional[str]:
    """
    寻找可用的PHREEQC数据库文件
    
    Returns:
        str or None: 找到的数据库路径，如果没找到返回None
    """
    for db_path in DATABASE_PATHS:
        if os.path.exists(db_path):
            _chem_logger.info(f"找到PHREEQC数据库: {db_path}")
            return os.path.abspath(db_path)
    
    _chem_logger.warning("未找到PHREEQC数据库文件")
    return None

def create_robust_phreeqcrm_instance(nxyz: int = DEFAULT_NXYZ, 
                                   nthreads: int = DEFAULT_NTHREADS) -> Optional[PhreeqcRM]:
    """
    创建健壮的PHREEQCRM实例，包含错误处理和配置优化
    
    Args:
        nxyz: 网格单元总数
        nthreads: 线程数
        
    Returns:
        PhreeqcRM实例或None（如果失败）
    """
    if not PHREEQC_AVAILABLE:
        _chem_logger.error("PHREEQCRM不可用")
        return None
        
    try:
        # 创建实例
        rm = PhreeqcRM(nxyz, nthreads)

        rm.SetComponentH2O(False)  # <-- 移到这里，最优先级
        
        # 基本配置
        rm.SetErrorOn(True)
        rm.SetErrorHandlerMode(1)  # 详细错误模式
        
        # 性能优化设置
        rm.SetRebalanceByCell(True)
        rm.SetRebalanceFraction(0.1)
        rm.SetSpeciesSaveOn(False)
        rm.SetPartitionUZSolids(False)
        
        # 单位设置
        rm.SetUnitsSolution(2)  # mol/L
        rm.SetUnitsPPassemblage(1)
        rm.SetUnitsExchange(1)
        
        _chem_logger.info("PHREEQCRM实例创建成功")
        return rm
        
    except Exception as e:
        _chem_logger.error(f"创建PHREEQCRM实例失败: {e}")
        return None

def load_phreeqc_database(rm: PhreeqcRM) -> bool:
    """
    加载PHREEQC数据库
    
    Args:
        rm: PhreeqcRM实例
        
    Returns:
        bool: 是否成功加载
    """
    db_path = find_phreeqc_database()
    if not db_path:
        _chem_logger.error("找不到PHREEQC数据库文件")
        return False
    rm.SetPrintChemistryOn(True, True, True)
    rm.SetFilePrefix("phreeqc")
    try:
        rm.LoadDatabase(db_path)
        _chem_logger.info(f"成功加载数据库: {db_path}")
        return True
    except Exception as e:
        _chem_logger.error(f"加载数据库失败 {db_path}: {e}")
        return False

def setup_default_chemical_system(rm: PhreeqcRM) -> bool:
    """
    设置默认的化学系统（酸碱中和反应）
    
    Args:
        rm: PhreeqcRM实例
        
    Returns:
        bool: 是否成功设置
    """
    # 1. 必须在 FindComponents 之前设置！
    rm.SetComponentH2O(False)
    try:
        # 定义简化的化学系统（使用mol/L单位，避免水组分计算）
        input_string = """
        SOLUTION 1 
            units mol/L
            temp 25.0 
            pH 7.0
            density 1.0
        END
        
        SOLUTION 2 Simple Acidic
            pH 3.0 charge
            temp 25.0
            units mol/L
            Cl 0.01 as HCl
        END
        
        SOLUTION 3 Simple Basic
            pH 11.0 charge
            temp 25.0
            units mol/L
            Na 0.01 as NaOH
        END
        
        SELECTED_OUTPUT 1
            -reset false
            -pH
        END
        """
        
        # 运行输入字符串
        rm.RunString(True, True, True, input_string)
        # 识别化学组分
        rm.FindComponents()
        
        ncomps = rm.GetComponentCount()
        comps = list(rm.GetComponents())
        
        _chem_logger.info(f"检测到 {ncomps} 个化学组分: {comps}")

        if ncomps == 0:
            _chem_logger.warning("未检测到任何化学组分，使用默认组分")
            comps = ['H+', 'OH-', 'Na+', 'Cl-']
            
        return True
        
    except Exception as e:
        _chem_logger.error(f"设置化学系统失败: {e}")
        return False

def initialize_grid_distribution(rm: PhreeqcRM, nx: int = DEFAULT_NX, 
                               ny: int = DEFAULT_NY, nz: int = DEFAULT_NZ) -> bool:
    """
    初始化网格分布和初始条件
    
    Args:
        rm: PhreeqcRM实例
        nx, ny, nz: 网格维度
        
    Returns:
        bool: 是否成功初始化
    """
    try:
        nxyz = nx * ny * nz
        
        # 分配初始溶液到网格
        for i in range(nxyz):
            # 计算3D索引
            z_idx = i // (nx * ny)
            remaining = i % (nx * ny)
            y_idx = remaining // nx
            x_idx = remaining % nx
            
            # 空间分布策略
            if x_idx < nx//3 and y_idx < ny//3 and z_idx < nz//3:
                # 左上角区域 - 酸性
                rm.InitialPhreeqcCell2Module(2, [i])
            elif x_idx > 2*nx//3 and y_idx > 2*ny//3 and z_idx > 2*nz//3:
                # 右下角区域 - 碱性
                rm.InitialPhreeqcCell2Module(3, [i])
            else:
                # 其他区域 - 背景
                rm.InitialPhreeqcCell2Module(1, [i])
        
        # 设置物理参数
        rm.SetTimeStep(0.017)  # 17ms
        rm.SetTemperature([25.0] * nxyz)
        rm.SetPressure([1.0] * nxyz)
        rm.SetPorosity([0.3] * nxyz)
        rm.SetSaturationUser([1.0] * nxyz)
        rm.SetRepresentativeVolume([1.0] * nxyz)
        rm.SetComponentH2O(False)

        # 创建映射
        grid2chem = list(range(nxyz))
        print(grid2chem)
        rm.CreateMapping(grid2chem)
        
        # 运行初始平衡
        _chem_logger.info("运行初始化学平衡...")
        status = rm.RunCells()
        if status != 0:
            _chem_logger.warning(f"初始平衡返回状态: {status}")
        
        _chem_logger.info("网格分布初始化完成")
        return True
        
    except Exception as e:
        _chem_logger.error(f"网格分布初始化失败: {e}")
        return False

def get_chemical_status(rm: PhreeqcRM) -> Dict:
    """
    获取化学系统的当前状态
    
    Args:
        rm: PhreeqcRM实例
        
    Returns:
        dict: 包含系统状态信息的字典
    """
    try:
        status = {
            'component_count': rm.GetComponentCount(),
            'components': list(rm.GetComponents()),
            'concentrations': rm.GetConcentrations(),
            'time': rm.GetTime(),
            'time_step': rm.GetTimeStep()
        }
        
        # 获取pH数据（如果可用）
        try:
            selected_out = rm.GetSelectedOutput()
            if selected_out and len(selected_out) > 0:
                status['ph_data'] = selected_out[0].get('pH', [])
        except:
            status['ph_data'] = []
            
        return status
        
    except Exception as e:
        _chem_logger.error(f"获取化学状态失败: {e}")
        return {}

def create_fully_initialized_phreeqcrm(nx: int = DEFAULT_NX, 
                                    ny: int = DEFAULT_NY, 
                                    nz: int = DEFAULT_NZ,
                                    nthreads: int = DEFAULT_NTHREADS) -> Optional[PhreeqcRM]:
    """
    创建完全初始化的PHREEQCRM实例
    
    这是一个便利函数，执行完整的初始化流程
    
    Returns:
        PhreeqcRM实例或None
    """
    _chem_logger.info("开始PHREEQCRM完整初始化流程...")
    
    # 1. 创建实例
    rm = create_robust_phreeqcrm_instance(nx * ny * nz, nthreads)
    if not rm:
        return None
    
    # 2. 加载数据库
    if not load_phreeqc_database(rm):
        _chem_logger.error("数据库加载失败，尝试继续...")
        # 即使数据库加载失败也继续，因为可以使用InitialPhreeqc定义
    
    # 3. 设置化学系统
    if not setup_default_chemical_system(rm):
        _chem_logger.error("化学系统设置失败")
        return None
    
    # 4. 初始化网格分布
    if not initialize_grid_distribution(rm, nx, ny, nz):
        _chem_logger.error("网格分布初始化失败")
        return None
    
    # 5. 验证状态
    status = get_chemical_status(rm)
    _chem_logger.info(f"最终状态 - 组分数: {status['component_count']}, 时间: {status['time']}")
    
    _chem_logger.info("PHREEQCRM完整初始化成功!")
    return rm

# 便利函数别名
initialize_phreeqcrm = create_fully_initialized_phreeqcrm