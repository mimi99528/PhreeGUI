"""
PhreeqcRM 物理计算模块
包含扩散计算和其他物理过程的实现
"""

import numpy as np
import time
from scipy.ndimage import convolve
from utils import NX, NY, NZ, NXYZ, DIFFUSION_COEFFICIENT


# physics.py - 内存安全版本
def apply_diffusion_vectorized_inplace(c_in, c_out, ncomps, nx=NX, ny=NY, nz=NZ, diffusion_coeff=DIFFUSION_COEFFICIENT):
    """
    原地向量化的 3D 扩散计算 - 内存安全版本
    
    Args:
        c_in (np.ndarray): 输入浓度数组，形状 (ncomps, nxyz)，float64，C连续
        c_out (np.ndarray): 输出浓度数组，形状 (ncomps, nxyz)，float64，C连续
        ncomps (int): 组件数量
        nx, ny, nz: 网格维度
        diffusion_coeff (float): 扩散系数
    
    注意：
    - 所有数组必须预先分配且持久存在
    - 不创建临时对象，避免内存碎片
    - 原地写入，提高性能和稳定性
    """
    # 验证输入参数
    nxyz = nx * ny * nz
    expected_size = ncomps * nxyz
    
    if c_in.size != expected_size or c_out.size != expected_size:
        raise RuntimeError(f"缓冲区大小不匹配: 期望 {expected_size}, 输入 {c_in.size}, 输出 {c_out.size}")
    
    if c_in.dtype != np.float64 or c_out.dtype != np.float64:
        raise RuntimeError("缓冲区必须为 float64 类型")
    
    if not c_in.flags['C_CONTIGUOUS'] or not c_out.flags['C_CONTIGUOUS']:
        raise RuntimeError("缓冲区必须为 C 连续内存布局")
    
    # 创建视图而不复制数据
    c_in_view = c_in.reshape((ncomps, nz, ny, nx))
    c_out_view = c_out.reshape((ncomps, nz, ny, nx))
    
    # 定义拉普拉斯算子核
    kernel = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ], dtype=np.float64) / 6.0
    
    start_time = time.time()
    
    # 对每个组件执行扩散计算
    for comp_idx in range(ncomps):
        # 获取当前组件的3D网格
        grid_current = c_in_view[comp_idx]
        
        # 计算拉普拉斯算子
        laplacian = convolve(grid_current, kernel, mode='nearest')
        
        # 应用扩散方程：c_new = c_old + D * laplacian
        grid_new = grid_current + diffusion_coeff * laplacian
        
        # 确保非负性
        np.maximum(grid_new, 0.0, out=grid_new)
        
        # 写入输出缓冲区
        c_out_view[comp_idx] = grid_new
    
    computation_time = time.time() - start_time
    
    # 验证输出数据的有效性
    if not np.isfinite(c_out).all():
        raise RuntimeError(f"扩散计算产生非有限值，在时间 {computation_time:.6f}s")
    
    return computation_time

def apply_diffusion_vectorized(c_flat, ncomps, nx=NX, ny=NY, nz=NZ, diffusion_coeff=DIFFUSION_COEFFICIENT):
    """
    向量化的 3D 扩散计算 - 兼容旧接口
    返回：扁平列表 [comp0_cell0, comp0_cell1, ..., comp1_cell0, ...]
    """
    nxyz = nx * ny * nz
    c_data = np.array(c_flat, dtype=np.float64).reshape((ncomps, nxyz))
    c_new = np.empty_like(c_data)

    kernel = np.zeros((3, 3, 3), dtype=np.float64)
    kernel[1, 1, 0] = 1
    kernel[1, 0, 1] = 1
    kernel[0, 1, 1] = 1
    kernel[2, 1, 1] = 1
    kernel[1, 2, 1] = 1
    kernel[1, 1, 2] = 1
    kernel[1, 1, 1] = -6
    kernel /= 6.0

    start_time = time.time()

    for comp_idx in range(ncomps):
        grid_3d = c_data[comp_idx, :].reshape((nz, ny, nx))
        laplacian = convolve(grid_3d, kernel, mode='nearest')
        grid_new_3d = grid_3d + diffusion_coeff * laplacian
        grid_new_3d = np.maximum(grid_new_3d, 0.0).astype(np.float64)
        c_new[comp_idx] = grid_new_3d.flatten()

    computation_time = time.time() - start_time

    # 🔥 关键：返回扁平列表，长度 = ncomps × nxyz
    c_new_flat = c_new.flatten().astype(np.float64)
    c_new_list = [float(v) for v in c_new_flat]

    return c_new_list, computation_time

def calculate_concentration_gradient(concentrations, nx=NX, ny=NY, nz=NZ):
    """
    计算浓度梯度场
    
    Args:
        concentrations (np.ndarray): 浓度场数据
        nx (int): X方向网格数
        ny (int): Y方向网格数
        nz (int): Z方向网格数
        
    Returns:
        dict: 包含各方向梯度的字典
    """
    # 重塑为3D网格
    conc_3d = concentrations.reshape((nz, ny, nx))
    
    # 计算各方向梯度
    grad_x = np.gradient(conc_3d, axis=2)
    grad_y = np.gradient(conc_3d, axis=1)
    grad_z = np.gradient(conc_3d, axis=0)
    
    return {
        'grad_x': grad_x,
        'grad_y': grad_y,
        'grad_z': grad_z,
        'magnitude': np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    }

def apply_boundary_conditions(concentrations, boundary_type='periodic'):
    """
    应用边界条件
    
    Args:
        concentrations (np.ndarray): 浓度数组
        boundary_type (str): 边界条件类型 ('periodic', 'reflective', 'fixed')
        
    Returns:
        np.ndarray: 应用边界条件后的浓度数组
    """
    if boundary_type == 'periodic':
        # 周期性边界条件
        return np.pad(concentrations, 1, mode='wrap')
    elif boundary_type == 'reflective':
        # 反射边界条件
        return np.pad(concentrations, 1, mode='reflect')
    elif boundary_type == 'fixed':
        # 固定边界条件（零通量）
        padded = np.pad(concentrations, 1, mode='edge')
        return padded
    else:
        return concentrations

def compute_mass_conservation(c_old, c_new, ncomps):
    """
    计算质量守恒误差
    
    Args:
        c_old (list): 旧浓度数组
        c_new (list): 新浓度数组
        ncomps (int): 组件数量
        
    Returns:
        dict: 各组件的质量变化信息
    """
    mass_changes = {}
    
    for comp_idx in range(ncomps):
        old_mass = sum(c_old[comp_idx::ncomps])
        new_mass = sum(c_new[comp_idx::ncomps])
        mass_change = abs(new_mass - old_mass)
        mass_changes[f'component_{comp_idx}'] = {
            'old_mass': old_mass,
            'new_mass': new_mass,
            'change': mass_change,
            'relative_change': mass_change / old_mass if old_mass != 0 else 0
        }
    
    return mass_changes