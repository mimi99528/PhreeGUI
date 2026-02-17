"""
PhreeqcRM 物理计算模块
包含扩散计算和其他物理过程的实现
"""

import numpy as np
import time
from scipy.ndimage import convolve
from utils import NX, NY, NZ, NXYZ, DIFFUSION_COEFFICIENT

def apply_diffusion_vectorized(c_flat, ncomps, nx=NX, ny=NY, nz=NZ, diffusion_coeff=DIFFUSION_COEFFICIENT):
    """
    向量化的3D扩散计算 - 使用卷积操作替代嵌套循环
    
    Args:
        c_flat (list): 扁平化的浓度数组
        ncomps (int): 组件数量
        nx (int): X方向网格数
        ny (int): Y方向网格数
        nz (int): Z方向网格数
        diffusion_coeff (float): 扩散系数
        
    Returns:
        tuple: (新的浓度数组, 计算时间)
    """
    nxyz = nx * ny * nz
    c_data = np.array(c_flat, dtype=float).reshape((ncomps, nxyz))
    c_new = np.empty_like(c_data)

    # 创建3D卷积核 (7点 stencil)
    kernel = np.zeros((3, 3, 3))
    kernel[1, 1, 0] = 1  # -z方向
    kernel[1, 0, 1] = 1  # -y方向
    kernel[0, 1, 1] = 1  # -x方向
    kernel[2, 1, 1] = 1  # +x方向
    kernel[1, 2, 1] = 1  # +y方向
    kernel[1, 1, 2] = 1  # +z方向
    kernel[1, 1, 1] = -6  # 中心点
    kernel /= 6.0  # 归一化

    start_time = time.time()

    for comp_idx in range(ncomps):
        # 重塑为3D网格
        grid_3d = c_data[comp_idx, :].reshape((nz, ny, nx))

        # 使用卷积计算拉普拉斯算子
        laplacian = convolve(grid_3d, kernel, mode='constant', cval=0.0)

        # 更新浓度 (显式欧拉方法)
        grid_new_3d = grid_3d + diffusion_coeff * laplacian

        # 非负约束
        grid_new_3d = np.maximum(grid_new_3d, 0)

        c_new[comp_idx] = grid_new_3d.flatten()

    computation_time = time.time() - start_time
    return c_new.flatten().tolist(), computation_time

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