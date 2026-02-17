"""
优化版的物理扩散计算
基于性能分析结果的改进版本
"""

import numpy as np
import time
from scipy.ndimage import convolve

def optimized_diffusion_vectorized(c_flat, ncomps, nx, ny, nz, diffusion_coeff=0.05):
    """
    向量化的3D扩散计算 - 使用卷积操作替代嵌套循环
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

def benchmark_comparison():
    """性能对比测试"""
    print("=== 物理扩散计算性能对比 ===")
    
    # 测试参数
    nx, ny, nz = 15, 15, 15
    ncomps = 3
    nxyz = nx * ny * nz
    c_flat = [0.1] * (ncomps * nxyz)
    iterations = 50
    
    # 原始方法测试
    print("测试原始方法...")
    original_times = []
    for i in range(iterations):
        _, calc_time = apply_diffusion_original(c_flat, ncomps, nx, ny, nz)
        original_times.append(calc_time)
    
    original_avg = np.mean(original_times)
    print(f"原始方法平均耗时: {original_avg*1000:.2f} ms")
    
    # 优化方法测试
    print("测试优化方法...")
    optimized_times = []
    for i in range(iterations):
        _, calc_time = optimized_diffusion_vectorized(c_flat, ncomps, nx, ny, nz)
        optimized_times.append(calc_time)
    
    optimized_avg = np.mean(optimized_times)
    print(f"优化方法平均耗时: {optimized_avg*1000:.2f} ms")
    
    # 性能提升计算
    speedup = original_avg / optimized_avg
    improvement = (original_avg - optimized_avg) / original_avg * 100
    
    print(f"\n性能对比结果:")
    print(f"加速比: {speedup:.2f}x")
    print(f"性能提升: {improvement:.1f}%")
    print(f"单次计算时间节省: {(original_avg - optimized_avg)*1000:.2f} ms")

def apply_diffusion_original(c_flat, ncomps, nx, ny, nz, diffusion_coeff=0.05):
    """原始的嵌套循环扩散计算（用于对比）"""
    nxyz = nx * ny * nz
    c_data = np.array(c_flat, dtype=float).reshape((ncomps, nxyz))
    c_new = c_data.copy()
    
    dx = dy = dz = 1.0
    D_x = D_y = D_z = diffusion_coeff
    
    computation_start = time.time()
    
    for comp_idx in range(ncomps):
        grid_3d = c_data[comp_idx, :].reshape((nz, ny, nx))
        grid_new_3d = np.zeros_like(grid_3d)
        
        # 原始的三重嵌套循环
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    val = grid_3d[z, y, x]
                    neighbors_sum = 0.0
                    weight_sum = 0.0
                    
                    # X方向邻居
                    if x > 0:
                        neighbors_sum += D_x * (grid_3d[z, y, x-1] - val) / (dx * dx)
                        weight_sum += D_x / (dx * dx)
                    if x < nx-1:
                        neighbors_sum += D_x * (grid_3d[z, y, x+1] - val) / (dx * dx)
                        weight_sum += D_x / (dx * dx)
                    
                    # Y方向邻居
                    if y > 0:
                        neighbors_sum += D_y * (grid_3d[z, y-1, x] - val) / (dy * dy)
                        weight_sum += D_y / (dy * dy)
                    if y < ny-1:
                        neighbors_sum += D_y * (grid_3d[z, y+1, x] - val) / (dy * dy)
                        weight_sum += D_y / (dy * dy)
                    
                    # Z方向邻居
                    if z > 0:
                        neighbors_sum += D_z * (grid_3d[z-1, y, x] - val) / (dz * dz)
                        weight_sum += D_z / (dz * dz)
                    if z < nz-1:
                        neighbors_sum += D_z * (grid_3d[z+1, y, x] - val) / (dz * dz)
                        weight_sum += D_z / (dz * dz)
                    
                    if weight_sum > 0:
                        laplacian = neighbors_sum / weight_sum
                        grid_new_3d[z, y, x] = val + diffusion_coeff * laplacian
                    else:
                        grid_new_3d[z, y, x] = val
        
        grid_new_3d = np.maximum(grid_new_3d, 0)
        c_new[comp_idx] = grid_new_3d.flatten()
    
    computation_time = time.time() - computation_start
    return c_new.flatten().tolist(), computation_time

if __name__ == "__main__":
    benchmark_comparison()