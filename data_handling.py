"""
PhreeqcRM 数据处理模块
包含数据获取、处理和状态监控功能
"""

import numpy as np
from utils import NX, NY, NZ, NXYZ

def get_ph_data(rm, nx=NX, ny=NY, nz=NZ):
    """
    鲁棒的 pH 获取函数 - 🔥 增强版数据验证
    专门处理只有pH的Selected Output (3D版本)
    
    Args:
        rm: PhreeqcRM实例
        nx (int): X方向网格数
        ny (int): Y方向网格数
        nz (int): Z方向网格数
        
    Returns:
        np.ndarray: 3D pH矩阵 (nz, ny, nx)
    """
    try:
        rm.SetCurrentSelectedOutputUserNumber(1)
        
        # 获取表头，了解返回了哪些变量
        headings = rm.GetSelectedOutputHeadings()
        n_vars = len(headings)
        
        # 🔥 关键验证：检查表头完整性
        if headings is None or len(headings) == 0:
            print("   [严重错误] Selected Output 表头为空")
            return np.full((nz, ny, nx), 7.0, dtype=np.float64)
        
        # 获取所有数据 (一维长数组)
        data = rm.GetSelectedOutput()
        
        # 🔥 关键验证：数据完整性检查
        if data is None:
            print("   [严重错误] Selected Output 数据为 None")
            return np.full((nz, ny, nx), 7.0, dtype=np.float64)
        
        if len(data) == 0:
            print("   [严重错误] Selected Output 数据为空数组")
            return np.full((nz, ny, nx), 7.0, dtype=np.float64)
        
        # 验证数据长度
        expected_length = NXYZ * n_vars
        if len(data) != expected_length:
            print(f"   [严重错误] 数据长度不匹配 (期望: {expected_length}, 实际: {len(data)})")
            return np.full((nz, ny, nx), 7.0, dtype=np.float64)
        
        # 找到 'pH' 所在的列索引
        try:
            ph_index = list(headings).index('pH')
        except ValueError:
            print(f"   [严重错误] 在输出变量中找不到 'pH'。可用变量: {headings}")
            return np.full((nz, ny, nx), 7.0, dtype=np.float64)
        
        # 🔥 性能优化：预分配数组
        ph_values = np.empty(NXYZ, dtype=np.float64)
        
        # 从长数组中提取所有单元格的 pH 值
        # 数据排列: [var1_c0, var2_c0, ..., pH_c0, ..., var1_c1, var2_c1, ...]
        for i in range(NXYZ):
            idx = i * n_vars + ph_index
            ph_val = data[idx]
            
            # 🔥 增强的数据验证
            if ph_val is None:
                ph_val = 7.0
            elif isinstance(ph_val, (int, float)):
                if np.isnan(ph_val) or np.isinf(ph_val):
                    ph_val = 7.0
                elif ph_val < 0 or ph_val > 14:
                    ph_val = 7.0  # 限制在合理pH范围内
            else:
                ph_val = 7.0  # 类型不匹配时使用默认值
                
            ph_values[i] = ph_val
        
        # 重塑为 3D矩阵 (Z, Y, X顺序)
        ph_matrix_3d = ph_values.reshape((nz, ny, nx))
        
        # 🔥 最终验证：确保返回数据的有效性
        if not np.isfinite(ph_matrix_3d).all():
            print("   [警告] pH数据包含非有限值，已替换为默认值")
            ph_matrix_3d = np.nan_to_num(ph_matrix_3d, nan=7.0, posinf=14.0, neginf=0.0)
        
        # 确保数据类型和范围正确
        ph_matrix_3d = ph_matrix_3d.astype(np.float64)
        ph_matrix_3d = np.clip(ph_matrix_3d, 0, 14)
        
        return ph_matrix_3d
        
    except Exception as e:
        print(f"   [致命错误] get_ph_data执行异常: {e}")
        # 发生异常时返回安全的默认数据
        return np.full((nz, ny, nx), 7.0, dtype=np.float64)

def print_grid_state(rm, nx=NX, ny=NY, nz=NZ, label="Grid State"):
    """
    打印 pH 值的3D网格分布（显示中间层切片）
    
    Args:
        rm: PhreeqcRM实例
        nx (int): X方向网格数
        ny (int): Y方向网格数
        nz (int): Z方向网格数
        label (str): 状态标签
    """
    print(f"\n--- {label} ---")

    # 获取3D pH数据
    ph_matrix_3d = get_ph_data(rm, nx, ny, nz)
    
    # 显示中间层切片 (z = nz//2)
    middle_z = nz // 2
    ph_slice = ph_matrix_3d[middle_z, :, :]
    
    print(f"   pH 值分布 (Z层 {middle_z}):")
    for y in range(ny):
        row_str = "   "
        for x in range(nx):
            val = ph_slice[y, x]
            # 根据pH值进行颜色编码
            if val < 4:  # 强酸性
                row_str += f"\033[91m{val:.2f}\033[0m  "  # 红色
            elif val < 6:  # 弱酸性
                row_str += f"\033[93m{val:.2f}\033[0m  "  # 黄色
            elif val > 10:  # 强碱性
                row_str += f"\033[94m{val:.2f}\033[0m  "  # 蓝色
            elif val > 8:  # 弱碱性
                row_str += f"\033[96m{val:.2f}\033[0m  "  # 青色
            else:  # 中性
                row_str += f"{val:.2f}  "  # 白色
        print(row_str)
    
    # 打印统计信息
    print(f"   Z层范围: 0-{nz-1}, 当前显示: {middle_z}")
    print(f"   pH范围: [{np.min(ph_matrix_3d):.2f}, {np.max(ph_matrix_3d):.2f}]")

def get_component_data(rm, component_name):
    """
    获取指定化学组件的浓度数据
    
    Args:
        rm: PhreeqcRM实例
        component_name (str): 组件名称
        
    Returns:
        np.ndarray: 组件浓度数据
    """
    # 这里可以根据需要扩展来获取其他组件数据
    pass

def calculate_statistics(data_3d):
    """
    计算3D数据的统计信息
    
    Args:
        data_3d (np.ndarray): 3D数据矩阵
        
    Returns:
        dict: 统计信息字典
    """
    return {
        'mean': np.mean(data_3d),
        'std': np.std(data_3d),
        'min': np.min(data_3d),
        'max': np.max(data_3d),
        'median': np.median(data_3d),
        'percentile_25': np.percentile(data_3d, 25),
        'percentile_75': np.percentile(data_3d, 75)
    }

def export_data_to_file(data_history, filename, format='csv'):
    """
    导出数据历史到文件
    
    Args:
        data_history (list): 数据历史列表
        filename (str): 输出文件名
        format (str): 文件格式 ('csv', 'npy', 'txt')
    """
    if format == 'npy':
        np.save(filename, np.array(data_history))
    elif format == 'csv':
        # 展平数据并保存为CSV
        flattened_data = [data.flatten() for data in data_history]
        np.savetxt(filename, np.array(flattened_data), delimiter=',')
    elif format == 'txt':
        with open(filename, 'w') as f:
            for i, data in enumerate(data_history):
                f.write(f"Frame {i}:\n")
                f.write(str(data) + "\n\n")

def validate_data_consistency(data_list):
    """
    验证数据一致性
    
    Args:
        data_list (list): 数据列表
        
    Returns:
        bool: 数据是否一致
    """
    if len(data_list) < 2:
        return True
    
    # 检查所有数据的形状是否一致
    first_shape = data_list[0].shape
    for data in data_list[1:]:
        if data.shape != first_shape:
            return False
    
    return True