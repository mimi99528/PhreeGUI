"""
PhreeqcRM 性能分析工具
使用 cProfile 对物理计算和化学计算进行详细的性能分析
"""

import cProfile
import pstats
import io
import time
import numpy as np
from phreeqcrm import PhreeqcRM

# 导入原始模拟函数的相关部分
nx, ny, nz = 15, 15, 15
nxyz = nx * ny * nz
nthreads = 8

def profile_physical_computation():
    """专门分析物理扩散计算的性能"""
    print("=== 物理计算性能分析 ===")
    
    # 创建测试数据
    ncomps = 3  # 假设有3个化学组分
    c_flat = [0.1] * (ncomps * nxyz)  # 模拟浓度数据
    
    # 运行多次以获得稳定的结果
    iterations = 100
    
    start_time = time.time()
    for i in range(iterations):
        result = apply_diffusion_numpy_profiled(c_flat, ncomps, nx, ny, nz)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    print(f"物理扩散计算平均耗时: {avg_time*1000:.2f} ms/次")
    print(f"总计算次数: {iterations}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")

def apply_diffusion_numpy_profiled(c_flat, ncomps, nx, ny, nz, diffusion_coeff=0.05):
    """带性能监控的物理扩散计算"""
    nxyz = nx * ny * nz
    c_data = np.array(c_flat, dtype=float).reshape((ncomps, nxyz))
    c_new = c_data.copy()
    
    # 计算网格间距
    dx = dy = dz = 1.0
    D_x = D_y = D_z = diffusion_coeff
    
    computation_start = time.time()
    
    for comp_idx in range(ncomps):
        grid_3d = c_data[comp_idx, :].reshape((nz, ny, nx))
        grid_new_3d = np.zeros_like(grid_3d)
        
        # 3D扩散计算
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
        
        # 非负约束
        grid_new_3d = np.maximum(grid_new_3d, 0)
        c_new[comp_idx] = grid_new_3d.flatten()
    
    computation_time = time.time() - computation_start
    return c_new.flatten().tolist(), computation_time

def profile_chemical_computation():
    """专门分析化学反应计算的性能"""
    print("\n=== 化学计算性能分析 ===")
    
    # 初始化PhreeqcRM
    rm = PhreeqcRM(nxyz, nthreads)
    
    try:
        # 基本设置
        rm.SetErrorOn(True)
        rm.SetErrorHandlerMode(1)
        rm.SetPrintChemistryOn(False, False, False)
        rm.OpenFiles()
        
        # 加载数据库
        database_paths = [
            r"C:\phreeqc\database\phreeqc.dat",
            r"C:\Program Files\USGS\phreeqc\database\phreeqc.dat",
            r"phreeqc.dat"
        ]
        
        database_loaded = False
        for db_path in database_paths:
            try:
                rm.LoadDatabase(db_path)
                database_loaded = True
                break
            except:
                continue
        
        if not database_loaded:
            print("警告: 无法加载数据库，跳过化学计算分析")
            return
        
        # 定义初始溶液
        initial_input = """
        SOLUTION 1 Background Water
            units mg/L; temp 25.0; pH 7.0; water 1.0
        END
        SOLUTION 2 Acidic Source
            units mg/L; temp 25.0; pH 3.0; Cl 1000.0 as HCl; water 1.0
        END
        SELECTED_OUTPUT 1
            -reset false
            -pH
        END
        """
        
        rm.RunString(True, True, True, initial_input)
        rm.FindComponents()
        
        # 分配初始溶液
        for i in range(nxyz):
            z_idx = i // (nx * ny)
            remaining = i % (nx * ny)
            y_idx = remaining // nx
            x_idx = remaining % nx
            
            if x_idx < nx//3 and y_idx < ny//3 and z_idx < nz//3:
                rm.InitialPhreeqcCell2Module(2, [i])
            else:
                rm.InitialPhreeqcCell2Module(1, [i])
        
        # 设置物理参数
        rm.SetTimeStep(86400.0)
        rm.SetTemperature([25.0] * nxyz)
        rm.SetPressure([1.0] * nxyz)
        rm.SetPorosity([0.2] * nxyz)
        rm.SetSaturationUser([1.0] * nxyz)
        rm.SetUnitsSolution(2)
        rm.SetRepresentativeVolume([1.0] * nxyz)
        
        grid2chem = list(range(nxyz))
        rm.CreateMapping(grid2chem)
        
        # 运行初始平衡
        print("运行初始化学平衡...")
        start_time = time.time()
        rm.RunCells()
        initial_balance_time = time.time() - start_time
        print(f"初始平衡耗时: {initial_balance_time:.2f} 秒")
        
        # 测试化学同步性能
        iterations = 10
        total_chemical_time = 0
        
        print(f"测试 {iterations} 次化学同步...")
        for i in range(iterations):
            start_time = time.time()
            rm.SetTime(i * 86400.0)
            rm.RunCells()
            chem_time = time.time() - start_time
            total_chemical_time += chem_time
            if i % 2 == 0:
                print(f"  第 {i+1} 次化学计算: {chem_time:.3f} 秒")
        
        avg_chemical_time = total_chemical_time / iterations
        print(f"平均化学计算耗时: {avg_chemical_time:.3f} 秒/次")
        print(f"总化学计算耗时: {total_chemical_time:.2f} 秒")
        
        # 清理
        rm.CloseFiles()
        
    except Exception as e:
        print(f"化学计算分析出错: {e}")

def comprehensive_performance_analysis():
    """综合性能分析 - 同时分析物理和化学计算"""
    print("=" * 60)
    print("PhreeqcRM 综合性能分析")
    print("=" * 60)
    
    # 创建性能分析器
    profiler = cProfile.Profile()
    
    def test_combined_simulation():
        """测试组合的物理-化学模拟"""
        rm = PhreeqcRM(nxyz, nthreads)
        
        try:
            # 基本初始化
            rm.SetErrorOn(True)
            rm.SetErrorHandlerMode(1)
            rm.SetPrintChemistryOn(False, False, False)
            rm.OpenFiles()
            
            # 加载数据库
            database_paths = [
                r"C:\phreeqc\database\phreeqc.dat",
                r"C:\Program Files\USGS\phreeqc\database\phreeqc.dat"
            ]
            
            for db_path in database_paths:
                try:
                    rm.LoadDatabase(db_path)
                    break
                except:
                    continue
            
            # 定义溶液
            initial_input = """
            SOLUTION 1 Background Water
                units mg/L; temp 25.0; pH 7.0; water 1.0
            END
            SOLUTION 2 Acidic Source
                units mg/L; temp 25.0; pH 3.0; Cl 1000.0 as HCl; water 1.0
            END
            SELECTED_OUTPUT 1
                -reset false
                -pH
            END
            """
            
            rm.RunString(True, True, True, initial_input)
            rm.FindComponents()
            ncomps = rm.GetComponentCount()
            
            # 分配溶液
            for i in range(nxyz):
                z_idx = i // (nx * ny)
                remaining = i % (nx * ny)
                y_idx = remaining // nx
                x_idx = remaining % nx
                
                if x_idx < nx//3 and y_idx < ny//3 and z_idx < nz//3:
                    rm.InitialPhreeqcCell2Module(2, [i])
                else:
                    rm.InitialPhreeqcCell2Module(1, [i])
            
            # 设置参数
            rm.SetTimeStep(86400.0)
            rm.SetTemperature([25.0] * nxyz)
            rm.SetPressure([1.0] * nxyz)
            rm.SetPorosity([0.2] * nxyz)
            rm.SetSaturationUser([1.0] * nxyz)
            rm.SetUnitsSolution(2)
            rm.SetRepresentativeVolume([1.0] * nxyz)
            
            grid2chem = list(range(nxyz))
            rm.CreateMapping(grid2chem)
            
            # 运行初始平衡
            rm.RunCells()
            
            # 模拟几个时间步
            steps = 20  # 减少步数以便分析
            chemical_sync_interval = 2
            
            physical_times = []
            chemical_times = []
            
            for step in range(1, steps + 1):
                # 物理计算
                phys_start = time.time()
                c_flat = list(rm.GetConcentrations())
                c_new = apply_diffusion_numpy_simple(c_flat, ncomps, nx, ny, nz)
                rm.SetConcentrations(c_new)
                physical_times.append(time.time() - phys_start)
                
                # 化学计算（按间隔执行）
                if step % chemical_sync_interval == 0:
                    chem_start = time.time()
                    rm.SetTime(step * 86400.0)
                    rm.RunCells()
                    chemical_times.append(time.time() - chem_start)
            
            # 输出统计
            print(f"\n性能统计:")
            print(f"物理计算次数: {len(physical_times)}")
            print(f"化学计算次数: {len(chemical_times)}")
            print(f"平均物理计算时间: {np.mean(physical_times)*1000:.2f} ms")
            print(f"平均化学计算时间: {np.mean(chemical_times):.3f} s")
            print(f"物理计算总时间: {sum(physical_times):.3f} s")
            print(f"化学计算总时间: {sum(chemical_times):.3f} s")
            
            rm.CloseFiles()
            
        except Exception as e:
            print(f"模拟过程中出错: {e}")
    
    # 运行性能分析
    print("开始全面性能分析...")
    profiler.enable()
    test_combined_simulation()
    profiler.disable()
    
    # 输出分析结果
    print("\n=== 详细性能分析报告 ===")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    
    # 输出前20个最耗时的函数
    print("Top 20 最耗时的函数:")
    ps.print_stats(20)
    
    # 保存详细报告
    with open('performance_report.txt', 'w', encoding='utf-8') as f:
        ps.print_stats()
        f.write(s.getvalue())
    
    print("\n详细性能报告已保存到 'performance_report.txt'")

def apply_diffusion_numpy_simple(c_flat, ncomps, nx, ny, nz, diffusion_coeff=0.05):
    """简化的物理扩散计算（用于性能测试）"""
    nxyz = nx * ny * nz
    c_data = np.array(c_flat, dtype=float).reshape((ncomps, nxyz))
    c_new = c_data.copy()
    
    dx = dy = dz = 1.0
    D_x = D_y = D_z = diffusion_coeff
    
    for comp_idx in range(ncomps):
        grid_3d = c_data[comp_idx, :].reshape((nz, ny, nx))
        grid_new_3d = np.zeros_like(grid_3d)
        
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    val = grid_3d[z, y, x]
                    # 简化计算，只考虑直接邻居
                    neighbors = []
                    if x > 0: neighbors.append(grid_3d[z, y, x-1])
                    if x < nx-1: neighbors.append(grid_3d[z, y, x+1])
                    if y > 0: neighbors.append(grid_3d[z, y-1, x])
                    if y < ny-1: neighbors.append(grid_3d[z, y+1, x])
                    if z > 0: neighbors.append(grid_3d[z-1, y, x])
                    if z < nz-1: neighbors.append(grid_3d[z+1, y, x])
                    
                    if neighbors:
                        avg_neighbor = sum(neighbors) / len(neighbors)
                        grid_new_3d[z, y, x] = val + diffusion_coeff * (avg_neighbor - val)
                    else:
                        grid_new_3d[z, y, x] = val
        
        grid_new_3d = np.maximum(grid_new_3d, 0)
        c_new[comp_idx] = grid_new_3d.flatten()
    
    return c_new.flatten().tolist()

def main():
    """主函数 - 运行所有性能分析"""
    print("开始 PhreeqcRM 性能分析...")
    
    # 1. 物理计算性能分析
    profile_physical_computation()
    
    # 2. 化学计算性能分析
    profile_chemical_computation()
    
    # 3. 综合性能分析
    comprehensive_performance_analysis()
    
    print("\n性能分析完成!")

if __name__ == "__main__":
    main()