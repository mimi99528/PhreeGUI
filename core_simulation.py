"""
PhreeqcRM 核心模拟模块
包含主要的模拟逻辑和流程控制
"""

import time
import phreeqcrm
from phreeqcrm import PhreeqcRM
from utils import (
    NX, NY, NZ, NXYZ, NTHREADS, TIME_STEP,
    initialize_phreeqcrm, load_database, configure_physical_parameters,
    assign_initial_solutions, get_solution_definition, create_mapping,
    get_elapsed_time, print_simulation_header, CHEMICAL_SYNC_INTERVAL, RECORD_INTERVAL
)
from physics import apply_diffusion_vectorized
from data_handling import get_ph_data, print_grid_state
from visualization import create_interactive_3d_visualization

def simulate_spatial_diffusion(steps=1000):
    """
    模拟15x15x15网格中 pH 的空间扩散过程，并收集数据用于可视化
    
    Args:
        steps (int): 模拟步数
        
    Returns:
        list: pH历史数据列表
    """
    print_simulation_header("PhreeqcRM 空间扩散模拟 (pH 示踪)")
    
    # 收集每步的pH数据用于可视化
    ph_history = []
    
    # 1. 初始化PhreeqcRM
    rm = initialize_phreeqcrm()
    
    # 2. 加载数据库
    database_loaded = load_database(rm)
    
    # 3. 定义初始溶液 + SELECTED_OUTPUT
    initial_input = get_solution_definition()
    rm.RunString(True, True, True, initial_input)
    
    # 4. 初始化组件
    rm.FindComponents()
    comps = rm.GetComponents()
    ncomps = rm.GetComponentCount()
    print(f"   检测到化学组件 ({ncomps}个): {list(comps)}")
    
    # 5. 分配初始溶液到网格
    assign_initial_solutions(rm)
    
    # 6. 设置物理参数和单位
    configure_physical_parameters(rm)
    
    # 固定负载分配以提高稳定性
    create_mapping(rm)
    
    # 7. 运行初始平衡
    print("   运行初始平衡...")
    rm.RunCells()
    
    # 记录初始状态
    ph_history.append(get_ph_data(rm, NX, NY, NZ))
    print_grid_state(rm, NX, NY, NZ, "Initial State")
    
    # 8. 循环模拟 - 物理-化学分离策略
    startTime = time.time()
    
    for step in range(1, steps + 1):
        # 物理传输 (每次都要执行)
        c_flat = list(rm.GetConcentrations())
        c_new, _ = apply_diffusion_vectorized(c_flat, ncomps, NX, NY, NZ)
        rm.SetConcentrations(c_new)
        
        # 化学反应同步
        if step % CHEMICAL_SYNC_INTERVAL == 0:
            rm.SetTime(step * TIME_STEP)
            rm.RunCells()
            sync_type = "[化学+物理]"
        else:
            sync_type = "[仅物理]"
        
        # 记录数据
        if step % RECORD_INTERVAL == 0:
            ph_history.append(get_ph_data(rm, NX, NY, NZ))
            print_grid_state(rm, NX, NY, NZ, f"Step {step} {sync_type}")
    
    print(f"模拟计算耗时: {get_elapsed_time(startTime):.2f} 秒")
    print(f"共记录 {len(ph_history)} 帧数据")
    print(f"化学同步频率: 每 {CHEMICAL_SYNC_INTERVAL} 帧一次")
    
    # 清理
    rm.CloseFiles()
    print("\n模拟完成。")
    
    return ph_history

def run_batch_simulation(parameter_sets):
    """
    运行批量模拟（不同参数组合）
    
    Args:
        parameter_sets (list): 参数设置列表
        
    Returns:
        list: 所有模拟结果
    """
    results = []
    
    for i, params in enumerate(parameter_sets):
        print(f"\n运行第 {i+1}/{len(parameter_sets)} 个模拟...")
        print(f"参数设置: {params}")
        
        # 可以在这里根据参数调整模拟设置
        steps = params.get('steps', 1000)
        ph_history = simulate_spatial_diffusion(steps=steps)
        results.append({
            'parameters': params,
            'results': ph_history
        })
    
    return results

def sensitivity_analysis(base_params, param_variations):
    """
    参数敏感性分析
    
    Args:
        base_params (dict): 基准参数
        param_variations (dict): 参数变化范围
        
    Returns:
        dict: 敏感性分析结果
    """
    results = {}
    
    # 对每个参数进行变化分析
    for param_name, variations in param_variations.items():
        results[param_name] = []
        for value in variations:
            # 修改基准参数
            test_params = base_params.copy()
            test_params[param_name] = value
            
            print(f"\n测试 {param_name} = {value}")
            simulation_result = run_batch_simulation([test_params])
            results[param_name].append({
                'value': value,
                'result': simulation_result[0]['results']
            })
    
    return results

def export_simulation_results(ph_history, filename_prefix="simulation"):
    """
    导出模拟结果
    
    Args:
        ph_history (list): pH历史数据
        filename_prefix (str): 文件名前缀
    """
    import numpy as np
    
    # 保存为numpy数组
    np.save(f"{filename_prefix}_data.npy", np.array(ph_history))
    
    # 保存元数据
    metadata = {
        'grid_size': [NX, NY, NZ],
        'total_steps': len(ph_history),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    import json
    with open(f"{filename_prefix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"结果已保存: {filename_prefix}_data.npy 和 {filename_prefix}_metadata.json")