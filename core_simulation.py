"""
PhreeqcRM 核心模拟模块
包含主要的模拟逻辑和流程控制
"""

import time

import numpy as np
import phreeqcrm
import gc
from phreeqcrm import PhreeqcRM
from utils import (
    NX, NY, NZ, NXYZ, NTHREADS, TIME_STEP,
    initialize_phreeqcrm, load_database, configure_physical_parameters,
    assign_initial_solutions, get_solution_definition, create_mapping,
    get_elapsed_time, print_simulation_header, CHEMICAL_SYNC_INTERVAL, RECORD_INTERVAL
)
from physics import apply_diffusion_vectorized, apply_diffusion_vectorized_inplace
from data_handling import get_ph_data, print_grid_state
# 模拟模块严格不导入任何可视化相关库
# 所有UI相关操作都在线程外处理
# 模拟只负责生成纯numpy数据
max_retries = 5

def simulate_spatial_diffusion_generator(steps=1000, data_buffer=None):
    """
    基于Generator的模拟15x15x15网格中 pH 的空间扩散过程
    
    模拟模块职责：
    - 仅负责科学计算
    - 通过yield输出纯numpy 3D数组数据
    - 可选地通过ThreadSafeDataBuffer与外部通信
    - 不涉及任何UI/可视化操作

    Args:
        steps (int): 模拟步数
        data_buffer (ThreadSafeDataBuffer): 可选的线程安全数据缓冲区

    Yields:
        np.ndarray: 每一步的pH数据（3D numpy数组）
    """
    print_simulation_header("PhreeqcRM 空间扩散模拟 (pH 示踪) - Generator版本")

    # 1. 初始化PhreeqcRM
    rm = initialize_phreeqcrm()

    # 关键修复设置
    print(rm.SetComponentH2O(False))
    rm.SetUnitsSolution(2)
    rm.SetUnitsPPassemblage(1)
    rm.SetUnitsExchange(1)
    rm.SetPartitionUZSolids(False)
    rm.SetSpeciesSaveOn(False)
    rm.SetRebalanceFraction(0.1)

    rm.SetErrorOn(True)
    rm.SetErrorHandlerMode(1)
    rm.SetPrintChemistryOn(False, False, False)
    rm.SetRebalanceByCell(True)

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
    
    # 获取并yield初始数据
    initial_data = get_ph_data(rm, NX, NY, NZ)
    print_grid_state(rm, NX, NY, NZ, "Initial State")
    
    # 通过数据缓冲区发送初始数据给主进程
    if data_buffer:
        try:
            success = data_buffer.put_data(initial_data)
            if success:
                print("[模拟进程] 初始pH数据已发送至主进程")
            else:
                print("[模拟进程] 初始数据发送失败")
        except Exception as e:
            print(f"[模拟进程] 数据发送失败: {e}")
            import traceback
            traceback.print_exc()
            # 继续运行，不影响计算
    
    yield initial_data  # yield初始帧
    
    # 8. 初始化双缓冲区 - 内存安全管理
    startTime = time.time()
    last_record_time = startTime
    frame_count = 1  # 从1开始，因为已经yield了初始帧
    
    # 🔥 关键改进：创建持久的双缓冲区
    ncells = NX * NY * NZ
    n_total = ncells * ncomps
    
    # 从RM获取初始浓度并创建第一个缓冲区
    c_buf = np.ascontiguousarray(np.array(rm.GetConcentrations(), dtype=np.float64))
    assert c_buf.size == n_total, f"初始浓度数组大小不匹配: 期望 {n_total}, 实际 {c_buf.size}"
    
    # 创建第二个持久缓冲区
    c_new = np.empty_like(c_buf)
    
    print(f"[DEBUG] 双缓冲区初始化完成")
    print(f"[DEBUG] 缓冲区大小: {n_total} (ncells={ncells}, ncomps={ncomps})")
    print(f"[DEBUG] c_buf 范围: [{c_buf.min():.6f}, {c_buf.max():.6f}]")
    
    # 9. 循环模拟 - 物理-化学分离策略
    print(f"[DEBUG] 开始模拟循环，总步数: {steps}")
    print(f"[DEBUG] 化学同步间隔: {CHEMICAL_SYNC_INTERVAL}, 记录间隔: {RECORD_INTERVAL}")

    for step in range(1, steps + 1):
        # 调试信息：显示当前步数
        if step % 100 == 0:  # 每100步打印一次进度
            print(f"[DEBUG] 当前步数: {step}/{steps}")

        # 检查主线程控制信号
        if data_buffer and data_buffer.is_simulation_paused():
            print(f"[DEBUG] 模拟暂停，等待恢复... (步数: {step})")
            time.sleep(0.1)  # 响应暂停信号
            continue

        # 物理传输 (每次都要执行) - 使用原地扩散
        try:
            # 🔥 关键改进：使用原地扩散函数，避免临时对象创建
            diffusion_time = apply_diffusion_vectorized_inplace(
                c_in=c_buf,
                c_out=c_new,
                ncomps=ncomps,
                nx=NX, ny=NY, nz=NZ
            )
            
            # 调试扩散过程
            if step == 1 or step % 100 == 0:
                print(f"[DEBUG] 扩散完成，耗时: {diffusion_time:.6f}s")
                print(f"[DEBUG] c_new 范围: [{c_new.min():.6f}, {c_new.max():.6f}]")
                
            # 🔥 关键验证：确保输出缓冲区符合RM要求
            if c_new.size != n_total or c_new.dtype != np.float64 or not c_new.flags['C_CONTIGUOUS']:
                raise RuntimeError(f"输出缓冲区不符合要求: size={c_new.size}, dtype={c_new.dtype}, contiguous={c_new.flags['C_CONTIGUOUS']}")
            
            # 将计算结果喂回RM
            rm.SetConcentrations(c_new)
            
            # 🔥 关键优化：交换缓冲区指针，下次迭代使用c_new作为输入
            c_buf, c_new = c_new, c_buf
            
        except Exception as e:
            print(f"[ERROR] 物理传输阶段出错 (步数: {step}): {e}")
            raise

        # 化学反应同步
        try:
            if step % CHEMICAL_SYNC_INTERVAL == 0:
                # print(f"[DEBUG] 执行化学反应同步 (步数: {step})")
                rm.SetTime(step * TIME_STEP)

                # ── 检测并排除问题单元 ──
                # 扩散后某些单元可能出现极端浓度，导致 RunCells 中水活度
                # 无法收敛（"A(H2O) Activity of water has not converged"）。
                # 通过 CreateMapping 将这些单元映射为 -1，RunCells 将跳过它们。
                current_conc = np.array(rm.GetConcentrations(), dtype=np.float64)
                conc_2d = current_conc.reshape((ncells, ncomps))
                cell_max = np.max(np.abs(conc_2d), axis=1)
                problematic = cell_max > 1e10
                if np.any(problematic):
                    n_prob = int(np.sum(problematic))
                    print(f"[SimThread] 排除 {n_prob}/{ncells} 个极端浓度单元")
                    mapping = [i if not problematic[i] else -1 for i in range(ncells)]
                    rm.CreateMapping(mapping)
                else:
                    rm.CreateMapping(list(range(ncells)))

                retry_count = 0
                while retry_count < max_retries:
                    try:
                        status = rm.RunCells()

                        if status == 0:
                            break
                        else:
                            print(f"[SimThread] Warning: RunCells returned status {status} at step {step + 1}")
                            retry_count += 1
                            if retry_count < max_retries:
                                rm.SetTimeStep(TIME_STEP * (0.5 ** retry_count))
                                print(f"[SimThread] Retry {retry_count} with smaller timestep")
                    except Exception as e:
                        print(f"[SimThread] RunCells error at step {step + 1}: {e}")
                        retry_count += 1
                        if retry_count < max_retries:
                            rm.SetTimeStep(TIME_STEP * (0.5 ** retry_count))
                            print(f"[SimThread] Exception retry {retry_count} with smaller timestep")

                sync_type = "[化学+物理]"
                print(f"[DEBUG] 化学反应完成")
            else:
                sync_type = "[仅物理]"
        except Exception as e:
            print(f"[ERROR] 化学反应同步出错 (步数: {step}): {e}")
            raise

        # 记录数据 - 🔥 优化版本：减少RM调用次数
        if step % RECORD_INTERVAL == 0:
            try:
                print(f"[DEBUG] 记录数据 (步数: {step})")
                current_data = get_ph_data(rm, NX, NY, NZ)
                
                # 调试pH数据
                if current_data is not None:
                    print(f"[DEBUG] pH数据形状: {current_data.shape}")
                    print(f"[DEBUG] pH范围: [{current_data.min():.3f}, {current_data.max():.3f}]"),
                    frame_count += 1
                else:
                    print(f"[WARNING] 获取pH数据失败 (步数: {step})")
                    continue  # 跳过本次记录
                                
                # 🔥 关键优化：使用已获取的current_data而不是再次调用RM
                # 只打印关键统计信息，避免大量输出
                if step % (RECORD_INTERVAL * 10) == 0:  # 每10次记录打印详细状态
                    print_grid_state(rm, NX, NY, NZ, f"Step {step} {sync_type}")
                else:
                    # 简化输出：只显示关键统计
                    z_middle = NZ // 2
                    middle_slice = current_data[z_middle, :, :]
                    valid_values = middle_slice[~np.isnan(middle_slice)]
                    if len(valid_values) > 0:
                        print(f"[INFO] Step {step} {sync_type} | Z={z_middle} 切片 | "
                              f"pH范围: [{valid_values.min():.2f}, {valid_values.max():.2f}] | "
                              f"平均: {valid_values.mean():.2f}")
                                
                # 将计算结果发送给主线程
                if data_buffer:
                    try:
                        success = data_buffer.put_data(current_data)
                        if success:
                            print(f"[模拟进程] 第{step}步数据已发送至主进程")
                            # 监控数据传输性能
                            current_time = time.time()
                            if current_time - last_record_time >= 1.0:
                                rate = frame_count / (current_time - startTime)
                                print(f"[模拟进程] 数据传输速率: {rate:.1f} 帧/秒")
                                last_record_time = current_time
                                frame_count = 0
                        else:
                            print(f"[WARNING] 数据缓冲区满，跳过本次传输 (步数: {step})")
                    except Exception as e:
                        print(f"[ERROR] 数据传输异常 (步数: {step}): {e}")
                        import traceback
                        traceback.print_exc()
                        # 计算任务优先，传输失败不中断模拟
                                
                # yield当前帧数据
                yield current_data
                
            except Exception as e:
                print(f"[ERROR] 数据记录阶段出错 (步数: {step}): {e}")
                raise

    print(f"模拟计算耗时: {get_elapsed_time(startTime):.2f} 秒")
    print(f"共生成 {frame_count} 帧数据")
    print(f"化学同步频率: 每 {CHEMICAL_SYNC_INTERVAL} 帧一次")
    print(f"[DEBUG] 模拟循环结束")

    # 清理
    rm.CloseFiles()
    print("\n模拟完成。")

    # 关键修复：强制垃圾回收确保PhreeqcRM对象完全析构
    # 避免与Dear PyGui渲染线程的内存冲突
    del rm
    gc.collect()
    print("内存清理完成")
    
    # 通知主进程模拟完成
    if data_buffer:
        data_buffer.mark_finished()
        print("[模拟进程] 模拟完成，已通知主进程")


def simulate_spatial_diffusion(steps=1000, data_buffer=None):
    """
    模拟15x15x15网格中 pH 的空间扩散过程
    
    模拟模块职责：
    - 仅负责科学计算
    - 输出纯numpy 3D数组数据
    - 通过ThreadSafeDataBuffer与外部通信
    - 不涉及任何UI/可视化操作

    Args:
        steps (int): 模拟步数
        data_buffer (ThreadSafeDataBuffer): 线程安全数据缓冲区

    Returns:
        list: pH历史数据列表（numpy 3D数组）
    """
    print_simulation_header("PhreeqcRM 空间扩散模拟 (pH 示踪) - 纯计算模块")

    # 收集每步的pH数据
    ph_history = []

    # 1. 初始化PhreeqcRM
    rm = initialize_phreeqcrm()

    # 关键修复设置
    print(rm.SetComponentH2O(False))
    rm.SetUnitsSolution(2)
    rm.SetUnitsPPassemblage(1)
    rm.SetUnitsExchange(1)
    rm.SetPartitionUZSolids(False)
    rm.SetSpeciesSaveOn(False)
    rm.SetRebalanceFraction(0.1)

    rm.SetErrorOn(True)
    rm.SetErrorHandlerMode(1)
    rm.SetPrintChemistryOn(False, False, False)
    rm.SetRebalanceByCell(True)

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
    
    # 通过数据缓冲区发送初始数据给主进程
    if data_buffer:
        try:
            initial_data = get_ph_data(rm, NX, NY, NZ)  # 获取初始数据
            data_buffer.put_data(initial_data)
            print("[模拟进程] 初始pH数据已发送至主进程")
        except Exception as e:
            print(f"[模拟进程] 数据发送失败: {e}")
            # 继续运行，不影响计算
    # 8. 初始化双缓冲区 - 内存安全管理
    startTime = time.time()
    last_record_time = startTime
    frame_count = 0
    
    # 🔥 关键改进：创建持久的双缓冲区
    ncells = NX * NY * NZ
    n_total = ncells * ncomps
    
    # 从RM获取初始浓度并创建第一个缓冲区
    c_buf = np.ascontiguousarray(np.array(rm.GetConcentrations(), dtype=np.float64))
    assert c_buf.size == n_total, f"初始浓度数组大小不匹配: 期望 {n_total}, 实际 {c_buf.size}"
    
    # 创建第二个持久缓冲区
    c_new = np.empty_like(c_buf)
    
    print(f"[DEBUG] 双缓冲区初始化完成")
    print(f"[DEBUG] 缓冲区大小: {n_total} (ncells={ncells}, ncomps={ncomps})")
    print(f"[DEBUG] c_buf 范围: [{c_buf.min():.6f}, {c_buf.max():.6f}]")
    
    # 9. 循环模拟 - 物理-化学分离策略
    print(f"[DEBUG] 开始模拟循环，总步数: {steps}")
    print(f"[DEBUG] 化学同步间隔: {CHEMICAL_SYNC_INTERVAL}, 记录间隔: {RECORD_INTERVAL}")

    for step in range(1, steps + 1):
        # 调试信息：显示当前步数
        if step % 100 == 0:  # 每100步打印一次进度
            print(f"[DEBUG] 当前步数: {step}/{steps}")

        # 检查主线程控制信号
        if data_buffer and data_buffer.is_simulation_paused():
            print(f"[DEBUG] 模拟暂停，等待恢复... (步数: {step})")
            time.sleep(0.1)  # 响应暂停信号
            continue

        # 物理传输 (每次都要执行) - 使用原地扩散
        try:
            # 🔥 关键改进：使用原地扩散函数，避免临时对象创建
            diffusion_time = apply_diffusion_vectorized_inplace(
                c_in=c_buf,
                c_out=c_new,
                ncomps=ncomps,
                nx=NX, ny=NY, nz=NZ
            )
            
            # 调试扩散过程
            if step == 1 or step % 100 == 0:
                print(f"[DEBUG] 扩散完成，耗时: {diffusion_time:.6f}s")
                print(f"[DEBUG] c_new 范围: [{c_new.min():.6f}, {c_new.max():.6f}]")
                
            # 🔥 关键验证：确保输出缓冲区符合RM要求
            if c_new.size != n_total or c_new.dtype != np.float64 or not c_new.flags['C_CONTIGUOUS']:
                raise RuntimeError(f"输出缓冲区不符合要求: size={c_new.size}, dtype={c_new.dtype}, contiguous={c_new.flags['C_CONTIGUOUS']}")
            
            # 将计算结果喂回RM
            rm.SetConcentrations(c_new)
            
            # 🔥 关键优化：交换缓冲区指针，下次迭代使用c_new作为输入
            c_buf, c_new = c_new, c_buf
            
        except Exception as e:
            print(f"[ERROR] 物理传输阶段出错 (步数: {step}): {e}")
            raise

        # 化学反应同步
        try:
            if step % CHEMICAL_SYNC_INTERVAL == 0:
                # print(f"[DEBUG] 执行化学反应同步 (步数: {step})")
                rm.SetTime(step * TIME_STEP)

                # ── 检测并排除问题单元 ──
                # 扩散后某些单元可能出现极端浓度，导致 RunCells 中水活度
                # 无法收敛（"A(H2O) Activity of water has not converged"）。
                # 通过 CreateMapping 将这些单元映射为 -1，RunCells 将跳过它们。
                current_conc = np.array(rm.GetConcentrations(), dtype=np.float64)
                conc_2d = current_conc.reshape((ncells, ncomps))
                cell_max = np.max(np.abs(conc_2d), axis=1)
                problematic = cell_max > 1e10
                if np.any(problematic):
                    n_prob = int(np.sum(problematic))
                    print(f"[SimThread] 排除 {n_prob}/{ncells} 个极端浓度单元")
                    mapping = [i if not problematic[i] else -1 for i in range(ncells)]
                    rm.CreateMapping(mapping)
                else:
                    rm.CreateMapping(list(range(ncells)))

                retry_count = 0
                while retry_count < max_retries:
                    try:
                        status = rm.RunCells()

                        if status == 0:
                            break
                        else:
                            print(f"[SimThread] Warning: RunCells returned status {status} at step {step + 1}")
                            retry_count += 1
                            if retry_count < max_retries:
                                rm.SetTimeStep(TIME_STEP * (0.5 ** retry_count))
                                print(f"[SimThread] Retry {retry_count} with smaller timestep")
                    except Exception as e:
                        print(f"[SimThread] RunCells error at step {step + 1}: {e}")
                        retry_count += 1
                        if retry_count < max_retries:
                            rm.SetTimeStep(TIME_STEP * (0.5 ** retry_count))
                            print(f"[SimThread] Exception retry {retry_count} with smaller timestep")

                sync_type = "[化学+物理]"
                print(f"[DEBUG] 化学反应完成")
            else:
                sync_type = "[仅物理]"
        except Exception as e:
            print(f"[ERROR] 化学反应同步出错 (步数: {step}): {e}")
            raise

        # 记录数据 - 🔥 优化版本：减少RM调用次数
        if step % RECORD_INTERVAL == 0:
            try:
                print(f"[DEBUG] 记录数据 (步数: {step})")
                current_data = get_ph_data(rm, NX, NY, NZ)
                
                # 调试pH数据
                if current_data is not None:
                    print(f"[DEBUG] pH数据形状: {current_data.shape}")
                    print(f"[DEBUG] pH范围: [{current_data.min():.3f}, {current_data.max():.3f}]")
                    ph_history.append(current_data)
                    frame_count += 1
                else:
                    print(f"[WARNING] 获取pH数据失败 (步数: {step})")
                    continue  # 跳过本次记录
                
                # 🔥 关键优化：使用已获取的current_data而不是再次调用RM
                # 只打印关键统计信息，避免大量输出
                if step % (RECORD_INTERVAL * 10) == 0:  # 每10次记录打印详细状态
                    print_grid_state(rm, NX, NY, NZ, f"Step {step} {sync_type}")
                else:
                    # 简化输出：只显示关键统计
                    z_middle = NZ // 2
                    middle_slice = current_data[z_middle, :, :]
                    valid_values = middle_slice[~np.isnan(middle_slice)]
                    if len(valid_values) > 0:
                        print(f"[INFO] Step {step} {sync_type} | Z={z_middle} 切片 | "
                              f"pH范围: [{valid_values.min():.2f}, {valid_values.max():.2f}] | "
                              f"平均: {valid_values.mean():.2f}")
                
                # 将计算结果发送给主线程
                if data_buffer:
                    try:
                        success = data_buffer.put_data(current_data)
                        if success:
                            # 监控数据传输性能
                            current_time = time.time()
                            if current_time - last_record_time >= 1.0:
                                rate = frame_count / (current_time - startTime)
                                print(f"[模拟进程] 数据传输速率: {rate:.1f} 帧/秒")
                                last_record_time = current_time
                                frame_count = 0
                        else:
                            print(f"[WARNING] 数据缓冲区满，跳过本次传输 (步数: {step})")
                    except Exception as e:
                        print(f"[ERROR] 数据传输异常 (步数: {step}): {e}")
                        # 计算任务优先，传输失败不中断模拟
            except Exception as e:
                print(f"[ERROR] 数据记录阶段出错 (步数: {step}): {e}")
                raise

    print(f"模拟计算耗时: {get_elapsed_time(startTime):.2f} 秒")
    print(f"共记录 {len(ph_history)} 帧数据")
    print(f"化学同步频率: 每 {CHEMICAL_SYNC_INTERVAL} 帧一次")
    print(f"[DEBUG] 模拟循环结束")

    # 清理
    rm.CloseFiles()
    print("\n模拟完成。")

    # 关键修复：强制垃圾回收确保PhreeqcRM对象完全析构
    # 避免与Dear PyGui渲染线程的内存冲突
    del rm
    gc.collect()
    print("内存清理完成")
    
    # 通知主进程模拟完成
    if data_buffer:
        data_buffer.mark_finished()
        print("[模拟进程] 模拟完成，已通知主进程")

    return ph_history

def run_batch_simulation(parameter_sets, data_buffers=None):
    """
    运行批量模拟（不同参数组合），支持线程安全数据传递

    Args:
        parameter_sets (list): 参数设置列表
        data_buffers (list): 对应的数据缓冲区列表

    Returns:
        list: 所有模拟结果
    """
    results = []

    for i, params in enumerate(parameter_sets):
        print(f"\n运行第 {i+1}/{len(parameter_sets)} 个模拟...")
        print(f"参数设置: {params}")

        # 获取对应的数据缓冲区
        data_buffer = data_buffers[i] if data_buffers and i < len(data_buffers) else None
        
        # 可以在这里根据参数调整模拟设置
        steps = params.get('steps', 1000)
        ph_history = simulate_spatial_diffusion(steps=steps, data_buffer=data_buffer)
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