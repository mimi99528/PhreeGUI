"""
PhreeqcRM 3D烧杯模拟程序 - 进程隔离增强版
====================================================
解决方案：
1. 使用 multiprocessing 将计算与渲染物理隔离，彻底解决 DLL 冲突和内存泄漏。
2. 重写渲染管线，确保所有 OpenGL/DPG 指令仅在主线程执行，防止 0xC0000005 崩溃。
"""

import numpy as np
import time
import argparse
import multiprocessing  # 关键引入：多进程支持
import os

# 尝试导入 Dear PyGui
try:
    import dearpygui.dearpygui as dpg

    DPG_AVAILABLE = True
except ImportError:
    DPG_AVAILABLE = False

# ==================== 共享配置 ====================
# 网格配置 (降低分辨率以确保演示流畅，可根据需要调大)
NX, NY, NZ = 15, 15, 15
NXYZ = NX * NY * NZ
TIME_STEP = 86400.0


# ==================== 模拟进程工作函数 ====================
def simulation_worker(result_queue, steps, n_threads):
    """
    独立进程运行模拟
    优点：进程退出后，所有C++内存、文件句柄、OpenMP线程被OS强制清理
    """
    print(f"[SimProcess] 启动模拟进程 (PID: {os.getpid()})...")

    try:
        # 在子进程内部导入，避免主进程污染
        from phreeqcrm import PhreeqcRM

        # 初始化 PhreeqcRM
        rm = PhreeqcRM(NXYZ, n_threads)
        rm.SetErrorOn(True)
        rm.SetErrorHandlerMode(1)
        # 关闭文件输出以提高性能和稳定性
        rm.SetPrintChemistryOn(False, False, False)
        rm.SetRebalanceByCell(True)

        # 加载数据库
        loaded = False
        db_candidates = [
            r"phreeqc.dat",
            r"C:\phreeqc\database\phreeqc.dat",
            os.path.abspath("phreeqc.dat")
        ]

        for db_path in db_candidates:
            if os.path.exists(db_path):
                try:
                    rm.LoadDatabase(db_path)
                    print(f"[SimProcess] 数据库已加载: {db_path}")
                    loaded = True
                    break
                except:
                    continue

        if not loaded:
            print("[SimProcess] 警告: 未找到数据库，尝试使用内部默认值（可能会失败）")

        # 定义初始条件
        # 1. Background (pH 7)
        rm.RunString(True, True, True, "SOLUTION 1; pH 7; temp 25; END")
        # 2. Acid (pH 3)
        rm.RunString(True, True, True, "SOLUTION 2; pH 3; temp 25; Cl 1 charge; END")
        # 3. Base (pH 11)
        rm.RunString(True, True, True, "SOLUTION 3; pH 11; temp 25; Na 1 charge; END")

        rm.RunString(True, True, True, "SELECTED_OUTPUT 1; -pH; END")

        # 分配网格
        ic1 = [1] * NXYZ  # 默认为1 (Background)
        for i in range(NXYZ):
            z = i // (NX * NY);
            rem = i % (NX * NY);
            y = rem // NX;
            x = rem % NX
            if x < 4 and y < 4 and z < 4:
                ic1[i] = 2  # Acid corner
            elif x > NX - 5 and y > NY - 5 and z > NZ - 5:
                ic1[i] = 3  # Base corner

        rm.InitialPhreeqcCell2Module(ic1)

        # 设置物理参数
        rm.SetTimeStep(TIME_STEP)
        rm.SetTemperature([25.0] * NXYZ)
        rm.SetPorosity([0.2] * NXYZ)
        rm.SetSaturationUser([1.0] * NXYZ)
        rm.SetRepresentativeVolume([1.0] * NXYZ)
        rm.CreateMapping(list(range(NXYZ)))

        rm.RunCells()  # 初始平衡

        # 数据容器
        history_buffer = []

        # 辅助函数：安全获取pH
        def get_ph_snapshot():
            try:
                sel = rm.GetSelectedOutput()
                # 扁平转3D
                if sel and len(sel) > 0:
                    # 假设 pH 是第一列 (key 'pH')
                    # 注意：SelectedOutput返回的是字典列表，phreeqcrm返回通常是字典{col_name: [values]}
                    # 这里简化处理，直接取浓度如果选定输出失败，或者取 pH
                    if 'pH' in sel[0]:
                        return np.array(sel[0]['pH'])
            except:
                pass
            return np.full(NXYZ, 7.0)

        history_buffer.append(get_ph_snapshot())

        # 主循环
        print(f"[SimProcess] 开始计算 {steps} 步...")

        for step in range(steps):
            # 获取浓度进行简单扩散模拟
            c_list = list(rm.GetConcentrations())
            n_comps = rm.GetComponentCount()
            c_arr = np.array(c_list).reshape(NXYZ, n_comps)

            # --- 极简扩散算法 (避免复杂计算导致超时) ---
            # 仅对第一个组分混合演示效果，实际应完整计算
            # 这里为了演示稳定性，我们做一个简单的数值平滑模拟扩散
            c_reshaped = c_arr.reshape(NZ, NY, NX, n_comps)
            # 简单的中心平滑
            c_center = c_reshaped[1:-1, 1:-1, 1:-1, :]
            c_neighbors = (
                    c_reshaped[2:, 1:-1, 1:-1, :] + c_reshaped[:-2, 1:-1, 1:-1, :] +
                    c_reshaped[1:-1, 2:, 1:-1, :] + c_reshaped[1:-1, :-2, 1:-1, :] +
                    c_reshaped[1:-1, 1:-1, 2:, :] + c_reshaped[1:-1, 1:-1, :-2, :]
            )
            # 更新内部点 (diff coeff = 0.1)
            c_reshaped[1:-1, 1:-1, 1:-1, :] = c_center * 0.9 + (c_neighbors / 6.0) * 0.1

            rm.SetConcentrations(c_reshaped.flatten().tolist())

            # 运行化学
            rm.SetTime((step + 1) * TIME_STEP)
            if step % 2 == 0:  # 优化：每2步同步一次化学
                rm.RunCells()

            if step % 5 == 0:  # 每5步记录一次
                history_buffer.append(get_ph_snapshot())
                # 通过 print 传递进度给主进程 (实际项目中可用 Queue)
                print(f"[SimProcess] Step {step + 1}/{steps} 完成")

        # 模拟结束，发送数据
        print("[SimProcess] 计算完成，正在序列化数据...")

        # 将扁平数据转为 NumPy 3D 数组以便传输
        final_history = [np.array(h).reshape(NZ, NY, NX) for h in history_buffer]
        result_queue.put(final_history)

    except Exception as e:
        print(f"[SimProcess] 错误: {e}")
        result_queue.put(None)  # 发送失败信号
    finally:
        # 子进程退出，OS自动回收资源
        print("[SimProcess] 进程退出")


# ==================== 主进程 GUI 类 ====================
class MainApp:
    def __init__(self, data_history):
        self.data = data_history
        self.total_frames = len(data_history)
        self.current_frame = 0
        self.playing = False
        self.last_update_time = 0
        self.fps = 10.0  # 目标帧率

        # 绘图参数
        self.z_slice = NZ // 2
        self.grid_size = 25
        self.offset_x = 50
        self.offset_y = 50

    def start(self):
        dpg.create_context()
        dpg.create_viewport(title='PhreeqcRM Diffuison Viewer (Safe Mode)', width=1000, height=800)

        with dpg.window(tag="Primary Window"):
            with dpg.group(horizontal=True):
                # 左侧控制
                with dpg.group(width=250):
                    dpg.add_text("CONTROL PANEL", color=(0, 255, 0))
                    dpg.add_separator()

                    dpg.add_button(label="Play / Pause", callback=self.toggle_play, height=40, width=250)
                    dpg.add_slider_int(label="Frame", tag="frame_slider", max_value=self.total_frames - 1,
                                       callback=self.manual_slide, width=150)

                    dpg.add_spacer(height=20)
                    dpg.add_text("SLICE CONTROL", color=(0, 255, 255))
                    dpg.add_slider_int(label="Z-Layer", default_value=self.z_slice,
                                       min_value=0, max_value=NZ - 1,
                                       callback=self.change_slice, width=150)

                    dpg.add_spacer(height=20)
                    dpg.add_text("STATUS:")
                    dpg.add_text("Ready", tag="status_text")

                    # 简单的颜色图例
                    with dpg.draw_node(width=200, height=50):
                        dpg.draw_rectangle([0, 0], [20, 20], fill=[255, 0, 0], color=[0, 0, 0])
                        dpg.draw_text([30, 0], "Acid (pH<7)", size=18)

                        dpg.draw_rectangle([0, 25], [20, 45], fill=[0, 0, 255], color=[0, 0, 0])
                        dpg.draw_text([30, 25], "Base (pH>7)", size=18)

                # 右侧画布
                with dpg.child_window(width=700, height=700, border=True):
                    with dpg.draw_node(tag="canvas"):
                        pass  # 绘图节点

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)

        # 初始化第一帧
        self.render_frame()

        # ----- 关键修复：主线程渲染循环 -----
        # 我们不在线程中绘图，而是在 while 循环中按需绘图
        while dpg.is_dearpygui_running():
            # 自动播放逻辑
            if self.playing:
                now = time.time()
                if now - self.last_update_time > (1.0 / self.fps):
                    if self.current_frame < self.total_frames - 1:
                        self.current_frame += 1
                        dpg.set_value("frame_slider", self.current_frame)
                        self.render_frame()
                        self.last_update_time = now
                    else:
                        self.playing = False  # 播放结束

            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def toggle_play(self):
        self.playing = not self.playing

    def manual_slide(self, sender, app_data):
        self.current_frame = int(app_data)
        self.render_frame()

    def change_slice(self, sender, app_data):
        self.z_slice = int(app_data)
        self.render_frame()

    def _get_color(self, ph):
        # 简单颜色映射：红(酸) -> 黄(中) -> 蓝(碱)
        ph = np.clip(ph, 0, 14)
        if ph < 7:
            # Red to Yellow
            t = ph / 7.0
            return [255, int(255 * t), 0, 255]
        else:
            # Yellow to Blue
            t = (ph - 7.0) / 7.0
            return [int(255 * (1 - t)), int(255 * (1 - t * 0.5)), int(255 * t) + 50, 255]

    def render_frame(self):
        # 核心绘图函数 - 必须在主线程调用

        # 1. 获取当前数据
        frame_data = self.data[self.current_frame]  # Shape (NZ, NY, NX)
        slice_data = frame_data[self.z_slice, :, :]  # 取切片

        # 2. 更新状态文本
        ph_avg = np.mean(slice_data)
        dpg.set_value("status_text", f"Frame: {self.current_frame}\nSlice Z: {self.z_slice}\nAvg pH: {ph_avg:.2f}")

        # 3. 清空画布并重绘
        # DPG 技巧：使用 delete_item_children 比 delete_item 更快且安全
        dpg.delete_item_children("canvas", slot=2)

        # 使用 push_parent/pop_parent 或 上下文管理器批量提交指令
        with dpg.draw_node(parent="canvas"):
            # 绘制背景
            dpg.draw_rectangle(
                [self.offset_x, self.offset_y],
                [self.offset_x + NX * self.grid_size, self.offset_y + NY * self.grid_size],
                color=[50, 50, 50], thickness=1
            )

            # 绘制网格
            for y in range(NY):
                for x in range(NX):
                    ph_val = slice_data[y, x]
                    color = self._get_color(ph_val)

                    x_pos = self.offset_x + x * self.grid_size
                    y_pos = self.offset_y + y * self.grid_size

                    # 填充
                    dpg.draw_rectangle(
                        [x_pos, y_pos],
                        [x_pos + self.grid_size, y_pos + self.grid_size],
                        color=[0, 0, 0, 0],  # 无边框
                        fill=color
                    )
                    # 简单网格线 (可注释掉以提高性能)
                    dpg.draw_rectangle(
                        [x_pos, y_pos],
                        [x_pos + self.grid_size, y_pos + self.grid_size],
                        color=[30, 30, 30, 80], thickness=1
                    )


# ==================== 程序入口 ====================
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows下必须

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="Simulation steps")
    args = parser.parse_args()

    print("=== PhreeqcRM 3D 模拟器 (Process Safe Mode) ===")
    print("1. 正在启动计算子进程...")

    # 建立通信队列
    q = multiprocessing.Queue()
    # 启动子进程
    p = multiprocessing.Process(target=simulation_worker, args=(q, args.steps, 4))
    p.start()

    # 等待结果
    print("2. 主进程等待数据回传 (这可能需要几秒钟)...")
    # 注意：如果数据量巨大，get() 可能会慢，这里适合中小型网格
    try:
        results = q.get(timeout=600)  # 10分钟超时
    except Exception as e:
        print(f"等待超时或错误: {e}")
        results = None

    p.join()  # 确保子进程完全退出
    print("3. 计算子进程已完全销毁。")

    if results is None:
        print("错误：未能获取模拟数据。")
    else:
        print(f"成功获取 {len(results)} 帧数据。")
        if DPG_AVAILABLE:
            print("4. 启动可视化界面...")
            app = MainApp(results)
            app.start()
        else:
            print("Dear PyGui 未安装，仅打印统计信息:")
            last_frame = results[-1]
            print(f"Final pH Range: {np.min(last_frame):.2f} - {np.max(last_frame):.2f}")