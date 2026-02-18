"""
集成的Generator同步系统
连接主进程GUI和计算进程的完整解决方案
"""
import os
import time
import threading
import multiprocessing as mp
import numpy as np
from typing import Generator, Optional, Callable, Any
import dearpygui.dearpygui as dpg

# 导入本地模块
from core_simulation import simulate_spatial_diffusion_generator
from generator_sync import FrameGenerator, FrameConsumer, SynchronizedFrameProcessor
from generator_consumer import GeneratorFrameConsumer, VisualizationFrameProcessor, FrameInfo
from utils import NX, NY, NZ


class IntegratedSimulationSystem:
    """
    集成的模拟系统
    统一管理计算进程、数据同步和GUI显示
    """
    
    def __init__(self):
        self.calculation_process = None
        self.sync_processor = None
        self.frame_consumer = None
        self.vis_processor = None
        self.gui_thread = None
        self.is_running = False
        self.is_paused = False
        
        # GUI相关
        self.current_z_layer = NZ // 2
        self.playback_speed = 1.0
        self.is_playing = False
        self.last_render_time = 0
        
    def setup_system(self, buffer_size: int = 50, max_history: int = 1000):
        """设置整个系统"""
        print("[集成系统] 初始化系统组件...")
        
        # 创建同步处理器
        self.sync_processor = SynchronizedFrameProcessor(buffer_size=buffer_size)
        
        # 创建帧消费者
        self.frame_consumer = GeneratorFrameConsumer(max_history=max_history)
        
        # 创建可视化处理器
        self.vis_processor = VisualizationFrameProcessor(z_layer=self.current_z_layer)
        
        # 添加处理器到消费者
        self.frame_consumer.add_frame_processor(self.vis_processor.process_frame)
        
        # 注册控制回调
        self._setup_callbacks()
        
        print("[集成系统] 系统初始化完成")
        
    def _setup_callbacks(self):
        """设置各种回调函数"""
        # 消费状态回调
        self.frame_consumer.register_control_callback('consumption_started', 
                                                    self._on_consumption_started)
        self.frame_consumer.register_control_callback('consumption_finished', 
                                                    self._on_consumption_finished)
        self.frame_consumer.register_control_callback('frame_updated', 
                                                    self._on_frame_updated)
        self.frame_consumer.register_control_callback('progress_updated', 
                                                    self._on_progress_updated)
        self.frame_consumer.register_control_callback('consumption_paused', 
                                                    self._on_consumption_paused)
        self.frame_consumer.register_control_callback('consumption_resumed', 
                                                    self._on_consumption_resumed)
        
    def _on_consumption_started(self):
        """消费开始回调"""
        print("[集成系统] 数据消费开始")
        self.is_running = True
        self._update_gui_status("模拟运行中...")
        
    def _on_consumption_finished(self, stats: dict):
        """消费完成回调"""
        print(f"[集成系统] 数据消费完成，处理了 {stats['processed_frames']} 帧")
        self.is_running = False
        self._update_gui_status(f"模拟完成 ({stats['processed_frames']} 帧)")
        
    def _on_frame_updated(self, frame_info: FrameInfo):
        """帧更新回调"""
        # 在主线程中更新GUI
        if hasattr(self, '_gui_ready') and self._gui_ready:
            self._schedule_gui_update(frame_info)
        else:
            # 如果GUI还没准备好，稍后重试
            threading.Timer(0.1, lambda: self._schedule_gui_update(frame_info)).start()
    
    def _notify_frame_update(self, frame_info: FrameInfo):
        """内部帧更新通知 - 直接触发回调"""
        try:
            self._on_frame_updated(frame_info)
        except Exception as e:
            print(f"[集成系统] 帧更新通知错误: {e}")
            
    def _on_progress_updated(self, stats: dict):
        """进度更新回调"""
        self._update_progress_bar(stats)
        
    def _on_consumption_paused(self):
        """消费暂停回调"""
        print("[集成系统] 数据消费暂停")
        self.is_paused = True
        self._update_gui_status("模拟已暂停")
        
    def _on_consumption_resumed(self):
        """消费恢复回调"""
        print("[集成系统] 数据消费恢复")
        self.is_paused = False
        self._update_gui_status("模拟运行中...")
        
    def start_calculation_process(self, steps: int = 1000, use_multiprocessing: bool = True):
        """
        启动计算进程
        
        Args:
            steps: 模拟步数
            use_multiprocessing: 是否使用多进程
        """
        if self.is_running:
            print("[集成系统] 模拟已在运行中")
            return False
            
        print(f"[集成系统] 启动计算进程，步数: {steps}")
        
        def simulation_wrapper():
            """模拟包装函数"""
            try:
                # 使用generator版本的模拟函数
                frame_gen = simulate_spatial_diffusion_generator(steps=steps)
                return frame_gen
            except Exception as e:
                print(f"[集成系统] 模拟启动错误: {e}")
                return None
        
        if use_multiprocessing:
            # 多进程版本
            self.calculation_process = mp.Process(
                target=self._multiprocess_worker,
                args=(simulation_wrapper, steps),
                daemon=True
            )
            self.calculation_process.start()
        else:
            # 线程版本
            calc_thread = threading.Thread(
                target=self._thread_worker,
                args=(simulation_wrapper,),
                daemon=True
            )
            calc_thread.start()
            
        return True
    
    def _multiprocess_worker(self, sim_func: Callable, steps: int):
        """多进程工作函数 - 修复版本"""
        try:
            print("[计算进程] 多进程模拟开始")
            
            # 创建进程间数据缓冲区
            from simple_process_buffer import SimpleProcessBuffer
            process_buffer = SimpleProcessBuffer(maxsize=50)
            
            # 修改模拟函数以使用进程缓冲区
            def modified_sim_func():
                # 重新导入必要的模块（避免序列化问题）
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                
                from core_simulation import simulate_spatial_diffusion_generator
                return simulate_spatial_diffusion_generator(steps=steps, data_buffer=process_buffer)
            
            frame_generator = modified_sim_func()
            if frame_generator:
                frame_count = 0
                for frame_data in frame_generator:
                    frame_count += 1
                    # 通过进程缓冲区发送数据给主进程
                    success = process_buffer.put_data(frame_data)
                    if success:
                        print(f"[计算进程] 发送帧 {frame_count} 到主进程")
                    else:
                        print(f"[计算进程] 帧 {frame_count} 发送失败")
                    
                # 标记完成
                process_buffer.mark_finished()
                print(f"[计算进程] 多进程模拟完成，共发送 {frame_count} 帧")
            
            # 保持进程活跃直到主进程获取完所有数据
            import time
            time.sleep(2)  # 给主进程一些时间获取剩余数据
            
        except Exception as e:
            print(f"[计算进程] 错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _thread_worker(self, sim_func: Callable):
        """线程工作函数 - 修复版本"""
        try:
            print("[计算线程] 线程模拟开始")
            
            # 创建线程安全数据缓冲区
            from thread_safe_buffer import ThreadSafeDataBuffer
            data_buffer = ThreadSafeDataBuffer(maxsize=50)
            
            # 启动消费者处理 - 修复数据流
            def consumer_worker():
                try:
                    print("[消费者线程] 启动帧消费处理")
                    # 直接使用数据缓冲区消费数据
                    frame_count = 0
                    while not data_buffer.is_finished() or not data_buffer.data_queue.empty():
                        # 获取数据
                        frame_data = data_buffer.get_data(timeout=0.1)
                        if frame_data is not None:
                            frame_count += 1
                            # 创建帧信息对象
                            from generator_consumer import FrameInfo
                            frame_info = FrameInfo(
                                frame_number=frame_count,
                                data=frame_data,
                                timestamp=time.time()
                            )
                            
                            # 直接添加到帧历史
                            self.frame_consumer.frame_history.append(frame_info)
                            self.frame_consumer.current_frame_index = len(self.frame_consumer.frame_history) - 1
                            
                            # 触发帧更新回调
                            self._notify_frame_update(frame_info)
                            
                            print(f"[消费者线程] 处理帧 {frame_count}")
                        
                        # 短暂休眠避免CPU占用过高
                        time.sleep(0.01)
                    
                    print(f"[消费者线程] 帧消费处理完成，共处理 {frame_count} 帧")
                except Exception as e:
                    print(f"[消费者线程] 错误: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 在单独线程中启动消费者
            consumer_thread = threading.Thread(target=consumer_worker, daemon=True)
            consumer_thread.start()
            
            # 启动模拟生成器
            frame_generator = sim_func()
            if frame_generator:
                frame_count = 0
                for frame_data in frame_generator:
                    frame_count += 1
                    # 通过数据缓冲区发送给消费者
                    success = data_buffer.put_data(frame_data)
                    if success:
                        print(f"[计算线程] 发送帧 {frame_count} 到消费者")
                    else:
                        print(f"[计算线程] 帧 {frame_count} 发送失败，缓冲区可能已满")
                    
                    # 适当延迟避免生产速度过快
                    time.sleep(0.001)
                
                # 标记完成
                data_buffer.mark_finished()
                print(f"[计算线程] 线程模拟完成，共发送 {frame_count} 帧")
                
                # 等待消费者完成
                consumer_thread.join(timeout=10)
                
        except Exception as e:
            print(f"[计算线程] 错误: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_gui(self):
        """设置GUI界面"""
        print("[集成系统] 设置GUI界面...")

        # 创建Dear PyGui上下文
        dpg.create_context()
        with dpg.font_registry():
            # 尝试常见路径
            font_path = "C:/Windows/Fonts/msyh.ttc"
            if not os.path.exists(font_path):
                font_path = "C:/Windows/Fonts/simsun.ttc"

            if os.path.exists(font_path):
                with dpg.font(font_path, 18) as font:
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                    dpg.bind_font(font)
        # 创建主窗口
        with dpg.window(label="PhreeqcRM Generator同步系统", width=1000, height=700, 
                       tag="main_window"):
            
            # 控制面板
            with dpg.group(horizontal=True):
                with dpg.group(width=250):
                    dpg.add_text("控制面板", color=(255, 255, 0))
                    dpg.add_separator()
                    
                    dpg.add_button(label="启动模拟", callback=self._start_simulation, 
                                 tag="start_btn", width=200, height=40)
                    dpg.add_button(label="暂停/继续", callback=self._toggle_pause, 
                                 tag="pause_btn", width=200, height=30)
                    dpg.add_button(label="停止模拟", callback=self._stop_simulation, 
                                 tag="stop_btn", width=200, height=30)
                    
                    dpg.add_spacer(height=10)
                    dpg.add_slider_float(label="播放速度", default_value=1.0, min_value=0.1, 
                                       max_value=3.0, tag="speed_slider", 
                                       callback=self._speed_changed, width=200)
                    
                    dpg.add_spacer(height=10)
                    dpg.add_slider_int(label="Z层", default_value=self.current_z_layer, 
                                     min_value=0, max_value=NZ-1, tag="z_slider",
                                     callback=self._z_layer_changed, width=200)
                    
                    dpg.add_spacer(height=10)
                    dpg.add_slider_int(label="帧索引", min_value=0, max_value=0, 
                                     tag="frame_slider", callback=self._frame_changed, 
                                     width=200)
                    
                    # 状态信息
                    dpg.add_spacer(height=10)
                    dpg.add_text("状态信息:", color=(0, 255, 255))
                    dpg.add_text("系统就绪", tag="status_text")
                    dpg.add_text("帧: 0/0", tag="frame_info")
                    dpg.add_text("pH: --", tag="ph_info")
                    dpg.add_text("FPS: 0.0", tag="fps_info")
                    
                    # 进度条
                    dpg.add_spacer(height=10)
                    dpg.add_progress_bar(tag="progress_bar", width=200, default_value=0.0)
                    
                # 显示区域
                with dpg.group():
                    dpg.add_text("3D pH分布可视化 (Z层切片)", color=(255, 255, 0))
                    with dpg.drawlist(width=700, height=550, tag="display_canvas"):
                        self._draw_empty_canvas()
        
        # 设置viewport
        dpg.create_viewport(title='PhreeqcRM Generator同步演示', width=1000, height=700)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        
        self._gui_ready = True
        print("[集成系统] GUI设置完成")
        
    def _draw_empty_canvas(self):
        """绘制空画布"""
        if dpg.does_item_exist("display_canvas"):
            dpg.delete_item("display_canvas", children_only=True)
            
        # 绘制网格背景
        cell_size = 700 // NX
        for y in range(NY):
            for x in range(NX):
                pos_x = x * cell_size
                pos_y = y * cell_size
                dpg.draw_rectangle([pos_x, pos_y], [pos_x + cell_size, pos_y + cell_size],
                                 color=[100, 100, 100, 100], fill=[200, 200, 200, 50],
                                 parent="display_canvas")
        
        # 添加提示文本
        center_x, center_y = 350, 275
        dpg.draw_text([center_x - 100, center_y], "等待数据...", size=20,
                     color=[100, 100, 100, 200], parent="display_canvas")
    
    def _start_simulation(self):
        """启动模拟回调 - 修复版本"""
        if not self.is_running:
            print("[GUI] 启动模拟按钮被点击")
            self.setup_system()
            
            # 启动计算进程
            success = self.start_calculation_process(steps=200, use_multiprocessing=False)
            if success:
                dpg.configure_item("start_btn", enabled=False, label="运行中...")
                self._update_gui_status("初始化模拟...")
                print("[GUI] 模拟启动成功")
            else:
                self._update_gui_status("模拟启动失败")
                print("[GUI] 模拟启动失败")
    
    def _toggle_pause(self):
        """暂停/继续回调"""
        if self.is_running:
            if self.is_paused:
                self.frame_consumer.resume_consumption()
                dpg.set_item_label("pause_btn", "暂停")
            else:
                self.frame_consumer.pause_consumption()
                dpg.set_item_label("pause_btn", "继续")
    
    def _stop_simulation(self):
        """停止模拟回调"""
        if self.is_running:
            self.frame_consumer.stop_consumption()
            if self.calculation_process and self.calculation_process.is_alive():
                self.calculation_process.terminate()
            dpg.configure_item("start_btn", enabled=True, label="启动模拟")
            self._update_gui_status("模拟已停止")
    
    def _speed_changed(self, sender, app_data):
        """速度改变回调"""
        self.playback_speed = float(app_data)
    
    def _z_layer_changed(self, sender, app_data):
        """Z层改变回调"""
        self.current_z_layer = int(app_data)
        self.vis_processor.z_layer = self.current_z_layer
        # 重新处理当前帧
        current_frame = self.frame_consumer.get_current_frame()
        if current_frame:
            self.vis_processor.process_frame(current_frame)
            self._update_display()
    
    def _frame_changed(self, sender, app_data):
        """帧改变回调"""
        frame_index = int(app_data)
        if self.frame_consumer.set_current_frame_index(frame_index):
            self._update_display()
    
    def _update_gui_status(self, message: str):
        """更新GUI状态"""
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", message)
    
    def _update_progress_bar(self, stats: dict):
        """更新进度条"""
        if dpg.does_item_exist("progress_bar"):
            if stats.get('total_frames', 0) > 0:
                progress = stats['processed_frames'] / stats['total_frames']
                dpg.set_value("progress_bar", min(1.0, progress))
    
    def _schedule_gui_update(self, frame_info: FrameInfo):
        """调度GUI更新 - 修复版本"""
        # 使用线程安全的方式更新GUI
        try:
            # 检查是否在主线程中
            if threading.current_thread() is threading.main_thread():
                # 直接更新
                self._update_display()
            else:
                # 在主线程中调度更新
                def safe_update():
                    try:
                        self._update_display()
                    except Exception as e:
                        print(f"[集成系统] GUI更新错误: {e}")
                
                # 使用DPG的异步机制
                if hasattr(dpg, '_callback_queue'):
                    # 如果有回调队列，使用它
                    dpg._callback_queue.put(safe_update)
                else:
                    # 否则直接调用
                    safe_update()
        except Exception as e:
            print(f"[集成系统] GUI调度错误: {e}")
    
    def _update_display(self):
        """更新显示 - 修复版本"""
        try:
            # 检查是否有数据
            if not self.frame_consumer:
                print("[GUI] 帧消费者未初始化")
                return
            
            # 检查帧历史
            if not self.frame_consumer.frame_history:
                print("[GUI] 帧历史为空")
                return
            
            current_frame = self.frame_consumer.get_current_frame()
            if not current_frame:
                # 尝试获取最后一帧
                if self.frame_consumer.frame_history:
                    current_frame = self.frame_consumer.frame_history[-1]
                    self.frame_consumer.current_frame_index = len(self.frame_consumer.frame_history) - 1
                else:
                    print("[GUI] 没有可用帧数据")
                    return
            
            print(f"[GUI] 更新显示 - 当前帧: {current_frame.frame_number}")
            
            # 处理当前帧数据
            if not hasattr(self.vis_processor, '_processed_frames'):
                self.vis_processor._processed_frames = set()
            
            # 检查是否已经处理过此帧
            if current_frame.frame_number not in self.vis_processor._processed_frames:
                success = self.vis_processor.process_frame(current_frame)
                if success:
                    self.vis_processor._processed_frames.add(current_frame.frame_number)
                    print(f"[GUI] 帧 {current_frame.frame_number} 处理成功")
                else:
                    print(f"[GUI] 帧 {current_frame.frame_number} 处理失败")
                    return
            
            # 获取可视化数据
            vis_data = self.vis_processor.get_cached_visualization(current_frame.frame_number)
            if not vis_data:
                print(f"[GUI] 没有找到帧 {current_frame.frame_number} 的可视化数据")
                return
            
            # 更新状态信息
            stats = vis_data['stats']
            if dpg.does_item_exist("frame_info"):
                total_frames = len(self.frame_consumer.frame_history)
                current_index = self.frame_consumer.current_frame_index + 1
                dpg.set_value("frame_info", f"帧: {current_index}/{total_frames}")
                
            if dpg.does_item_exist("ph_info"):
                dpg.set_value("ph_info", f"pH: {stats['min']:.2f} ~ {stats['max']:.2f} "
                                       f"(平均: {stats['mean']:.2f})")
            
            # 更新FPS信息
            if dpg.does_item_exist("fps_info"):
                consumption_stats = self.frame_consumer.get_consumption_stats()
                dpg.set_value("fps_info", f"FPS: {consumption_stats.get('average_fps', 0):.1f}")
            
            # 更新帧滑块
            if dpg.does_item_exist("frame_slider"):
                max_frames = max(0, len(self.frame_consumer.frame_history) - 1)
                dpg.configure_item("frame_slider", max_value=max_frames)
                dpg.set_value("frame_slider", self.frame_consumer.current_frame_index)
            
            # 绘制可视化
            self._draw_ph_visualization(vis_data)
            print(f"[GUI] 显示更新完成 - 帧 {current_frame.frame_number}")
            
        except Exception as e:
            print(f"[GUI] 显示更新错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_ph_visualization(self, vis_data: dict):
        """绘制pH可视化"""
        if not dpg.does_item_exist("display_canvas"):
            return
            
        # 清除现有内容
        dpg.delete_item("display_canvas", children_only=True)
        
        color_data = vis_data['color_data']
        ny, nx = color_data.shape[:2]
        cell_size = 700 // max(nx, ny)
        
        # 绘制颜色网格
        for y in range(min(ny, 15)):  # 限制显示大小
            for x in range(min(nx, 15)):
                color = color_data[y, x].tolist()
                pos_x = x * cell_size
                pos_y = y * cell_size
                
                dpg.draw_rectangle([pos_x, pos_y], [pos_x + cell_size, pos_y + cell_size],
                                 color=[80, 80, 80, 150], fill=color,
                                 parent="display_canvas")
    
    def run_gui_loop(self):
        """运行GUI主循环 - 修复版本"""
        print("[集成系统] 启动GUI主循环...")
        
        last_data_check = time.time()
        check_interval = 0.5  # 每0.5秒检查一次新数据
        
        try:
            while dpg.is_dearpygui_running():
                current_time = time.time()
                
                # 定期检查新数据
                if current_time - last_data_check > check_interval:
                    self._check_for_new_data()
                    last_data_check = current_time
                
                # 处理自动播放
                self._handle_auto_playback()
                
                # 渲染
                dpg.render_dearpygui_frame()
                time.sleep(0.016)  # ~60 FPS
                
        except KeyboardInterrupt:
            print("[集成系统] 收到中断信号")
        except Exception as e:
            print(f"[集成系统] GUI循环错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def _check_for_new_data(self):
        """检查新数据 - 修复版本"""
        try:
            # 检查帧消费者是否有新数据
            if self.frame_consumer and hasattr(self.frame_consumer, 'frame_history'):
                current_history_size = len(self.frame_consumer.frame_history)
                if hasattr(self, '_last_history_size'):
                    if current_history_size > self._last_history_size:
                        print(f"[GUI] 检测到新数据: {current_history_size - self._last_history_size} 帧")
                        # 确保在主线程中更新显示
                        if threading.current_thread() is threading.main_thread():
                            self._update_display()
                        else:
                            # 在主线程中调度更新
                            dpg.split_frame(callback=self._update_display)
                self._last_history_size = current_history_size
            
            # 检查计算进程状态
            if self.calculation_process and not self.calculation_process.is_alive():
                if self.is_running:
                    print("[GUI] 检测到计算进程结束")
                    self.is_running = False
                    self._update_gui_status("计算完成")
                    # 启用启动按钮
                    if dpg.does_item_exist("start_btn"):
                        dpg.configure_item("start_btn", enabled=True, label="启动模拟")
        
        except Exception as e:
            print(f"[GUI] 数据检查错误: {e}")
    
    def _handle_auto_playback(self):
        """处理自动播放逻辑"""
        if self.is_playing and self.frame_consumer.frame_history:
            current_time = time.time()
            if current_time - self.last_render_time > (1.0 / 15.0 / self.playback_speed):
                if self.frame_consumer.next_frame():
                    self._update_display()
                else:
                    # 播放完成
                    self.is_playing = False
                self.last_render_time = current_time
    
    def cleanup(self):
        """清理资源"""
        print("[集成系统] 清理系统资源...")
        
        # 停止所有活动
        if self.frame_consumer:
            self.frame_consumer.stop_consumption()
            
        if self.calculation_process and self.calculation_process.is_alive():
            self.calculation_process.terminate()
            self.calculation_process.join(timeout=5)
            
        # 清理GUI
        try:
            if dpg.is_dearpygui_running():
                dpg.stop_dearpygui()
            dpg.destroy_context()
        except:
            pass
            
        print("[集成系统] 清理完成")


def main():
    """主函数"""
    print("=== PhreeqcRM Generator同步系统 ===")
    
    # 创建集成系统
    system = IntegratedSimulationSystem()
    
    try:
        # 设置并运行GUI
        system.setup_gui()
        system.run_gui_loop()
        
    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        print(f"程序错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()