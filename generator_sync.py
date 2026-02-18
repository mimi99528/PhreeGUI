"""
基于Generator的帧数据同步机制
替代传统的回调方式，提供更直观的数据流控制
"""

import time
import threading
import queue
from typing import Generator, Optional, Any, Callable
import numpy as np


class FrameGenerator:
    """
    帧数据生成器
    负责产生模拟帧数据并通过generator方式输出
    """
    
    def __init__(self, buffer_size: int = 50):
        self.buffer_size = buffer_size
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.control_queue = queue.Queue()
        self.is_running = threading.Event()
        self.is_paused = threading.Event()
        self.frame_count = 0
        self.generator_lock = threading.Lock()
        
    def start_generation(self):
        """启动数据生成"""
        self.is_running.set()
        self.is_paused.clear()
        
    def pause_generation(self):
        """暂停数据生成"""
        self.is_paused.set()
        
    def resume_generation(self):
        """恢复数据生成"""
        self.is_paused.clear()
        
    def stop_generation(self):
        """停止数据生成"""
        self.is_running.clear()
        self.is_paused.clear()
        
    def put_frame(self, frame_data: np.ndarray) -> bool:
        """
        放入帧数据
        
        Args:
            frame_data: 3D numpy数组帧数据
            
        Returns:
            bool: 是否成功放入
        """
        if not self.is_running.is_set():
            return False
            
        try:
            # 如果队列满了，等待有空间
            while self.data_queue.full() and self.is_running.is_set():
                if self.is_paused.is_set():
                    time.sleep(0.1)
                    continue
                try:
                    # 移除最旧的数据为新数据腾出空间
                    self.data_queue.get_nowait()
                except queue.Empty:
                    break
            
            if self.is_running.is_set():
                self.data_queue.put_nowait({
                    'frame_number': self.frame_count,
                    'data': frame_data,
                    'timestamp': time.time()
                })
                self.frame_count += 1
                return True
        except Exception as e:
            print(f"[FrameGenerator] 数据放入失败: {e}")
            
        return False
    
    def send_control_command(self, command: str, **kwargs):
        """发送控制命令"""
        try:
            self.control_queue.put_nowait({
                'command': command,
                'timestamp': time.time(),
                'params': kwargs
            })
        except queue.Full:
            pass
    
    def get_control_commands(self) -> list:
        """获取控制命令"""
        commands = []
        while not self.control_queue.empty():
            try:
                commands.append(self.control_queue.get_nowait())
            except queue.Empty:
                break
        return commands
    
    def frame_generator(self) -> Generator[dict, None, None]:
        """
        帧数据生成器
        yield格式: {'frame_number': int, 'data': np.ndarray, 'timestamp': float}
        """
        print("[FrameGenerator] 启动帧数据生成器")
        
        while self.is_running.is_set():
            # 处理暂停状态
            if self.is_paused.is_set():
                time.sleep(0.1)
                continue
                
            # 处理控制命令
            commands = self.get_control_commands()
            for cmd in commands:
                if cmd['command'] == 'pause':
                    self.pause_generation()
                elif cmd['command'] == 'resume':
                    self.resume_generation()
                elif cmd['command'] == 'stop':
                    self.stop_generation()
                    break
                    
            if not self.is_running.is_set():
                break
                
            # 尝试获取数据
            try:
                frame_data = self.data_queue.get(timeout=0.1)
                yield frame_data
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"[FrameGenerator] 数据获取错误: {e}")
                continue
                
        print("[FrameGenerator] 帧数据生成器已停止")


class FrameConsumer:
    """
    帧数据消费者
    负责消费generator产生的帧数据并进行处理
    """
    
    def __init__(self, frame_processor: Optional[Callable] = None):
        self.frame_processor = frame_processor or self._default_processor
        self.consumed_frames = 0
        self.last_consumption_time = 0
        self.consumption_rate = 0.0
        
    def _default_processor(self, frame_data: dict) -> bool:
        """
        默认帧处理器
        只是简单地计数和打印信息
        
        Args:
            frame_data: 帧数据字典
            
        Returns:
            bool: 处理是否成功
        """
        try:
            frame_num = frame_data['frame_number']
            data_shape = frame_data['data'].shape
            timestamp = frame_data['timestamp']
            
            # 计算处理速率
            current_time = time.time()
            if self.last_consumption_time > 0:
                time_diff = current_time - self.last_consumption_time
                if time_diff > 0:
                    self.consumption_rate = 1.0 / time_diff
            
            self.last_consumption_time = current_time
            self.consumed_frames += 1
            
            # 打印基本信息（每10帧打印一次）
            if frame_num % 10 == 0:
                print(f"[FrameConsumer] 处理帧 {frame_num} | 形状: {data_shape} | "
                      f"速率: {self.consumption_rate:.1f} fps")
                      
            return True
        except Exception as e:
            print(f"[FrameConsumer] 帧处理错误: {e}")
            return False
    
    def consume_frames(self, frame_generator: Generator[dict, None, None]) -> int:
        """
        消费帧数据
        
        Args:
            frame_generator: 帧数据生成器
            
        Returns:
            int: 成功消费的帧数
        """
        consumed_count = 0
        print("[FrameConsumer] 开始消费帧数据...")
        
        try:
            for frame_data in frame_generator:
                if self.frame_processor(frame_data):
                    consumed_count += 1
                else:
                    print(f"[FrameConsumer] 帧处理失败，帧号: {frame_data.get('frame_number', 'unknown')}")
                    
        except KeyboardInterrupt:
            print("[FrameConsumer] 收到中断信号")
        except Exception as e:
            print(f"[FrameConsumer] 消费过程中发生错误: {e}")
        finally:
            print(f"[FrameConsumer] 消费完成，共处理 {consumed_count} 帧")
            
        return consumed_count


class SynchronizedFrameProcessor:
    """
    同步帧处理器
    结合生成器和消费者，提供完整的同步处理流程
    """
    
    def __init__(self, buffer_size: int = 50):
        self.frame_generator = FrameGenerator(buffer_size)
        self.frame_consumer = FrameConsumer()
        self.processing_thread = None
        self.is_active = False
        
    def set_frame_processor(self, processor_func: Callable[[dict], bool]):
        """设置自定义帧处理器"""
        self.frame_consumer.frame_processor = processor_func
        
    def start_processing(self, simulation_func: Callable[[], Generator[np.ndarray, None, None]]):
        """
        启动同步处理
        
        Args:
            simulation_func: 返回帧数据generator的模拟函数（函数而非generator对象）
        """
        if self.is_active:
            print("[SyncProcessor] 处理已在运行中")
            return False
            
        def processing_worker():
            try:
                # 启动生成器
                self.frame_generator.start_generation()
                print("[SyncProcessor] 帧生成器已启动")
                
                # 启动模拟并获取帧数据generator
                print("[SyncProcessor] 启动模拟函数...")
                sim_generator = simulation_func()
                print("[SyncProcessor] 模拟generator已获取")
                
                # 创建帧数据注入器
                def frame_injector():
                    frame_counter = 0
                    print("[SyncProcessor] 注入器开始工作")
                    try:
                        for frame_data in sim_generator:
                            frame_counter += 1
                            # 直接传递numpy数组，让帧消费者自己构造FrameInfo
                            success = self.frame_generator.put_frame(frame_data)
                            if not success:
                                print(f"[SyncProcessor] 数据注入失败，帧 {frame_counter}，可能已停止")
                                break
                            print(f"[SyncProcessor] 注入帧 {frame_counter}")
                            # 不需要yield，因为我们不需要这个generator的输出
                    except Exception as e:
                        print(f"[SyncProcessor] 注入器错误: {e}")
                    finally:
                        print(f"[SyncProcessor] 注入器完成，共注入 {frame_counter} 帧")
                
                # 启动注入器线程
                print("[SyncProcessor] 启动注入器线程")
                injector_thread = threading.Thread(target=frame_injector, daemon=True)
                injector_thread.start()
                
                # 等待注入器开始工作
                time.sleep(0.5)
                
                # 消费处理
                print("[SyncProcessor] 启动消费者处理")
                consumed_count = self.frame_consumer.consume_frames(self.frame_generator.frame_generator())
                print(f"[SyncProcessor] 消费完成，处理了 {consumed_count} 帧")
                
                # 等待注入器完成
                injector_thread.join(timeout=10)
                print("[SyncProcessor] 注入器线程已结束")
                
            except Exception as e:
                print(f"[SyncProcessor] 处理过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.frame_generator.stop_generation()
                self.is_active = False
                print("[SyncProcessor] 同步处理已结束")
        
        self.is_active = True
        self.processing_thread = threading.Thread(target=processing_worker, daemon=True)
        self.processing_thread.start()
        return True
        
    def pause_processing(self):
        """暂停处理"""
        self.frame_generator.send_control_command('pause')
        
    def resume_processing(self):
        """恢复处理"""
        self.frame_generator.send_control_command('resume')
        
    def stop_processing(self):
        """停止处理"""
        self.frame_generator.send_control_command('stop')
        self.is_active = False
        
    def get_status(self) -> dict:
        """获取处理状态"""
        return {
            'is_active': self.is_active,
            'consumed_frames': self.frame_consumer.consumed_frames,
            'consumption_rate': self.frame_consumer.consumption_rate,
            'queue_size': self.frame_generator.data_queue.qsize(),
            'frame_count': self.frame_generator.frame_count
        }


# 示例使用函数
def example_simulation_generator(total_frames: int = 100) -> Generator[np.ndarray, None, None]:
    """
    示例模拟生成器
    生成测试用的3D数据帧
    """
    print(f"[示例模拟] 开始生成 {total_frames} 帧数据...")
    
    for frame_num in range(total_frames):
        # 生成模拟的3D pH数据 (15x15x15)
        data_3d = np.random.uniform(6.5, 7.5, (15, 15, 15))
        
        # 添加一些变化模拟扩散过程
        time_factor = frame_num / total_frames
        data_3d += np.sin(time_factor * np.pi) * 0.5
        
        # 确保数据在合理范围内
        data_3d = np.clip(data_3d, 0, 14)
        
        print(f"[示例模拟] 生成帧 {frame_num + 1}/{total_frames}")
        yield data_3d
        time.sleep(0.1)  # 模拟计算时间


def example_frame_processor(frame_data: dict) -> bool:
    """
    示例帧处理器
    处理接收到的帧数据
    """
    try:
        frame_num = frame_data['frame_number']
        data = frame_data['data']
        timestamp = frame_data['timestamp']
        
        # 计算基本统计信息
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            ph_min = np.min(valid_data)
            ph_max = np.max(valid_data)
            ph_avg = np.mean(valid_data)
            
            print(f"[示例处理器] 帧 {frame_num}: pH范围 [{ph_min:.2f}, {ph_max:.2f}], "
                  f"平均值 {ph_avg:.2f}")
        else:
            print(f"[示例处理器] 帧 {frame_num}: 无有效数据")
            
        return True
    except Exception as e:
        print(f"[示例处理器] 处理错误: {e}")
        return False


if __name__ == "__main__":
    # 测试示例
    print("=== Generator同步机制测试 ===")
    
    # 创建同步处理器
    sync_processor = SynchronizedFrameProcessor(buffer_size=20)
    
    # 设置自定义处理器
    sync_processor.set_frame_processor(example_frame_processor)
    
    # 启动处理
    sync_processor.start_processing(lambda: example_simulation_generator(50))
    
    # 监控处理状态
    try:
        while sync_processor.is_active:
            status = sync_processor.get_status()
            print(f"状态: 活跃={status['is_active']}, 已消费={status['consumed_frames']}帧, "
                  f"速率={status['consumption_rate']:.1f}fps")
            time.sleep(2)
    except KeyboardInterrupt:
        print("用户中断")
        sync_processor.stop_processing()
        
    print("测试完成")