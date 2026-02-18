"""
Generator数据消费者模块
专门处理基于generator的帧数据流
"""

import time
import threading
import numpy as np
from typing import Generator, Callable, Optional, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class FrameInfo:
    """帧信息数据类"""
    frame_number: int
    data: np.ndarray
    timestamp: float
    processing_time: float = 0.0


class GeneratorFrameConsumer:
    """
    Generator帧数据消费者
    负责消费、处理和管理从generator来的帧数据
    """
    
    def __init__(self, max_history: int = 1000, buffer_size: int = 50):
        self.max_history = max_history
        self.buffer_size = buffer_size
        self.frame_history = deque(maxlen=max_history)
        self.current_frame_index = 0
        self.is_consuming = False
        self.is_paused = False
        self.consumption_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'dropped_frames': 0,
            'start_time': 0,
            'last_frame_time': 0
        }
        self.frame_processors = []
        self.control_callbacks = {}
        
    def add_frame_processor(self, processor_func: Callable[[FrameInfo], bool], priority: int = 0):
        """
        添加帧处理器
        
        Args:
            processor_func: 处理函数，接受FrameInfo返回bool
            priority: 处理优先级，数值越大优先级越高
        """
        self.frame_processors.append((priority, processor_func))
        # 按优先级排序
        self.frame_processors.sort(key=lambda x: x[0], reverse=True)
        
    def remove_frame_processor(self, processor_func: Callable):
        """移除帧处理器"""
        self.frame_processors = [(p, func) for p, func in self.frame_processors 
                               if func != processor_func]
        
    def register_control_callback(self, event_name: str, callback: Callable):
        """注册控制回调"""
        self.control_callbacks[event_name] = callback
        
    def _notify_control_event(self, event_name: str, *args, **kwargs):
        """通知控制事件"""
        if event_name in self.control_callbacks:
            try:
                self.control_callbacks[event_name](*args, **kwargs)
            except Exception as e:
                print(f"[FrameConsumer] 控制回调执行错误: {e}")
    
    def consume_frames(self, frame_generator: Generator[np.ndarray, None, None]) -> int:
        """
        消费帧数据生成器
        
        Args:
            frame_generator: 帧数据生成器
            
        Returns:
            int: 处理的帧数
        """
        self.is_consuming = True
        self.is_paused = False
        self.consumption_stats['start_time'] = time.time()
        self.consumption_stats['total_frames'] = 0
        self.consumption_stats['processed_frames'] = 0
        self.consumption_stats['dropped_frames'] = 0
        
        self._notify_control_event('consumption_started')
        
        try:
            frame_counter = 0
            for frame_data in frame_generator:
                if not self.is_consuming:
                    break
                    
                # 处理暂停状态
                while self.is_paused and self.is_consuming:
                    time.sleep(0.1)
                    
                if not self.is_consuming:
                    break
                
                frame_counter += 1
                process_start_time = time.time()
                
                # 创建帧信息对象
                frame_info = FrameInfo(
                    frame_number=frame_counter,
                    data=frame_data,
                    timestamp=time.time()
                )
                
                # 处理帧数据
                success = self._process_frame(frame_info)
                frame_info.processing_time = time.time() - process_start_time
                
                if success:
                    self.consumption_stats['processed_frames'] += 1
                    # 添加到历史记录
                    self.frame_history.append(frame_info)
                    self.current_frame_index = len(self.frame_history) - 1
                    
                    # 通知帧更新
                    self._notify_control_event('frame_updated', frame_info)
                else:
                    self.consumption_stats['dropped_frames'] += 1
                    print(f"[FrameConsumer] 帧 {frame_counter} 处理失败")
                
                self.consumption_stats['total_frames'] = frame_counter
                self.consumption_stats['last_frame_time'] = time.time()
                
                # 通知进度更新
                if frame_counter % 10 == 0:  # 每10帧通知一次
                    self._notify_control_event('progress_updated', self.get_consumption_stats())
                    
        except KeyboardInterrupt:
            print("[FrameConsumer] 收到中断信号")
        except Exception as e:
            print(f"[FrameConsumer] 消费过程中发生错误: {e}")
        finally:
            self.is_consuming = False
            self._notify_control_event('consumption_finished', self.get_consumption_stats())
            
        return self.consumption_stats['processed_frames']
    
    def _process_frame(self, frame_info: FrameInfo) -> bool:
        """
        处理单个帧
        
        Args:
            frame_info: 帧信息
            
        Returns:
            bool: 处理是否成功
        """
        try:
            # 依次调用所有处理器
            for _, processor in self.frame_processors:
                # 直接传递FrameInfo对象给所有处理器
                # 处理器应该自己决定如何处理数据
                if not processor(frame_info):
                    return False
                        
            return True
        except Exception as e:
            print(f"[FrameConsumer] 帧处理错误: {e}")
            return False
    
    def pause_consumption(self):
        """暂停消费"""
        self.is_paused = True
        self._notify_control_event('consumption_paused')
        
    def resume_consumption(self):
        """恢复消费"""
        self.is_paused = False
        self._notify_control_event('consumption_resumed')
        
    def stop_consumption(self):
        """停止消费"""
        self.is_consuming = False
        self.is_paused = False
        self._notify_control_event('consumption_stopped')
        
    def get_consumption_stats(self) -> dict:
        """获取消费统计信息"""
        stats = self.consumption_stats.copy()
        if stats['start_time'] > 0:
            elapsed_time = time.time() - stats['start_time']
            stats['elapsed_time'] = elapsed_time
            if elapsed_time > 0:
                stats['average_fps'] = stats['processed_frames'] / elapsed_time
            else:
                stats['average_fps'] = 0
        else:
            stats['elapsed_time'] = 0
            stats['average_fps'] = 0
            
        stats['current_frame'] = self.current_frame_index
        stats['history_size'] = len(self.frame_history)
        stats['is_active'] = self.is_consuming
        stats['is_paused'] = self.is_paused
        
        return stats
    
    def get_current_frame(self) -> Optional[FrameInfo]:
        """获取当前帧"""
        if self.frame_history and 0 <= self.current_frame_index < len(self.frame_history):
            return self.frame_history[self.current_frame_index]
        return None
    
    def get_frame_at_index(self, index: int) -> Optional[FrameInfo]:
        """获取指定索引的帧"""
        if self.frame_history and 0 <= index < len(self.frame_history):
            return self.frame_history[index]
        return None
    
    def set_current_frame_index(self, index: int) -> bool:
        """设置当前帧索引"""
        if 0 <= index < len(self.frame_history):
            self.current_frame_index = index
            current_frame = self.frame_history[index]
            self._notify_control_event('frame_navigated', current_frame)
            return True
        return False
    
    def next_frame(self) -> bool:
        """下一帧"""
        return self.set_current_frame_index(self.current_frame_index + 1)
    
    def previous_frame(self) -> bool:
        """上一帧"""
        return self.set_current_frame_index(self.current_frame_index - 1)
    
    def jump_to_frame(self, frame_number: int) -> bool:
        """跳转到指定帧号"""
        # 在历史记录中查找对应的帧号
        for i, frame_info in enumerate(self.frame_history):
            if frame_info.frame_number == frame_number:
                return self.set_current_frame_index(i)
        return False


class VisualizationFrameProcessor:
    """
    可视化专用帧处理器
    负责将帧数据转换为可视化格式
    """
    
    def __init__(self, z_layer: int = 7, color_mapper: Optional[Callable] = None):
        self.z_layer = z_layer
        self.color_mapper = color_mapper or self._default_color_mapper
        self.visualization_cache = {}
        
    def _default_color_mapper(self, ph_value: float) -> tuple:
        """默认颜色映射器"""
        ph = max(0, min(14, ph_value))
        if ph <= 7:
            # 酸性：红色到黄色
            t = ph / 7.0
            return (255, int(255 * t), 0, 255)
        else:
            # 碱性：黄色到蓝色
            t = (ph - 7.0) / 7.0
            return (int(255 * (1 - t)), int(255 * (1 - t * 0.5)), int(255 * t) + 50, 255)
    
    def process_frame(self, frame_info: FrameInfo) -> bool:
        """
        处理帧数据用于可视化
        
        Args:
            frame_info: 帧信息
            
        Returns:
            bool: 处理是否成功
        """
        try:
            data_3d = frame_info.data
            
            # 验证数据维度
            if data_3d.ndim != 3:
                print(f"[VisProcessor] 数据维度错误: {data_3d.ndim}")
                return False
                
            nz, ny, nx = data_3d.shape
            z_idx = max(0, min(self.z_layer, nz - 1))
            
            # 提取Z层切片
            slice_data = data_3d[z_idx, :, :]
            
            # 计算统计数据
            valid_data = slice_data[~np.isnan(slice_data)]
            if len(valid_data) > 0:
                stats = {
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data))
                }
            else:
                stats = {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
            
            # 生成颜色映射
            color_data = np.zeros((ny, nx, 4), dtype=np.uint8)
            for y in range(ny):
                for x in range(nx):
                    ph_value = slice_data[y, x]
                    if not np.isnan(ph_value):
                        color_data[y, x] = self.color_mapper(ph_value)
            
            # 缓存处理结果
            self.visualization_cache[frame_info.frame_number] = {
                'slice_data': slice_data,
                'color_data': color_data,
                'stats': stats,
                'z_layer': z_idx
            }
            
            return True
            
        except Exception as e:
            print(f"[VisProcessor] 可视化处理错误: {e}")
            return False
    
    def get_cached_visualization(self, frame_number: int) -> Optional[dict]:
        """获取缓存的可视化数据"""
        return self.visualization_cache.get(frame_number)


# 示例使用
def example_frame_processor(frame_info: FrameInfo) -> bool:
    """示例帧处理器"""
    try:
        data = frame_info.data
        frame_num = frame_info.frame_number
        
        # 基本验证
        if data is None or data.size == 0:
            return False
            
        # 计算统计信息
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            ph_min = np.min(valid_data)
            ph_max = np.max(valid_data)
            ph_avg = np.mean(valid_data)
            
            if frame_num % 20 == 0:  # 每20帧打印一次
                print(f"[示例处理器] 帧 {frame_num}: "
                      f"pH范围 [{ph_min:.2f}, {ph_max:.2f}], 平均 {ph_avg:.2f}")
        
        return True
    except Exception as e:
        print(f"[示例处理器] 处理错误: {e}")
        return False


if __name__ == "__main__":
    # 简单测试
    print("=== Generator消费者测试 ===")
    
    # 创建消费者
    consumer = GeneratorFrameConsumer(max_history=100)
    
    # 添加处理器
    consumer.add_frame_processor(example_frame_processor)
    
    # 注册回调
    def on_frame_update(frame_info):
        print(f"帧更新: {frame_info.frame_number}")
    
    consumer.register_control_callback('frame_updated', on_frame_update)
    
    # 创建测试数据生成器
    def test_generator():
        for i in range(50):
            # 生成测试数据
            test_data = np.random.uniform(6.0, 8.0, (15, 15, 15))
            yield test_data
            time.sleep(0.05)
    
    # 启动消费
    processed_count = consumer.consume_frames(test_generator())
    print(f"处理完成，共处理 {processed_count} 帧")
    
    # 查看统计
    stats = consumer.get_consumption_stats()
    print(f"统计信息: {stats}")