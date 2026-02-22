"""
辅助工具函数
Helper Utility Functions
"""

import time
import numpy as np
from functools import wraps
from typing import Dict, List, Optional, Tuple

class ThrottleLog:
    """节流日志类，避免频繁打印导致IO阻塞"""
    def __init__(self, interval=1.0):
        self.interval = interval
        self.last_print = 0

    def print(self, msg):
        now = time.perf_counter()
        if now - self.last_print >= self.interval:
            print(msg)
            self.last_print = now

def profile_function(func):
    """性能分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        from ..utils.config import CONFIG
        if not CONFIG.ENABLE_PERFORMANCE_PROFILING:
            return func(*args, **kwargs)
            
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            if hasattr(args[0], 'performance_capture') and args[0].performance_capture:
                args[0].performance_capture.record_function_time(func.__name__, end_time - start_time)
    return wrapper

def create_perspective_matrix(fov, aspect, near, far, out=None):
    """创建透视投影矩阵 - 原地操作优化版"""
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    nf = 1.0 / (near - far)

    if out is None:
        out = np.empty((4, 4), dtype=np.float32)

    # 原地填充矩阵
    out[0, 0] = f / aspect
    out[0, 1] = 0.0
    out[0, 2] = 0.0
    out[0, 3] = 0.0

    out[1, 0] = 0.0
    out[1, 1] = f
    out[1, 2] = 0.0
    out[1, 3] = 0.0

    out[2, 0] = 0.0
    out[2, 1] = 0.0
    out[2, 2] = (far + near) * nf
    out[2, 3] = -1.0

    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 2.0 * far * near * nf
    out[3, 3] = 0.0

    return out

def create_lookat_matrix(eye, target, up, out=None):
    """创建观察矩阵 - 向量化优化版"""
    if out is None:
        out = np.empty((4, 4), dtype=np.float32)

    # 向量化计算
    f = np.array(target, dtype=np.float32) - np.array(eye, dtype=np.float32)
    f_norm = np.linalg.norm(f)
    if f_norm > 1e-8:
        f /= f_norm

    up_arr = np.array(up, dtype=np.float32)
    s = np.cross(f, up_arr)
    s_norm = np.linalg.norm(s)
    if s_norm > 1e-8:
        s /= s_norm

    u = np.cross(s, f)

    # 原地填充矩阵
    out[0, 0] = s[0]
    out[0, 1] = u[0]
    out[0, 2] = -f[0]
    out[0, 3] = 0.0

    out[1, 0] = s[1]
    out[1, 1] = u[1]
    out[1, 2] = -f[1]
    out[1, 3] = 0.0

    out[2, 0] = s[2]
    out[2, 1] = u[2]
    out[2, 2] = -f[2]
    out[2, 3] = 0.0

    dot_s_eye = np.dot(s, eye)
    dot_u_eye = np.dot(u, eye)
    dot_f_eye = np.dot(f, eye)

    out[3, 0] = -dot_s_eye
    out[3, 1] = -dot_u_eye
    out[3, 2] = dot_f_eye
    out[3, 3] = 1.0

    return out

def calculate_camera_position(distance, angle_x, angle_y, target):
    """计算相机位置"""
    cam_x = target[0] + distance * np.sin(angle_x) * np.cos(angle_y)
    cam_y = target[1] + distance * np.sin(angle_y)
    cam_z = target[2] + distance * np.cos(angle_x) * np.cos(angle_y)
    return [cam_x, cam_y, cam_z]

def safe_copy_arrays(source, destination, count):
    """安全的数组拷贝"""
    try:
        np.copyto(destination[:count], source[:count], casting="same_kind")
        return True
    except Exception as e:
        print(f"Array copy error: {e}")
        return False

def validate_buffer_shape(buffer, expected_shape, buffer_name="Buffer"):
    """验证缓冲区形状"""
    if buffer.ndim != len(expected_shape):
        raise ValueError(f"{buffer_name} dimension mismatch: {buffer.ndim}, expected {len(expected_shape)}")
    
    for i, (actual, expected) in enumerate(zip(buffer.shape, expected_shape)):
        if expected > 0 and actual < expected:
            raise ValueError(f"{buffer_name} dimension {i} size mismatch: {actual}, expected at least {expected}")

def get_system_stats_formatted(stats_dict):
    """格式化系统统计信息"""
    formatted_stats = {
        'frame': stats_dict.get('frame', 0),
        'particle_count': stats_dict.get('particle_count', 0),
        'mesh_vertices': stats_dict.get('mesh_vertices', 0),
        'mesh_triangles': stats_dict.get('mesh_triangles', 0),
        'memory_usage_mb': round(stats_dict.get('memory_usage_mb', 0), 2),
        'fps': round(stats_dict.get('performance', {}).get('current_fps', 0), 1),
        'avg_frame_time_ms': round(stats_dict.get('performance', {}).get('avg_frame_time', 0) * 1000, 2)
    }
    return formatted_stats