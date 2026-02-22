"""
PySide6 + ModernGL网格重建系统
使用PySide6作为GUI框架，ModernGL进行OpenGL渲染
"""

import sys
import numpy as np
import math
import time
import json
from typing import Dict, List, Optional, Tuple
from functools import wraps
import threading

from OpenGL.GL import glGetString
# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QCheckBox, QComboBox, QGroupBox,
    QTextEdit, QSplitter, QFrame, QTabWidget, QGridLayout
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject
from PySide6.QtGui import QFont
from PySide6.QtOpenGLWidgets import QOpenGLWidget

use_ssfr = True
# ModernGL for OpenGL rendering
import moderngl
from moderngl_window import WindowConfig
import moderngl_window as mglw
from PIL import Image

# PySPlisHSPlasH integration
import pysplishsplash as sph

# PySplashSurf
import pysplashsurf
from pyglet.gl import GL_VERSION
import pysplishsplash as sph
import pysplishsplash.Utilities.SceneLoaderStructs as Scenes

# Enable PySPlisHSPlasH
PYSPLISH_AVAILABLE = True
SPLASHSURF_AVAILABLE = True

# 导入性能分析组件
from performance_analyzer_component import (
    DetailedPerformanceCapture,
    profile_function,
    _ThrottleLog,
    ENABLE_PERFORMANCE_PROFILING,
    ENABLE_DETAILED_PERFORMANCE_CAPTURE,
    ENABLE_ADVANCED_PERFORMANCE_ANALYSIS,
    PERFORMANCE_HISTORY_SIZE,
    PERFORMANCE_SAMPLING_RATE,
    ENABLE_REALTIME_ANALYSIS,
    ENABLE_MEMORY_PROFILING,
    ENABLE_GPU_PROFILING
)

class _ThrottleLog:
    """节流日志类，避免频繁打印导致IO阻塞"""
    def __init__(self, interval=1.0):
        self.interval = interval
        self.last_print = 0

    def print(self, msg):
        import time
        now = time.perf_counter()
        if now - self.last_print >= self.interval:
            print(msg)
            self.last_print = now

def profile_function(func):
    """性能分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not ENABLE_PERFORMANCE_PROFILING:
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

class AdvancedPerformanceAnalyzer:
    """高级性能分析器 - 提供深度性能洞察和优化建议"""
    
    def __init__(self, capture_system):
        self.capture_system = capture_system
        self.analysis_cache = {}
        self.trend_analysis = {
            'performance_trends': [],
            'bottleneck_patterns': [],
            'optimization_impact': []
        }
        
    def analyze_performance_patterns(self):
        """分析性能模式和趋势"""
        if len(self.capture_system.metrics['frame_times']) < 50:
            return None
            
        recent_data = {}
        for metric_name, values in self.capture_system.metrics.items():
            if len(values) >= 50:
                recent_data[metric_name] = values[-50:]
                
        patterns = {
            'stability': self._analyze_stability(recent_data),
            'scaling': self._analyze_scaling(recent_data),
            'correlations': self._analyze_correlations(recent_data),
            'trends': self._analyze_trends(recent_data)
        }
        
        return patterns
        
    def _analyze_stability(self, data):
        """分析性能稳定性"""
        stability_metrics = {}
        for metric_name, values in data.items():
            if 'times' in metric_name:
                std_dev = np.std(values)
                mean_val = np.mean(values)
                cv = std_dev / mean_val if mean_val > 0 else 0  # 变异系数
                
                if cv < 0.1:
                    stability = 'very_stable'
                elif cv < 0.25:
                    stability = 'stable'
                elif cv < 0.5:
                    stability = 'moderately_unstable'
                else:
                    stability = 'unstable'
                    
                stability_metrics[metric_name] = {
                    'stability_level': stability,
                    'coefficient_of_variation': cv,
                    'standard_deviation': std_dev,
                    'mean': mean_val
                }
        
        return stability_metrics
        
    def _analyze_scaling(self, data):
        """分析性能随负载的变化"""
        scaling_analysis = {}
        
        # 分析粒子数量与性能的关系
        if 'particle_counts' in data and 'frame_times' in data:
            particles = data['particle_counts']
            frames = data['frame_times']
            
            if len(particles) == len(frames) and len(particles) > 10:
                # 计算相关性
                correlation = np.corrcoef(particles, frames)[0, 1]
                
                # 线性回归分析
                slope, intercept = np.polyfit(particles, frames, 1)
                
                scaling_analysis['particle_scaling'] = {
                    'correlation': correlation,
                    'slope': slope,  # 每增加一个粒子的时间成本
                    'intercept': intercept,
                    'scaling_type': 'linear' if abs(correlation) > 0.7 else 'non_linear'
                }
        
        return scaling_analysis
        
    def _analyze_correlations(self, data):
        """分析不同性能指标间的相关性"""
        correlations = {}
        time_metrics = [k for k in data.keys() if 'times' in k]
        
        if len(time_metrics) >= 2:
            # 创建相关性矩阵
            metric_data = np.array([data[k] for k in time_metrics]).T
            corr_matrix = np.corrcoef(metric_data.T)
            
            correlations['time_correlations'] = {}
            for i, metric1 in enumerate(time_metrics):
                for j, metric2 in enumerate(time_metrics):
                    if i < j:  # 避免重复和自相关
                        corr_value = corr_matrix[i, j]
                        correlations['time_correlations'][f'{metric1}_vs_{metric2}'] = {
                            'correlation': corr_value,
                            'strength': self._classify_correlation(abs(corr_value))
                        }
        
        return correlations
        
    def _analyze_trends(self, data):
        """分析性能趋势"""
        trends = {}
        
        for metric_name, values in data.items():
            if len(values) >= 20:
                # 移动平均分析
                window_size = min(10, len(values) // 2)
                moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                
                # 趋势分析
                x = np.arange(len(moving_avg))
                slope, _ = np.polyfit(x, moving_avg, 1)
                
                # 判断趋势方向
                if slope > 0.001:
                    trend = 'increasing'
                elif slope < -0.001:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                trends[metric_name] = {
                    'trend': trend,
                    'slope': slope,
                    'recent_average': np.mean(values[-10:]),
                    'overall_average': np.mean(values)
                }
        
        return trends
        
    def _classify_correlation(self, corr_value):
        """分类相关性强弱"""
        if abs(corr_value) >= 0.8:
            return 'very_strong'
        elif abs(corr_value) >= 0.6:
            return 'strong'
        elif abs(corr_value) >= 0.4:
            return 'moderate'
        elif abs(corr_value) >= 0.2:
            return 'weak'
        else:
            return 'very_weak'

    def generate_advanced_recommendations(self, patterns):
        """生成高级优化建议"""
        recommendations = []
        
        if not patterns:
            return recommendations
            
        # 基于稳定性分析的建议
        if 'stability' in patterns:
            unstable_metrics = [k for k, v in patterns['stability'].items() 
                              if v['stability_level'] in ['unstable', 'moderately_unstable']]
            if unstable_metrics:
                recommendations.append({
                    'category': 'stability',
                    'priority': 'high',
                    'message': f'检测到以下组件性能不稳定: {", ".join(unstable_metrics)}',
                    'actions': [
                        '检查是否存在竞争条件或同步问题',
                        '优化算法以减少波动性',
                        '考虑增加缓存机制',
                        '分析是否受外部因素影响'
                    ]
                })
        
        # 基于扩展性分析的建议
        if 'scaling' in patterns and 'particle_scaling' in patterns['scaling']:
            scaling = patterns['scaling']['particle_scaling']
            if scaling['scaling_type'] == 'linear' and scaling['slope'] > 0.0001:
                recommendations.append({
                    'category': 'scalability',
                    'priority': 'medium',
                    'message': f'粒子系统呈现线性扩展，每粒子增加约{scaling["slope"]*1000:.2f}ms计算时间',
                    'actions': [
                        '考虑使用空间分区优化邻居搜索',
                        '实现粒子LOD系统',
                        '评估是否可以使用并行计算',
                        '优化数据结构访问模式'
                    ]
                })
        
        # 基于趋势分析的建议
        if 'trends' in patterns:
            degrading_metrics = [k for k, v in patterns['trends'].items() 
                               if v['trend'] == 'increasing']
            if degrading_metrics:
                recommendations.append({
                    'category': 'performance_degradation',
                    'priority': 'high',
                    'message': f'检测到性能逐渐恶化: {", ".join(degrading_metrics)}',
                    'actions': [
                        '立即调查性能退化原因',
                        '检查内存泄漏',
                        '分析算法复杂度是否随时间增长',
                        '考虑定期重启或重置策略'
                    ]
                })
        
        return recommendations


class DetailedPerformanceCapture:
    """详细的性能捕捉和分析系统"""
    
    def __init__(self, history_size=PERFORMANCE_HISTORY_SIZE):
        self.history_size = history_size
        self.sampling_rate = PERFORMANCE_SAMPLING_RATE
        self.sample_counter = 0
        
        # 高级分析器
        self.advanced_analyzer = AdvancedPerformanceAnalyzer(self) if ENABLE_ADVANCED_PERFORMANCE_ANALYSIS else None
        self.last_analysis_time = 0
        self.analysis_interval = 2.0  # 每2秒进行一次高级分析
        
        # 性能指标存储
        self.metrics = {
            'frame_times': [],           # 帧时间
            'render_times': [],          # 渲染时间
            'physics_times': [],         # 物理计算时间
            'mesh_times': [],            # 网格生成时间
            'gpu_upload_times': [],      # GPU上传时间
            'particle_counts': [],       # 粒子数量
            'vertex_counts': [],         # 顶点数量
            'triangle_counts': [],       # 三角形数量
            'memory_usage': [],          # 内存使用
            'fps_history': [],           # FPS历史
            'cpu_usage': [],             # CPU使用率
            'gpu_usage': []              # GPU使用率（如果有）
        }
        
        # 实时统计
        self.real_time_stats = {
            'current_fps': 0,
            'avg_frame_time': 0,
            'max_frame_time': 0,
            'min_frame_time': float('inf'),
            'frame_count': 0,
            'start_time': time.time(),
            'last_frame_time': time.time()
        }
        
        # 性能瓶颈检测
        self.bottlenecks = {
            'physics_bottleneck': False,
            'rendering_bottleneck': False,
            'memory_bottleneck': False,
            'gpu_bottleneck': False
        }
        
        # 优化建议
        self.optimization_suggestions = []
        
        # 计时器缓存
        self._timers = {}
        self._frame_start_time = 0
        
    def start_frame_timer(self):
        """开始帧计时"""
        self._frame_start_time = time.perf_counter()
        
    def end_frame_timer(self):
        """结束帧计时并记录数据"""
        if self._frame_start_time == 0:
            return
            
        frame_time = time.perf_counter() - self._frame_start_time
        self._record_metric('frame_times', frame_time)
        
        # 更新实时统计
        self.real_time_stats['frame_count'] += 1
        current_time = time.time()
        
        # 计算FPS
        if current_time - self.real_time_stats['last_frame_time'] >= 1.0:
            elapsed = current_time - self.real_time_stats['start_time']
            self.real_time_stats['current_fps'] = self.real_time_stats['frame_count'] / elapsed
            self._record_metric('fps_history', self.real_time_stats['current_fps'])
            self.real_time_stats['last_frame_time'] = current_time
            
        # 更新帧时间统计
        self.real_time_stats['avg_frame_time'] = np.mean(self.metrics['frame_times'][-100:]) if self.metrics['frame_times'] else 0
        self.real_time_stats['max_frame_time'] = max(self.metrics['frame_times'][-100:]) if self.metrics['frame_times'] else 0
        if self.metrics['frame_times']:
            self.real_time_stats['min_frame_time'] = min(self.metrics['frame_times'][-100:])
        
        self._frame_start_time = 0
        
    def start_timer(self, timer_name):
        """开始特定任务计时"""
        self._timers[timer_name] = time.perf_counter()
        
    def end_timer(self, timer_name):
        """结束特定任务计时并记录"""
        if timer_name in self._timers:
            elapsed = time.perf_counter() - self._timers[timer_name]
            metric_name = f"{timer_name}_times"
            if metric_name in self.metrics:
                self._record_metric(metric_name, elapsed)
            del self._timers[timer_name]
            return elapsed
        return 0
        
    def record_function_time(self, function_name, elapsed_time):
        """记录函数执行时间"""
        metric_name = f"{function_name}_times"
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self._record_metric(metric_name, elapsed_time)
        
    def _record_metric(self, metric_name, value):
        """记录性能指标"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
            # 保持历史大小限制
            if len(self.metrics[metric_name]) > self.history_size:
                self.metrics[metric_name] = self.metrics[metric_name][-self.history_size:]
                
    def record_snapshot(self, snapshot_data):
        """记录系统快照数据"""
        # 记录各种系统状态
        if 'particle_count' in snapshot_data:
            self._record_metric('particle_counts', snapshot_data['particle_count'])
        if 'vertex_count' in snapshot_data:
            self._record_metric('vertex_counts', snapshot_data['vertex_count'])
        if 'triangle_count' in snapshot_data:
            self._record_metric('triangle_counts', snapshot_data['triangle_count'])
        if 'memory_usage_mb' in snapshot_data:
            self._record_metric('memory_usage', snapshot_data['memory_usage_mb'])
            
    def detect_bottlenecks(self):
        """检测性能瓶颈"""
        if len(self.metrics['frame_times']) < 10:
            return
            
        recent_frames = self.metrics['frame_times'][-100:]
        recent_physics = self.metrics['physics_times'][-100:] if self.metrics['physics_times'] else []
        recent_render = self.metrics['render_times'][-100:] if self.metrics['render_times'] else []
        recent_memory = self.metrics['memory_usage'][-100:] if self.metrics['memory_usage'] else []
        
        avg_frame_time = np.mean(recent_frames)
        avg_physics_time = np.mean(recent_physics) if recent_physics else 0
        avg_render_time = np.mean(recent_render) if recent_render else 0
        
        # 物理计算瓶颈检测
        if avg_physics_time > avg_frame_time * 0.6:  # 物理计算占帧时间60%以上
            self.bottlenecks['physics_bottleneck'] = True
        else:
            self.bottlenecks['physics_bottleneck'] = False
            
        # 渲染瓶颈检测
        if avg_render_time > avg_frame_time * 0.5:  # 渲染占帧时间50%以上
            self.bottlenecks['rendering_bottleneck'] = True
        else:
            self.bottlenecks['rendering_bottleneck'] = False
            
        # 内存瓶颈检测
        if recent_memory and np.mean(recent_memory) > 1000:  # 内存使用超过1GB
            self.bottlenecks['memory_bottleneck'] = True
        else:
            self.bottlenecks['memory_bottleneck'] = False
            
        # GPU瓶颈检测（基于GPU上传时间）
        recent_gpu = self.metrics['gpu_upload_times'][-100:] if self.metrics['gpu_upload_times'] else []
        if recent_gpu and np.mean(recent_gpu) > 0.005:  # GPU上传时间超过5ms
            self.bottlenecks['gpu_bottleneck'] = True
        else:
            self.bottlenecks['gpu_bottleneck'] = False
            
    def generate_optimization_suggestions(self):
        """生成优化建议"""
        self.optimization_suggestions = []
        
        # 基于瓶颈检测生成建议
        if self.bottlenecks['physics_bottleneck']:
            self.optimization_suggestions.append({
                'type': 'physics',
                'severity': 'high',
                'message': '物理计算成为性能瓶颈，建议优化SPH算法或降低粒子数量',
                'actions': [
                    '减少粒子数量',
                    '优化邻居搜索算法',
                    '调整网格分辨率',
                    '使用更高效的积分方法'
                ]
            })
            
        if self.bottlenecks['rendering_bottleneck']:
            self.optimization_suggestions.append({
                'type': 'rendering',
                'severity': 'medium',
                'message': '渲染性能不足，建议优化渲染管线',
                'actions': [
                    '降低渲染质量设置',
                    '优化着色器代码',
                    '减少批次数量',
                    '使用实例化渲染'
                ]
            })
            
        if self.bottlenecks['memory_bottleneck']:
            self.optimization_suggestions.append({
                'type': 'memory',
                'severity': 'high',
                'message': '内存使用过高，可能导致性能下降',
                'actions': [
                    '优化数据结构',
                    '及时释放不需要的资源',
                    '使用内存池',
                    '降低纹理分辨率'
                ]
            })
            
        if self.bottlenecks['gpu_bottleneck']:
            self.optimization_suggestions.append({
                'type': 'gpu',
                'severity': 'medium',
                'message': 'GPU传输成为瓶颈，建议优化数据传输',
                'actions': [
                    '批量传输数据',
                    '使用缓冲区对象',
                    '减少不必要的GPU调用',
                    '优化顶点格式'
                ]
            })
            
        # 基于具体指标的建议
        recent_particles = self.metrics['particle_counts'][-100:] if self.metrics['particle_counts'] else []
        if recent_particles and np.mean(recent_particles) > 10000:
            self.optimization_suggestions.append({
                'type': 'particles',
                'severity': 'medium',
                'message': f'当前粒子数量较高({int(np.mean(recent_particles))})，可能影响性能',
                'actions': [
                    '适当降低粒子半径',
                    '减少发射粒子数量',
                    '优化粒子生命周期管理'
                ]
            })
            
        # FPS相关建议
        if self.real_time_stats['current_fps'] < 30:
            self.optimization_suggestions.append({
                'type': 'general',
                'severity': 'high',
                'message': f'当前FPS较低({self.real_time_stats["current_fps"]:.1f})，需要整体性能优化',
                'actions': [
                    '综合调整各项参数',
                    '考虑硬件升级',
                    '优化算法复杂度',
                    '使用多线程处理'
                ]
            })
            
    def get_performance_report(self):
        """获取完整性能报告"""
        self.detect_bottlenecks()
        self.generate_optimization_suggestions()
        
        # 执行高级分析
        advanced_patterns = None
        advanced_recommendations = []
        if (self.advanced_analyzer and 
            time.time() - self.last_analysis_time > self.analysis_interval and
            len(self.metrics['frame_times']) >= 50):
            
            advanced_patterns = self.advanced_analyzer.analyze_performance_patterns()
            if advanced_patterns:
                advanced_recommendations = self.advanced_analyzer.generate_advanced_recommendations(advanced_patterns)
            self.last_analysis_time = time.time()
        
        # 计算统计信息
        stats = {}
        for metric_name, values in self.metrics.items():
            if values:
                stats[metric_name] = {
                    'avg': np.mean(values[-100:]),
                    'min': np.min(values[-100:]),
                    'max': np.max(values[-100:]),
                    'std': np.std(values[-100:]) if len(values) > 1 else 0
                }
            
        report = {
            'real_time_stats': self.real_time_stats.copy(),
            'statistics': stats,
            'bottlenecks': self.bottlenecks.copy(),
            'optimization_suggestions': self.optimization_suggestions.copy(),
            'advanced_analysis': {
                'patterns': advanced_patterns,
                'recommendations': advanced_recommendations,
                'analysis_timestamp': time.time() if advanced_patterns else None
            },
            'system_info': {
                'platform': sys.platform,
                'python_version': sys.version,
                'timestamp': time.time()
            }
        }
        
        return report
        
    def export_performance_data(self, filename=None):
        """导出性能数据用于分析"""
        if filename is None:
            filename = f"performance_data_{int(time.time())}.json"
            
        report = self.get_performance_report()
        
        # 添加原始数据
        report['raw_data'] = {
            'timestamps': [self.real_time_stats['start_time'] + i * 0.016 for i in range(len(self.metrics['frame_times']))],
            'frame_times_ms': [t * 1000 for t in self.metrics['frame_times']],
            'fps_values': self.metrics['fps_history']
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"性能数据已导出到: {filename}")
            return filename
        except Exception as e:
            print(f"导出性能数据失败: {e}")
            return None
            
    def clear_history(self):
        """清空历史数据"""
        for metric_name in self.metrics:
            self.metrics[metric_name] = []
        self.real_time_stats = {
            'current_fps': 0,
            'avg_frame_time': 0,
            'max_frame_time': 0,
            'min_frame_time': float('inf'),
            'frame_count': 0,
            'start_time': time.time(),
            'last_frame_time': time.time()
        }

# System parameters
MC_GRID_RES = 16

# 粒子半径：需与 PySPlisHSPlasH 一致，0.05 可在 2×2.9×2 流体块中
# 产生 ~11 600 个粒子，兼顾密度与性能
PARTICLE_RADIUS = 0.05

MAX_PARTICLES = 30000
H = PARTICLE_RADIUS * 2.0
GRID_REBUILD_INTERVAL = 1
MC_MESH_UPDATE_INTERVAL = 1

# SSFR shadow map resolution (fixed size for consistent quality & performance)
SHADOW_MAP_SIZE = 512

# Performance optimization constants
BATCH_SIZE = 1000  # 批量处理大小
VERTEX_BUFFER_SIZE = 50000  # 预分配顶点缓冲区大小
INDEX_BUFFER_SIZE = 100000  # 预分配索引缓冲区大小

class MeshSystem:
    """网格重建核心系统"""

    def __init__(self):
        # PySPlisHSPlasH components
        self.pysplish_simulation = None
        self.pysplish_time_manager = None
        self.pysplish_fluid_model = None

        # Manual boundary handling (fallback)
        self.manual_boundary_enabled = False
        self.boundary_planes = []

        # PySPlisHSPlasH configuration
        self.pysplish_config = {
            'time_step': 0.008,
            'gravity': [0.0, -9.8, 0.0],
            'viscosity': 0.01,
            'surface_tension': 0.05,
            'density0': 1000.0
        }

        # Particle data storage - 预分配批量缓冲区
        self.particle_positions = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)  # 使用float32提高性能
        self.particle_velocities = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)
        self.particle_accelerations = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)
        self.particle_materials = np.zeros(MAX_PARTICLES, dtype=np.int32)

        # 批量处理临时缓冲区
        self.temp_positions = np.empty((BATCH_SIZE, 3), dtype=np.float32)
        self.temp_velocities = np.empty((BATCH_SIZE, 3), dtype=np.float32)

        # System state
        self.current_particle_num = 0
        self.current_frame = 0
        self.last_grid_rebuild = -GRID_REBUILD_INTERVAL
        self.last_mesh_update = -MC_MESH_UPDATE_INTERVAL
        self.mesh_enabled = True

        # Boundary parameters
        self.boundary_min = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        self.boundary_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        # Boundary container components
        self.boundary_walls = []  # Store boundary wall objects
        self.boundary_size = self.boundary_max - self.boundary_min
        self.mc_cell_size = self.boundary_size / MC_GRID_RES

        # Grid structure for neighbor search optimization
        self.grid_resolution = np.array([10, 10, 10], dtype=int)
        self.grid_cell_size = self.boundary_size / self.grid_resolution
        self.grid_total_cells = np.prod(self.grid_resolution)
        self.particle_grid_indices = np.zeros(MAX_PARTICLES, dtype=int)
        self.grid_cell_particles = [[] for _ in range(self.grid_total_cells)]

        # Performance statistics
        self.performance_stats = {
            'grid_rebuild_time': 0.0,
            'force_computation_time': 0.0,
            'integration_time': 0.0,
            'mesh_generation_time': 0.0,
            'gpu_upload_time': 0.0,
            'data_processing_time': 0.0,
            'total_mesh_vertices': 0,
            'total_mesh_triangles': 0,
            'memory_usage_mb': 0.0
        }

        # 性能计时器缓存
        self._timer_cache = {
            'start_times': {},
            'accumulated_times': {}
        }

        # 详细的性能捕捉系统
        self.performance_capture = DetailedPerformanceCapture() if ENABLE_DETAILED_PERFORMANCE_CAPTURE else None
        
        # 性能分析标志
        self.frame_sample_counter = 0

        # PySplashSurf configuration
        if SPLASHSURF_AVAILABLE:
            self.splashsurf_config = {
                'particle_radius': PARTICLE_RADIUS,
                'rest_density': 1000.0,
                # 按建议调整：smoothing_length = 2.0~2.5 * particle_radius
                'smoothing_length': 2.5 * PARTICLE_RADIUS,
                # cube_size 设为 ~ particle_radius * (1.0~2.0) 的量级
                'cube_size': PARTICLE_RADIUS * 1.5,
                'iso_surface_threshold': 0.6,
                'enable_multithreading': True,
                'enable_simd': True,
                'enable_zsort': False,
            }

        # Initialize PySPlisHSPlasH
        global PYSPLISH_AVAILABLE
        if PYSPLISH_AVAILABLE:
            try:
                self._initialize_pysplish_components()
                print("✓ PySPlisHSPlasH components initialized successfully")
            except Exception as e:
                print(f"✗ PySPlisHSPlasH initialization failed: {e}")
                PYSPLISH_AVAILABLE = False

        # 物理仿真线程化初始化
        self._init_threading()

    def _setup_boundary_with_scenes_module(self):
        """使用官方Scenes模块设置边界（推荐方式）- 修复版"""

        # 根据 particleRadius 动态计算 mapResolution
        # 默认 particleRadius=0.025 时 mapResolution=[25,25,25]，
        # particleRadius 增大时，supportRadius 也增大，
        # 需要按比例提高分辨率以维持边界精度。
        #
        # 基准：radius=0.025 → res=25
        # 公式：res = max(25, int(25 * 0.025 / particleRadius * 2))
        # 当 particleRadius=0.05 时 → res = max(25, 25) = 25 ... 不够
        #
        # 更好的公式：确保每个 supportRadius 内至少有足够的采样
        # supportRadius = 4 * particleRadius
        # boundary box size ≈ [4, 6, 4] (scale)
        # 需要 cellSize ≈ supportRadius 量级
        # 所以 res ≈ boxSize / supportRadius ≈ boxSize / (4 * particleRadius)

        support_radius = 4.0 * PARTICLE_RADIUS
        # 边界 box 的 scale 是 [4, 6, 4]，实际范围需要的最小分辨率
        box_scale = [4.0, 6.0, 4.0]
        map_res = [
            max(25, int(box_scale[0] / support_radius) + 1),
            max(25, int(box_scale[1] / support_radius) + 1),
            max(25, int(box_scale[2] / support_radius) + 1),
        ]

        boundary_data = Scenes.BoundaryData()
        boundary_data.meshFile = "../models/UnitBox.obj"
        boundary_data.translation = [0.0, 3.0, 0.0]
        boundary_data.scale = [4.0, 6.0, 4.0]
        boundary_data.color = [0.1, 0.4, 0.5, 1.0]
        boundary_data.isWall = True
        boundary_data.mapInvert = True
        boundary_data.mapResolution = map_res  # 动态分辨率

        self.scene.boundaryModels.append(boundary_data)
        print(f"✓ Boundary container added (mapResolution={map_res}, supportRadius={support_radius:.3f})")

    # --- 修复2：在 _initialize_pysplish_components 中，
    #     确保 initBoundaryData + updateVMVelocity 在 particleRadius 设定之后正确执行，
    #     并且清除旧缓存（如果存在） ---

    def _initialize_pysplish_components(self):
        """Initialize PySPlisHSPlasH - 修复版：确保边界与 particleRadius 匹配"""

        self.pysplish_base = sph.Exec.SimulatorBase()
        self.pysplish_base.init(useGui=False, sceneFile=sph.Extras.Scenes.Empty)

        scene_config = sph.Exec.SceneConfiguration.getCurrent()
        self.scene = scene_config.getScene()

        # 关键：设置 particleRadius 必须在添加 boundary/fluid 之前
        self.scene.particleRadius = PARTICLE_RADIUS

        # 添加边界（使用动态 mapResolution）
        self._setup_boundary_with_scenes_module()

        # 添加流体块
        self._setup_fluid_blocks()

        # 禁用缓存，强制重新生成 Volume Map（重要！）
        # 因为旧缓存是用不同 particleRadius 生成的
        self.pysplish_base.setUseParticleCaching(False)

        # 初始化仿真（内部会调用 sim->init(scene.particleRadius, ...)）
        self.pysplish_base.initSimulation()

        self.pysplish_simulation = sph.Simulation.getCurrent()

        self.pysplish_simulation.setValueBool(
            self.pysplish_simulation.ENABLE_Z_SORT, False
        )
        print("✓ Disabled Z-Sort (eliminates periodic stutter)")

        # 验证 particleRadius 已正确传播到 Simulation 对象
        actual_radius = self.pysplish_simulation.getParticleRadius()
        actual_support = self.pysplish_simulation.getSupportRadius()
        print(f"  Simulation particleRadius={actual_radius}, supportRadius={actual_support}")
        assert abs(actual_radius - PARTICLE_RADIUS) < 1e-6, \
            f"particleRadius mismatch: expected {PARTICLE_RADIUS}, got {actual_radius}"

        # 配置仿真方法（DFSPH）
        if self.pysplish_simulation.getSimulationMethod() != -1:
            self.pysplish_simulation.setValueInt(self.pysplish_simulation.SIMULATION_METHOD, 4)
            print("  Set simulation method to DFSPH")

        # 初始化边界数据（Volume Map 生成发生在这里！）
        self.pysplish_base.getBoundarySimulator().initBoundaryData()

        # 对于 Bender2019，必须调用 updateVMVelocity
        self.pysplish_base.updateVMVelocity()
        print("  Boundary simulator initialized")

        # 配置 DFSPH 参数
        self._configure_dfsph_parameters()

        # 获取流体模型引用
        self.pysplish_fluid_model = self.pysplish_simulation.getFluidModel(0)

        if self.pysplish_fluid_model:
            particle_count = self.pysplish_fluid_model.numActiveParticles()
            print(f"  Fluid model initialized with {particle_count} particles")
            if particle_count == 0:
                print("  WARNING: Fluid model has 0 particles!")
        else:
            raise RuntimeError("Failed to get fluid model")

        num_boundary_models = self.pysplish_simulation.numberOfBoundaryModels()
        print(f"  Boundary models count: {num_boundary_models}")

        # 重新启用缓存（后续运行可以使用缓存）
        self.pysplish_base.setUseParticleCaching(True)

        print("  All components initialized successfully!")

    def _init_threading(self):
        """物理仿真线程化初始化"""
        # 物理仿真线程化
        self._physics_lock = threading.Lock()
        self._physics_thread = None
        self._physics_running = False
        self._physics_result_ready = False
        self._last_physics_time = 0.0
        self._physics_budget_ms = 17.0  # 每帧物理计算时间预算（毫秒）
        self._skip_physics_count = 0

        # 配置 CFL 和时间步长限制
        self._configure_timestep_limits()

    def _configure_timestep_limits(self):
        """配置时间步长限制，减少自适应时间步长导致的波动"""
        if not self.pysplish_simulation:
            return
        try:
            sim = self.pysplish_simulation
            # 设置最小时间步长，避免 CFL 过度缩小导致单步耗时暴增
            # SPlisHSPlasH 的 TimeManager 有 MIN_TIMESTEP / MAX_TIMESTEP
            tm = sph.TimeManager.getCurrent()
            # 设置合理的时间步长范围
            tm.setTimeStepSize(0.004)  # 初始时间步长

            # 通过 Simulation 的参数设置 CFL
            # CFL_METHOD: 0=none, 1=CFL, 2=CFL_with_pause
            # CFL_FACTOR: 通常 0.5
            # CFL_MIN_TIMESTEPSIZE / CFL_MAX_TIMESTEPSIZE
            try:
                sim.setValueInt(sim.CFL_METHOD, 1)  # 启用 CFL
                sim.setValueFloat(sim.CFL_FACTOR, 0.25)  # 降低 CFL 系数（更保守）
                sim.setValueFloat(sim.CFL_MIN_TIMESTEPSIZE, 0.0002)  # 最小时间步长
                sim.setValueFloat(sim.CFL_MAX_TIMESTEPSIZE, 0.005)  # 最大时间步长
                print("✓ Configured CFL time step limits")
            except Exception as e:
                print(f"⚠ CFL configuration: {e}")

        except Exception as e:
            print(f"⚠ Time step limit configuration: {e}")

    def _setup_fluid_blocks(self):
        """设置流体块（保证initSimulation内就创建fluid model）"""
        try:
            # 更紧凑、更厚：避免"平铺"
            self.scene.fluidBlocks.append(
                sph.Utilities.SceneLoaderStructs.FluidBlock(
                    id='Fluid',
                    boxMin=[-1.0, 0.3, -1.0],
                    boxMax=[ 1.0, 3.2,  1.0],
                    mode=0,
                    initialVelocity=[0.0, 0.0, 0.0]
                )
            )
            print("✓ Fluid block added to scene")

        except Exception as e:
            print(f"✗ Fluid block setup failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _configure_dfsph_parameters(self):
        """配置DFSPH参数（在fluid model创建后调用）"""
        try:
            # 配置流体模型参数
            if self.pysplish_simulation.numberOfFluidModels() > 0:
                fluid = self.pysplish_simulation.getFluidModel(0)
                if fluid:
                    # 设置流体密度（水的密度 = 1000 kg/m^3）
                    fluid.setValueFloat(fluid.DENSITY0, 1000.0)
                    print("✓ Configured fluid model density")

            # 配置时间步长参数
            time_step = self.pysplish_simulation.getTimeStep()
            if time_step:
                # DFSPH参数
                time_step.setValueUInt(time_step.MIN_ITERATIONS, 2)
                time_step.setValueUInt(time_step.MAX_ITERATIONS, 50)
                time_step.setValueFloat(time_step.MAX_ERROR, 0.05)
                time_step.setValueUInt(time_step.MAX_ITERATIONS_V, 50)
                time_step.setValueFloat(time_step.MAX_ERROR_V, 0.1)
                time_step.setValueBool(time_step.USE_DIVERGENCE_SOLVER, True)
                print("✓ Configured DFSPH time step parameters")

        except Exception as e:
            print(f"⚠ DFSPH parameter configuration warning: {e}")

    def compute_forces(self):
        """执行一步物理仿真 - 修复版：带时间预算控制"""
        try:
            start_time = time.perf_counter()

            # 验证仿真对象有效
            if not self.pysplish_simulation:
                raise RuntimeError("Simulation object is None")

            if self.pysplish_simulation.numberOfFluidModels() == 0:
                raise RuntimeError("No fluid models in simulation")

            # 验证流体模型有效
            fluid_model = self.pysplish_simulation.getFluidModel(0)
            if not fluid_model:
                raise RuntimeError("Fluid model is None")

            particle_count = fluid_model.numActiveParticles()

            if particle_count == 0:
                print("⚠ WARNING: No active particles before timeStep!")
                return
            
            # 确保 STOP_AT 设置得足够远，避免 timeStepNoGUI 返回 false
            
            # 执行时间步
            result = self.pysplish_base.timeStepNoGUI()

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.performance_stats['force_computation_time'] = elapsed_ms / 1000.0


            self._last_physics_time = elapsed_ms

        except Exception as e:
            print(f"✗ Error in compute_forces: {e}")
            import traceback
            traceback.print_exc()

    def _create_boundary_container(self):
        """Create box-shaped boundary container using PySPlisHSPlasH scene configuration"""
        if not PYSPLISH_AVAILABLE:
            return

        try:
            # Clear existing boundaries
            self.boundary_walls = []


            # Clear existing boundary models
            # self.scene.boundaryModels.clear()

            # Add a simple boundary box
            self.scene.boundaryModels.append(
                sph.Utilities.SceneLoaderStructs.BoundaryData(
                    meshFile="../models/UnitBox.obj",
                    translation=[0., 3.0, 0.],
                    scale=[4., 6., 4.],
                    color=[0.1, 0.4, 0.5, 1.0],
                    isWall=True,
                    mapInvert=True,
                    mapResolution=[25, 25, 25],
                    isDynamic=False
                )
            )

            # Store wall definitions for reference
            wall_definitions = [
                {'name': 'container_wall', 'type': 'box'}
            ]

            for wall_def in wall_definitions:
                self.boundary_walls.append({
                    'definition': wall_def,
                    'created': True
                })

            print(f"Created {len(self.boundary_walls)} boundary walls using PySPlisHSPlasH scene configuration")

        except Exception as e:
            print(f"Boundary container creation failed: {e}")
            # Fallback to manual boundary handling
            self._setup_manual_boundary_handling()

    def _setup_manual_boundary_handling(self):
        """设置手动边界处理作为备用方案"""
        print("Setting up manual boundary handling as fallback")
        raise NotImplementedError("Manual boundary handling not implemented")

    def _apply_manual_boundary_collision(self):
        """应用手动边界碰撞检测和响应"""
        if not hasattr(self, 'manual_boundary_enabled') or not self.manual_boundary_enabled:
            return

        damping = 0.8  # 阻尼系数

        for i in range(self.current_particle_num):
            pos = self.particle_positions[i]
            vel = self.particle_velocities[i]

            # 检查每个边界平面
            for plane in self.boundary_planes:
                normal = plane['normal']
                distance = plane['distance']

                # 计算点到平面的距离
                point_distance = np.dot(pos, normal) + distance

                # 如果粒子穿透了边界
                if point_distance < 0:
                    # 将粒子推回边界内
                    penetration_depth = abs(point_distance)
                    pos = pos - normal * point_distance

                    # 反射速度
                    vel = vel - (1.0 + damping) * np.dot(vel, normal) * normal

                    # 更新粒子状态
                    self.particle_positions[i] = pos
                    self.particle_velocities[i] = vel

    def _create_boundary_wall(self, wall_definition):
        """Create a single boundary wall using PySPlisHSPlasH scene loading approach"""
        try:
            # Since direct API usage is problematic, let's use the scene-based approach
            # PySPlisHSPlasH typically loads boundaries from scene files
            # For now, we'll rely on the default scene initialization which should include boundaries
            print(f"Boundary wall '{wall_definition['name']}' will be handled by default scene")

            # Store wall definition for reference
            self.boundary_walls.append({
                'definition': wall_definition,
                'created': True
            })

        except Exception as e:
            print(f"Boundary wall creation note: {e}")

    # Removed _create_box_mesh method since we're using scene-based boundary handling

    def initialize_fluid_drops(self):
        """Initialize multiple fluid drops"""
        # Let PySPlisHSPlasH use its default scene initialization
        # Sync initial particle data
        self.sync_particle_data()

        print(f"Using PySPlisHSPlasH default scene initialization, particle count: {self.current_particle_num}")
        return self.current_particle_num

    def sync_particle_data(self):
        t = time.perf_counter()
        fm = self.pysplish_fluid_model
        if fm is None:
            print("Warning: pysplish_fluid_model is None")
            return

        try:
            n_active = int(fm.numActiveParticles())
            self.current_particle_num = min(n_active, MAX_PARTICLES)
            n = self.current_particle_num
            if n == 0:
                return

            # 安全的粒子数据同步 - 每次都重新获取buffer并进行安全拷贝
            # 获取位置buffer
            pos_buffer = fm.getFieldBuffer("position")
            pos_view = np.asarray(pos_buffer)  # 让numpy决定是否需要复制
            
            # 获取速度buffer
            vel_buffer = fm.getFieldBuffer("velocity")
            vel_view = np.asarray(vel_buffer)  # 让numpy决定是否需要复制
            
            # 严格验证buffer形状和类型
            if pos_view.ndim != 2 or pos_view.shape[1] != 3:
                raise ValueError(f"Position buffer shape invalid: {pos_view.shape}, expected (N, 3)")
            if vel_view.ndim != 2 or vel_view.shape[1] != 3:
                raise ValueError(f"Velocity buffer shape invalid: {vel_view.shape}, expected (N, 3)")
            if pos_view.shape[0] < n or vel_view.shape[0] < n:
                raise ValueError(f"Buffer size insufficient: pos={pos_view.shape[0]}, vel={vel_view.shape[0]}, need={n}")
            
            # 确保数据类型为float32，如果不是则转换
            if pos_view.dtype != np.float32:
                pos_view = pos_view.astype(np.float32)
            if vel_view.dtype != np.float32:
                vel_view = vel_view.astype(np.float32)
            
            # 安全拷贝到预分配的数组中
            np.copyto(self.particle_positions[:n], pos_view[:n], casting="same_kind")
            np.copyto(self.particle_velocities[:n], vel_view[:n], casting="same_kind")

            # materials (vectorized)
            self.particle_materials[:n].fill(0)

        except Exception as e:
            print(f"Error syncing particle data: {e}")
            import traceback
            traceback.print_exc()
            self.current_particle_num = 0

        # print(f"Particle sync time: {time.perf_counter() - t:.4f} seconds")

    def step_simulation(self, dt: float = 0.008) -> Tuple[List, List]:
        """Execute one simulation step - 修复版：带跳帧机制"""
        # 性能采样控制
        self.frame_sample_counter += 1
        should_sample = (self.performance_capture and 
                        self.frame_sample_counter % self.performance_capture.sampling_rate == 0)
        
        # 开始帧计时
        if should_sample:
            self.performance_capture.start_frame_timer()
            
        self.current_frame += 1

        # 跳帧机制：如果上一帧物理计算太慢，跳过本帧的物理
        if self._skip_physics_count > 0 and self.current_frame % 2 == 0:
            # 只做数据同步，不做物理计算
            self.sync_particle_data()
            if should_sample:
                self.performance_capture.end_frame_timer()
            return [], []

        # 开始物理计算计时
        if should_sample:
            self.performance_capture.start_timer('physics')
            
        # Physics simulation
        forces_start = time.perf_counter()
        self.compute_forces()
        self.performance_stats['force_computation_time'] = time.perf_counter() - forces_start

        # Sync particle data from SPH solver
        self.sync_particle_data()

        # Apply boundary collision handling
        # self._apply_manual_boundary_collision()

        # 结束物理计算计时
        if should_sample:
            physics_time = self.performance_capture.end_timer('physics')
            self.performance_capture.record_snapshot({
                'particle_count': self.current_particle_num
            })

        # Conditional grid rebuilding
        if (self.current_frame - self.last_grid_rebuild) >= GRID_REBUILD_INTERVAL:
            grid_start = time.perf_counter()
            self.rebuild_spatial_grid()
            self.last_grid_rebuild = self.current_frame
            self.performance_stats['grid_rebuild_time'] = time.perf_counter() - grid_start

        # Conditional mesh update (only needed in non-SSFR mode)
        mesh_vertices, mesh_triangles = [], []
        if not use_ssfr:
            if (self.current_frame - self.last_mesh_update) >= MC_MESH_UPDATE_INTERVAL:
                # 开始网格生成计时
                if should_sample:
                    self.performance_capture.start_timer('mesh')
                    
                mesh_start = time.perf_counter()
                mesh_vertices, mesh_triangles = self.update_mesh()
                self.last_mesh_update = self.current_frame
                self.performance_stats['mesh_generation_time'] = time.perf_counter() - mesh_start
                
                # 结束网格生成计时
                if should_sample:
                    mesh_time = self.performance_capture.end_timer('mesh')
                    self.performance_capture.record_snapshot({
                        'vertex_count': len(mesh_vertices),
                        'triangle_count': len(mesh_triangles)
                    })
        
        # 结束帧计时
        if should_sample:
            self.performance_capture.end_frame_timer()
            
        return mesh_vertices, mesh_triangles
    def rebuild_spatial_grid(self):
        """Rebuild spatial grid structure for neighbor search optimization - 向量化优化版"""
        start_time = time.perf_counter()

        # 批量清理网格
        for cell_list in self.grid_cell_particles:
            cell_list.clear()

        if self.current_particle_num == 0:
            self.performance_stats['grid_rebuild_time'] = 0.0
            return

        # 向量化网格索引计算
        positions = self.particle_positions[:self.current_particle_num]

        # 批量计算网格坐标
        grid_coords = ((positions - self.boundary_min) / self.grid_cell_size).astype(np.int32)
        grid_coords = np.clip(grid_coords, 0, self.grid_resolution - 1)

        # 向量化计算1D索引
        cell_indices = (grid_coords[:, 0] +
                       grid_coords[:, 1] * self.grid_resolution[0] +
                       grid_coords[:, 2] * self.grid_resolution[0] * self.grid_resolution[1])

        # 批量分配粒子到网格单元
        valid_mask = cell_indices < len(self.grid_cell_particles)
        valid_indices = cell_indices[valid_mask]
        valid_particle_ids = np.arange(self.current_particle_num)[valid_mask]

        for cell_idx, particle_idx in zip(valid_indices, valid_particle_ids):
            self.grid_cell_particles[cell_idx].append(particle_idx)
            self.particle_grid_indices[particle_idx] = cell_idx

        self.performance_stats['grid_rebuild_time'] = time.perf_counter() - start_time

    def update_mesh(self) -> Tuple[List, List]:
        """Update mesh data"""
        if not self.mesh_enabled or self.current_particle_num == 0:
            return [], []

        return self.generate_mesh_splashsurf()

    def generate_mesh_splashsurf(self) -> Tuple[List, List]:
        """Generate high-quality mesh using PySplashSurf - 内存优化版"""
        if not SPLASHSURF_AVAILABLE or self.current_particle_num == 0:
            return [], []

        try:
            # 直接使用活动粒子（避免布尔索引开销）
            n = self.current_particle_num
            fluid_positions = self.particle_positions[:n]
            if n == 0:
                return [], []

            start_time = time.perf_counter()

            # Use PySplashSurf for surface reconstruction
            result = pysplashsurf.reconstruct_surface(
                particles=fluid_positions,
                particle_radius=self.splashsurf_config['particle_radius'],
                rest_density=self.splashsurf_config['rest_density'],
                smoothing_length=self.splashsurf_config['smoothing_length'],
                cube_size=self.splashsurf_config['cube_size'],
                iso_surface_threshold=self.splashsurf_config['iso_surface_threshold'],
                multi_threading=True,
                simd=self.splashsurf_config['enable_simd']
            )

            # 提取网格数据（避免不必要的类型转换）
            mesh = result.mesh
            vertices = mesh.vertices  # 保持numpy数组格式
            triangles = mesh.triangles  # 保持numpy数组格式

            # 性能统计
            gen_time = time.perf_counter() - start_time
            self.performance_stats['mesh_generation_time'] = gen_time
            self.performance_stats['total_mesh_vertices'] = len(vertices)
            self.performance_stats['total_mesh_triangles'] = len(triangles)

            # 估算内存使用
            vertex_memory = vertices.nbytes / (1024 * 1024)  # MB
            triangle_memory = triangles.nbytes / (1024 * 1024)  # MB
            self.performance_stats['memory_usage_mb'] = vertex_memory + triangle_memory

            return vertices, triangles

        except Exception as e:
            print(f"PySplashSurf mesh generation failed: {e}")
            return [], []

    def get_system_stats(self) -> Dict:
        """Get system statistics - 增强版"""
        return {
            'frame': self.current_frame,
            'particle_count': self.current_particle_num,
            'mesh_vertices': self.performance_stats['total_mesh_vertices'],
            'mesh_triangles': self.performance_stats['total_mesh_triangles'],
            'memory_usage_mb': self.performance_stats['memory_usage_mb'],
            'performance': self.performance_stats.copy()
        }

class OpenGLRenderer:
    """ModernGL OpenGL渲染器"""

    def __init__(self, mesh_system):
        self.mesh_system = mesh_system
        self.ctx = None
        self.prog = None
        self.vao = None
        self.fbo = None

        # 节流日志
        self._tlog = _ThrottleLog(0.5)

        # 深度图保存相关变量
        self.last_depth_save_time = 0
        self.depth_save_interval = 1.0  # 每秒保存一次
        self.depth_frame_counter = 0

        # 顶点和索引计数
        self.vertex_count = 0
        self.index_count = 0

        # Camera parameters
        self.camera_distance = 10.0
        self.camera_angle_x = 0.5
        self.camera_angle_y = 0.3
        self.camera_target = [0.0, 1.5, 0.0]

        # Mesh data - 预分配缓冲区
        self.vertices = []
        self.triangles = []
        self.vertex_buffer = None
        self.index_buffer = None

        # 预分配的GPU缓冲区
        self.max_vertices = VERTEX_BUFFER_SIZE
        self.max_indices = INDEX_BUFFER_SIZE

        # 临时缓冲区用于数据准备
        self._temp_vertex_data = None
        self._temp_index_data = None
        self._temp_verts = None
        self._temp_normals = None
        self._temp_counts = None
        self._vertex_data = None

        # ---- SSFR (Screen Space Fluid Rendering) ----
        self.use_ssfr = True  # 默认使用SSFR（性能更好）

        # ★ 关键修改：全分辨率渲染，不再降采样
        self.ssfr_scale = 1.0  # ★ 从 0.5 改为 1.0（全分辨率）

        # ★ 多尺度滤波参数（参考 FluidSimulationDemo）
        self.ssfr_filter_iterations = 5   # 5×5 kernel 的迭代次数
        self.ssfr_filter_intervals = [1, 2, 4, 8, 16]  # 指数增长的采样间距
        self.ssfr_blur_sigma = 8.0     # ★ 从 5.0 提高到 8.0（匹配 FluidSimulationDemo）
        self.ssfr_depth_sigma = 0.025  # ★ 从 0.2 降低到 0.025（匹配 FluidSimulationDemo）
        self.ssfr_blur_radius = 4      # ★ kernel 半径改为 4（匹配 9×9 kernel）

        # 粒子渲染半径不变
        self.ssfr_particle_radius_world = PARTICLE_RADIUS * 1.2

        # runtime size cache
        self._ssfr_w = 0
        self._ssfr_h = 0

        # particle GPU resources
        self.particle_vbo = None
        self.particle_vao_depth = None
        self.particle_vao_thick = None
        self.particle_depth_prog = None
        self.particle_thick_prog = None
        self.particle_count = 0

        # ssfr passes
        self.bilateral_prog = None
        self.quad_vao_bilateral = None
        self.quad_vao_comp = None
        self.quad_vbo = None

        self.composite_prog = None

        # ssfr textures/fbos (分离depth/thickness pass)
        self.depth_tex0 = None
        self.depth_tex1 = None
        self.thick_tex = None
        self.depth_fbo0 = None
        self.depth_fbo1 = None
        self.thick_fbo = None

        # ---- Shadow mapping (light-space depth) ----
        self.shadow_scale = 1.0     # shadow map 分辨率倍率（1.0 = 跟 ssfr 一样大；也可独立）
        self.shadow_w = 0
        self.shadow_h = 0
        self.shadow_tex = None      # R32F depth (stored in color attachment)
        self.shadow_fbo = None
        self.shadow_prog = None
        self.shadow_vao = None

        # light config
        self.light_pos = np.array([3.5, 7.0, 3.5], dtype=np.float32)
        self.light_target = np.array([0.0, 1.5, 0.0], dtype=np.float32)

    def initialize(self, ctx):
        """Initialize OpenGL context and shaders"""
        self.ctx = ctx

        # --- 替换 shader：增加菲涅尔、透明、简单吸收，让水更像水 ---
        vertex_shader = '''
            #version 330
            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 model;

            layout(location = 0) in vec3 in_position;
            layout(location = 1) in vec3 in_normal;

            out vec3 v_normal_ws;
            out vec3 v_pos_ws;

            void main() {
                vec4 world_pos = model * vec4(in_position, 1.0);
                v_pos_ws = world_pos.xyz;

                // model 目前是单位阵，但保留正确写法
                mat3 nmat = mat3(transpose(inverse(model)));
                v_normal_ws = normalize(nmat * in_normal);

                gl_Position = projection * view * world_pos;
            }
        '''

        fragment_shader = '''
            #version 330
            uniform vec3 light_pos;
            uniform vec3 camera_pos;

            // 水参数
            uniform vec3 water_color;    // 基色（浅蓝）
            uniform float alpha_base;    // 基础透明度（越小越透明）
            uniform float fresnel_power; // 菲涅尔强度
            uniform float spec_power;    // 高光锐度
            uniform float absorption;    // 吸收强度（简单厚度近似）

            in vec3 v_normal_ws;
            in vec3 v_pos_ws;

            out vec4 frag_color;

            void main() {
                vec3 N = normalize(v_normal_ws);
                vec3 V = normalize(camera_pos - v_pos_ws);
                vec3 L = normalize(light_pos - v_pos_ws);

                // Blinn-Phong 半角向量高光（比你原来的 reflect 更稳定）
                vec3 H = normalize(L + V);

                float NdotL = max(dot(N, L), 0.0);
                float NdotV = max(dot(N, V), 0.0);
                float spec = pow(max(dot(N, H), 0.0), spec_power);

                // 菲涅尔（Schlick近似）
                float F0 = 0.02; // 水的基础反射率近似
                float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, fresnel_power);

                // 简单"吸收/厚度"近似：视角越刁钻看起来越"厚"（纯近似，但很像水）
                float thickness = 1.0 / max(NdotV, 0.15);
                float atten = exp(-absorption * (thickness - 1.0)); // 0~1

                vec3 diffuse = water_color * NdotL * atten;
                vec3 specular = vec3(1.0) * spec;

                // 让边缘更亮（反射更强），中心更透
                vec3 color = mix(diffuse, diffuse + specular, fresnel);

                // alpha：中心更透明，边缘更不透明（菲涅尔带来的"边缘更实"）
                float alpha = clamp(alpha_base + fresnel * 0.6, 0.02, 0.95);

                frag_color = vec4(color, alpha);
            }
        '''

        try:
            self.prog = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            print("✓ OpenGL shaders compiled successfully")
        except Exception as e:
            print(f"✗ Shader compilation failed: {e}")
            return False

        # 你原来的 mesh shader 仍保留（可用于debug），下面创建SSFR资源
        self._init_ssfr_resources()
        return True

    def _init_ssfr_resources(self):
        """替换 _init_ssfr_resources - 新增多尺度双边滤波 shader"""
        # ----------------------------------------------------------------
        # Fullscreen quad vertex shader (shared by bilateral + composite)
        # ----------------------------------------------------------------
        quad_vs = r'''
            #version 330
            in vec2 in_pos;
            out vec2 v_uv;
            void main() {
                v_uv = in_pos * 0.5 + 0.5;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
        '''

        # ----------------------------------------------------------------
        # Particle VS: perspective-correct point size from world radius
        # Outputs view-space center position for sphere impostor in FS
        # ----------------------------------------------------------------
        particle_vs = r'''
        #version 330
        uniform mat4 projection;
        uniform mat4 view;
        uniform float u_particle_radius_world;
        uniform float u_viewport_h;
        uniform float u_point_scale;
        layout(location=0) in vec3 in_pos;
        out vec3 v_center_vs;
        void main() {
            vec4 view_pos = view * vec4(in_pos, 1.0);
            v_center_vs = view_pos.xyz;
            gl_Position = projection * view_pos;
            float z = max(-view_pos.z, 1e-4);
            float proj_y = projection[1][1];
            // 修正点大小计算，避免粒子过大
            float r_px = (proj_y * u_particle_radius_world / z) * (u_viewport_h * 1.2);  // 降低系数从0.5到0.3
            gl_PointSize = clamp(1.5 * r_px * u_point_scale, 1.0, 512.0);  // 降低倍数和最大值
        }
        '''

        # ----------------------------------------------------------------
        # Depth FS: stores LINEAR (view-space) depth as positive float.
        # Background = 0.0, fluid = positive camera-distance value.
        # Sphere impostor correction gives physically accurate depths.
        # ----------------------------------------------------------------
        particle_depth_fs = r'''
        #version 330
        uniform mat4 projection;
        uniform float u_particle_radius_world;
        in vec3 v_center_vs;
        out float out_depth;
        void main() {
            vec2 p = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(p, p);
            if (r2 > 1.0) discard;
            float zcap = sqrt(1.0 - r2);
            // Sphere surface in view space (z is negative for visible geometry)
            float surf_z = v_center_vs.z + u_particle_radius_world * zcap;
            // Linear depth = positive distance from camera
            out_depth = max(-surf_z, 0.0);
            // Update hardware depth buffer for proper particle-particle occlusion
            vec4 clip = projection * vec4(v_center_vs.xy, surf_z, 1.0);
            gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
        }
        '''

        # ----------------------------------------------------------------
        # Thickness FS: additive per-particle contribution (chord length)
        # ----------------------------------------------------------------
        particle_thick_fs = r'''
        #version 330
        uniform float u_particle_radius_world;
        out float out_thick;
        void main() {
            vec2 p = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(p, p);
            if (r2 > 1.0) discard;
            // 2 * radius * cos(theta) gives chord length through sphere
            out_thick = 2.0 * u_particle_radius_world * sqrt(1.0 - r2);
        }
        '''

        # ================================================================
        # ★★★ 关键修复：多尺度 2D 双边滤波 shader ★★★
        # 参考 FluidSimulationDemo 的 BlurZ.comp 设计：
        # - 使用 2D kernel（不是可分离 H+V）
        # - 权重 = exp(-d_spatial²/(2σ₁²) - d_depth²/(2σ₂²))
        # - 通过 u_filter_interval 控制采样间距实现多尺度
        # ================================================================
        bilateral_fs = r'''
        #version 330
        uniform sampler2D u_depth;
        uniform vec2 u_texel;
        uniform int u_filter_interval;
        uniform float u_sigma_xy;     // spatial sigma (pixels)
        uniform float u_sigma_z;      // depth sigma (world units)
        uniform int u_kernel_radius;  // half-kernel size (2 for 5x5, 4 for 9x9)
        in vec2 v_uv;
        out float frag_out;
        void main() {
            float center = texture(u_depth, v_uv).r;
            // Background: no fluid here
            if (center < 1e-4) { frag_out = 0.0; return; }

            float sum = 0.0;
            float wsum = 0.0;

            // 2D kernel loop（参考 FluidSimulationDemo 的 BlurZ.comp）
            for (int dy = -u_kernel_radius; dy <= u_kernel_radius; dy++) {
                for (int dx = -u_kernel_radius; dx <= u_kernel_radius; dx++) {
                    // 多尺度：采样间距 = kernel offset * filter_interval
                    vec2 offset = vec2(float(dx), float(dy)) * float(u_filter_interval);
                    vec2 uv = v_uv + offset * u_texel;

                    float d = texture(u_depth, uv).r;
                    // Skip background pixels
                    if (d < 1e-4) continue;

                    // Spatial weight: based on pixel distance
                    float d_spatial = length(offset);
                    float w_spatial = exp(-(d_spatial * d_spatial) / (2.0 * u_sigma_xy * u_sigma_xy));

                    // Depth weight: bilateral term
                    float d_depth = abs(d - center);
                    float w_depth = exp(-(d_depth * d_depth) / (2.0 * u_sigma_z * u_sigma_z));

                    float w = w_spatial * w_depth;
                    sum += d * w;
                    wsum += w;
                }
            }

            frag_out = (wsum > 1e-6) ? (sum / wsum) : center;
        }
        '''

        # ----------------------------------------------------------------
        # Shadow depth pass: from light's perspective
        # ----------------------------------------------------------------
        shadow_vs = r'''
        #version 330
        uniform mat4 u_light_proj;
        uniform mat4 u_light_view;
        uniform float u_particle_radius_world;
        uniform float u_viewport_h;
        layout(location=0) in vec3 in_pos;
        out vec3 v_center_vs_light;
        void main() {
            vec4 view_pos = u_light_view * vec4(in_pos, 1.0);
            v_center_vs_light = view_pos.xyz;
            gl_Position = u_light_proj * view_pos;
            float z = max(-view_pos.z, 1e-3);
            float proj_y = u_light_proj[1][1];
            // 修正阴影贴图中的点大小计算
            float point_size = (proj_y * u_particle_radius_world / z) * (u_viewport_h * 0.3) * 1.2;  // 降低系数和倍数
            gl_PointSize = clamp(point_size, 1.0, 256.0);  // 降低最大值
        }
        '''

        shadow_fs = r'''
        #version 330
        uniform mat4 u_light_proj;
        uniform float u_particle_radius_world;
        in vec3 v_center_vs_light;
        out float out_depth;
        void main() {
            vec2 p = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(p, p);
            if (r2 > 1.0) discard;
            float zcap = sqrt(1.0 - r2);
            // Sphere impostor correction for shadow map too
            float surf_z = v_center_vs_light.z + u_particle_radius_world * zcap;
            vec4 clip = u_light_proj * vec4(v_center_vs_light.xy, surf_z, 1.0);
            out_depth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
            gl_FragDepth = out_depth;
        }
        '''

        # ----------------------------------------------------------------
        # Composite FS — full photorealistic water shader
        # Inspired by FluidSimulationDemo (xiaodoubao22):
        #   • Reconstruct view-space position from linear depth
        #   • Accurate normal from cross product of position gradients
        #   • Schlick Fresnel (reflection / refraction mix)
        #   • Beer-Lambert absorption (thickness buffer)
        #   • Procedural sky environment (no cubemap asset needed)
        #   • Shadow map sampling with bias
        #   • Approximate caustics from thickness gradient
        # ----------------------------------------------------------------
        comp_fs = r'''
        #version 330
        uniform sampler2D u_depth;      // linear depth (positive = camera distance)
        uniform sampler2D u_thick;      // thickness buffer
        uniform sampler2D u_shadow;     // shadow map (hardware depth 0-1)
        uniform vec2 u_texel;           // 1/viewport_size for normal reconstruction
        uniform vec2 u_shadow_texel;    // 1/shadow_map_size for PCF kernel

        // Camera: projection factors for position reconstruction
        uniform float u_proj_x;         // projection[0][0] = f/aspect
        uniform float u_proj_y;         // projection[1][1] = f
        uniform mat4 u_view_inv;        // inverse view matrix: view-space -> world-space

        // Light
        uniform mat4 u_light_vp;        // light projection * light view
        uniform vec3 u_light_dir_ws;    // normalised light direction in world space

        // Water material
        uniform vec3  u_water_color;
        uniform float u_alpha_base;
        uniform float u_fresnel_power;
        uniform float u_spec_power;
        uniform float u_sigma_a;        // Beer-Lambert absorption coefficient
        uniform float u_sigma_s;        // scattering coefficient

        // Effect knobs
        uniform float u_shadow_strength;
        uniform float u_caustic_strength;
        uniform float u_caustic_power;

        in vec2 v_uv;
        out vec4 frag_color;

        // ------ Position reconstruction ----------------------------------
        vec3 vs_pos(vec2 uv, float lin_d) {
            vec2 ndc = uv * 2.0 - 1.0;
            return vec3(ndc.x * lin_d / u_proj_x,
                        ndc.y * lin_d / u_proj_y,
                        -lin_d);  // view-space Z is negative for visible geometry
        }

        // ★★★ 法线重建 - 参考 FluidSimulationDemo ★★★
        // 关键改进：使用 2 像素间距采样，选择深度变化更小的方向
        vec3 vs_normal(vec2 uv, float dc) {
            vec3 pc = vs_pos(uv, dc);
        
            // ★ 使用 2 像素间距（参考 FluidSimulationDemo 的 `* 2`）
            float dr = texture(u_depth, uv + vec2( u_texel.x * 2.0, 0.0)).r;
            float dl = texture(u_depth, uv - vec2( u_texel.x * 2.0, 0.0)).r;
            float du = texture(u_depth, uv + vec2( 0.0, u_texel.y * 2.0)).r;
            float dd = texture(u_depth, uv - vec2( 0.0, u_texel.y * 2.0)).r;
        
            vec3 pr = vs_pos(uv + vec2( u_texel.x * 2.0, 0.0), dr);
            vec3 pl = vs_pos(uv - vec2( u_texel.x * 2.0, 0.0), dl);
            vec3 pu = vs_pos(uv + vec2( 0.0, u_texel.y * 2.0), du);
            vec3 pd = vs_pos(uv - vec2( 0.0, u_texel.y * 2.0), dd);
        
            // 计算两个方向的切线
            vec3 tangentU_right = pr - pc;
            vec3 tangentU_left  = pc - pl;
            vec3 tangentV_up    = pu - pc;
            vec3 tangentV_down  = pc - pd;
        
            // ★ 选择深度变化更小的方向（参考 FluidSimulationDemo CalculateNormal）
            // 这避免了流体边缘的法线跳变
            bool use_left = (dl > 1e-4) && (dr < 1e-4 || abs(tangentU_left.z) < abs(tangentU_right.z));
            bool use_down = (dd > 1e-4) && (du < 1e-4 || abs(tangentV_down.z) < abs(tangentV_up.z));
        
            vec3 dx, dy;
        
            if (use_left && dl > 1e-4)        dx = normalize(tangentU_left);
            else if (dr > 1e-4)                dx = normalize(tangentU_right);
            else return vec3(0.0, 0.0, 1.0);
        
            if (use_down && dd > 1e-4)         dy = normalize(tangentV_down);
            else if (du > 1e-4)                dy = normalize(tangentV_up);
            else return vec3(0.0, 0.0, 1.0);
        
            vec3 n = cross(dx, dy);
            float len = length(n);
            if (len < 1e-8) return vec3(0.0, 0.0, 1.0);
            return n / len;
        }

        // ------ Procedural sky (replaces cubemap) ------------------------
        vec3 sky(vec3 dir) {
            float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
            vec3 zenith  = vec3(0.10, 0.30, 0.80);
            vec3 horizon = vec3(0.70, 0.85, 1.00);
            vec3 ground  = vec3(0.08, 0.08, 0.10);
            vec3 col = (dir.y >= 0.0)
                ? mix(horizon, zenith,  pow(t, 0.6))
                : mix(horizon, ground, pow(1.0 - t, 1.5));
            // Sun disc
            vec3 sun_dir = normalize(u_light_dir_ws);
            float sun = pow(max(dot(dir, sun_dir), 0.0), 256.0);
            col += vec3(1.0, 0.95, 0.80) * sun * 2.0;
            return col;
        }

        // ------ Shadow sampling ------------------------------------------
        float sample_shadow(vec3 pos_ws) {
            vec4 lp = u_light_vp * vec4(pos_ws, 1.0);
            vec3 ndc = lp.xyz / lp.w;
            vec2 suv = ndc.xy * 0.5 + 0.5;
            if (suv.x < 0.01 || suv.x > 0.99 || suv.y < 0.01 || suv.y > 0.99)
                return 1.0;
            float cur = ndc.z * 0.5 + 0.5;
            // 3x3 PCF kernel — u_shadow_texel keeps this in sync with shadow map size
            float shadow = 0.0;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    float s = texture(u_shadow, suv + vec2(dx, dy) * u_shadow_texel).r;
                    shadow += (s + 0.003 < cur) ? (1.0 - u_shadow_strength) : 1.0;
                }
            }
            return shadow / 9.0;
        }

        void main() {
            float lin_d = texture(u_depth, v_uv).r;
            if (lin_d < 1e-4) discard;

            // Reconstruct view-space position and normal
            vec3 pos_vs = vs_pos(v_uv, lin_d);
            vec3 N_vs   = vs_normal(v_uv, lin_d);

            // Transform to world space
            vec3 pos_ws = (u_view_inv * vec4(pos_vs, 1.0)).xyz;
            vec3 N_ws   = normalize(mat3(u_view_inv) * N_vs);
            vec3 cam_ws = (u_view_inv * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
            vec3 V      = normalize(cam_ws - pos_ws);

            float NdotV = max(dot(N_ws, V), 0.0);

            // Schlick Fresnel — water F0 ≈ 0.02
            float F0 = 0.02;
            float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, u_fresnel_power);
            fresnel = min(fresnel, 0.85);  // 防止纯白反射

            // Thickness & Beer-Lambert absorption
            float thick = clamp(texture(u_thick, v_uv).r, 0.0, 10.0);
            float trans  = exp(-u_sigma_a * thick);
            float scatter = (1.0 - trans) * u_sigma_s;

            // Lighting
            vec3  L = normalize(u_light_dir_ws);
            vec3  H = normalize(L + V);
            float NdotL = max(dot(N_ws, L), 0.0);
            float spec  = pow(max(dot(N_ws, H), 0.0), u_spec_power);

            // Shadow
            float shad = sample_shadow(pos_ws);

            // Fluid body color (absorption + scatter + diffuse)
            vec3 fluid_body = u_water_color * ((1.0 - trans) * 0.5 + scatter);
            fluid_body += u_water_color * 0.35 * NdotL * shad;

            // Environment reflection & refraction (procedural sky)
            vec3 refl_dir = reflect(-V, N_ws);
            vec3 refr_dir = refract(-V, N_ws, 1.0 / 1.333);
            vec3 env_refl = sky(refl_dir);
            vec3 env_refr = (length(refr_dir) > 0.5) ? sky(refr_dir) : sky(-V);

            // Blend refracted environment with fluid body by depth
            float fluid_opacity = clamp((1.0 - trans) * 2.0 + scatter, 0.0, 1.0);
            vec3 refr_color = mix(env_refr, fluid_body, fluid_opacity);

            // Final mix: refraction + Fresnel reflection
            vec3 color = mix(refr_color, env_refl, fresnel);

            // Specular highlight on top
            color += vec3(1.0) * spec * (0.3 + 0.7 * fresnel) * shad;

            // Caustics: brightness spikes at thickness gradient edges
            float tC = texture(u_thick, v_uv).r;
            float tR = texture(u_thick, v_uv + vec2( u_texel.x, 0.0)).r;
            float tU = texture(u_thick, v_uv + vec2( 0.0, u_texel.y)).r;
            float cgrad   = length(vec2(tR - tC, tU - tC));
            // 限制梯度值避免过度明亮
            float caustic_factor = clamp(cgrad * 4.0, 0.0, 1.0);  // 降低乘数从8.0到4.0
            float caustic = pow(caustic_factor, u_caustic_power) * u_caustic_strength;
            // 限制焦散贡献的最大值
            caustic = min(caustic, 0.4);
            color += caustic * vec3(1.0, 0.96, 0.82) * shad;

            // Alpha driven by thickness + Fresnel
            float alpha = clamp(u_alpha_base + (1.0 - trans) * 0.85 + fresnel * 0.25, 0.05, 0.98);

            frag_color = vec4(color, alpha);
        }
        '''

        # Compile programs
        self.particle_depth_prog = self.ctx.program(vertex_shader=particle_vs, fragment_shader=particle_depth_fs)
        self.particle_thick_prog = self.ctx.program(vertex_shader=particle_vs, fragment_shader=particle_thick_fs)
        self.bilateral_prog = self.ctx.program(vertex_shader=quad_vs, fragment_shader=bilateral_fs)
        self.composite_prog = self.ctx.program(vertex_shader=quad_vs, fragment_shader=comp_fs)
        self.shadow_prog = self.ctx.program(vertex_shader=shadow_vs, fragment_shader=shadow_fs)

        # Fullscreen quad
        quad = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
        ], dtype=np.float32)
        self.quad_vbo = self.ctx.buffer(quad.tobytes())

        self.quad_vao_bilateral = self.ctx.vertex_array(self.bilateral_prog, [(self.quad_vbo, '2f', 'in_pos')])
        self.quad_vao_comp = self.ctx.vertex_array(self.composite_prog, [(self.quad_vbo, '2f', 'in_pos')])

        # Particle VBO / VAOs (one VBO shared across passes for zero-copy)
        self.particle_vbo = self.ctx.buffer(reserve=MAX_PARTICLES * 3 * 4)
        self.particle_vao_depth = self.ctx.vertex_array(self.particle_depth_prog, [(self.particle_vbo, '3f', 'in_pos')])
        self.particle_vao_thick = self.ctx.vertex_array(self.particle_thick_prog, [(self.particle_vbo, '3f', 'in_pos')])
        self.shadow_vao = self.ctx.vertex_array(self.shadow_prog, [(self.particle_vbo, '3f', 'in_pos')])

    def update_mesh_data(self, vertices, triangles):
        """Update mesh vertex and index data - GPU优化版"""
        if vertices is None or triangles is None:
            return

        # vertices/triangles 来自 SplashSurf，通常已经是 numpy array
        if len(vertices) == 0 or len(triangles) == 0:
            self.vertices = []
            self.triangles = []
            self.vertex_count = 0
            self.index_count = 0
            return

        self.vertices = vertices
        self.triangles = triangles

        t0 = time.perf_counter()
        vertex_data = self._create_vertex_data(vertices, triangles)
        indices_u32 = np.asarray(triangles, dtype=np.uint32).reshape(-1)
        t_cpu = time.perf_counter() - t0

        vertex_bytes = vertex_data.nbytes
        index_bytes = indices_u32.nbytes

        recreate_buffers = (
            self.vertex_buffer is None or
            self.index_buffer is None or
            vertex_bytes > self.vertex_buffer.size or
            index_bytes > self.index_buffer.size
        )

        if recreate_buffers:
            if self.vertex_buffer:
                self.vertex_buffer.release()
            if self.index_buffer:
                self.index_buffer.release()
            if self.vao:
                self.vao.release()

            self.vertex_buffer = self.ctx.buffer(reserve=max(vertex_bytes, self.max_vertices * 24))
            self.index_buffer = self.ctx.buffer(reserve=max(index_bytes, self.max_indices * 4))

            self.vao = self.ctx.vertex_array(
                self.prog,
                [(self.vertex_buffer, '3f 3f', 0, 1)],
                self.index_buffer,
                index_element_size=4,
            )

        t1 = time.perf_counter()
        self.vertex_buffer.write(vertex_data.tobytes())
        self.index_buffer.write(indices_u32.tobytes())
        t_gpu = time.perf_counter() - t1

        self.vertex_count = int(vertex_data.shape[0])
        self.index_count = int(indices_u32.size)

        # 节流打印，避免 I/O 卡死
        self._tlog.print(
            f"upload: cpu_prep={t_cpu*1000:.1f}ms gpu_write={t_gpu*1000:.1f}ms "
            f"verts={self.vertex_count} idx={self.index_count}"
        )

        # 把 GPU 上传时间写回 stats（可选）
        self.mesh_system.performance_stats['gpu_upload_time'] = t_gpu

    def _ensure_ssfr_targets(self, width: int, height: int):
        w = max(1, int(width * self.ssfr_scale))
        h = max(1, int(height * self.ssfr_scale))
        if w == self._ssfr_w and h == self._ssfr_h and getattr(self, "depth_fbo0", None) is not None:
            return

        self._ssfr_w, self._ssfr_h = w, h

        # Release old resources
        for obj in (
            getattr(self, "depth_fbo0", None),
            getattr(self, "depth_fbo1", None),
            getattr(self, "thick_fbo", None),
            getattr(self, "depth_tex0", None),
            getattr(self, "depth_tex1", None),
            getattr(self, "thick_tex", None),
            getattr(self, "_depth_rb0", None),
            getattr(self, "_depth_rb1", None),
        ):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass

        # Linear depth color textures (R32F): background=0, fluid=positive distance
        self.depth_tex0 = self.ctx.texture((w, h), components=1, dtype='f4')
        self.depth_tex1 = self.ctx.texture((w, h), components=1, dtype='f4')
        for t in (self.depth_tex0, self.depth_tex1):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Depth renderbuffers: enable hardware depth-test for sphere impostor z-fighting
        self._depth_rb0 = self.ctx.depth_renderbuffer((w, h))
        self._depth_rb1 = self.ctx.depth_renderbuffer((w, h))

        # Thickness texture (R16F: sufficient for additive thickness, saves VRAM)
        self.thick_tex = self.ctx.texture((w, h), components=1, dtype='f2')
        self.thick_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # FBOs: depth passes get hardware depth buffer for proper occlusion
        self.depth_fbo0 = self.ctx.framebuffer(
            color_attachments=[self.depth_tex0], depth_attachment=self._depth_rb0)
        self.depth_fbo1 = self.ctx.framebuffer(
            color_attachments=[self.depth_tex1], depth_attachment=self._depth_rb1)
        # Thickness pass: no depth test (additive blending)
        self.thick_fbo = self.ctx.framebuffer(color_attachments=[self.thick_tex])

    def _ensure_shadow_targets(self, width: int, height: int):
        # Fixed shadow map resolution defined by SHADOW_MAP_SIZE constant
        w = SHADOW_MAP_SIZE
        h = SHADOW_MAP_SIZE
        if w == self.shadow_w and h == self.shadow_h and self.shadow_fbo is not None:
            return

        self.shadow_w, self.shadow_h = w, h

        # Release old shadow resources
        for obj in (self.shadow_fbo, self.shadow_tex, getattr(self, "_shadow_depth_rb", None)):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass

        self.shadow_tex = self.ctx.texture((w, h), components=1, dtype='f4')
        self.shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow_tex.repeat_x = False
        self.shadow_tex.repeat_y = False

        self._shadow_depth_rb = self.ctx.depth_renderbuffer((w, h))
        self.shadow_fbo = self.ctx.framebuffer(
            color_attachments=[self.shadow_tex], depth_attachment=self._shadow_depth_rb)

    def update_particles(self, positions: np.ndarray):
        """positions: shape (N,3) float32"""
        if positions is None:
            self.particle_count = 0
            return
        n = int(len(positions))
        self.particle_count = n
        if n == 0 or self.particle_vbo is None:
            return

        pos = np.asarray(positions, dtype=np.float32)
        # ensure contiguous
        if not pos.flags['C_CONTIGUOUS']:
            pos = np.ascontiguousarray(pos)

        byte_len = n * 3 * 4
        if byte_len > self.particle_vbo.size:
            # grow buffer (rare)
            self.particle_vbo.release()
            self.particle_vbo = self.ctx.buffer(reserve=max(byte_len, MAX_PARTICLES * 3 * 4))

            # rebuild ALL VAOs with the new VBO (包括shadow_vao)
            if self.particle_vao_depth:
                self.particle_vao_depth.release()
            if self.particle_vao_thick:
                self.particle_vao_thick.release()
            if self.shadow_vao:
                self.shadow_vao.release()

            self.particle_vao_depth = self.ctx.vertex_array(
                self.particle_depth_prog, [(self.particle_vbo, '3f', 'in_pos')]
            )
            self.particle_vao_thick = self.ctx.vertex_array(
                self.particle_thick_prog, [(self.particle_vbo, '3f', 'in_pos')]
            )
            # 修复：重建shadow_vao以避免访问已释放的buffer对象
            self.shadow_vao = self.ctx.vertex_array(
                self.shadow_prog, [(self.particle_vbo, '3f', 'in_pos')]
            )

        self.particle_vbo.write(pos[:n].tobytes())

    def _create_vertex_data(self, vertices, triangles):
        """
        更快/更正确的法线计算版本：
        - 直接使用 vertices 的 numpy 数组（避免 Python for-loop）
        - 用 np.add.at 做 scatter-add，避免高级索引 += 的大坑
        """
        verts = np.asarray(vertices, dtype=np.float32)
        tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
        vert_count = verts.shape[0]
        tri_count = tris.shape[0]

        if vert_count == 0 or tri_count == 0:
            if self._vertex_data is None or self._vertex_data.shape[0] < vert_count:
                self._vertex_data = np.empty((max(vert_count, 1), 6), dtype=np.float32)
            vd = self._vertex_data[:vert_count]
            if vert_count:
                vd[:, :3] = verts
                vd[:, 3:] = 0
            return vd

        # 预分配缓存
        if self._temp_normals is None or self._temp_normals.shape[0] < vert_count:
            self._temp_normals = np.zeros((vert_count, 3), dtype=np.float32)
            self._temp_counts = np.zeros((vert_count,), dtype=np.int32)
        normals = self._temp_normals[:vert_count]
        counts = self._temp_counts[:vert_count]
        normals.fill(0.0)
        counts.fill(0)

        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        lens = np.linalg.norm(face_normals, axis=1)
        mask = lens > 1e-12
        face_normals[mask] /= lens[mask][:, None]

        # scatter-add：对每个角点累加面法线
        np.add.at(normals, tris[:, 0], face_normals)
        np.add.at(normals, tris[:, 1], face_normals)
        np.add.at(normals, tris[:, 2], face_normals)
        np.add.at(counts, tris[:, 0], 1)
        np.add.at(counts, tris[:, 1], 1)
        np.add.at(counts, tris[:, 2], 1)

        # 平均并归一化
        m = counts > 0
        normals[m] /= counts[m][:, None]
        nlen = np.linalg.norm(normals, axis=1)
        m2 = nlen > 1e-12
        normals[m2] /= nlen[m2][:, None]

        if self._vertex_data is None or self._vertex_data.shape[0] < vert_count:
            self._vertex_data = np.empty((vert_count, 6), dtype=np.float32)
        vertex_data = self._vertex_data[:vert_count]
        vertex_data[:, :3] = verts
        vertex_data[:, 3:] = normals
        return vertex_data

    def render(self, width, height, screen_fbo=None):
        """Render the scene. screen_fbo must be the Qt/ModernGL final framebuffer captured in paintGL()."""
        if not self.ctx:
            return

        if self.use_ssfr:
            self._render_ssfr(width, height, screen_fbo)
            return

        # ---- 原网格渲染逻辑保留（debug用） ----
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.clear(0.1, 0.1, 0.15, 1.0)

        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        if self.vao is None or self.index_count == 0:
            return

        projection = self._perspective(60.0, width / height, 0.1, 100.0)
        cam_pos = self._calculate_camera_position()
        view = self._look_at(cam_pos, self.camera_target, [0.0, 1.0, 0.0])
        model = np.eye(4, dtype=np.float32)

        self.prog['projection'].write(projection.astype('f4').tobytes())
        self.prog['view'].write(view.astype('f4').tobytes())
        self.prog['model'].write(model.astype('f4').tobytes())
        self.prog['light_pos'].value = (5.0, 8.0, 5.0)
        self.prog['camera_pos'].value = tuple(cam_pos)

        # 水材质参数
        self.prog['water_color'].value = (0.25, 0.65, 0.95)
        self.prog['alpha_base'].value = 0.06
        self.prog['fresnel_power'].value = 5.0
        self.prog['spec_power'].value = 128.0
        self.prog['absorption'].value = 2.0

        # Render mesh：明确告诉 VAO 画多少个"顶点/索引元素"
        self.vao.render(moderngl.TRIANGLES, vertices=self.index_count)

    def _calculate_camera_position(self):
        """Calculate camera position based on angles and distance"""
        cam_x = self.camera_target[0] + self.camera_distance * math.sin(self.camera_angle_x) * math.cos(self.camera_angle_y)
        cam_y = self.camera_target[1] + self.camera_distance * math.sin(self.camera_angle_y)
        cam_z = self.camera_target[2] + self.camera_distance * math.cos(self.camera_angle_x) * math.cos(self.camera_angle_y)
        return [cam_x, cam_y, cam_z]

    def _perspective(self, fov, aspect, near, far, out=None):
        """Create perspective projection matrix - 原地操作优化版"""
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
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

    def _render_ssfr(self, width: int, height: int, screen_fbo):
        self._ensure_ssfr_targets(width, height)
        self._ensure_shadow_targets(width, height)

        near, far = 0.1, 100.0
        projection = self._perspective(60.0, width / height, near, far)
        cam_pos    = self._calculate_camera_position()
        view       = self._look_at(cam_pos, self.camera_target, [0.0, 1.0, 0.0])
        # Inverse view for world-space reconstruction in composite pass
        view_inv   = np.linalg.inv(view).astype(np.float32)

        r_world = float(self.ssfr_particle_radius_world)

        # 关键：启用着色器控制的点大小，否则 gl_PointSize 被忽略，粒子只有 1px
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # ----------------------------------------------------------------
        # Pass 1 – Depth (sphere impostor → linear depth in R32F color tex)
        # Hardware depth test enabled so closer spheres overwrite far ones
        # ----------------------------------------------------------------
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        self.depth_fbo0.use()
        self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)   # clear color=0 (background), depth=1 (far)

        if self.particle_count > 0:
            p = self.particle_depth_prog
            p['projection'].write(projection.tobytes())
            p['view'].write(view.tobytes())
            p['u_particle_radius_world'].value = r_world
            p['u_viewport_h'].value = float(self._ssfr_h)
            p['u_point_scale'].value = 1.0
            self.particle_vao_depth.render(mode=moderngl.POINTS, vertices=self.particle_count)

        # ================================================================
        # Pass 2 – ★★★ 多尺度 2D 双边滤波 ★★★
        # 参考 FluidSimulationDemo 的 DepthFilter::Filter：
        # 1. 用 5×5 kernel (radius=2) 做 5 次，interval = 1,2,4,8,16
        # 2. 用 9×9 kernel (radius=4) 做 1 次，interval = 1
        # ================================================================
        self.ctx.disable(moderngl.DEPTH_TEST)

        bp = self.bilateral_prog
        bp['u_texel'].value = (1.0 / self._ssfr_w, 1.0 / self._ssfr_h)
        bp['u_sigma_xy'].value = float(self.ssfr_blur_sigma)    # 8.0
        bp['u_sigma_z'].value  = float(self.ssfr_depth_sigma)   # 0.025

        # ---- 阶段 1: 5×5 kernel, 5 次迭代，间距指数增长 ----
        bp['u_kernel_radius'].value = 2  # 5×5 kernel
        for i, interval in enumerate(self.ssfr_filter_intervals):
            # Ping-pong: depth_fbo0 ↔ depth_fbo1
            src_tex = self.depth_tex0 if (i % 2 == 0) else self.depth_tex1
            dst_fbo = self.depth_fbo1 if (i % 2 == 0) else self.depth_fbo0

            dst_fbo.use()
            self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)

            bp['u_filter_interval'].value = interval
            src_tex.use(location=0)
            bp['u_depth'].value = 0
            self.quad_vao_bilateral.render(moderngl.TRIANGLES, vertices=6)

        # 5 次迭代后，结果在 depth_tex1 中（因为 5 是奇数，最后一次写到 fbo1）
        # ---- 阶段 2: 9×9 kernel, 1 次精细平滑 ----
        bp['u_kernel_radius'].value = 4  # 9×9 kernel
        bp['u_filter_interval'].value = 1

        self.depth_fbo0.use()
        self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        self.depth_tex1.use(location=0)
        bp['u_depth'].value = 0
        self.quad_vao_bilateral.render(moderngl.TRIANGLES, vertices=6)

        # 最终平滑后的深度图在 depth_tex0 中
        
        # ---- 添加深度图保存功能 ----
        current_time = time.time()
        if current_time - self.last_depth_save_time >= self.depth_save_interval:
            self._save_depth_texture()
            self.last_depth_save_time = current_time
            self.depth_frame_counter += 1

        # ----------------------------------------------------------------
        # Pass 3 – Thickness  (additive blending, no depth test)
        # ----------------------------------------------------------------
        self.thick_fbo.use()
        self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)

        if self.particle_count > 0:
            p = self.particle_thick_prog
            p['projection'].write(projection.tobytes())
            p['view'].write(view.tobytes())
            p['u_particle_radius_world'].value = r_world
            p['u_viewport_h'].value = float(self._ssfr_h)
            p['u_point_scale'].value = 1.0
            self.particle_vao_thick.render(mode=moderngl.POINTS, vertices=self.particle_count)

        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # ----------------------------------------------------------------
        # Pass 4 – Shadow map  (from light's perspective)
        # ----------------------------------------------------------------
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)

        light_proj = self._perspective(70.0, 1.0, 0.5, 60.0)
        light_view = self._look_at(self.light_pos.tolist(), self.light_target.tolist(), [0.0, 1.0, 0.0])
        light_vp   = (light_proj @ light_view).astype(np.float32)
        # World-space light direction (from light toward scene centre)
        light_dir_ws = (self.light_target - self.light_pos)
        light_dir_ws = light_dir_ws / (np.linalg.norm(light_dir_ws) + 1e-8)

        self.shadow_fbo.use()
        self.ctx.viewport = (0, 0, self.shadow_w, self.shadow_h)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)   # shadow map: 1.0 = nothing blocking

        if self.particle_count > 0:
            p = self.shadow_prog
            p['u_light_proj'].write(light_proj.tobytes())
            p['u_light_view'].write(light_view.tobytes())
            p['u_particle_radius_world'].value = r_world
            p['u_viewport_h'].value = float(self.shadow_h)
            self.shadow_vao.render(mode=moderngl.POINTS, vertices=self.particle_count)

        # ----------------------------------------------------------------
        # Pass 5 – Composite  (depth + thick + shadow → final water color)
        # ----------------------------------------------------------------
        screen_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.clear(0.06, 0.08, 0.12, 1.0)   # dark blue-grey background

        cp = self.composite_prog

        # Camera reconstruction uniforms
        cp['u_texel'].value    = (1.0 / self._ssfr_w, 1.0 / self._ssfr_h)
        cp['u_proj_x'].value   = float(projection[0, 0])
        cp['u_proj_y'].value   = float(projection[1, 1])
        cp['u_view_inv'].write(view_inv.tobytes())

        # Light uniforms
        cp['u_light_vp'].write(light_vp.tobytes())
        cp['u_light_dir_ws'].value  = tuple(light_dir_ws.astype(np.float32).tolist())
        cp['u_shadow_texel'].value  = (1.0 / self.shadow_w, 1.0 / self.shadow_h)

        # Water material
        cp['u_water_color'].value    = (0.04, 0.28, 0.60)   # 深蓝色水体
        cp['u_alpha_base'].value     = 0.08  # 稍微提高基础透明度
        cp['u_fresnel_power'].value  = 4.0   # 降低菲涅尔强度
        cp['u_spec_power'].value     = 150.0 # 降低高光锐度
        cp['u_sigma_a'].value        = 3.5   # 降低吸收系数
        cp['u_sigma_s'].value        = 0.15  # 降低散射系数

        # Effect knobs
        cp['u_shadow_strength'].value  = 0.55
        cp['u_caustic_strength'].value = 0.5  # 进一步降低焦散强度
        cp['u_caustic_power'].value    = 0.6  # 进一步降低焦散幂次

        # Bind textures
        self.depth_tex0.use(location=0)
        self.thick_tex.use(location=1)
        self.shadow_tex.use(location=2)
        cp['u_depth'].value  = 0
        cp['u_thick'].value  = 1
        cp['u_shadow'].value = 2

        self.quad_vao_comp.render(moderngl.TRIANGLES, vertices=6)

    def _look_at(self, eye, target, up, out=None):
        """Create look-at view matrix - 向量化优化版"""
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

    def render_default_scene(self, width, height):
        """渲染默认场景（坐标轴等）- 预分配优化版"""
        if not self.ctx or not self.prog:
            return

        try:
            # Clear screen
            # self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.clear(0.1, 0.1, 0.15, 1.0)

            # 显式绑定framebuffer
            fbo = self.ctx.detect_framebuffer()
            fbo.use()

            # 预分配相机矩阵缓存
            if not hasattr(self, '_cached_projection'):
                self._cached_projection = np.empty((4, 4), dtype=np.float32)
                self._cached_view = np.empty((4, 4), dtype=np.float32)
                self._cached_model = np.eye(4, dtype=np.float32)
                # 预分配坐标轴数据
                # self._axis_vertices = np.array([
                #     # X-axis
                #     0.0, 0.0, 0.0,  1.0, 0.0, 0.0,
                #     2.0, 0.0, 0.0,  1.0, 0.0, 0.0,
                #     # Y-axis
                #     0.0, 0.0, 0.0,  0.0, 1.0, 0.0,
                #     0.0, 2.0, 0.0,  0.0, 1.0, 0.0,
                #     # Z-axis
                #     0.0, 0.0, 0.0,  0.0, 0.0, 1.0,
                #     0.0, 0.0, 2.0,  0.0, 0.0, 1.0
                # ], dtype=np.float32)
                # self._axis_buffer = self.ctx.buffer(self._axis_vertices)
                # self._axis_vao = self.ctx.vertex_array(
                #     self.prog,
                #     [(self._axis_buffer, '3f 3f', 0, 1)]
                # )

            # 计算相机矩阵（使用预分配数组）
            self._perspective(60.0, width / height, 0.1, 100.0, out=self._cached_projection)
            camera_pos = self._calculate_camera_position()
            self._look_at(
                camera_pos,
                self.camera_target,
                [0.0, 1.0, 0.0],
                out=self._cached_view
            )

            # 设置uniforms
            self.prog['projection'].write(self._cached_projection.astype('f4').tobytes())
            self.prog['view'].write(self._cached_view.astype('f4').tobytes())
            self.prog['model'].write(self._cached_model.astype('f4').tobytes())
            self.prog['light_pos'].value = (5.0, 5.0, 5.0)
            self.prog['camera_pos'].value = tuple(camera_pos)

            # 渲染坐标轴（无需重新创建资源）
            # self._axis_vao.render(moderngl.LINES, vertices=6)

        except Exception as e:
            print(f"Default scene rendering error: {e}")

    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.vao:
            self.vao.release()
        if self.vertex_buffer:
            self.vertex_buffer.release()
        if self.index_buffer:
            self.index_buffer.release()
        if self.prog:
            self.prog.release()

        # Clean up SSFR resources
        for obj in (
            getattr(self, "depth_fbo0", None),
            getattr(self, "depth_fbo1", None),
            getattr(self, "thick_fbo", None),
            getattr(self, "depth_tex0", None),
            getattr(self, "depth_tex1", None),
            getattr(self, "thick_tex", None),
            getattr(self, "particle_vao_depth", None),
            getattr(self, "particle_vao_thick", None),
            getattr(self, "particle_depth_prog", None),
            getattr(self, "particle_thick_prog", None),
            getattr(self, "bilateral_prog", None),
            getattr(self, "quad_vao_bilateral", None),
            getattr(self, "quad_vao_comp", None),
        ):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass

    def _save_depth_texture(self):
        pass


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.mesh_system = MeshSystem()
        self.opengl_renderer = OpenGLRenderer(self.mesh_system)
        self.is_running = False
        self.is_paused = False
        self.fps_counter = 0
        self.last_fps_time = time.time()

        self.init_ui()
        self.setup_timer()

    def init_ui(self):
        """Initialize UI with separated panels"""
        self.setWindowTitle("PySide6 + ModernGL 网格重建系统")
        self.setGeometry(100, 100, 1400, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Left control panel (basic controls)
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=0)

        # Center area - OpenGL widget and performance panel splitter
        center_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # OpenGL widget (main rendering area)
        self.opengl_widget = QOpenGLWidgetWrapper(self.opengl_renderer)
        center_splitter.addWidget(self.opengl_widget)
        
        # Right performance analysis panel
        performance_panel = self.create_performance_panel()
        center_splitter.addWidget(performance_panel)
        
        # Set initial sizes (70% for OpenGL, 30% for performance)
        center_splitter.setSizes([700, 300])
        
        main_layout.addWidget(center_splitter, stretch=1)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("系统就绪")

    def create_control_panel(self):
        """Create control panel"""
        panel = QFrame()
        panel.setFixedWidth(300)
        panel.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #444444;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666666;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QSlider {
                background-color: #333333;
            }
            QCheckBox {
                color: white;
            }
            QComboBox {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666666;
            }
        """)

        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("控制面板")
        title.setStyleSheet("color: #4fc3f7; font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始模拟")
        self.start_btn.clicked.connect(self.start_simulation)
        self.pause_btn = QPushButton("暂停/继续")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_simulation)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.reset_btn)
        layout.addLayout(button_layout)

        # Speed control
        layout.addWidget(QLabel("播放速度:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 30)  # 0.1x to 3.0x
        self.speed_slider.setValue(10)  # 1.0x default
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("1.0x")
        self.speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.speed_label)

        # Mesh toggle
        self.mesh_toggle = QCheckBox("启用网格显示")
        self.mesh_toggle.setChecked(True)
        self.mesh_toggle.stateChanged.connect(self.toggle_mesh)
        layout.addWidget(self.mesh_toggle)

        # Rendering mode toggle
        self.render_mode_toggle = QCheckBox("使用SSFR渲染(直接粒子渲染)")
        self.render_mode_toggle.setChecked(True)
        self.render_mode_toggle.stateChanged.connect(self.toggle_render_mode)
        layout.addWidget(self.render_mode_toggle)

        # Boundary handling toggle
        self.boundary_toggle = QCheckBox("启用边界碰撞")
        self.boundary_toggle.setChecked(True)
        self.boundary_toggle.stateChanged.connect(self.toggle_boundary)
        layout.addWidget(self.boundary_toggle)

        # Display mode
        layout.addWidget(QLabel("显示模式:"))
        self.display_combo = QComboBox()
        self.display_combo.addItems(["实体", "线框", "点"])
        self.display_combo.setCurrentText("实体")
        layout.addWidget(self.display_combo)

        # Camera controls
        camera_group = QGroupBox("相机控制")
        camera_layout = QVBoxLayout()

        camera_layout.addWidget(QLabel("距离:"))
        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(20, 200)  # 2.0 to 20.0
        self.distance_slider.setValue(100)  # 10.0 default
        self.distance_slider.valueChanged.connect(self.update_camera)
        camera_layout.addWidget(self.distance_slider)

        camera_layout.addWidget(QLabel("水平角度:"))
        self.angle_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_x_slider.setRange(0, 314)  # 0 to π
        self.angle_x_slider.setValue(50)  # 0.5 radians
        self.angle_x_slider.valueChanged.connect(self.update_camera)
        camera_layout.addWidget(self.angle_x_slider)

        camera_layout.addWidget(QLabel("垂直角度:"))
        self.angle_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_y_slider.setRange(-157, 157)  # -π/2 to π/2
        self.angle_y_slider.setValue(30)  # 0.3 radians
        self.angle_y_slider.valueChanged.connect(self.update_camera)
        camera_layout.addWidget(self.angle_y_slider)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # SSFR 参数调节
        ssfr_group = QGroupBox("SSFR参数调节")
        ssfr_layout = QVBoxLayout()

        # 世界空间粒子半径
        ssfr_layout.addWidget(QLabel("粒子世界半径:"))
        self.ssfr_radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_radius_slider.setRange(5, 100)  # 0.05 to 1.00
        radius_slider_init = int(PARTICLE_RADIUS * 3.5 * 100)
        self.ssfr_radius_slider.setValue(radius_slider_init)
        self.ssfr_radius_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_radius_slider)
        self.ssfr_radius_label = QLabel(f"{PARTICLE_RADIUS * 3.5:.2f}")
        ssfr_layout.addWidget(self.ssfr_radius_label)

        # 模糊强度（空间sigma，像素）
        ssfr_layout.addWidget(QLabel("模糊强度:"))
        self.ssfr_sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_sigma_slider.setRange(5, 80)  # 0.5 to 8.0
        self.ssfr_sigma_slider.setValue(50)      # 5.0 default
        self.ssfr_sigma_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_sigma_slider)
        self.ssfr_sigma_label = QLabel("5.0")
        ssfr_layout.addWidget(self.ssfr_sigma_label)

        # 双边滤波深度sigma（世界空间单位）
        ssfr_layout.addWidget(QLabel("深度sigma(世界单位):"))
        self.ssfr_depth_sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_depth_sigma_slider.setRange(1, 100)  # 0.1 to 10.0
        self.ssfr_depth_sigma_slider.setValue(2)        # 0.2 default
        self.ssfr_depth_sigma_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_depth_sigma_slider)
        self.ssfr_depth_sigma_label = QLabel("0.2")
        ssfr_layout.addWidget(self.ssfr_depth_sigma_label)

        # 吸收系数
        ssfr_layout.addWidget(QLabel("吸收系数:"))
        self.ssfr_absorption_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_absorption_slider.setRange(5, 50)  # 0.5 to 5.0
        self.ssfr_absorption_slider.setValue(40)  # 4.0 default
        self.ssfr_absorption_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_absorption_slider)
        self.ssfr_absorption_label = QLabel("4.0")
        ssfr_layout.addWidget(self.ssfr_absorption_label)

        # 散射系数
        ssfr_layout.addWidget(QLabel("散射系数:"))
        self.ssfr_scatter_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_scatter_slider.setRange(1, 20)  # 0.1 to 2.0
        self.ssfr_scatter_slider.setValue(2)  # 0.2 default
        self.ssfr_scatter_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_scatter_slider)
        self.ssfr_scatter_label = QLabel("0.20")
        ssfr_layout.addWidget(self.ssfr_scatter_label)

        ssfr_group.setLayout(ssfr_layout)
        layout.addWidget(ssfr_group)

        # Basic statistics (minimal)
        stats_group = QGroupBox("基本统计")
        stats_layout = QVBoxLayout()

        self.frame_label = QLabel("帧: 0")
        self.particle_label = QLabel("粒子数: 0")
        self.fps_label = QLabel("FPS: 0")

        stats_layout.addWidget(self.frame_label)
        stats_layout.addWidget(self.particle_label)
        stats_layout.addWidget(self.fps_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()
        return panel

    def create_performance_panel(self):
        """Create performance analysis panel for the right side"""
        panel = QFrame()
        panel.setMinimumHeight(200)
        panel.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #444444;
            }
            QLabel {
                color: white;
                font-size: 11px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666666;
                padding: 6px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QTabWidget::pane {
                border: 1px solid #444444;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3a3a3a;
                color: white;
                padding: 8px 15px;
                margin: 2px;
                border: 1px solid #555555;
            }
            QTabBar::tab:selected {
                background-color: #4a4a4a;
                border-bottom: 2px solid #4fc3f7;
            }
            QGroupBox {
                color: #4fc3f7;
                font-weight: bold;
                border: 1px solid #555555;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Panel title
        title = QLabel("性能分析面板")
        title.setStyleSheet("color: #4fc3f7; font-size: 14px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Tab widget for different performance views
        tab_widget = QTabWidget()
        
        # Basic Performance Tab
        basic_tab = self.create_basic_performance_tab()
        tab_widget.addTab(basic_tab, "基础性能")
        
        # Advanced Analysis Tab
        advanced_tab = self.create_advanced_analysis_tab()
        tab_widget.addTab(advanced_tab, "高级分析")
        
        # Export/Data Tab
        export_tab = self.create_export_tab()
        tab_widget.addTab(export_tab, "数据导出")
        
        layout.addWidget(tab_widget)
        return panel

    def create_basic_performance_tab(self):
        """Create basic performance monitoring tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Performance metrics grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(8)
        
        # Row 1
        metrics_grid.addWidget(QLabel("帧时间:"), 0, 0)
        self.avg_frame_time_label = QLabel("0.00ms")
        metrics_grid.addWidget(self.avg_frame_time_label, 0, 1)
        
        metrics_grid.addWidget(QLabel("当前FPS:"), 0, 2)
        self.current_fps_label = QLabel("0")
        metrics_grid.addWidget(self.current_fps_label, 0, 3)
        
        # Row 2
        metrics_grid.addWidget(QLabel("网格重建:"), 1, 0)
        self.grid_time_label = QLabel("0ms")
        metrics_grid.addWidget(self.grid_time_label, 1, 1)
        
        metrics_grid.addWidget(QLabel("力计算:"), 1, 2)
        self.force_time_label = QLabel("0ms")
        metrics_grid.addWidget(self.force_time_label, 1, 3)
        
        # Row 3
        metrics_grid.addWidget(QLabel("网格生成:"), 2, 0)
        self.mesh_time_label = QLabel("0ms")
        metrics_grid.addWidget(self.mesh_time_label, 2, 1)
        
        metrics_grid.addWidget(QLabel("GPU上传:"), 2, 2)
        self.gpu_time_label = QLabel("0ms")
        metrics_grid.addWidget(self.gpu_time_label, 2, 3)
        
        # Row 4
        metrics_grid.addWidget(QLabel("内存使用:"), 3, 0)
        self.memory_label = QLabel("0MB")
        metrics_grid.addWidget(self.memory_label, 3, 1)
        
        metrics_grid.addWidget(QLabel("顶点数:"), 3, 2)
        self.vertex_label = QLabel("0")
        metrics_grid.addWidget(self.vertex_label, 3, 3)
        
        # Row 5
        metrics_grid.addWidget(QLabel("三角形数:"), 4, 0)
        self.triangle_label = QLabel("0")
        metrics_grid.addWidget(self.triangle_label, 4, 1)
        
        layout.addLayout(metrics_grid)
        
        # Optimization suggestions section
        suggestions_group = QGroupBox("优化建议")
        suggestions_layout = QVBoxLayout()
        
        self.optimization_label = QLabel("暂无优化建议")
        self.optimization_label.setWordWrap(True)
        self.optimization_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        suggestions_layout.addWidget(self.optimization_label)
        
        suggestions_group.setLayout(suggestions_layout)
        layout.addWidget(suggestions_group)
        
        layout.addStretch()
        return widget

    def create_advanced_analysis_tab(self):
        """Create advanced performance analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Advanced metrics
        advanced_grid = QGridLayout()
        advanced_grid.setSpacing(8)
        
        # Stability analysis
        advanced_grid.addWidget(QLabel("稳定性:"), 0, 0)
        self.stability_label = QLabel("未知")
        self.stability_label.setStyleSheet("color: #a0a0a0;")
        advanced_grid.addWidget(self.stability_label, 0, 1)
        
        # Scaling analysis
        advanced_grid.addWidget(QLabel("扩展性:"), 0, 2)
        self.scaling_label = QLabel("未知")
        self.scaling_label.setStyleSheet("color: #a0a0a0;")
        advanced_grid.addWidget(self.scaling_label, 0, 3)
        
        # Trend analysis
        advanced_grid.addWidget(QLabel("性能趋势:"), 1, 0)
        self.trend_label = QLabel("稳定")
        self.trend_label.setStyleSheet("color: green;")
        advanced_grid.addWidget(self.trend_label, 1, 1)
        
        # Bottleneck detection
        advanced_grid.addWidget(QLabel("瓶颈检测:"), 1, 2)
        self.bottleneck_label = QLabel("未检测")
        self.bottleneck_label.setStyleSheet("color: #a0a0a0;")
        advanced_grid.addWidget(self.bottleneck_label, 1, 3)
        
        layout.addLayout(advanced_grid)
        
        # Advanced recommendations
        recommendations_group = QGroupBox("高级建议")
        recommendations_layout = QVBoxLayout()
        
        self.advanced_recommendation_label = QLabel("暂无高级建议")
        self.advanced_recommendation_label.setWordWrap(True)
        self.advanced_recommendation_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        recommendations_layout.addWidget(self.advanced_recommendation_label)
        
        recommendations_group.setLayout(recommendations_layout)
        layout.addWidget(recommendations_group)
        
        # Analysis controls
        controls_layout = QHBoxLayout()
        self.refresh_analysis_btn = QPushButton("刷新分析")
        self.refresh_analysis_btn.clicked.connect(self.refresh_advanced_analysis)
        controls_layout.addWidget(self.refresh_analysis_btn)
        
        self.clear_history_btn = QPushButton("清除历史")
        self.clear_history_btn.clicked.connect(self.clear_performance_history)
        controls_layout.addWidget(self.clear_history_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        layout.addStretch()
        return widget

    def create_export_tab(self):
        """Create data export and visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Export controls
        export_group = QGroupBox("性能数据导出")
        export_layout = QVBoxLayout()
        
        # Export buttons
        buttons_layout = QHBoxLayout()
        
        self.export_perf_btn = QPushButton("导出JSON数据")
        self.export_perf_btn.clicked.connect(self.export_performance_data)
        buttons_layout.addWidget(self.export_perf_btn)
        
        self.export_csv_btn = QPushButton("导出CSV数据")
        self.export_csv_btn.clicked.connect(self.export_performance_csv)
        buttons_layout.addWidget(self.export_csv_btn)
        
        buttons_layout.addStretch()
        export_layout.addLayout(buttons_layout)
        
        # Export info
        self.export_info_label = QLabel("点击按钮导出性能数据用于进一步分析")
        self.export_info_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        export_layout.addWidget(self.export_info_label)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Data visualization placeholder
        viz_group = QGroupBox("数据可视化")
        viz_layout = QVBoxLayout()
        
        self.viz_info_label = QLabel("性能图表和趋势分析功能待实现")
        self.viz_info_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        viz_layout.addWidget(self.viz_info_label)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        layout.addStretch()
        return widget

    def setup_timer(self):
        """Setup simulation timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.setInterval(16)  # ~60 FPS

    def start_simulation(self):
        """Start simulation"""
        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.timer.start()

            # Initialize system
            particle_count = self.mesh_system.initialize_fluid_drops()

            self.start_btn.setEnabled(False)

            self.start_btn.setText("运行中...")
            self.status_bar.showMessage(f"模拟运行中... 粒子数: {particle_count}")

    def toggle_pause(self):
        """Pause/resume simulation"""
        if self.is_running:
            self.is_paused = not self.is_paused
            text = "继续" if self.is_paused else "暂停"
            self.pause_btn.setText(text)

    def reset_simulation(self):
        """Reset simulation"""
        self.is_running = False
        self.is_paused = False
        self.timer.stop()

        # Reset system
        self.mesh_system.current_frame = 0
        self.mesh_system.current_particle_num = 0
        self.mesh_system.last_grid_rebuild = -GRID_REBUILD_INTERVAL
        self.mesh_system.last_mesh_update = -MC_MESH_UPDATE_INTERVAL

        # Update UI
        self.start_btn.setEnabled(True)
        self.start_btn.setText("开始模拟")
        self.pause_btn.setText("暂停/继续")

        self.status_bar.showMessage("系统已重置")

        # Clear mesh data
        self.opengl_renderer.update_mesh_data([], [])
        self.opengl_widget.update()

    def toggle_mesh(self, state):
        """Toggle mesh display"""
        self.mesh_system.mesh_enabled = bool(state)

    def toggle_render_mode(self, state):

        global use_ssfr
        """Toggle between mesh rendering and SSFR rendering"""
        self.opengl_renderer.use_ssfr = bool(state)
        use_ssfr = bool(state)
        mode = "SSFR直接粒子渲染" if state else "网格生成渲染"
        if state:
            print(f"SSFR参数: radius={self.opengl_renderer.ssfr_particle_radius_world:.3f}, \
                  sigma={self.opengl_renderer.ssfr_blur_sigma:.1f}, \
                  depth_sigma={self.opengl_renderer.ssfr_depth_sigma:.3f}")
            print(f"渲染参数: sigma_a={5.0}, sigma_s={0.15}, water_color=(0.08,0.35,0.65)")
        print(f"渲染模式切换到: {mode}")
        self.status_bar.showMessage(f"渲染模式: {mode}")

    def toggle_boundary(self, state):
        """Toggle boundary collision handling"""
        if hasattr(self.mesh_system, 'manual_boundary_enabled'):
            self.mesh_system.manual_boundary_enabled = bool(state)
        print(f"Boundary collision handling: {'enabled' if state else 'disabled'}")

    def update_speed_label(self, value):
        """Update speed label"""
        speed = value / 10.0
        self.speed_label.setText(f"{speed:.1f}x")

    def update_camera(self):
        """Update camera parameters"""
        self.opengl_renderer.camera_distance = self.distance_slider.value() / 10.0
        self.opengl_renderer.camera_angle_x = self.angle_x_slider.value() / 100.0
        self.opengl_renderer.camera_angle_y = self.angle_y_slider.value() / 100.0
        self.opengl_widget.update()

    def update_ssfr_params(self):
        """Update SSFR rendering parameters"""
        radius = self.ssfr_radius_slider.value() / 100.0
        self.opengl_renderer.ssfr_particle_radius_world = radius
        self.ssfr_radius_label.setText(f"{radius:.2f}")

        sigma = self.ssfr_sigma_slider.value() / 10.0
        self.opengl_renderer.ssfr_blur_sigma = sigma
        self.ssfr_sigma_label.setText(f"{sigma:.1f}")

        # Depth sigma now in world-space units (slider / 10.0 gives 0.1–10.0)
        depth_sigma = self.ssfr_depth_sigma_slider.value() / 10.0
        self.opengl_renderer.ssfr_depth_sigma = depth_sigma
        self.ssfr_depth_sigma_label.setText(f"{depth_sigma:.2f}")

        absorption = self.ssfr_absorption_slider.value() / 10.0
        if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_sigma_a' in self.opengl_renderer.composite_prog.uniforms:
            self.opengl_renderer.composite_prog['u_sigma_a'].value = absorption
        self.ssfr_absorption_label.setText(f"{absorption:.1f}")

        scatter = self.ssfr_scatter_slider.value() / 10.0
        if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_sigma_s' in self.opengl_renderer.composite_prog.uniforms:
            self.opengl_renderer.composite_prog['u_sigma_s'].value = scatter
        self.ssfr_scatter_label.setText(f"{scatter:.2f}")

        self.opengl_widget.update()

    def update_simulation(self):
        """Update simulation"""
        if not self.is_running or self.is_paused:
            return

        speed = self.speed_slider.value() / 10.0
        dt = 0.008 / speed

        # 仍然执行物理仿真（SPH）
        t = time.perf_counter()
        _new_vertices, _new_triangles = self.mesh_system.step_simulation(dt)
        print(f"Simulation step time: {time.perf_counter() - t:.3f} s")
        # SSFR：直接上传粒子位置（强耦合、低延迟、避免重建网格）
        t = time.perf_counter()
        self.opengl_widget.makeCurrent()
        try:
            n = self.mesh_system.current_particle_num
            positions = self.mesh_system.particle_positions[:n]
            self.opengl_renderer.update_particles(positions)
        finally:
            self.opengl_widget.doneCurrent()

        self.update_statistics()
        self.opengl_widget.update()
        print(f"Mesh update time: {time.perf_counter() - t:.3f} s")
        
    def export_performance_data(self):
        """导出性能数据"""
        if self.mesh_system.performance_capture:
            filename = self.mesh_system.performance_capture.export_performance_data()
            if filename:
                self.status_bar.showMessage(f"性能数据已导出到: {filename}")
            else:
                self.status_bar.showMessage("性能数据导出失败")
    def update_statistics(self):
        """Update statistics display with separated UI panels"""
        stats = self.mesh_system.get_system_stats()

        # Update basic labels in left panel
        self.frame_label.setText(f"帧: {stats['frame']}")
        self.particle_label.setText(f"粒子数: {stats['particle_count']}")
        
        # Update FPS in left panel
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.fps_counter = 0
            self.last_fps_time = current_time

        # Update performance metrics in right panel
        perf = stats['performance']
        
        # Basic performance tab updates
        if hasattr(self, 'avg_frame_time_label'):
            self.avg_frame_time_label.setText(f"{perf.get('avg_frame_time', 0)*1000:.2f}ms")
        if hasattr(self, 'current_fps_label'):
            self.current_fps_label.setText(f"{perf.get('current_fps', 0):.1f}")
        if hasattr(self, 'grid_time_label'):
            self.grid_time_label.setText(f"{perf['grid_rebuild_time']*1000:.1f}ms")
        if hasattr(self, 'force_time_label'):
            self.force_time_label.setText(f"{perf['force_computation_time']*1000:.1f}ms")
        if hasattr(self, 'mesh_time_label'):
            self.mesh_time_label.setText(f"{perf['mesh_generation_time']*1000:.1f}ms")
        if hasattr(self, 'gpu_time_label'):
            self.gpu_time_label.setText(f"{perf.get('gpu_upload_time', 0)*1000:.1f}ms")
        if hasattr(self, 'memory_label'):
            self.memory_label.setText(f"{stats.get('memory_usage_mb', 0):.1f}MB")
        if hasattr(self, 'vertex_label'):
            self.vertex_label.setText(f"{stats['mesh_vertices']}")
        if hasattr(self, 'triangle_label'):
            self.triangle_label.setText(f"{stats['mesh_triangles']}")
        
        # Update optimization suggestions
        if self.mesh_system.performance_capture:
            # 获取实时性能统计
            real_time_stats = self.mesh_system.performance_capture.real_time_stats
            
            # Update optimization suggestions in basic tab
            if hasattr(self, 'optimization_label'):
                suggestions = self.mesh_system.performance_capture.optimization_suggestions
                if suggestions:
                    # 显示最严重的建议
                    critical_suggestions = [s for s in suggestions if s['severity'] == 'high']
                    if critical_suggestions:
                        suggestion_text = f"⚠ {critical_suggestions[0]['message']}"
                        self.optimization_label.setText(suggestion_text)
                        self.optimization_label.setStyleSheet("color: red; font-weight: bold;")
                    else:
                        # 显示中等严重性的建议
                        medium_suggestions = [s for s in suggestions if s['severity'] == 'medium']
                        if medium_suggestions:
                            suggestion_text = f"⚠ {medium_suggestions[0]['message']}"
                            self.optimization_label.setText(suggestion_text)
                            self.optimization_label.setStyleSheet("color: orange;")
                        else:
                            self.optimization_label.setText("✓ 性能良好")
                            self.optimization_label.setStyleSheet("color: green;")
                else:
                    self.optimization_label.setText("✓ 性能良好")
                    self.optimization_label.setStyleSheet("color: green;")
            
            # 更新高级性能分析信息
            self._update_advanced_performance_info()

    def _update_advanced_performance_info(self):
        """更新高级性能分析信息显示"""
        if not self.mesh_system.performance_capture:
            return
            
        # 获取最新性能报告
        report = self.mesh_system.performance_capture.get_performance_report()
        advanced_analysis = report.get('advanced_analysis', {})
        
        # 更新稳定性信息
        if hasattr(self, 'stability_label'):
            patterns = advanced_analysis.get('patterns', {})
            if patterns and 'stability' in patterns:
                stability_info = patterns['stability']
                unstable_metrics = [k for k, v in stability_info.items() 
                                  if v['stability_level'] in ['unstable', 'moderately_unstable']]
                if unstable_metrics:
                    self.stability_label.setText(f"⚠ 不稳定 ({len(unstable_metrics)}个组件)")
                    self.stability_label.setStyleSheet("color: orange;")
                else:
                    self.stability_label.setText("✓ 良好")
                    self.stability_label.setStyleSheet("color: green;")
            else:
                self.stability_label.setText("未知")
                self.stability_label.setStyleSheet("color: gray;")
            
        # 更新扩展性信息
        if hasattr(self, 'scaling_label'):
            patterns = advanced_analysis.get('patterns', {})
            if patterns and 'scaling' in patterns and 'particle_scaling' in patterns['scaling']:
                scaling = patterns['scaling']['particle_scaling']
                if scaling['scaling_type'] == 'linear':
                    cost_per_particle = scaling['slope'] * 1000  # 转换为毫秒
                    self.scaling_label.setText(f"线性 ({cost_per_particle:.2f}ms/粒子)")
                    if cost_per_particle > 0.1:
                        self.scaling_label.setStyleSheet("color: orange;")
                    else:
                        self.scaling_label.setStyleSheet("color: green;")
                else:
                    self.scaling_label.setText(f"非线性")
                    self.scaling_label.setStyleSheet("color: yellow;")
            else:
                self.scaling_label.setText("未知")
                self.scaling_label.setStyleSheet("color: gray;")
            
        # 更新趋势信息
        if hasattr(self, 'trend_label'):
            patterns = advanced_analysis.get('patterns', {})
            if patterns and 'trends' in patterns:
                degrading_metrics = [k for k, v in patterns['trends'].items() 
                                   if v['trend'] == 'increasing']
                improving_metrics = [k for k, v in patterns['trends'].items() 
                                   if v['trend'] == 'decreasing']
                                    
                if degrading_metrics:
                    self.trend_label.setText(f"⚠ 下降 ({len(degrading_metrics)}个指标)")
                    self.trend_label.setStyleSheet("color: red;")
                elif improving_metrics:
                    self.trend_label.setText(f"✓ 改善 ({len(improving_metrics)}个指标)")
                    self.trend_label.setStyleSheet("color: green;")
                else:
                    self.trend_label.setText("稳定")
                    self.trend_label.setStyleSheet("color: green;")
            else:
                self.trend_label.setText("未知")
                self.trend_label.setStyleSheet("color: gray;")
            
        # 更新高级建议
        if hasattr(self, 'advanced_recommendation_label'):
            recommendations = advanced_analysis.get('recommendations', [])
            if recommendations:
                # 显示最高优先级的建议
                high_priority = [r for r in recommendations if r['priority'] == 'high']
                if high_priority:
                    rec_text = f"⚠ {high_priority[0]['message'][:30]}..."
                    self.advanced_recommendation_label.setText(rec_text)
                    self.advanced_recommendation_label.setStyleSheet("color: red; font-weight: bold;")
                else:
                    medium_priority = [r for r in recommendations if r['priority'] == 'medium']
                    if medium_priority:
                        rec_text = f"⚠ {medium_priority[0]['message'][:30]}..."
                        self.advanced_recommendation_label.setText(rec_text)
                        self.advanced_recommendation_label.setStyleSheet("color: orange;")
                    else:
                        self.advanced_recommendation_label.setText("✓ 无紧急建议")
                        self.advanced_recommendation_label.setStyleSheet("color: green;")
            else:
                self.advanced_recommendation_label.setText("✓ 无高级建议")
                self.advanced_recommendation_label.setStyleSheet("color: green;")

    def refresh_advanced_analysis(self):
        """手动刷新高级性能分析"""
        if self.mesh_system.performance_capture:
            self.mesh_system.performance_capture.last_analysis_time = 0  # 强制重新分析
            self._update_advanced_performance_info()
            self.status_bar.showMessage("高级性能分析已刷新")

    def clear_performance_history(self):
        """清除性能历史数据"""
        if self.mesh_system.performance_capture:
            self.mesh_system.performance_capture.clear_history()
            self.status_bar.showMessage("性能历史数据已清除")
            # 重置显示
            self._reset_performance_display()

    def _reset_performance_display(self):
        """重置性能显示为初始状态"""
        # 重置基本性能标签
        if hasattr(self, 'avg_frame_time_label'):
            self.avg_frame_time_label.setText("0.00ms")
        if hasattr(self, 'current_fps_label'):
            self.current_fps_label.setText("0")
        if hasattr(self, 'grid_time_label'):
            self.grid_time_label.setText("0ms")
        if hasattr(self, 'force_time_label'):
            self.force_time_label.setText("0ms")
        if hasattr(self, 'mesh_time_label'):
            self.mesh_time_label.setText("0ms")
        if hasattr(self, 'gpu_time_label'):
            self.gpu_time_label.setText("0ms")
        if hasattr(self, 'memory_label'):
            self.memory_label.setText("0MB")
        if hasattr(self, 'vertex_label'):
            self.vertex_label.setText("0")
        if hasattr(self, 'triangle_label'):
            self.triangle_label.setText("0")
        
        # 重置优化建议
        if hasattr(self, 'optimization_label'):
            self.optimization_label.setText("暂无优化建议")
            self.optimization_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        
        # 重置高级分析标签
        if hasattr(self, 'stability_label'):
            self.stability_label.setText("未知")
            self.stability_label.setStyleSheet("color: gray;")
        if hasattr(self, 'scaling_label'):
            self.scaling_label.setText("未知")
            self.scaling_label.setStyleSheet("color: gray;")
        if hasattr(self, 'trend_label'):
            self.trend_label.setText("稳定")
            self.trend_label.setStyleSheet("color: green;")
        if hasattr(self, 'advanced_recommendation_label'):
            self.advanced_recommendation_label.setText("暂无高级建议")
            self.advanced_recommendation_label.setStyleSheet("color: #a0a0a0; font-style: italic;")

    def export_performance_csv(self):
        """导出性能数据为CSV格式"""
        if self.mesh_system.performance_capture:
            try:
                import csv
                from datetime import datetime
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"performance_data_{timestamp}.csv"
                
                report = self.mesh_system.performance_capture.get_performance_report()
                raw_data = report.get('raw_data', {})
                
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # 写入头部信息
                    writer.writerow(['Timestamp', 'Frame Time (ms)', 'FPS'])
                    
                    # 写入数据
                    timestamps = raw_data.get('timestamps', [])
                    frame_times = raw_data.get('frame_times_ms', [])
                    fps_values = raw_data.get('fps_values', [])
                    
                    max_len = max(len(timestamps), len(frame_times), len(fps_values))
                    for i in range(max_len):
                        row = [
                            timestamps[i] if i < len(timestamps) else '',
                            frame_times[i] if i < len(frame_times) else '',
                            fps_values[i] if i < len(fps_values) else ''
                        ]
                        writer.writerow(row)
                
                self.status_bar.showMessage(f"CSV数据已导出到: {filename}")
                print(f"Performance CSV exported to: {filename}")
                
            except Exception as e:
                self.status_bar.showMessage(f"CSV导出失败: {str(e)}")
                print(f"CSV export failed: {e}")

class QOpenGLWidgetWrapper(QOpenGLWidget):
    """OpenGL widget wrapper for ModernGL - 修复版"""

    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer
        self.ctx = None
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        # self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen)

    def initializeGL(self):
        """Initialize OpenGL context - 修复版"""
        try:
            # 确保OpenGL上下文已激活
            self.makeCurrent()
            # Create ModernGL context with proper backend
            self.ctx = moderngl.create_context(standalone=False)
            print(glGetString(GL_VERSION))
            print(self.ctx.version_code, self.ctx.info)
            print("✓ ModernGL context created successfully")

            # Initialize renderer
            if not self.renderer.initialize(self.ctx):
                print("✗ Renderer initialization failed")
                return

            # 设置默认的OpenGL状态
            # self.ctx.enable(moderngl.DEPTH_TEST)
            # self.ctx.enable(moderngl.CULL_FACE)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

            print("✓ OpenGL state configured successfully")
                    
        except Exception as e:
            print(f"✗ OpenGL initialization failed: {e}" )

    def paintGL(self):
        """Paint OpenGL scene - 修复版"""
        if not self.ctx:
            return
        # self.ctx.disable(moderngl.CULL_FACE)
        # self.ctx.disable(moderngl.DEPTH_TEST)

        # 捕获屏幕FBO，作为最终输出目标
        screen_fbo = self.ctx.detect_framebuffer()
        screen_fbo.use()


        # 确保上下文激活
        self.makeCurrent()
        # Get widget size
        size = self.size()
        width, height = size.width(), size.height()

        # Ensure framebuffer is the right size
        if width > 0 and height > 0:
            self.ctx.viewport = (0, 0, width, height)
            self.ctx.clear(0.1, 0.1, 0.15, 1.0)  # 清除背景色

            # 渲染场景，传入屏幕FBO确保最终输出一致性
            self.renderer.render(width, height, screen_fbo)
        else:
            print(f"⚠ Invalid viewport size: {width}x{height}" )



    def resizeGL(self, width, height):
        """Handle resize event - 修复版"""
        if self.ctx and width > 0 and height > 0:
            self.ctx.viewport = (0, 0, width, height)
            print(f"✓ Resize OpenGL viewport to {width}x{height}" )


def main():
    # 先初始化 SPH（在 Qt 之前）
    # 这样避免 Qt 事件循环和 SPlisHSPlasH 内部 Qt 组件冲突

    from PySide6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # MeshSystem 在 QApplication 创建后初始化（SPH 不依赖 Qt）
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()