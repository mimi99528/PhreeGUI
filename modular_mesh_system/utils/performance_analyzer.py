"""
性能分析模块
Performance Analysis Module
"""

import sys
import time
import json
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from ..utils.config import CONFIG
from ..utils.helpers import ThrottleLog

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
    
    def __init__(self, history_size=CONFIG.PERFORMANCE_HISTORY_SIZE):
        self.history_size = history_size
        self.sampling_rate = CONFIG.PERFORMANCE_SAMPLING_RATE
        self.sample_counter = 0
        
        # 高级分析器
        self.advanced_analyzer = AdvancedPerformanceAnalyzer(self) if CONFIG.ENABLE_ADVANCED_PERFORMANCE_ANALYSIS else None
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

# 兼容性导出
PerformanceAnalyzer = DetailedPerformanceCapture