"""
系统配置和常量定义
System Configuration and Constants
"""

import numpy as np

class Config:
    """系统配置类"""
    
    # 物理仿真参数
    PARTICLE_RADIUS = 0.05
    MAX_PARTICLES = 30000
    H = PARTICLE_RADIUS * 2.0
    
    # 网格参数
    MC_GRID_RES = 16
    GRID_REBUILD_INTERVAL = 1
    MC_MESH_UPDATE_INTERVAL = 1
    
    # SSFR参数
    SHADOW_MAP_SIZE = 512
    BATCH_SIZE = 1000
    VERTEX_BUFFER_SIZE = 50000
    INDEX_BUFFER_SIZE = 100000
    
    # 性能优化参数
    ENABLE_PERFORMANCE_PROFILING = True
    ENABLE_DETAILED_PERFORMANCE_CAPTURE = True
    ENABLE_ADVANCED_PERFORMANCE_ANALYSIS = True
    PERFORMANCE_HISTORY_SIZE = 1000
    PERFORMANCE_SAMPLING_RATE = 5
    ENABLE_REALTIME_ANALYSIS = True
    ENABLE_MEMORY_PROFILING = True
    ENABLE_GPU_PROFILING = True
    
    # 深度纹理保存参数
    ENABLE_DEPTH_TEXTURE_SAVE = True
    DEPTH_SAVE_INTERVAL = 1.0  # 秒
    DEPTH_SAVE_PATH = "./depth_textures"
    DEPTH_SAVE_FORMAT = "png"  # png, npy
    DEPTH_NORMALIZE_RANGE = True  # 是否将深度值归一化到[0,1]范围
    
    # 边界参数
    BOUNDARY_MIN = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    BOUNDARY_MAX = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    
    # PySPlisHSPlasH配置
    PYSPLISH_AVAILABLE = True
    SPLASHSURF_AVAILABLE = True
    USE_SSFR = True
    
    # 时间步长配置
    DEFAULT_TIME_STEP = 0.008
    MIN_TIME_STEP = 0.0002
    MAX_TIME_STEP = 0.005
    CFL_FACTOR = 0.25
    
    # DFSPH参数
    DFSPH_MIN_ITERATIONS = 2
    DFSPH_MAX_ITERATIONS = 50
    DFSPH_MAX_ERROR = 0.05
    DFSPH_MAX_ITERATIONS_V = 50
    DFSPH_MAX_ERROR_V = 0.1
    
    # 渲染参数
    CAMERA_DISTANCE = 10.0
    CAMERA_ANGLE_X = 0.5
    CAMERA_ANGLE_Y = 0.3
    CAMERA_TARGET = [0.0, 1.5, 0.0]
    
    # SSFR渲染参数
    SSFR_SCALE = 1.0
    SSFR_FILTER_ITERATIONS = 5
    SSFR_FILTER_INTERVALS = [1, 2, 4, 8, 16]
    SSFR_BLUR_SIGMA = 8.0
    SSFR_DEPTH_SIGMA = 0.025
    SSFR_BLUR_RADIUS = 4
    SSFR_PARTICLE_RADIUS_WORLD = PARTICLE_RADIUS * 1.5
    
    # 光照参数
    LIGHT_POSITION = np.array([3.5, 7.0, 3.5], dtype=np.float32)
    LIGHT_TARGET = np.array([0.0, 1.5, 0.0], dtype=np.float32)
    
    # 材质参数 (已根据SSFR渲染参数面板调整)
    alpha_liquid = 2
    WATER_COLOR = (0.04 / alpha_liquid, 0.28 / alpha_liquid, 0.60 / alpha_liquid)
    ALPHA_BASE = 0.05              # 基础透明度: 0.05 (原0.08)
    FRESNEL_POWER = 4.0            # 菲涅尔幂次: 4.0 (不变)
    SPEC_POWER = 137.0             # 镜面反射强度: 137 (原150)
    SIGMA_A = 1.1                  # 吸收系数: 1.1 (原3.5)
    SIGMA_S = 0.20                 # 散射系数: 0.20 (原0.15)
    SHADOW_STRENGTH = 0.50         # 阴影强度: 0.50 (原0.55)
    CAUSTIC_STRENGTH = 0.60        # 焦散强度: 0.60 (原0.5)
    CAUSTIC_POWER = 0.2            # 焦散幂次: 0.2 (原0.6)

# 全局配置实例
CONFIG = Config()