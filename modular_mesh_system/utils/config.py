"""
系统配置和常量定义
System Configuration and Constants
"""

import os
import numpy as np

class Config:
    """系统配置类"""

    # ── 物理后端选择 ────────────────────────────────────────────────
    # "cpu"    = SPlisHSPlasH DFSPH（Bender2019 体积图边界，原实现，默认）
    # "taichi" = erizmr/SPH_Taichi DFSPH（Akinci2012 边界粒子，GPU 并行）
    #
    # 可用环境变量覆盖，便于不改代码切换：
    #   PHREEGUI_PHYSICS_BACKEND=taichi
    #   PHREEGUI_TAICHI_ARCH=cuda|cpu   （默认 cuda；CUDA 不可用时硬失败，
    #                                    不会静默退回 CPU —— 决策 D4）
    #
    # 注意：两条路径的边界模型不同，数值结果不会逐位一致；
    # 化学耦合只依赖 (position, particleId)，两者契约相同。
    PHYSICS_BACKEND = os.environ.get(
        "PHREEGUI_PHYSICS_BACKEND", "taichi").strip().lower()
    TAICHI_ARCH = os.environ.get(
        "PHREEGUI_TAICHI_ARCH", "cuda").strip().lower()
    
    # 物理仿真参数
    # SPlisHSPlasH 约定：supportRadius = 4 * particleRadius，
    # 而 FluidBlock 采样步长 = 2 * particleRadius，所以静止密度下的
    # 粒子间距就是 PARTICLE_SPACING。新增粒子必须按这个间距落格，
    # 否则初帧密度远超 density0，DFSPH 会把粒子炸开/压成一层。
    PARTICLE_RADIUS = 0.01
    PARTICLE_SPACING = PARTICLE_RADIUS * 2.0
    # 容量上限。原来 MAX_EMITTER_PARTICLES=5000，点满 5000 个就再也加不进去
    # （状态栏报"求解器数组空间不足：要加 8 个，只剩 0"）。
    #
    # 实测依据（probe_capacity.py，本机 32.6 GB / 可用 11.2 GB）：
    #   引擎空场按 maxEmitterParticles 预分配，边际成本 ~72 B/粒子，
    #   1M 容量 = 108 MB RSS，initSimulation 耗时与容量无关（恒 ~5 s）。
    #   numpy 侧 MAX_PARTICLES × 32 B = 32 MB，渲染 VBO 2 × 12 MB。
    #
    # 为什么是 1M 而不是 10M：容器 1.6×2.4×1.6 m、间距 0.02 m 只有
    #   80×120×80 = 768,000 个晶格位（点满整杯就这么多，超出容器的
    #   落点会被生成校验拒绝）。10M 个 r=0.01 的粒子根本放不进这个杯子；
    #   而且 DFSPH 实测 200K 活跃时已 984 ms/步，1M 以上是十秒级，不可交互。
    #   要真的跑数 M，必须先缩小 PARTICLE_RADIUS 并换更大的容器。
    MAX_PARTICLES = 1000000
    MAX_EMITTER_PARTICLES = 1000000  # 引擎预留容量，必须 <= MAX_PARTICLES
    H = PARTICLE_RADIUS * 2.0
    
    # 网格参数  
    MC_GRID_RES = 16
    MC_MESH_UPDATE_INTERVAL = 1
    
    # SSFR参数
    SHADOW_MAP_SIZE = 512
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
    ENABLE_DEPTH_TEXTURE_SAVE = False
    DEPTH_SAVE_INTERVAL = 1.0  # 秒
    DEPTH_SAVE_PATH = "./depth_textures"
    DEPTH_SAVE_FORMAT = "png"  # png, npy
    DEPTH_NORMALIZE_RANGE = True  # 是否将深度值归一化到[0,1]范围
    
    # 容器参数
    # 原来的容器是 4 x 6 x 4 的大箱体（地板 16 m²），同样多的水只会铺成
    # 薄薄一层，看起来"永远是一个平面"。改成杯状容器（1.6 x 2.4 x 1.6，
    # 地板 2.56 m²）后，同样的粒子数就能堆出可见的水深。
    # CONTAINER_SIZE 是内部净尺寸，也是 mesh_system 边界体与点击生成范围
    # 的唯一真源；BOUNDARY_MIN/MAX 由它推导，不要各写一份。
    CONTAINER_SIZE = np.array([1.6, 2.4, 1.6], dtype=np.float64)
    CONTAINER_BASE_Y = 0.0
    BOUNDARY_MIN = np.array([-CONTAINER_SIZE[0] * 0.5,
                             CONTAINER_BASE_Y,
                             -CONTAINER_SIZE[2] * 0.5], dtype=np.float64)
    BOUNDARY_MAX = np.array([CONTAINER_SIZE[0] * 0.5,
                             CONTAINER_BASE_Y + CONTAINER_SIZE[1],
                             CONTAINER_SIZE[2] * 0.5], dtype=np.float64)
    
    # PySPlisHSPlasH配置
    PYSPLISH_AVAILABLE = True
    SPLASHSURF_AVAILABLE = True
    USE_SSFR = True
    
    # 时间步长配置 - 修复：SPH模拟需要更小的时间步长
    # 典型值：0.001-0.005秒（基于CFL条件和粒子半径）
    DEFAULT_TIME_STEP = 0.003  # 3ms，符合SPH稳定性要求
    MIN_TIME_STEP = 0.001      # 1ms 最小步长
    MAX_TIME_STEP = 0.01       # 10ms 最大步长（保守上限）
    CFL_FACTOR = 0.5           # CFL条件因子（稍微放宽以提高性能）
    # Taichi 后端没有引擎内的自适应 CFL（CPU 走 C++ TimeManager，会把
    # 传入的步长夹回稳定区间），所以必须自己按上游公式夹一次：
    #   SPlisHSPlasH/Simulation.cpp:487
    #   h_cfl = cflFactor * 0.4 * diameter / sqrt(maxVel)
    CFL_DIAMETER_FACTOR = 0.4  # 上游 CFL 公式里的 0.4 系数

    # Taichi 专用的步长硬上限（与 MAX_TIME_STEP 不是一回事）。
    # MAX_TIME_STEP=0.01 只下发给 CPU 引擎的 CFL_MAX_TIMESTEPSIZE，由引擎自己
    # 按 CFL 再夹；Taichi 没有这层引擎保护，dt 会原样进求解器。
    # 而显式粘性/表面张力对 dt 是刚性的，实测（8000 粒子同一灌注）：
    #   0.003（本值）→ 稳；0.0075（2.5x）→ 峰值 2.7e4 m/s，水炸满容器；
    #   0.015 → NaN/Inf。
    # 所以 Taichi 路径必须先按这个上限硬夹，再做 CFL 收缩（只能更小）。
    TAICHI_MAX_TIME_STEP = DEFAULT_TIME_STEP
    
    # DFSPH参数
    DFSPH_MIN_ITERATIONS = 2
    DFSPH_MAX_ITERATIONS = 50
    DFSPH_MAX_ERROR = 0.05
    DFSPH_MAX_ITERATIONS_V = 50
    DFSPH_MAX_ERROR_V = 0.1

    # Taichi 后端的流体材质参数。
    # 【为什么必须显式写出来】_build_scene() 只下发 Configuration 段，
    # 下面三个值原本是 DFSPHSolver/SPHBase 的构造默认（DFSPH.py 构造函数、
    # sph_base.py），scene dict 里没有对应 key，调用方无从配置——
    # 出问题只能去改求解器源码。
    #
    # 粘性 viscosity：显式格式，稳定性上限直接受时间步长约束。实测
    #   同一灌注（8000 粒子，dt=0.003）峰值 |v|：
    #     0.01 → 2.49 m/s（稳）   0.03 → 1.5e5 m/s（炸）   0.1 → 3.3e5（炸）
    #   即 0.01 已是当前步长下的可用上限，不能为了"更黏"直接调大。
    #   上游 SPlisHSPlasH dam-break 场景同样用 0.01。
    VISCOSITY = 0.01

    # 边界粘性（Akinci2012 的壁面 no-slip 项）。
    # 【这是"粒子很轻、一点力就到处窜"的直接原因】该分支原本硬编码
    # boundary_viscosity = 0.0，壁面完全无粘（free-slip）。流体-流体粘性只在
    # 支撑半径内有邻居时才有效，而默认点击量是 8 个粒子（2x2x2），彼此很快
    # 散开失去耦合 → 整团水在无摩擦地面上滑行/弹跳，永远不停。
    # 实测（同一提交，仅改这个系数；每个配置独立进程跑）：
    #              S1 8粒子静置(200帧)          S2 8000粒子灌注
    #   bv=0.0   mean|v|=0.695 足迹1.523m         t=2s mean=0.243 >0.3速占31.3%
    #            后续100帧漂移 0.817m(仍滑行)     t=4s mean=0.035（勉强停）
    #   bv=0.005 mean|v|=0.0066 足迹0.078m        t=2s mean=0.018 >0.3速占 0.0%
    #            后续100帧漂移 0.0007m(静止)      t=4s mean=0.006
    #   bv=0.05  8粒子同样静止，但灌注 t=4s 残余 0.051（过阻尼抖动，未采用）
    # 取 0.005：两个场景都干净收敛。上游 Viscosity_Standard.cpp 也有
    # "viscosityBoundary" 参数（默认 0，因为它的示例用 Bender2019 体积图）。
    BOUNDARY_VISCOSITY = 0.005

    # 表面张力（内聚力项）。保持上游默认 0.01：实测在边界粘性修好后，
    # 0.01/0.05/0.08 的静置收敛差别可以忽略（t=2s 均为 fast%≈0），
    # 0.01 反而收敛更快（t=4s 残余 0.007 vs 0.011）。
    SURFACE_TENSION = 0.01
    
    # 渲染参数
    # 容器改小之后相机要跟着靠近：原来 distance=10、target y=1.5 是为
    # 4x6x4 大箱体配的，1.6 m 宽的杯子在 10 m 外只剩屏幕中间一小块。
    CAMERA_DISTANCE = 4.2
    CAMERA_ANGLE_X = 0.5
    CAMERA_ANGLE_Y = 0.22
    CAMERA_TARGET = [0.0, 1.0, 0.0]
    
    # SSFR渲染参数
    SSFR_SCALE = 1.0
    SSFR_FILTER_ITERATIONS = 6  # 增加迭代次数
    SSFR_FILTER_INTERVALS = [1, 1, 2, 2, 4, 4]  # 小步长优先，密集采样
    SSFR_BLUR_SIGMA = 3.0
    # 深度 sigma 的单位是米（深度 pass 存的是视图空间线性深度）。
    # 原来 0.015 m 远小于粒子半径 0.05 m，双边滤波把每个粒子的深度起伏
    # 都当成"边缘"保留下来，于是水面呈一颗颗圆包。放宽到 ~0.04 m 后
    # 相邻粒子被当成同一张表面平滑掉。
    SSFR_DEPTH_SIGMA = 0.04
    SSFR_BLUR_RADIUS = 4
    # 1.5× 会把粒子渲染成互相重叠的大圆包（看起来像糖浆）。
    # 1.2× 刚好让相邻粒子轻微搭接、表面连续又不臃肿；调试面板可实时微调。
    SSFR_PARTICLE_RADIUS_WORLD = PARTICLE_RADIUS * 1.2
    
    # 光照参数
    LIGHT_POSITION = np.array([3.5, 7.0, 3.5], dtype=np.float32)
    LIGHT_TARGET = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    # 材质参数 (已根据SSFR渲染参数面板调整)
    alpha_liquid = 2
    WATER_COLOR = (0.04 / alpha_liquid, 0.28 / alpha_liquid, 0.60 / alpha_liquid)
    ALPHA_BASE = 0.05              # 基础透明度: 0.05 (原0.08)
    FRESNEL_POWER = 8.0            # 菲涅尔幂次: 4.0 (不变)
    SPEC_POWER = 70.0             # 镜面反射强度: 137 (原150)
    SIGMA_A = 1.1                  # 吸收系数: 1.1 (原3.5)
    SIGMA_S = 0.20                 # 散射系数: 0.20 (原0.15)
    SHADOW_STRENGTH = 0.50         # 阴影强度: 0.50 (原0.55)
    CAUSTIC_STRENGTH = 0.2        # 焦散强度: 0.60 (原0.5)
    CAUSTIC_POWER = 0.1            # 焦散幂次: 0.2 (原0.6)

# 全局配置实例
CONFIG = Config()