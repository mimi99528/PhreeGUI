"""
网格系统核心模块
Core Mesh System Module
"""
import os
import time
import numpy as np
import threading
from typing import Dict, List, Optional, Tuple


# PySPlisHSPlasH integration
try:
    import pysplishsplash as sph
    import pysplishsplash.Utilities.SceneLoaderStructs as Scenes
    PYSPLISH_AVAILABLE = True
except ImportError:
    PYSPLISH_AVAILABLE = False
    print("Warning: PySPlisHSPlasH not available")

# PySplashSurf
try:
    import pysplashsurf
    SPLASHSURF_AVAILABLE = True
except ImportError:
    SPLASHSURF_AVAILABLE = False
    print("Warning: PySplashSurf not available")

from ..utils.config import CONFIG
from ..utils.helpers import profile_function, ThrottleLog
from ..utils.performance_analyzer import PerformanceAnalyzer

# ---------------------------------------------------------------------------
# 引擎数据资产路径（必须 __file__ 基，不能用 cwd，也不能用 sph.Extras.Scenes.Empty）
# ---------------------------------------------------------------------------
# pysplishsplash 的 SimulatorBase.getExePath() 返回 venv 的 base-prefix
# （本机实测 C:\Python312），引擎会把相对 sceneFile 按该目录解析，
# 导致 "Cannot open file" 后在 initBoundaryData() 里直接结束进程（exit 127）。
# pybind 层没有任何 data-path setter，唯一可靠的办法是传绝对路径。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPH_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPH_SCENE_EMPTY = os.path.join(SPH_DATA_DIR, "Scenes", "Empty.json")
SPH_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

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

        # 【清理 c2】删掉 self.pysplish_config 字典：全仓库仅此一处赋值、从未被读取，
        # 而且里面的 gravity=[0.0,-9.8,0.0] 与真正下发给引擎的 -9.81 不一致，
        # 留着只会被当成"重力在这里配置"误导下一个人。
        # 真正生效的重力在 _initialize_pysplish_components() 里显式调用。

        # Particle data storage - 预分配批量缓冲区
        self.particle_positions = np.zeros((CONFIG.MAX_PARTICLES, 3), dtype=np.float32)
        self.particle_velocities = np.zeros((CONFIG.MAX_PARTICLES, 3), dtype=np.float32)
        self.particle_types = np.ones(CONFIG.MAX_PARTICLES, dtype=np.int32)  # 1=水, 2=酸性, 3=碱性
        # 【修复 b3 / 决策 D3】引擎分配的稳定粒子号：particle_ids[r] = 第 r 行那个粒子的 id。
        # 引擎做 z-sort 时会把 id 连同 position/velocity 一起搬走，所以只有它才是身份，
        # 行号不是。化学层一律按这个 id 存状态。
        self.particle_ids = np.zeros(CONFIG.MAX_PARTICLES, dtype=np.int64)

        # 身份护栏的期望 id 序列缓存（见 _expected_ids_u32）
        self._expected_ids_cache = np.zeros(0, dtype=np.uint32)

        # System state
        self.current_particle_num = 0
        self.current_frame = 0
        self.last_mesh_update = -CONFIG.MC_MESH_UPDATE_INTERVAL
        self.mesh_enabled = True

        # Boundary parameters
        # 与容器内部净尺寸严格一致（CONFIG.BOUNDARY_MIN/MAX 由 CONTAINER_SIZE
        # 推导）。化学网格的 domain 也直接从这里取，保证粒子永远不会落在
        # 化学域之外被 clip 到边缘单元。
        self.boundary_min = CONFIG.BOUNDARY_MIN.copy()
        self.boundary_max = CONFIG.BOUNDARY_MAX.copy()
        # 新增粒子的落格间距（= 引擎 FluidBlock 采样步长 = 2r）。
        self.particle_spacing = float(CONFIG.PARTICLE_SPACING)

        # Boundary container components
        self.boundary_walls = []  # Store boundary wall objects
        self.boundary_size = self.boundary_max - self.boundary_min
        self.mc_cell_size = self.boundary_size / CONFIG.MC_GRID_RES

        # Performance statistics
        self.performance_stats = {
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
        self.performance_capture = PerformanceAnalyzer() if CONFIG.ENABLE_DETAILED_PERFORMANCE_CAPTURE else None
        
        # 性能分析标志
        self.frame_sample_counter = 0

        # PySplashSurf configuration
        if SPLASHSURF_AVAILABLE:
            self.splashsurf_config = {
                'particle_radius': CONFIG.PARTICLE_RADIUS,
                'rest_density': 1000.0,
                'smoothing_length': 2.5 * CONFIG.PARTICLE_RADIUS,
                'cube_size': CONFIG.PARTICLE_RADIUS * 1.5,
                'iso_surface_threshold': 0.6,
                'enable_multithreading': True,
                'enable_simd': True,
                'enable_zsort': False,
            }

        # Initialize PySPlisHSPlasH
        # 【决策 D4】引擎初始化失败必须硬失败并向上传播。
        # 原实现 except 后把全局 CONFIG.PYSPLISH_AVAILABLE 置 False，
        # 于是 GUI 照常启动、按钮照常可点、粒子数照常显示，
        # 但底下没有任何真实求解器在跑 —— 这是"跑起来了但跑不对"的最坏形态。
        # 现在：pysplishsplash 不可导入 / 引擎初始化抛错 → 直接 raise，
        # GUI 侧由 create_right_panel 的 except 显示成可见错误面板。
        if not PYSPLISH_AVAILABLE:
            raise RuntimeError(
                "pysplishsplash 不可用，无法进行真实 SPH/DFSPH 计算。"
                "请确认该包安装在当前解释器中。")
        if not CONFIG.PYSPLISH_AVAILABLE:
            raise RuntimeError(
                "CONFIG.PYSPLISH_AVAILABLE 被显式禁用，"
                "拒绝以降级（非真实物理）模式启动。")
        self._initialize_pysplish_components()
        print("PySPlisHSPlasH components initialized successfully")

        # 物理仿真线程化初始化
        self._init_threading()

    def _setup_boundary_with_scenes_module(self):
        """使用官方Scenes模块设置边界（推荐方式）- 修复版"""
        support_radius = 4.0 * CONFIG.PARTICLE_RADIUS
        # 容器尺寸唯一真源 = CONFIG.CONTAINER_SIZE（内部净尺寸）。
        # UnitBox.obj 是边长 1、中心在原点的闭合盒子，放大到 CONTAINER_SIZE
        # 后只有沿 y 平移 size_y/2 才能让地板落在 y=0。
        box_scale = CONFIG.CONTAINER_SIZE.astype(float).tolist()
        map_res = [
            max(25, int(box_scale[0] / support_radius) + 1),
            max(25, int(box_scale[1] / support_radius) + 1),
            max(25, int(box_scale[2] / support_radius) + 1),
        ]

        boundary_data = Scenes.BoundaryData()
        boundary_data.meshFile = "../models/UnitBox.obj"
        boundary_data.translation = [0.0, box_scale[1] * 0.5, 0.0]
        boundary_data.scale = box_scale
        boundary_data.color = [0.1, 0.4, 0.5, 1.0]
        boundary_data.isWall = True
        boundary_data.mapInvert = True
        boundary_data.mapResolution = map_res

        self.scene.boundaryModels.append(boundary_data)
        print(f"✓ Boundary container added (scale={box_scale}, "
              f"mapResolution={map_res}, supportRadius={support_radius:.3f})")

    def _initialize_pysplish_components(self):
        """Initialize PySPlisHSPlasH - 修复版：确保边界与 particleRadius 匹配"""
        # D4：数据资产缺失必须硬失败，不能静默退化
        if not os.path.isfile(SPH_SCENE_EMPTY):
            raise RuntimeError(
                f"SPlisHSPlasH scene file missing: {SPH_SCENE_EMPTY!r} "
                "(repo-local data/ assets are required)")
        os.makedirs(SPH_OUTPUT_DIR, exist_ok=True)

        self.pysplish_base = sph.Exec.SimulatorBase()
        self.pysplish_base.init(useGui=False,
                                sceneFile=SPH_SCENE_EMPTY,
                                outputDir=SPH_OUTPUT_DIR)

        scene_config = sph.Exec.SceneConfiguration.getCurrent()
        self.scene = scene_config.getScene()

        # 关键：设置 particleRadius 必须在添加 boundary/fluid 之前
        self.scene.particleRadius = CONFIG.PARTICLE_RADIUS

        # 添加边界（使用动态 mapResolution）
        self._setup_boundary_with_scenes_module()

        # 添加流体块
        self._setup_fluid_blocks()

        # 禁用缓存，强制重新生成 Volume Map（重要！）
        self.pysplish_base.setUseParticleCaching(False)

        # 初始化仿真（内部会调用 sim->init(scene.particleRadius, ...)）
        self.pysplish_base.initSimulation()

        self.pysplish_simulation = sph.Simulation.getCurrent()
        self.pysplish_time_manager = sph.TimeManager.getCurrent()

        self.pysplish_simulation.setValueBool(
            self.pysplish_simulation.ENABLE_Z_SORT, False
        )
        # 【裁决 D3'-3 补充：前提锁死】写完必须读回并当场断言。
        # 探针 E2（D:\tmp_pg_verify\impl\e2.log）实测：一旦 z-sort 被打开，
        # performNeighborhoodSearch() 会把 id 连同粒子一起搬到新行（64/64 行不符），
        # 而随后 add_particles 内部的 FluidModel::reset() 又按新行号把 id 全部重编回
        # 0..n-1 —— 于是 sync_particle_data 里"ids==行号"那条护栏会重新"通过"，
        # 身份却早已被摧毁（各行位置在 reset 前后一字不动，只有 id 被改写）。
        # 结论：ids==arange 护栏是必要不充分条件，真正的前提是 z-sort 必须为关。
        # 这里让它在初始化阶段就炸，而不是留到运行期静默错位。
        if bool(self.pysplish_simulation.zSortEnabled()):
            raise RuntimeError(
                "Z-Sort 未能按预期关闭（setValueBool(ENABLE_Z_SORT, False) 之后 "
                "zSortEnabled() 仍为 True）。粒子身份契约（particleId==行号）"
                "在此配置下不成立，化学状态映射必然张冠李戴，禁止启动仿真。"
            )
        print("✓ Disabled Z-Sort (eliminates periodic stutter) "
              "- zSortEnabled() read back: %s"
              % bool(self.pysplish_simulation.zSortEnabled()))

        # 【关键修复】显式设置重力。
        # 原 WIP 写的 setValueVector3(...) / GRAVITY 在 pysplishsplash 2.16.0 中都不存在，
        # AttributeError 会中止本函数后续所有初始化（:196 之后的 radius 校验、
        # 边界 VolumeMap、DFSPH 参数、流体模型引用全部没跑）。
        # 真实 API：常量 GRAVITATION，setter setVec3ValueReal(int, [3]float)。
        # 该参数在 C++ 侧直接绑定到 Simulation::m_gravitation.data()
        # （SPlisHSPlasH/Simulation.cpp:188），因此写入即时生效；
        # 必须在 initSimulation() 之后调用，之前所有大写常量都是 -1。
        gravity = [0.0, -9.81, 0.0]
        self.pysplish_simulation.setVec3ValueReal(
            self.pysplish_simulation.GRAVITATION, gravity)
        _g_now = self.pysplish_simulation.getVec3ValueReal(
            self.pysplish_simulation.GRAVITATION)
        print(f"  Gravity set to {gravity}, read back {list(_g_now)}")

        # 验证 particleRadius 已正确传播到 Simulation 对象
        actual_radius = self.pysplish_simulation.getParticleRadius()
        actual_support = self.pysplish_simulation.getSupportRadius()
        print(f"  Simulation particleRadius={actual_radius}, supportRadius={actual_support}")
        assert abs(actual_radius - CONFIG.PARTICLE_RADIUS) < 1e-6, \
            f"particleRadius mismatch: expected {CONFIG.PARTICLE_RADIUS}, got {actual_radius}"

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

        # 【修复 a4】fluid model 必须由 initSimulation() 之前的退化 FluidBlock 产生，
        # 这里绝不再"运行时补模型"：
        # SimulationDataDFSPH::init() 只在 initSimulation() 内被调用一次，
        # 那一刻 nFM==0 就意味着 solver 的 per-model 缓冲区永久为空，
        # 之后运行时 addFluidModel 只补得回粒子缓冲、补不回 solver 缓冲，
        # emittedParticles()/timeStepNoGUI() 访问 m_pressure_rho2[0][j] 直接 SIGSEGV(139)。
        num_fluid_models = self.pysplish_simulation.numberOfFluidModels()
        if num_fluid_models == 0:
            raise RuntimeError(
                "initSimulation() 后没有任何流体模型：退化 FluidBlock 未生效。"
                "此时 DFSPH 的 per-model 缓冲区不会被 sizing，运行时加粒子必然崩溃，"
                "拒绝继续（禁止回退为运行时 addFluidModel）。")

        self.pysplish_fluid_model = self.pysplish_simulation.getFluidModel(0)
        if not self.pysplish_fluid_model:
            raise RuntimeError(
                f"numberOfFluidModels()={num_fluid_models} 但 getFluidModel(0) 返回 None")

        capacity = int(self.pysplish_fluid_model.numParticles())
        active = int(self.pysplish_fluid_model.numActiveParticles())
        print(f"  Fluid model 0: capacity={capacity}, active={active}")

        # 【决策 D1】启动必须是空场，粒子只在用户点击后出现。
        if active != 0:
            raise RuntimeError(
                f"启动时场里不该有粒子（实际 active={active}）。"
                "FluidBlock 必须是退化块（boxMin==boxMax），不允许生成实体初始粒子。")
        if capacity < CONFIG.MAX_EMITTER_PARTICLES:
            raise RuntimeError(
                f"流体模型容量 {capacity} < MAX_EMITTER_PARTICLES "
                f"{CONFIG.MAX_EMITTER_PARTICLES}：Material.maxEmitterParticles 未生效，"
                "运行时加粒子会越界崩溃。")

        # 【修复 a5】首次邻域搜索：把 point set 注册进 NeighborhoodSearch。
        # 未做这一步，后续 FluidModel::reset() 会抛
        # RuntimeError: Neighborhood search was not initialized.
        self.pysplish_simulation.performNeighborhoodSearch()
        print("  Initial neighborhood search performed")

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
            tm = sph.TimeManager.getCurrent()
            tm.setTimeStepSize(CONFIG.DEFAULT_TIME_STEP)

            try:
                sim.setValueInt(sim.CFL_METHOD, 1)
                sim.setValueFloat(sim.CFL_FACTOR, CONFIG.CFL_FACTOR)
                sim.setValueFloat(sim.CFL_MIN_TIMESTEPSIZE, CONFIG.MIN_TIME_STEP)
                sim.setValueFloat(sim.CFL_MAX_TIMESTEPSIZE, CONFIG.MAX_TIME_STEP)
                print("✓ Configured CFL time step limits")
            except Exception as e:
                print(f"⚠ CFL configuration: {e}")

        except Exception as e:
            print(f"⚠ Time step limit configuration: {e}")

    def _setup_fluid_blocks(self):
        """设置流体块和材质配置（保证initSimulation内就创建fluid model）"""
        try:
            # 添加材质配置，预留 maxEmitterParticles 空间以便运行时动态添加粒子
            material = Scenes.MaterialData()
            material.id = 'Fluid'
            material.maxEmitterParticles = CONFIG.MAX_EMITTER_PARTICLES
            material.emitterReuseParticles = False
            material.emitterBoxMin = CONFIG.BOUNDARY_MIN.tolist()
            material.emitterBoxMax = CONFIG.BOUNDARY_MAX.tolist()
            self.scene.materials.append(material)
            print(f"✓ Material added with maxEmitterParticles={CONFIG.MAX_EMITTER_PARTICLES}")

            # 【决策 D1 / 修复 a4】用一个"退化 FluidBlock"只换取 sizing，
            # 启动时场里必须没有粒子。
            #
            # 为什么不能只在运行时 addFluidModel：
            #   SimulationDataDFSPH::init()（DFSPH/SimulationDataDFSPH.cpp:22-42）
            #   按 initSimulation() 当时的 numberOfFluidModels() resize 外层 vector、
            #   按各 FluidModel 的 numParticles() resize 内层 vector。
            #   initSimulation() 时 nFM==0 → m_pressure_rho2 是空的，
            #   之后运行时补模型只补得回"粒子缓冲"，补不回"solver 缓冲"，
            #   emittedParticles/timeStepNoGUI 访问 m_pressure_rho2[0][j] 即越界 → SIGSEGV(139)。
            #
            # 引擎源码给出的合法通道（Simulator/SimulatorBase.cpp）：
            #   :1109-1112  fluidIDs 先按 scene.fluidBlocks 的 id 建好（与是否采样成功无关）
            #   :1460-1464  stepsX/Y/Z <= 1 时打 "FluidBlock ... is empty." 并 continue（0 个粒子）
            #   :1264-1274  仍对每个 fluidID 调 addFluidModel(id, 0, ..., maxEmitterParticles)
            #   FluidModel::initModel（FluidModel.cpp:285-327）把各字段 resize 到
            #   nFluidParticles + nMaxEmitterParticles = 0 + 5000
            #   → numParticles()=5000（solver 缓冲够用）、numActiveParticles()=0（空场）
            #
            # boxMin == boxMax == (0,0,0) → diff=(0,0,0) → steps=-1 → 走 empty 分支。
            # 严禁改回 HEAD 的实体块（会在启动时生成约 324 个粒子，用户已明确否决）。
            self.scene.fluidBlocks.append(
                Scenes.FluidBlock(
                    id='Fluid',
                    boxMin=[0.0, 0.0, 0.0],
                    boxMax=[0.0, 0.0, 0.0],
                    mode=0,
                    initialVelocity=[0.0, 0.0, 0.0]
                )
            )
            print("✓ Degenerate FluidBlock appended (sizing-only: 0 active particles, "
                  f"capacity={CONFIG.MAX_EMITTER_PARTICLES})")

        except Exception as e:
            print(f"✗ Fluid block setup failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _configure_dfsph_parameters(self):
        """配置DFSPH参数（在fluid model创建后调用）"""
        try:
            if self.pysplish_simulation.numberOfFluidModels() > 0:
                fluid = self.pysplish_simulation.getFluidModel(0)
                if fluid:
                    fluid.setValueFloat(fluid.DENSITY0, 1000.0)
                    print("✓ Configured fluid model density")

            time_step = self.pysplish_simulation.getTimeStep()
            if time_step:
                time_step.setValueUInt(time_step.MIN_ITERATIONS, CONFIG.DFSPH_MIN_ITERATIONS)
                time_step.setValueUInt(time_step.MAX_ITERATIONS, CONFIG.DFSPH_MAX_ITERATIONS)
                time_step.setValueFloat(time_step.MAX_ERROR, CONFIG.DFSPH_MAX_ERROR)
                time_step.setValueUInt(time_step.MAX_ITERATIONS_V, CONFIG.DFSPH_MAX_ITERATIONS_V)
                time_step.setValueFloat(time_step.MAX_ERROR_V, CONFIG.DFSPH_MAX_ERROR_V)
                time_step.setValueBool(time_step.USE_DIVERGENCE_SOLVER, True)
                print("✓ Configured DFSPH time step parameters")

        except Exception as e:
            print(f"⚠ DFSPH parameter configuration warning: {e}")

    @profile_function
    def compute_forces(self):
        """执行一步物理仿真 - 修复版：带时间预算控制"""
        try:
            start_time = time.perf_counter()

            if not self.pysplish_simulation:
                print("[DEBUG compute_forces] Simulation object is None")
                return

            # 【修复 a6】nFM==0 时绝不能调 getFluidModel(0)：
            # pybind 层没有边界检查，实测 exit 139（SIGSEGV）。
            # 也不再"运行时重新补一个流体模型"（见 _initialize_pysplish_components 的说明），
            # 那种模型没有 solver 缓冲区，走到 emittedParticles 一样崩。
            # 按决策 D4：这是状态被破坏的信号，硬失败，不静默跳过物理。
            if self.pysplish_simulation.numberOfFluidModels() == 0:
                raise RuntimeError(
                    "compute_forces: numberOfFluidModels()==0，"
                    "fluid model 在初始化之后丢失，无法安全继续物理计算。")

            fluid_model = self.pysplish_simulation.getFluidModel(0)
            if not fluid_model:
                raise RuntimeError(
                    "compute_forces: getFluidModel(0) 返回 None "
                    "（numberOfFluidModels()>0）")
            self.pysplish_fluid_model = fluid_model

            particle_count = fluid_model.numActiveParticles()

            if particle_count == 0:
                return

            try:
                self.pysplish_base.timeStepNoGUI()
            except Exception as e:
                print(f"[DEBUG compute_forces] timeStepNoGUI 异常: {e}")
                import traceback
                traceback.print_exc()
                # 不抛出异常，让仿真继续运行
                return

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.performance_stats['force_computation_time'] = elapsed_ms / 1000.0

            self._last_physics_time = elapsed_ms

        except Exception as e:
            print(f"✗ Error in compute_forces: {e}")
            import traceback
            traceback.print_exc()

    def _expected_ids_u32(self, n: int) -> np.ndarray:
        """身份护栏用的 0..n-1 期望序列（uint32，与引擎 id 字段同类型）。

        按需几何增长并缓存：原实现每帧 np.arange(n) 都要新分配一块 N 大小的
        内存，10M 粒子时是每帧 80 MB 的分配+写入。用 uint32 而非 int64 是为了
        和引擎缓冲同宽，比较时少一半内存流量。
        """
        cache = self._expected_ids_cache
        if cache.size < n:
            grow = max(n, cache.size * 2, 1024)
            cache = np.arange(grow, dtype=np.uint32)
            self._expected_ids_cache = cache
        return cache[:n]

    def sync_particle_data(self):
        """同步粒子数据 - 修复版：正确处理手动添加的粒子与PySPlisHSPlasH的集成"""
        t = time.perf_counter()
        fm = self.pysplish_fluid_model
        if fm is None:
            print("Warning: pysplish_fluid_model is None")
            return

        try:
            # 获取PySPlisHSPlasH中的当前粒子数
            pysplish_active_count = int(fm.numActiveParticles())
            
            # 从PySPlisHSPlasH同步粒子数据到我们的系统
            if pysplish_active_count > 0:
                # 安全的粒子数据同步
                pos_buffer = fm.getFieldBuffer("position")
                pos_view = np.asarray(pos_buffer)
                
                vel_buffer = fm.getFieldBuffer("velocity")
                vel_view = np.asarray(vel_buffer)

                # 【修复 b3/D3】粒子身份：引擎字段注册表里的键名是 "id"（uint32）。
                # 注意陷阱：getFieldBuffer("particleId") 和 ("objectId") 都会静默
                # 别名到 density 缓冲区（实测读出 443.89 这类密度值），不报错、
                # 不返回 None —— 用它当粒子身份会神不知鬼不觉地张冠李戴。
                id_view = np.asarray(fm.getFieldBuffer("id"))
                if id_view.ndim != 1:
                    raise ValueError(
                        f"particle id buffer 不是一维: shape={id_view.shape}")
                if id_view.shape[0] < pysplish_active_count:
                    raise ValueError(
                        f"id buffer 太短: {id_view.shape[0]} < {pysplish_active_count}")
                
                # 严格验证buffer形状和类型
                if pos_view.ndim != 2 or pos_view.shape[1] != 3:
                    raise ValueError(f"Position buffer shape invalid: {pos_view.shape}, expected (N, 3)")
                if vel_view.ndim != 2 or vel_view.shape[1] != 3:
                    raise ValueError(f"Velocity buffer shape invalid: {vel_view.shape}, expected (N, 3)")
                if pos_view.shape[0] < pysplish_active_count or vel_view.shape[0] < pysplish_active_count:
                    raise ValueError(f"Buffer size insufficient: pos={pos_view.shape[0]}, vel={vel_view.shape[0]}, need={pysplish_active_count}")
                
                # 确保数据类型为float32
                if pos_view.dtype != np.float32:
                    pos_view = pos_view.astype(np.float32)
                if vel_view.dtype != np.float32:
                    vel_view = vel_view.astype(np.float32)
                
                # 拷贝PySPlisHSPlasH的粒子数据到我们的系统
                pysplish_count = min(pysplish_active_count, CONFIG.MAX_PARTICLES)
                
                np.copyto(self.particle_positions[:pysplish_count], pos_view[:pysplish_count], casting="same_kind")
                np.copyto(self.particle_velocities[:pysplish_count], vel_view[:pysplish_count], casting="same_kind")
                self.particle_ids[:pysplish_count] = id_view[:pysplish_count].astype(np.int64, copy=False)

                # ===【裁决 D3'-3】粒子身份硬护栏 ===
                # 化学状态(_particle_solutions)按 particleId 建行，而 particleId 之所以
                # 能当"从 0 开始的稠密编号"用，前提是：
                #   (1) ENABLE_Z_SORT == False —— 本文件初始化处显式关闭并读回断言，
                #       所以引擎不会 sort_field 把行搬走；
                #   (2) FluidModel::reset() 每次都把全部容量行重写成 m_particleId[i]=i
                #       （SPlisHSPlasH/SPlisHSPlasH/FluidModel.cpp:210），
                #       而 add_particles 每批都调 reset()。
                # 实测（探针 E1，D:\tmp_pg_verify\impl\e1.log）：关掉 z-sort 后连续
                # 3 批 add(8/8/8) 与随后 20 帧，id 数组恒等于 list(range(active))。
                # 本护栏是【必要不充分】条件：探针 E2 证明 z-sort 开着时 sort 先把 id
                # 随粒子搬走、紧接着 add_particles 内部的 reset() 又按新行号把 id 全部
                # 重编（各行位置一字不动），这一条会重新"通过"而身份早已被摧毁。
                # 真正的保护是初始化处对 zSortEnabled() 的读回断言；这里兜住的是
                # "其它机制在非尾部插入粒子"（例如改用官方 Emitter）这类情况。
                expected_ids = self._expected_ids_u32(pysplish_count)
                actual_ids = self.particle_ids[:pysplish_count]
                if not np.array_equal(id_view[:pysplish_count], expected_ids):
                    n_diff = int((actual_ids != expected_ids).sum())
                    raise RuntimeError(
                        "检测到粒子身份被重排/重编号，化学状态映射已失效："
                        "期望 particleId==行号(0..%d)，实际有 %d/%d 行不符，"
                        "实际 ids 前 16 个=%s。"
                        "当前 zSortEnabled()=%r（初始化处已断言其必须为 False）："
                        "True 说明有人绕过初始化直接改开关；"
                        "False 则说明引擎改用了在非尾部插入粒子的机制（如 Emitter）。"
                        "此时按 id 索引的 _particle_solutions 会张冠李戴，禁止继续仿真。"
                        % (pysplish_count - 1, n_diff, pysplish_count,
                           actual_ids[:16].tolist(),
                           bool(self.pysplish_simulation.zSortEnabled())))

                # 更新粒子计数
                self.current_particle_num = pysplish_count
            else:
                # 如果PySPlisHSPlasH中没有粒子，则清空我们的系统
                self.current_particle_num = 0

        except (ValueError, RuntimeError):
            # 【决策 D4】缓冲区形状/身份数据异常不能退化成"当成没有粒子"：
            # 那会让上层看到 0 个粒子、以为一切正常，实际是同步失败。
            raise
        except Exception as e:
            print(f"Error syncing particle data: {e}")
            import traceback
            traceback.print_exc()
            self.current_particle_num = 0

    def step_simulation(self, dt: float = CONFIG.DEFAULT_TIME_STEP) -> Tuple[List, List]:
        """Execute one simulation step - 修复版：带跳帧机制"""
        # 性能采样控制
        self.frame_sample_counter += 1
        should_sample = (self.performance_capture and
                        self.frame_sample_counter % self.performance_capture.sampling_rate == 0)
        
        # 开始帧计时
        if should_sample:
            self.performance_capture.start_frame_timer()
        
        self.current_frame += 1

        # 【修复】确保时间步长在每帧开始时正确设置
        if self.pysplish_simulation:
            try:
                tm = sph.TimeManager.getCurrent()
                # 使用配置的时间步长，而不是依赖内部默认值
                tm.setTimeStepSize(CONFIG.DEFAULT_TIME_STEP)
                # 注意：不需要手动设置 DT，时间步长由 TimeManager 管理
            except Exception as e:
                print(f"[DEBUG] 时间步长设置警告: {e}")

        # 跳帧机制 - 修复版：减少跳过帧数，缩短稳定期
        if self._skip_physics_count > 0:
            self._skip_physics_count -= 1
            self.sync_particle_data()
            if should_sample:
                self.performance_capture.end_frame_timer()
            return [], []

        # 开始物理计算计时
        if should_sample:
            self.performance_capture.start_timer('physics')
            
        # Physics simulation - 添加异常保护
        forces_start = time.perf_counter()
        try:
            self.compute_forces()
        except Exception as e:
            print(f"[DEBUG step_simulation] compute_forces 异常: {e}")
            import traceback
            traceback.print_exc()
        self.performance_stats['force_computation_time'] = time.perf_counter() - forces_start

        # Sync particle data from SPH solver
        self.sync_particle_data()

        # 结束物理计算计时
        if should_sample:
            physics_time = self.performance_capture.end_timer('physics')
            self.performance_capture.record_snapshot({
                'particle_count': self.current_particle_num
            })

        # Conditional mesh update
        mesh_vertices, mesh_triangles = [], []
        if not CONFIG.USE_SSFR:
            if (self.current_frame - self.last_mesh_update) >= CONFIG.MC_MESH_UPDATE_INTERVAL:
                if should_sample:
                    self.performance_capture.start_timer('mesh')
                    
                mesh_start = time.perf_counter()
                mesh_vertices, mesh_triangles = self.update_mesh()
                self.last_mesh_update = self.current_frame
                self.performance_stats['mesh_generation_time'] = time.perf_counter() - mesh_start
                
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

    def capacity(self) -> int:
        """流体粒子容量（不含边界）。后端无关的容量查询入口。"""
        fm = self.pysplish_fluid_model
        if fm is not None:
            return int(fm.numParticles())
        return int(CONFIG.MAX_EMITTER_PARTICLES)

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
            n = self.current_particle_num
            fluid_positions = self.particle_positions[:n]
            if n == 0:
                return [], []

            start_time = time.perf_counter()

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

            mesh = result.mesh
            vertices = mesh.vertices
            triangles = mesh.triangles

            gen_time = time.perf_counter() - start_time
            self.performance_stats['mesh_generation_time'] = gen_time
            self.performance_stats['total_mesh_vertices'] = len(vertices)
            self.performance_stats['total_mesh_triangles'] = len(triangles)

            vertex_memory = vertices.nbytes / (1024 * 1024)
            triangle_memory = triangles.nbytes / (1024 * 1024)
            self.performance_stats['memory_usage_mb'] = vertex_memory + triangle_memory

            return vertices, triangles

        except Exception as e:
            print(f"PySplashSurf mesh generation failed: {e}")
            return [], []

    def initialize_fluid_drops(self):
        """Initialize multiple fluid drops"""
        self.sync_particle_data()
        print(f"Using PySPlisHSPlasH default scene initialization, particle count: {self.current_particle_num}")
        return self.current_particle_num

    def add_particles(self, positions: np.ndarray, fluid_type: int = 1):
        """向引擎添加新粒子，并返回它们的稳定 particleId 列表。

        由于 CompactNSearch::NeighborhoodSearch 未在 pybind11 中绑定，
        resize_point_set 不可从 Python 直接调用。

        绕过方案：
        1. 将当前所有粒子的物理状态存入 position0/velocity0（作为"快照"）
        2. 将新粒子同样写入 position0/velocity0
        3. 调用 fm.reset()：内部会触发 resize_point_set + 从 position0 恢复
        4. 执行邻域搜索

        这个方案的代价：reset() 会清零 acceleration，重置 density，
        但 position 和 velocity 会从 position0/velocity0 完整恢复，
        物理仿真连续性基本不受影响（损失当前帧的 acceleration，可接受）。

        在添加粒子后设置稳定期（_skip_physics_count），跳过最初的一帧物理计算，
        让 C++ 扩展完成内部初始化，避免崩溃。

        Returns:
            list[int]: 新增粒子的稳定 particleId（顺序与 positions 一致）。
            调用方（GUI）必须用这些 id 去播种化学状态，
            不许自己拿"添加前的粒子数"当粒子号（决策 D3）。

        Raises:
            RuntimeError: 任何一条没能真正把粒子加进引擎的路径都直接抛错。
            （决策 D4：过去是 print 一句警告然后 return，界面上表现为
            "点了没反应但不报错"，甚至掉进没有物理的本地回退分支。）
        """
        if positions is None or len(positions) == 0:
            raise RuntimeError("add_particles: 没有任何位置可添加")

        n_new = len(positions)

        if self.pysplish_fluid_model is not None:
            try:
                # 使用顶部已经导入的 sph 模块，避免重复导入
                fm = self.pysplish_fluid_model
                current_active = int(fm.numActiveParticles())
                max_particles = int(fm.numParticles())
                available = max_particles - current_active

                if available < n_new:
                    raise RuntimeError(
                        f"求解器数组空间不足：要加 {n_new} 个，只剩 {available} "
                        f"(活跃={current_active}, 容量={max_particles})。"
                        "（决策 D4：不再 n_to_add=min(...) 静默截断——那样调用方"
                        "会按'请求的个数'去播种化学，多出来的下标全是空头。）")

                n_to_add = n_new
                positions_to_add = np.asarray(positions, dtype=np.float32)
                new_active = current_active + n_to_add

                # === 步骤 1：批量快照 + 批量写入（零拷贝字段视图） ===
                # getFieldBuffer 返回的是直达 C++ 缓冲的可写 memoryview
                # （pySPlisHSPlasH/FluidModelModule.cpp:103），所以整块赋值就是
                # 一次 memcpy。原实现对每个老粒子调 setPosition0+setVelocity0，
                # 是 2N 次 pybind 往返 + 2N 次 .tolist()，点击卡顿的主因。
                #
                # 【陷阱】"velocity0" 不在引擎字段注册表里（FluidModel.cpp:61-67
                # 只注册了 id/state/object_id/position/position0/velocity/density）。
                # getFieldBuffer("velocity0") 不报错，而是静默别名到别的字段：
                # 实测返回 (capacity,) 而不是 (capacity,3)，写进去等于污染密度场。
                # 所以速度不能走视图，改为在 reset() 之后整块还原（见步骤 4）。
                pos_view = np.asarray(fm.getFieldBuffer("position"))
                vel_view = np.asarray(fm.getFieldBuffer("velocity"))
                pos0_view = np.asarray(fm.getFieldBuffer("position0"))

                vel_snapshot = vel_view[:current_active].copy()
                pos0_view[:current_active] = pos_view[:current_active]

                # === 步骤 2：写入新粒子（position/position0 同值） ===
                pos_view[current_active:new_active] = positions_to_add
                pos0_view[current_active:new_active] = positions_to_add

                # === 步骤 3：更新 numActiveParticles0（reset() 会从这里读取目标数量） ===
                fm.setNumActiveParticles0(new_active)

                # === 步骤 4：调用 fm.reset()，随后整块写回速度 ===
                # reset() 内部执行（SPlisHSPlasH/FluidModel.cpp:196-232）：
                #   setNumActiveParticles(numActiveParticles0)  ← = new_active ✓
                #   for i in 0..numParticles: pos[i]=pos0[i], vel[i]=vel0[i],
                #                             acceleration=0, density=0, particleId[i]=i
                #   neighborhoodSearch->resize_point_set(pointSetIndex, &pos[0], new_active)  ← 核心！✓
                #   emitterSystem->reset()  ← emitter 计数器重置，无害
                # 位置由步骤 1 的 position0 快照恢复。
                #
                # 速度不走 m_v0：reset() 会把 vel0 拷进 vel，但 vel0 没有可写视图
                # （getFieldBuffer("velocity0") 会静默别名到别的字段，见步骤 1 注释），
                # 逐个 setVelocity0 又是 O(N) 次 pybind 往返。既然 reset() 之后
                # 我们本来就要整块写回 vel，就直接在这里定终态：老粒子=快照，
                # 新粒子=0。m_v0 里留什么值都无所谓——全仓库只有本函数调 reset()，
                # 而每次 reset() 后 vel 都会被下面这两行覆盖。
                fm.reset()
                vel_after = np.asarray(fm.getFieldBuffer("velocity"))
                vel_after[:current_active] = vel_snapshot
                vel_after[current_active:new_active] = 0.0

                # === 步骤 5：通知 solver 缓冲区（在 reset 之后） ===
                # reset 之后 numActiveParticles = new_active，emittedParticles 从 current_active 开始
                self.pysplish_simulation.emittedParticles(fm, current_active)

                # === 步骤 6：执行邻域搜索 ===
                self.pysplish_simulation.performNeighborhoodSearch()

                print(f"Adding {n_to_add} particles. New total: {new_active}, Capacity: {max_particles}")
                self.sync_particle_data()

                # === 步骤 7：取回这批新粒子的稳定 particleId（决策 D3） ===
                # 必须在 sync_particle_data() 之后取：那时 particle_ids 才是
                # 引擎 reset/sort 之后的真实行→id 映射。
                actual_active = int(fm.numActiveParticles())
                if actual_active != new_active:
                    raise RuntimeError(
                        f"引擎没有接受全部新粒子: 期望 {new_active} 个活跃，"
                        f"实际 {actual_active} 个")
                new_ids = self.particle_ids[current_active:new_active].tolist()
                if len(new_ids) != n_to_add:
                    raise RuntimeError(
                        f"取新粒子 id 失败: 需要 {n_to_add} 个，拿到 {len(new_ids)} 个")
                if len(set(new_ids)) != len(new_ids):
                    raise RuntimeError(
                        f"新粒子 id 出现重复: {new_ids} —— 粒子身份不再唯一，"
                        "按 id 存化学状态会互相覆盖")

                print(f"[OK] 成功添加 {n_to_add}/{n_new} 个粒子，"
                      f"总活跃粒子数: {actual_active}，新 ids: {new_ids}")

                # === 设置稳定期：跳过添加后的 1 帧物理，让 C++ 侧完成内部初始化 ===
                self._skip_physics_count = 1

                return new_ids

            except RuntimeError:
                raise
            except Exception as e:
                # 【决策 D4】以前这里打印一句警告然后"掉到下面的本地回退分支"，
                # 造出一批没有物理、只有数组位置的假粒子。添加失败必须让调用方知道。
                raise RuntimeError(
                    f"向 PySPlisHSPlasH 添加粒子失败: {e!r}") from e

        # 【决策 D4 / 修复 a7】下面这段"本地回退"以前会造出一批只有数组位置、
        # 没有任何物理和化学的假粒子，界面看起来"点出东西了"，实际是死数据。
        # 现在 a7 已经保证引擎缺失时在构造阶段就抛错，走到这里只可能是
        # FluidModel 指针丢失——那是必须硬失败的内部不一致，不再回退。
        raise RuntimeError(
            f"MeshSystem 没有可用的 FluidModel，无法添加 {n_new} 个粒子"
            "（禁止回退到无物理的本地假粒子数组）")

    def get_particle_colors_by_type(self) -> np.ndarray:
        """根据粒子类型获取颜色
        
        Returns:
            colors: (N, 3) RGB颜色数组，值范围[0,1]
        """
        if self.current_particle_num == 0:
            return np.zeros((0, 3), dtype=np.float32)
            
        colors = np.zeros((self.current_particle_num, 3), dtype=np.float32)
        types = self.particle_types[:self.current_particle_num]
        
        # 根据类型设置颜色
        # 1=水(蓝色), 2=酸性(红色), 3=碱性(绿色)
        colors[types == 1] = [0.2, 0.6, 1.0]  # 蓝色
        colors[types == 2] = [1.0, 0.3, 0.3]  # 红色
        colors[types == 3] = [0.3, 1.0, 0.3]  # 绿色
        
        return colors
    
    def get_system_stats(self) -> Dict:
        """Get system statistics - 增强版"""
        # 统计各类型粒子数量
        type_counts = {}
        if self.current_particle_num > 0:
            types = self.particle_types[:self.current_particle_num]
            unique_types, counts = np.unique(types, return_counts=True)
            for t, c in zip(unique_types, counts):
                type_counts[f'type_{int(t)}'] = int(c)
        
        return {
            'frame': self.current_frame,
            'particle_count': self.current_particle_num,
            'particle_types': type_counts,
            'mesh_vertices': self.performance_stats['total_mesh_vertices'],
            'mesh_triangles': self.performance_stats['total_mesh_triangles'],
            'memory_usage_mb': self.performance_stats['memory_usage_mb'],
            'performance': self.performance_stats.copy()
        }