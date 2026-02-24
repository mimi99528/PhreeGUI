"""
网格系统核心模块
Core Mesh System Module
"""
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
            'time_step': CONFIG.DEFAULT_TIME_STEP,
            'gravity': [0.0, -9.8, 0.0],
            'viscosity': 0.01,
            'surface_tension': 0.05,
            'density0': 1000.0
        }

        # Particle data storage - 预分配批量缓冲区
        self.particle_positions = np.zeros((CONFIG.MAX_PARTICLES, 3), dtype=np.float32)
        self.particle_velocities = np.zeros((CONFIG.MAX_PARTICLES, 3), dtype=np.float32)
        self.particle_accelerations = np.zeros((CONFIG.MAX_PARTICLES, 3), dtype=np.float32)
        self.particle_materials = np.zeros(CONFIG.MAX_PARTICLES, dtype=np.int32)
        self.particle_types = np.ones(CONFIG.MAX_PARTICLES, dtype=np.int32)  # 1=水, 2=酸性, 3=碱性

        # 批量处理临时缓冲区
        self.temp_positions = np.empty((CONFIG.BATCH_SIZE, 3), dtype=np.float32)
        self.temp_velocities = np.empty((CONFIG.BATCH_SIZE, 3), dtype=np.float32)

        # System state
        self.current_particle_num = 0
        self.current_frame = 0
        self.last_grid_rebuild = -CONFIG.GRID_REBUILD_INTERVAL
        self.last_mesh_update = -CONFIG.MC_MESH_UPDATE_INTERVAL
        self.mesh_enabled = True

        # Boundary parameters
        self.boundary_min = CONFIG.BOUNDARY_MIN.copy()
        self.boundary_max = CONFIG.BOUNDARY_MAX.copy()

        # Boundary container components
        self.boundary_walls = []  # Store boundary wall objects
        self.boundary_size = self.boundary_max - self.boundary_min
        self.mc_cell_size = self.boundary_size / CONFIG.MC_GRID_RES

        # Grid structure for neighbor search optimization
        self.grid_resolution = np.array([10, 10, 10], dtype=int)
        self.grid_cell_size = self.boundary_size / self.grid_resolution
        self.grid_total_cells = np.prod(self.grid_resolution)
        self.particle_grid_indices = np.zeros(CONFIG.MAX_PARTICLES, dtype=int)
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
        if CONFIG.PYSPLISH_AVAILABLE and PYSPLISH_AVAILABLE:
            try:
                self._initialize_pysplish_components()
                print("✓ PySPlisHSPlasH components initialized successfully")
            except Exception as e:
                print(f"✗ PySPlisHSPlasH initialization failed: {e}")
                CONFIG.PYSPLISH_AVAILABLE = False

        # 物理仿真线程化初始化
        self._init_threading()

    def _setup_boundary_with_scenes_module(self):
        """使用官方Scenes模块设置边界（推荐方式）- 修复版"""
        support_radius = 4.0 * CONFIG.PARTICLE_RADIUS
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
        boundary_data.mapResolution = map_res

        self.scene.boundaryModels.append(boundary_data)
        print(f"✓ Boundary container added (mapResolution={map_res}, supportRadius={support_radius:.3f})")

    def _initialize_pysplish_components(self):
        """Initialize PySPlisHSPlasH - 修复版：确保边界与 particleRadius 匹配"""
        self.pysplish_base = sph.Exec.SimulatorBase()
        self.pysplish_base.init(useGui=False, sceneFile=sph.Extras.Scenes.Empty)

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
        print("✓ Disabled Z-Sort (eliminates periodic stutter)")

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

            self.scene.fluidBlocks.append(
                sph.Utilities.SceneLoaderStructs.FluidBlock(
                    id='Fluid',
                    boxMin=[-0.5, 0, -0.5],
                    boxMax=[0.5, 6, 0.5],
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
                raise RuntimeError("Simulation object is None")

            if self.pysplish_simulation.numberOfFluidModels() == 0:
                raise RuntimeError("No fluid models in simulation")

            fluid_model = self.pysplish_simulation.getFluidModel(0)
            if not fluid_model:
                raise RuntimeError("Fluid model is None")

            particle_count = fluid_model.numActiveParticles()

            if particle_count == 0:
                print("⚠ WARNING: No active particles before timeStep!")
                return
            
            result = self.pysplish_base.timeStepNoGUI()

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.performance_stats['force_computation_time'] = elapsed_ms / 1000.0

            self._last_physics_time = elapsed_ms

        except Exception as e:
            print(f"✗ Error in compute_forces: {e}")
            import traceback
            traceback.print_exc()

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
                self.particle_materials[:pysplish_count].fill(0)
                self.particle_types[:pysplish_count].fill(1)  # PySPlisHSPlasH粒子默认为类型1(水)
                
                # 更新粒子计数
                self.current_particle_num = pysplish_count
                print(f"从PySPlisHSPlasH同步了 {pysplish_count} 个粒子")
            else:
                # 如果PySPlisHSPlasH中没有粒子，则清空我们的系统
                self.current_particle_num = 0
                print("PySPlisHSPlasH中没有活动粒子")

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

        # 跳帧机制
        if self._skip_physics_count > 0 and self.current_frame % 2 == 0:
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

        # 结束物理计算计时
        if should_sample:
            physics_time = self.performance_capture.end_timer('physics')
            self.performance_capture.record_snapshot({
                'particle_count': self.current_particle_num
            })

        # Conditional grid rebuilding
        if (self.current_frame - self.last_grid_rebuild) >= CONFIG.GRID_REBUILD_INTERVAL:
            grid_start = time.perf_counter()
            self.rebuild_spatial_grid()
            self.last_grid_rebuild = self.current_frame
            self.performance_stats['grid_rebuild_time'] = time.perf_counter() - grid_start

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
        """添加新的粒子到系统中 - 使用PySPlisHSPlasH官方FluidModel API直接注入粒子
        
        正确方式：通过 FluidModel.setPosition / setVelocity / setNumActiveParticles
        直接在已初始化的仿真中激活预分配的粒子槽位（需在场景初始化时配置
        MaterialData.maxEmitterParticles 以预留足够空间）。
        
        注意：scene.emitters.append() 仅在 initSimulation() 调用前有效，
        运行时无法通过该方式动态添加粒子。

        Args:
            positions: 粒子位置数组，shape (N, 3)
            fluid_type: 液体类型 1=水, 2=酸性溶液, 3=碱性溶液
        """
        if positions is None or len(positions) == 0:
            return

        n_new = len(positions)
        if n_new <= 0:
            return

        if self.pysplish_fluid_model is not None:
            try:
                fm = self.pysplish_fluid_model
                current_active = int(fm.numActiveParticles())
                max_particles = int(fm.numParticles())
                available = max_particles - current_active

                if available <= 0:
                    print(f"⚠ PySPlisHSPlasH空间不足: 当前活跃={current_active}, 最大容量={max_particles}")
                    return

                n_to_add = min(n_new, available)
                positions_to_add = np.asarray(positions[:n_to_add], dtype=np.float64)

                # 1. 写入粒子位置和速度
                for i in range(n_to_add):
                    idx = current_active + i
                    fm.setPosition(idx, positions_to_add[i].tolist())
                    fm.setVelocity(idx, [0.0, 0.0, 0.0])

                # 2. 更新活跃粒子计数
                fm.setNumActiveParticles(current_active + n_to_add)

                # 3. 关键补丁 A：初始化解算器缓冲区（j 从 current_active 开始）
                # 注意：第二个参数必须是新粒子开始的索引！
                self.pysplish_simulation.emittedParticles(fm, current_active)

                print(f"Adding {n_to_add} particles. New total: {current_active + n_to_add}, Capacity: {fm.numParticles()}")
                # 5. 执行邻域搜索
                self.pysplish_simulation.performNeighborhoodSearch()

                # 同步PySPlisHSPlasH数据到本地缓冲区
                self.sync_particle_data()

                print(f"✓ 成功添加 {n_to_add}/{n_new} 个粒子，总活跃粒子数: {fm.numActiveParticles()}")
                return

            except Exception as e:
                print(f"⚠ PySPlisHSPlasH粒子添加失败: {e}")
                import traceback
                traceback.print_exc()

        # PySPlisHSPlasH不可用时的回退：仅添加到本地缓冲区
        # 注意：此模式下粒子不参与物理仿真
        available_space = CONFIG.MAX_PARTICLES - self.current_particle_num
        n_to_add = min(n_new, available_space)
        if n_to_add <= 0:
            print(f"⚠ 本地系统空间不足")
            return

        start_idx = self.current_particle_num
        end_idx = start_idx + n_to_add
        self.particle_positions[start_idx:end_idx] = positions[:n_to_add]
        self.particle_velocities[start_idx:end_idx] = 0.0
        self.particle_types[start_idx:end_idx] = fluid_type
        self.particle_materials[start_idx:end_idx] = 0
        self.current_particle_num = end_idx
        print(f"⚠ 回退到本地系统（无物理仿真）: 添加 {n_to_add} 个粒子，总数: {self.current_particle_num}")
    
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