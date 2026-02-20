"""
PySide6 + ModernGL网格重建系统
使用PySide6作为GUI框架，ModernGL进行OpenGL渲染
"""

import sys
import numpy as np
import math
import time
from typing import Dict, List, Optional, Tuple
from functools import wraps

from OpenGL.GL import glGetString
# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QCheckBox, QComboBox, QGroupBox,
    QTextEdit, QSplitter, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject
from PySide6.QtGui import QFont
from PySide6.QtOpenGLWidgets import QOpenGLWidget

use_ssfr = False
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
import pysplishsplash.Utilities.SceneLoaderStructs as Scenes

# Enable PySPlisHSPlasH
PYSPLISH_AVAILABLE = True
SPLASHSURF_AVAILABLE = True

# 性能分析装饰器
ENABLE_PERFORMANCE_PROFILING = False  # 设置为True启用详细性能分析

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
            end_time = time.perf_counter()
            elapsed = (end_time - start_time) * 1000  # 转换为毫秒
            print(f"{func.__name__}: {elapsed:.2f}ms")
            return result
        except Exception as e:
            end_time = time.perf_counter()
            elapsed = (end_time - start_time) * 1000
            print(f"{func.__name__} ERROR: {elapsed:.2f}ms - {e}")
            raise
    return wrapper

# System parameters
MC_GRID_RES = 16

# 放大粒子半径解决"星星太小"问题
PARTICLE_RADIUS = 0.3

MAX_PARTICLES = 15000
H = PARTICLE_RADIUS * 2.0
GRID_REBUILD_INTERVAL = 1
MC_MESH_UPDATE_INTERVAL = 1

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
        
        # PySplashSurf configuration
        if SPLASHSURF_AVAILABLE:
            self.splashsurf_config = {
                'particle_radius': PARTICLE_RADIUS,
                'rest_density': 1000.0,
                # 按建议调整：smoothing_length = 2.0~2.5 * particle_radius
                'smoothing_length': 2.5 * PARTICLE_RADIUS,
                # cube_size 设为 ~ particle_radius * (1.0~2.0) 的量级
                'cube_size': PARTICLE_RADIUS * 1.5,
                'iso_surface_threshold': 0.05,
                'enable_multithreading': True,
                'enable_simd': True
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

    def _initialize_pysplish_components(self):
        """Initialize PySPlisHSPlasH using recommended Empty scene + fluidBlocks approach"""
        try:
            # 第1步：创建仿真器
            self.pysplish_base = sph.Exec.SimulatorBase()
            
            # 第2步：使用Empty场景初始化（重要：useGui=False）
            self.pysplish_base.init(useGui=False, sceneFile=sph.Extras.Scenes.Empty)
            
            # 第3步：获取场景配置（必须在init之后）
            scene_config = sph.Exec.SceneConfiguration.getCurrent()
            self.scene = scene_config.getScene()
            
            # 第4步：清理边界模型列表（可选，确保干净）
            self.scene.boundaryModels.clear()
            
            # 第5步：添加边界到场景（使用官方Scenes模块构造）
            self._setup_boundary_with_scenes_module()
            
            # 第6步：添加流体块到场景（至少一个fluidBlock，保证initSimulation内就创建fluid model）
            self._setup_fluid_blocks()
            
            # 第7步：初始化仿真场景（此时C++会创建Simulation/TimeStep/FluidModel）
            self.pysplish_base.initSimulation()
            
            # 第8步：获取仿真对象
            self.pysplish_simulation = sph.Simulation.getCurrent()
            
            # 第9步：配置边界处理方法（优先用直接setter）
            self.pysplish_simulation.setBoundaryHandlingMethod(sph.BoundaryHandlingMethods.Bender2019)
            print("✓ Set boundary handling method to Bender2019")
            
            # 第10步：配置仿真方法（DFSPH）
            # 等确认SIMULATION_METHOD != -1后再setValueInt
            if self.pysplish_simulation.getSimulationMethod() != -1:
                self.pysplish_simulation.setValueInt(self.pysplish_simulation.SIMULATION_METHOD, 4)  # DFSPH
                print("✓ Set simulation method to DFSPH")
            
            # 第11步：手动补deferredInit的核心
            self.pysplish_base.getBoundarySimulator().initBoundaryData()
            self.pysplish_base.updateVMVelocity()  # 对于Bender2019
            print("✓ Boundary simulator initialized")
            
            # 第12步：配置DFSPH参数
            self._configure_dfsph_parameters()
            
            # 第13步：获取流体模型引用
            self.pysplish_fluid_model = self.pysplish_simulation.getFluidModel(0)
            
            # 第14步：验证初始化
            if self.pysplish_fluid_model:
                particle_count = self.pysplish_fluid_model.numActiveParticles()
                print(f"✓ Fluid model initialized with {particle_count} particles")
                if particle_count == 0:
                    print("⚠ WARNING: Fluid model has 0 particles!")
            else:
                raise RuntimeError("Failed to get fluid model")
            
            # 第15步：验证边界已创建
            num_boundary_models = self.pysplish_simulation.numberOfBoundaryModels()
            print(f"✓ Boundary models count: {num_boundary_models}")
            
            print("✓ All components initialized successfully!")
            
        except Exception as e:
            print(f"✗ PySPlisHSPlasH initialization error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _setup_boundary_with_scenes_module(self):
        """使用官方Scenes模块设置边界（推荐方式）"""
        try:
            # 使用官方Scenes模块构造BoundaryData
            boundary_data = sph.Utilities.SceneLoaderStructs.BoundaryData()
            # 使用相对路径避免文件找不到问题
            boundary_data.meshFile = "../models/UnitBox.obj"  # 单位盒子（相对路径）
            boundary_data.translation = [0.0, 3.0, 0.0]  # 中心位置
            boundary_data.scale = [4.0, 6.0, 4.0]  # 尺寸：宽4, 高6, 深4
            boundary_data.color = [0.1, 0.4, 0.5, 1.0]  # 蓝色
            boundary_data.isWall = True  # 是墙壁
            boundary_data.mapInvert = True  # 反转SDF（内部为负）
            boundary_data.mapResolution = [25, 25, 25]  # SDF分辨率
            
            # 添加到场景
            self.scene.boundaryModels.append(boundary_data)
            print("✓ Boundary container added using Scenes module")
            
        except Exception as e:
            print(f"✗ Boundary setup with Scenes module failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _setup_fluid_blocks(self):
        """设置流体块（保证initSimulation内就创建fluid model）"""
        try:
            # 更紧凑、更厚：避免"平铺"
            self.scene.fluidBlocks.append(
                Scenes.FluidBlock(
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
        """执行一步物理仿真"""
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

            # 调用时间步长
            result = self.pysplish_base.timeStepNoGUI()

            if not result:
                print("⚠ timeStepNoGUI returned False (simulation end reached)")

            self.performance_stats['force_computation_time'] = time.perf_counter() - start_time

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
            self.scene.boundaryModels.clear()
            
            # Add a simple boundary box
            self.scene.boundaryModels.append(
                sph.Utilities.SceneLoaderStructs.BoundaryData(
                    meshFile="SPlisHSPlasH/data/models/UnitBox.obj", 
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

            # zero-copy views into C++ particle arrays (shape: (numParticles, 3))
            pos_view = np.array(fm.getFieldBuffer("position"), copy=False)
            vel_view = np.array(fm.getFieldBuffer("velocity"), copy=False)

            # copy into your preallocated float32 arrays (only active range)
            np.copyto(self.particle_positions[:n], pos_view[:n], casting="unsafe")
            np.copyto(self.particle_velocities[:n], vel_view[:n], casting="unsafe")

            # materials (vectorized)
            self.particle_materials[:n].fill(0)

        except Exception as e:
            print(f"Error syncing particle data: {e}")
            self.current_particle_num = 0

        # print(f"Particle sync time: {time.perf_counter() - t:.4f} seconds")
        
    def step_simulation(self, dt: float = 0.008) -> Tuple[List, List]:
        """Execute one simulation step with boundary handling"""
        self.current_frame += 1
        
        # Time tracking for each step
        step_start = time.perf_counter()
        
        # Use PySPlisHSPlasH for physics simulation
        forces_start = time.perf_counter()
        self.compute_forces()
        forces_time = time.perf_counter() - forces_start
        print(f"Force computation time: {forces_time*1000:.2f}ms")
        
        # Sync particle data
        sync_start = time.perf_counter()
        self.sync_particle_data()
        sync_time = time.perf_counter() - sync_start
        print(f"Particle data sync time: {sync_time*1000:.2f}ms")
        
        # Apply boundary collision handling
        boundary_start = time.perf_counter()
        self._apply_manual_boundary_collision()
        boundary_time = time.perf_counter() - boundary_start
        print(f"Boundary collision time: {boundary_time*1000:.2f}ms")
        
        # Conditional grid rebuilding
        grid_start = time.perf_counter()
        if (self.current_frame - self.last_grid_rebuild) >= GRID_REBUILD_INTERVAL:
            self.rebuild_spatial_grid()
            self.last_grid_rebuild = self.current_frame
        grid_time = time.perf_counter() - grid_start
        print(f"Grid rebuild time: {grid_time*1000:.2f}ms")
        
        # Conditional mesh update
        if not use_ssfr:
            mesh_start = time.perf_counter()
            mesh_vertices, mesh_triangles = [], []
            if (self.current_frame - self.last_mesh_update) >= MC_MESH_UPDATE_INTERVAL:
                mesh_vertices, mesh_triangles = self.update_mesh()
                self.last_mesh_update = self.current_frame
            mesh_time = time.perf_counter() - mesh_start
            print(f"Mesh update time: {mesh_time*1000:.2f}ms")

            # Total step time
            total_time = time.perf_counter() - step_start
            print(f"Total step time: {total_time*1000:.2f}ms")

            return mesh_vertices, mesh_triangles
        return [], []
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
            
            print(f"PySplashSurf mesh: {len(vertices)} verts, {len(triangles)} tris, {gen_time*1000:.1f}ms")
            print(f"  - smoothing_length: {self.splashsurf_config['smoothing_length']:.3f}")
            print(f"  - cube_size: {self.splashsurf_config['cube_size']:.3f}")
            print(f"  - iso_threshold: {self.splashsurf_config['iso_surface_threshold']:.3f}")
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
        
        # 顶点和索引计数
        self.vertex_count = 0
        self.index_count = 0
        
        # Camera parameters
        self.camera_distance = 16.0
        self.camera_angle_x = 0.5
        self.camera_angle_y = 0.3
        self.camera_target = [0.0, 0.0, 0.0]
        
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
        self.use_ssfr = False  # 默认使用网格渲染，可以通过UI切换到SSFR

        # SSFR quality/perf knobs (甜蜜点)
        self.ssfr_scale = 0.5          # 0.5 = 半分辨率（强烈建议，性价比最高）
        self.ssfr_particle_radius_world = 0.05  # 世界空间粒子半径(默认0.10，适合[-1,1]世界)
        self.ssfr_blur_sigma = 1.5     # 模糊强度
        self.ssfr_blur_radius = 7      # kernel半径（separable，性能还行）
        self.ssfr_depth_sigma = 0.008  # 双边滤波深度sigma

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
        # Fullscreen quad vertex shader
        quad_vs = r'''
            #version 330
            in vec2 in_pos;
            out vec2 v_uv;
            void main() {
                v_uv = in_pos * 0.5 + 0.5;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
        '''
        
        # --- Bilateral blur on depth (separable) ---
        bilateral_fs = r'''
            #version 330
            uniform sampler2D u_depth;
            uniform vec2 u_texel;     // 1/w, 1/h
            uniform vec2 u_dir;       // (1,0) or (0,1)
            uniform int u_radius;     // e.g. 7
            uniform float u_sigma_xy; // spatial sigma in pixels (e.g. 2.5)
            uniform float u_sigma_z;  // depth sigma (e.g. 0.01)

            in vec2 v_uv;
            out float frag_out;

            float w_xy(float x, float sigma) {
                return exp(-(x*x) / (2.0*sigma*sigma));
            }

            void main() {
                float center = texture(u_depth, v_uv).r;

                // background: we clear to 1.0
                if (center >= 0.9999) {
                    frag_out = 1.0;
                    return;
                }

                float sum = 0.0;
                float wsum = 0.0;

                for (int i = -64; i <= 64; ++i) {
                    if (i < -u_radius) continue;
                    if (i >  u_radius) break;

                    vec2 uv = v_uv + u_dir * u_texel * float(i);
                    float d = texture(u_depth, uv).r;

                    // ignore background samples to avoid pulling toward 1.0
                    if (d >= 0.9999) continue;

                    float wx = w_xy(float(i), u_sigma_xy);

                    float dz = d - center;
                    float wz = exp(-(dz*dz) / (2.0*u_sigma_z*u_sigma_z));

                    float w = wx * wz;
                    sum += d * w;
                    wsum += w;
                }

                frag_out = (wsum > 1e-6) ? (sum / wsum) : center;
            }
        '''

        # --- Particle VS: perspective-correct point size from world radius ---
        particle_vs = r'''
        #version 330
        
        uniform mat4 projection;
        uniform mat4 view;
        
        // particle radius in WORLD units (same units as positions)
        uniform float u_particle_radius_world;
        
        // viewport height in PIXELS of the current SSFR target (e.g. _ssfr_h)
        uniform float u_viewport_h;
        
        // extra tweak (1.0 = physically derived size)
        uniform float u_point_scale;
        
        layout(location=0) in vec3 in_pos;
        
        // Pass view-space center position to fragment shader for sphere impostor
        out vec3 v_center_vs;
        
        void main() {
            vec4 view_pos = view * vec4(in_pos, 1.0);
            v_center_vs = view_pos.xyz;
            gl_Position = projection * view_pos;
        
            // right-handed look-at: in front of camera => view_pos.z negative
            float z = max(-view_pos.z, 1e-4);
        
            // projection[1][1] = cot(fov_y/2)
            float proj_y = projection[1][1];
        
            // Pixel radius derived from perspective projection:
            // r_px = (proj_y * r_world / z) * (viewport_h / 2)
            float r_px = (proj_y * u_particle_radius_world / z) * (u_viewport_h * 0.5);
        
            // gl_PointSize is DIAMETER in pixels
            gl_PointSize = clamp(2.0 * r_px * u_point_scale, 1.0, 2048.0);
        }
        '''

        # --- Depth FS: output real depth (0..1) using gl_FragCoord.z, with sphere impostor correction ---
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
        
            float zcap = sqrt(max(1.0 - r2, 0.0)); // 0..1
        
            // In view space, camera looks down -Z, so points closer have larger (less negative) z.
            // Sphere surface point along view direction:
            vec3 surf_vs = v_center_vs + vec3(0.0, 0.0, u_particle_radius_world * zcap);
        
            vec4 clip = projection * vec4(surf_vs, 1.0);
            float ndc_z = clip.z / clip.w;          // -1..1
            float depth = ndc_z * 0.5 + 0.5;        // 0..1
        
            out_depth = clamp(depth, 0.0, 1.0);
        }
        '''

        # --- Thickness FS: linear additive thickness field (方案1) ---
        particle_thick_fs = r'''
            #version 330
            out float out_thick;

            void main() {
                vec2 p = gl_PointCoord * 2.0 - 1.0;
                float r2 = dot(p,p);
                if (r2 > 1.0) discard;

                float zcap = sqrt(1.0 - r2); // 0..1
                // 方案1：输出线性可累加的厚度
                out_thick = zcap * zcap;  // 线性厚度，可直接累加
            }
        '''

        # --- Shadow depth pass: from light view ---
        shadow_vs = r'''
            #version 330
            uniform mat4 u_light_proj;
            uniform mat4 u_light_view;
            uniform float u_particle_radius_world;
            uniform float u_viewport_h;

            layout(location=0) in vec3 in_pos;

            void main() {
                vec4 view_pos = u_light_view * vec4(in_pos, 1.0);
                gl_Position = u_light_proj * view_pos;

                float z = -view_pos.z;
                z = max(z, 1e-3);

                float proj_y = u_light_proj[1][1];
                float point_size =
                    (proj_y * u_particle_radius_world / z) * (u_viewport_h * 0.5) * 2.0;
                gl_PointSize = clamp(point_size, 1.0, 1024.0);
            }
        '''

        shadow_fs = r'''
            #version 330
            out float out_depth;

            void main() {
                vec2 p = gl_PointCoord * 2.0 - 1.0;
                float r2 = dot(p,p);
                if (r2 > 1.0) discard;

                // store light-space depth in 0..1
                out_depth = gl_FragCoord.z;
            }
        '''

        # --- Composite FS: enhanced version with shadows/caustics/env mapping ---
        comp_fs = r'''
            #version 330
            uniform sampler2D u_depth;   // blurred depth (camera)
            uniform sampler2D u_thick;   // thickness
            uniform sampler2D u_shadow;  // shadow map (light)
            uniform vec2 u_texel;

            uniform mat4 u_light_vp;     // light_proj * light_view

            uniform vec3 u_water_color;
            uniform float u_alpha_base;
            uniform float u_fresnel_power;
            uniform float u_spec_power;
            uniform float u_sigma_a;
            uniform float u_sigma_s;

            // caustics/shadow knobs
            uniform float u_shadow_strength;  // 0..1 (how dark)
            uniform float u_caustic_strength; // 0..something
            uniform float u_caustic_power;    // 1..4

            in vec2 v_uv;
            out vec4 frag_color;

            vec3 normal_from_depth(vec2 uv) {
                float dC = texture(u_depth, uv).r;
                float dR = texture(u_depth, uv + vec2(u_texel.x, 0.0)).r;
                float dU = texture(u_depth, uv + vec2(0.0, u_texel.y)).r;

                if (dC >= 0.9999) return vec3(0.0, 0.0, 1.0);

                // 使用线性深度进行梯度计算，避免非线性depth的不稳定梯度
                float dx = dR - dC;
                float dy = dU - dC;
                // 调整梯度缩放因子以适应线性深度
                return normalize(vec3(-dx * 80.0, -dy * 80.0, 1.0));
            }

            // cheap procedural environment (no cubemap asset required)
            vec3 env_color(vec3 dir) {
                float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
                vec3 sky = vec3(0.55, 0.70, 0.95);
                vec3 ground = vec3(0.10, 0.10, 0.12);
                return mix(ground, sky, t);
            }

            float sample_shadow(vec3 world_pos) {
                vec4 lp = u_light_vp * vec4(world_pos, 1.0);
                vec3 ndc = lp.xyz / lp.w;
                vec2 suv = ndc.xy * 0.5 + 0.5;
                float cur = ndc.z * 0.5 + 0.5; // to 0..1

                // outside shadow map => lit
                if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) return 1.0;

                float sd = texture(u_shadow, suv).r;

                // bias reduces acne; tune with particle radius / scale
                float bias = 0.0025;
                float lit = (sd + bias < cur) ? (1.0 - u_shadow_strength) : 1.0;
                return lit;
            }

            void main() {
                float d = texture(u_depth, v_uv).r;
                if (d >= 0.9999) discard;

                float thick_raw = texture(u_thick, v_uv).r;
                // 方案1：thickness pass已输出线性厚度，composite只做一次映射
                float thick = clamp(thick_raw * 2.5, 0.0, 1.0);  // 线性映射到[0,1]

                // 暂时禁用阴影采样以定位问题（按建议）
                vec3 world_pos = vec3((v_uv - 0.5) * 6.0, (1.0 - d) * 6.0);
                
                vec3 N = normal_from_depth(v_uv);
                vec3 V = vec3(0.0, 0.0, 1.0);
                vec3 L = normalize(vec3(0.35, 0.85, 0.25));
                vec3 H = normalize(L + V);
                
                float NdotL = max(dot(N, L), 0.0);
                float NdotV = max(dot(N, V), 0.0);
                
                float F0 = 0.02;
                float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, u_fresnel_power);
                float spec = pow(max(dot(N, H), 0.0), u_spec_power);
                
                // Beer-Lambert
                float trans = exp(-u_sigma_a * thick);
                float scatter = (1.0 - trans) * u_sigma_s;
                
                vec3 base = u_water_color * (0.25 + 0.75 * NdotL) * (0.35 + 0.65 * (1.0 - trans) + scatter);
                
                // Caustics approx: thickness gradient -> brighten
                float tC = thick;
                float tR = texture(u_thick, v_uv + vec2(u_texel.x, 0.0)).r;
                float tU = texture(u_thick, v_uv + vec2(0.0, u_texel.y)).r;
                float grad = length(vec2(tR - tC, tU - tC));
                float caustic = pow(grad * 12.0, u_caustic_power) * u_caustic_strength;
                
                // Env reflection/refraction approx
                vec3 R = reflect(-V, N);
                vec3 refr = refract(-V, N, 1.0 / 1.33);
                vec3 envR = env_color(R);
                vec3 envT = env_color(refr);
                
                vec3 color = base;
                color += envR * (0.12 + 0.35 * fresnel);
                color = mix(color, envT, 0.08 * (1.0 - fresnel));
                color += vec3(1.0) * spec * (0.35 + 0.65 * fresnel);
                
                // 暂时禁用阴影和焦散以改善水体基础外观
                float shadow = sample_shadow(world_pos);
                color *= shadow;
                // float shadow = 1.0;  // 临时设为无阴影
                
                // Add caustics after shadow (looks like light focusing on lit regions)
                color += caustic * vec3(1.0, 0.98, 0.90) * shadow;

                float alpha = u_alpha_base + (1.0 - trans) * 0.90 + fresnel * 0.20;
                alpha = clamp(alpha, 0.03, 0.98);

                frag_color = vec4(color, alpha);
            }
        '''

        # compile programs
        self.particle_depth_prog = self.ctx.program(vertex_shader=particle_vs, fragment_shader=particle_depth_fs)
        self.particle_thick_prog = self.ctx.program(vertex_shader=particle_vs, fragment_shader=particle_thick_fs)
        self.bilateral_prog = self.ctx.program(vertex_shader=quad_vs, fragment_shader=bilateral_fs)
        self.composite_prog = self.ctx.program(vertex_shader=quad_vs, fragment_shader=comp_fs)
        self.shadow_prog = self.ctx.program(vertex_shader=shadow_vs, fragment_shader=shadow_fs)

        # fullscreen quad
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

        # particle vbo/vao (reuse one vbo, create two vaos for two programs)
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

        # release old
        for obj in (
            getattr(self, "depth_fbo0", None),
            getattr(self, "depth_fbo1", None),
            getattr(self, "thick_fbo", None),
            getattr(self, "depth_tex0", None),
            getattr(self, "depth_tex1", None),
            getattr(self, "thick_tex", None),
        ):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass

        # Create new SSFR textures and FBOs
        # depth: R32F for precision (bilateral uses dz)
        self.depth_tex0 = self.ctx.texture((w, h), components=1, dtype='f4')
        self.depth_tex1 = self.ctx.texture((w, h), components=1, dtype='f4')
        for t in (self.depth_tex0, self.depth_tex1):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # thickness: R16F is fine
        self.thick_tex = self.ctx.texture((w, h), components=1, dtype='f2')
        self.thick_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.depth_fbo0 = self.ctx.framebuffer(color_attachments=[self.depth_tex0])
        self.depth_fbo1 = self.ctx.framebuffer(color_attachments=[self.depth_tex1])
        self.thick_fbo = self.ctx.framebuffer(color_attachments=[self.thick_tex])

    def _ensure_shadow_targets(self, width: int, height: int):
        w = max(1, int(width * self.ssfr_scale * self.shadow_scale))
        h = max(1, int(height * self.ssfr_scale * self.shadow_scale))
        if w == self.shadow_w and h == self.shadow_h and self.shadow_fbo is not None:
            return

        self.shadow_w, self.shadow_h = w, h

        # release old shadow resources
        for obj in (self.shadow_fbo, self.shadow_tex):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass

        # Create shadow map resources
        self.shadow_tex = self.ctx.texture((w, h), components=1, dtype='f4')
        self.shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # 采样超出边界时，避免引入错误阴影
        self.shadow_tex.repeat_x = False
        self.shadow_tex.repeat_y = False

        self.shadow_fbo = self.ctx.framebuffer(color_attachments=[self.shadow_tex])

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

            # rebuild VAOs with the new VBO
            if self.particle_vao_depth:
                self.particle_vao_depth.release()
            if self.particle_vao_thick:
                self.particle_vao_thick.release()

            self.particle_vao_depth = self.ctx.vertex_array(
                self.particle_depth_prog, [(self.particle_vbo, '3f', 'in_pos')]
            )
            self.particle_vao_thick = self.ctx.vertex_array(
                self.particle_thick_prog, [(self.particle_vbo, '3f', 'in_pos')]
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

        projection = self._perspective(60.0, width / height, 0.1, 100.0)
        cam_pos = self._calculate_camera_position()
        view = self._look_at(cam_pos, self.camera_target, [0.0, 1.0, 0.0])

        # SSFR passes 需要深度测试（点精灵互相遮挡正确）
        self.ctx.enable(moderngl.DEPTH_TEST)

        # ----------------------------
        # 1) Depth pass (no blending)
        # ----------------------------
        self.depth_fbo0.use()
        self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
        # clear to 1.0 -> background far
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)
        self.ctx.disable(moderngl.BLEND)

        if self.particle_count > 0:
            prog = self.particle_depth_prog
            prog['projection'].write(projection.astype('f4').tobytes())
            prog['view'].write(view.astype('f4').tobytes())
            prog['u_particle_radius_world'].value = float(self.ssfr_particle_radius_world)
            prog['u_viewport_h'].value = float(self._ssfr_h)
            prog['u_point_scale'].value = 1.0  # 物理正确的点大小缩放
            self.particle_vao_depth.render(mode=moderngl.POINTS, vertices=self.particle_count)

        # ------------------------------------
        # 2) Bilateral blur depth (H then V)
        # ------------------------------------
        self.bilateral_prog['u_texel'].value = (1.0 / self._ssfr_w, 1.0 / self._ssfr_h)
        self.bilateral_prog['u_radius'].value = int(self.ssfr_blur_radius)
        self.bilateral_prog['u_sigma_xy'].value = float(max(1.5, self.ssfr_blur_sigma))
        # Depth sigma: important knob. Bigger = more blur across depth changes (worse edges)
        self.bilateral_prog['u_sigma_z'].value = self.ssfr_depth_sigma

        # H: depth0 -> depth1
        self.depth_fbo1.use()
        self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)
        self.bilateral_prog['u_dir'].value = (1.0, 0.0)
        self.depth_tex0.use(location=0)
        self.bilateral_prog['u_depth'].value = 0
        self.quad_vao_bilateral.render(moderngl.TRIANGLES, vertices=6)

        # V: depth1 -> depth0
        self.depth_fbo0.use()
        self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)
        self.bilateral_prog['u_dir'].value = (0.0, 1.0)
        self.depth_tex1.use(location=0)
        self.bilateral_prog['u_depth'].value = 0
        self.quad_vao_bilateral.render(moderngl.TRIANGLES, vertices=6)

        # ------------------------------------
        # 3) Thickness pass (additive blending)
        # ------------------------------------
        self.thick_fbo.use()
        self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)

        if self.particle_count > 0:
            prog = self.particle_thick_prog
            prog['projection'].write(projection.astype('f4').tobytes())
            prog['view'].write(view.astype('f4').tobytes())
            prog['u_particle_radius_world'].value = float(self.ssfr_particle_radius_world)
            prog['u_viewport_h'].value = float(self._ssfr_h)
            prog['u_point_scale'].value = 1.0  # 物理正确的点大小缩放
            self.particle_vao_thick.render(mode=moderngl.POINTS, vertices=self.particle_count)

        # restore blending for composite
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # ----------------------------
        # 4) Shadow map pass (light view)
        # ----------------------------
        self._ensure_shadow_targets(width, height)
        self.shadow_fbo.use()
        self.ctx.viewport = (0, 0, self.shadow_w, self.shadow_h)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)
        self.ctx.disable(moderngl.BLEND)

        # light matrices (simple spot-like)
        light_proj = self._perspective(70.0, self.shadow_w / self.shadow_h, 0.1, 50.0)
        light_view = self._look_at(self.light_pos.tolist(), self.light_target.tolist(), [0.0, 1.0, 0.0])

        if self.particle_count > 0:
            prog = self.shadow_prog
            prog['u_light_proj'].write(light_proj.astype('f4').tobytes())
            prog['u_light_view'].write(light_view.astype('f4').tobytes())
            prog['u_particle_radius_world'].value = float(self.ssfr_particle_radius_world)
            prog['u_viewport_h'].value = float(self.shadow_h)
            self.shadow_vao.render(mode=moderngl.POINTS, vertices=self.particle_count)

        # ----------------------------
        # 5) Composite to screen
        # ----------------------------
        screen_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.clear(0.10, 0.12, 0.16, 1.0)

        self.composite_prog['u_texel'].value = (1.0 / self._ssfr_w, 1.0 / self._ssfr_h)

        # 暂时注释掉light_vp设置，因为暂时禁用了阴影
        # light vp for shadow sampling
        # light_vp = (light_proj @ light_view).astype(np.float32)
        # self.composite_prog['u_light_vp'].write(light_vp.tobytes())

        # 按建议调整的水体渲染参数
        self.composite_prog['u_water_color'].value = (0.08, 0.35, 0.65)  # 更深的水色
        self.composite_prog['u_alpha_base'].value = 0.03  # 稍微提高基础透明度
        self.composite_prog['u_fresnel_power'].value = 5.0
        self.composite_prog['u_spec_power'].value = 128.0  # 提高高光锐度
        self.composite_prog['u_sigma_a'].value = 5.0  # 提高吸收系数(3~8)
        self.composite_prog['u_sigma_s'].value = 0.15  # 降低散射系数(0.05~0.2)

        # 暂时禁用阴影和焦散相关的uniform设置，因为可能存在编译问题
        self.composite_prog['u_shadow_strength'].value = 0.0  # 禁用阴影
        self.composite_prog['u_caustic_strength'].value = 0.0  # 禁用焦散
        self.composite_prog['u_caustic_power'].value = 2.0

        self.depth_tex0.use(location=0)
        self.thick_tex.use(location=1)
        self.shadow_tex.use(location=2)

        self.composite_prog['u_depth'].value = 0
        self.composite_prog['u_thick'].value = 1
        # self.composite_prog['u_shadow'].value = 2  # 暂时禁用，因为u_shadow sampler被注释掉了

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
        """Initialize UI"""
        self.setWindowTitle("PySide6 + ModernGL 网格重建系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=0)
        
        # Right OpenGL widget
        self.opengl_widget = QOpenGLWidgetWrapper(self.opengl_renderer)
        main_layout.addWidget(self.opengl_widget, stretch=1)
        
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
        self.render_mode_toggle.setChecked(False)
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
        self.distance_slider.setValue(80)  # 8.0 default
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
        self.ssfr_radius_slider.setRange(5, 100)  # 0.05 to 1.00 (更适合[-1,1]世界)
        self.ssfr_radius_slider.setValue(10)  # 0.10 default
        self.ssfr_radius_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_radius_slider)
        self.ssfr_radius_label = QLabel("0.10")
        ssfr_layout.addWidget(self.ssfr_radius_label)
        
        # 模糊强度
        ssfr_layout.addWidget(QLabel("模糊强度:"))
        self.ssfr_sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_sigma_slider.setRange(5, 50)  # 0.5 to 5.0
        self.ssfr_sigma_slider.setValue(15)  # 1.5 default
        self.ssfr_sigma_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_sigma_slider)
        self.ssfr_sigma_label = QLabel("1.5")
        ssfr_layout.addWidget(self.ssfr_sigma_label)
        
        # 双边滤波深度sigma
        ssfr_layout.addWidget(QLabel("深度sigma:"))
        self.ssfr_depth_sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_depth_sigma_slider.setRange(1, 50)  # 0.001 to 0.05
        self.ssfr_depth_sigma_slider.setValue(8)  # 0.008 default
        self.ssfr_depth_sigma_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_depth_sigma_slider)
        self.ssfr_depth_sigma_label = QLabel("0.008")
        ssfr_layout.addWidget(self.ssfr_depth_sigma_label)
        
        # 吸收系数
        ssfr_layout.addWidget(QLabel("吸收系数:"))
        self.ssfr_absorption_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_absorption_slider.setRange(5, 50)  # 0.5 to 5.0
        self.ssfr_absorption_slider.setValue(23)  # 2.3 default
        self.ssfr_absorption_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_absorption_slider)
        self.ssfr_absorption_label = QLabel("2.3")
        ssfr_layout.addWidget(self.ssfr_absorption_label)
        
        # 散射系数
        ssfr_layout.addWidget(QLabel("散射系数:"))
        self.ssfr_scatter_slider = QSlider(Qt.Orientation.Horizontal)
        self.ssfr_scatter_slider.setRange(1, 20)  # 0.1 to 2.0
        self.ssfr_scatter_slider.setValue(6)  # 0.55 default
        self.ssfr_scatter_slider.valueChanged.connect(self.update_ssfr_params)
        ssfr_layout.addWidget(self.ssfr_scatter_slider)
        self.ssfr_scatter_label = QLabel("0.55")
        ssfr_layout.addWidget(self.ssfr_scatter_label)
        
        ssfr_group.setLayout(ssfr_layout)
        layout.addWidget(ssfr_group)
        
        # Statistics
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout()
        
        self.frame_label = QLabel("帧: 0")
        self.particle_label = QLabel("粒子数: 0")
        self.vertex_label = QLabel("顶点数: 0")
        self.triangle_label = QLabel("三角形数: 0")
        self.fps_label = QLabel("FPS: 0")
        
        stats_layout.addWidget(self.frame_label)
        stats_layout.addWidget(self.particle_label)
        stats_layout.addWidget(self.vertex_label)
        stats_layout.addWidget(self.triangle_label)
        stats_layout.addWidget(self.fps_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Performance timing
        perf_group = QGroupBox("性能时间")
        perf_layout = QVBoxLayout()
        
        self.grid_time_label = QLabel("网格重建: 0ms")
        self.force_time_label = QLabel("力计算: 0ms")
        self.integration_time_label = QLabel("积分: 0ms")
        self.mesh_time_label = QLabel("网格生成: 0ms")
        self.gpu_time_label = QLabel("GPU上传: 0ms")
        self.memory_label = QLabel("内存使用: 0MB")
        
        perf_layout.addWidget(self.grid_time_label)
        perf_layout.addWidget(self.force_time_label)
        perf_layout.addWidget(self.integration_time_label)
        perf_layout.addWidget(self.mesh_time_label)
        perf_layout.addWidget(self.gpu_time_label)
        perf_layout.addWidget(self.memory_label)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        layout.addStretch()
        return panel
    
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
        # 更新世界空间粒子半径
        radius = self.ssfr_radius_slider.value() / 100.0
        self.opengl_renderer.ssfr_particle_radius_world = radius
        self.ssfr_radius_label.setText(f"{radius:.3f}")
        
        # 更新模糊强度
        sigma = self.ssfr_sigma_slider.value() / 10.0
        self.opengl_renderer.ssfr_blur_sigma = sigma
        self.ssfr_sigma_label.setText(f"{sigma:.1f}")
        
        # 更新双边滤波深度sigma
        depth_sigma = self.ssfr_depth_sigma_slider.value() / 1000.0
        self.opengl_renderer.ssfr_depth_sigma = depth_sigma
        self.ssfr_depth_sigma_label.setText(f"{depth_sigma:.3f}")
        
        # 更新吸收系数
        absorption = self.ssfr_absorption_slider.value() / 10.0
        if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_sigma_a' in self.opengl_renderer.composite_prog.uniforms:
            self.opengl_renderer.composite_prog['u_sigma_a'].value = absorption
        self.ssfr_absorption_label.setText(f"{absorption:.1f}")
        
        # 更新散射系数
        scatter = self.ssfr_scatter_slider.value() / 10.0
        if hasattr(self.opengl_renderer.composite_prog, 'uniforms') and 'u_sigma_s' in self.opengl_renderer.composite_prog.uniforms:
            self.opengl_renderer.composite_prog['u_sigma_s'].value = scatter
        self.ssfr_scatter_label.setText(f"{scatter:.2f}")
        
        # 实时刷新渲染
        self.opengl_widget.update()
    
    def update_simulation(self):
        """Update simulation"""
        if not self.is_running or self.is_paused:
            return
        
        speed = self.speed_slider.value() / 10.0
        dt = 0.008 / speed

        # 仍然执行物理仿真（SPH）
        _new_vertices, _new_triangles = self.mesh_system.step_simulation(dt)

        # SSFR：直接上传粒子位置（强耦合、低延迟、避免重建网格）
        self.opengl_widget.makeCurrent()
        try:
            n = self.mesh_system.current_particle_num
            positions = self.mesh_system.particle_positions[:n]
            self.opengl_renderer.update_particles(positions)
        finally:
            self.opengl_widget.doneCurrent()

        self.update_statistics()
        self.opengl_widget.update()
    
    def update_statistics(self):
        """Update statistics display"""
        stats = self.mesh_system.get_system_stats()
        
        # Update labels
        self.frame_label.setText(f"帧: {stats['frame']}")
        self.particle_label.setText(f"粒子数: {stats['particle_count']}")
        self.vertex_label.setText(f"顶点数: {stats['mesh_vertices']}")
        self.triangle_label.setText(f"三角形数: {stats['mesh_triangles']}")
        
        # Update FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.fps_counter = 0
            self.last_fps_time = current_time
        
        # Update performance times
        perf = stats['performance']
        self.grid_time_label.setText(f"网格重建: {perf['grid_rebuild_time']*1000:.1f}ms")
        self.force_time_label.setText(f"力计算: {perf['force_computation_time']*1000:.1f}ms")
        self.integration_time_label.setText(f"积分: {perf['integration_time']*1000:.1f}ms")
        self.mesh_time_label.setText(f"网格生成: {perf['mesh_generation_time']*1000:.1f}ms")
        self.gpu_time_label.setText(f"GPU上传: {perf.get('gpu_upload_time', 0)*1000:.1f}ms")
        self.memory_label.setText(f"内存使用: {stats.get('memory_usage_mb', 0):.1f}MB")

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
    """Main function"""
    print("Initializing PySide6 + ModernGL mesh reconstruction system...")
    print(f"PySplashSurf availability: {'Yes' if SPLASHSURF_AVAILABLE else 'No'}")
    print(f"PySPlisHSPlasH availability: {'Yes' if PYSPLISH_AVAILABLE else 'No'}")
    
    if PYSPLISH_AVAILABLE:
        print("✓ Will use PySPlisHSPlasH for professional SPH force calculation")
        print("  Including: Standard kernel functions, professional pressure forces, viscous force calculations")
        print("✓ Boundary handling: Using PySPlisHSPlasH scene configuration with manual fallback")
    else:
        print("⚠ Will use manually implemented simplified force calculations")
        print("⚠ Boundary handling: Manual collision detection only")
    
    print("\nFeatures:")
    print("- Real-time fluid simulation with SPH physics")
    print("- High-quality surface reconstruction using PySplashSurf")
    print("- Interactive 3D visualization with ModernGL")
    print("- Configurable boundary collision handling")
    print("- Performance monitoring and statistics")
    
    # 设置OpenGL格式
    from PySide6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapInterval(1)  # vsync
    QSurfaceFormat.setDefaultFormat(fmt)
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()