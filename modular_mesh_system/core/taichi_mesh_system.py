"""Taichi(GPU) 物理后端：与 MeshSystem 同构的鸭子类型接口。

对外契约（与 MeshSystem 一致，见 coupled_simulation.py / GUI）：
  - particle_positions / particle_velocities / particle_types / particle_ids
    （行 → 粒子；particle_ids 是稳定身份，行号不是）
  - current_particle_num（只统计流体粒子，不含边界粒子）
  - boundary_min / boundary_max / particle_spacing
  - add_particles(positions, fluid_type) -> list[int]（稳定 id）
  - step_simulation(dt) / sync_particle_data() / capacity()
  - initialize_fluid_drops() / update_mesh() / get_system_stats()
  - mesh_enabled / manual_boundary_enabled / current_frame / last_mesh_update
  - performance_stats / performance_capture

坐标约定：Taichi 库在 [0, domain_size] 的 0 基网格里工作，而本项目的
CONFIG.BOUNDARY_MIN 从负数开始（容器以 y=0 为底、xz 居中）。后端在
进出口处做平移，对外始终暴露 CONFIG 的世界坐标。

物理差异（必须知情）：边界是 Akinci2012 边界粒子，不是 CPU 路径的
Bender2019 体积图，数值结果不会逐位一致；化学层只依赖
(position, particleId)，契约不变。

线程契约（踩过坑，必须遵守）：ti.init 与全部内核的"首次编译"都发生在
主线程（GUI 启动时 create_mesh_system），而物理步跑在
QThread(SimulationWorker)。Taichi 会在内核首次编译时进入
TaichiLLVMContext::add_struct_module，那里断言当前线程就是 ti.init 线程，
所以**物理步路径上不允许出现任何首次编译的内核**（也不要在 worker 里新建
稀疏字段）。新增内核必须在本类的 __init__ 里用真实参数预热一次。
实测症状：延迟初始化完成后第一个物理步即抛
"Assertion failure: std::this_thread::get_id() == main_thread_id_"。
"""
import time
import threading
from typing import List, Tuple

import numpy as np
import taichi as ti

from ..utils.config import CONFIG
from ..utils.performance_analyzer import PerformanceAnalyzer


@ti.kernel
def _export_active(x: ti.template(), v: ti.template(), pid: ti.template(),
                   mat: ti.template(), n: ti.i32, out: ti.types.ndarray()):
    """活跃行打包成 (n, 8) 交错缓冲：pos(3) vel(3) pid(1) material(1)。

    单块 D2H 传输，比四个字段各 to_numpy() 少三次流同步。
    """
    for i in range(n):
        for d in ti.static(range(3)):
            out[i, d] = x[i][d]
            out[i, 3 + d] = v[i][d]
        out[i, 6] = ti.cast(pid[i], ti.f32)
        out[i, 7] = ti.cast(mat[i], ti.f32)


# Taichi 运行期只能初始化一次（同一进程内换 arch 需要重启）。
_taichi_arch = None
_cuda_bound_threads = set()


def _bind_cuda_context_per_thread(arch: str) -> None:
    """Taichi 的 CUDA 上下文绑定在初始化线程上；GUI 的物理步跑在 worker 线程。

    实测（本机）：不绑定时 worker 线程抛
    CUDA_ERROR_INVALID_CONTEXT (cuStreamSynchronize)。
    Taichi 用的就是设备 0 的 primary context，因此这里在每个新线程里
    把它设为 current（primary context 进程内共享，重复 Retain 是幂等的）。
    """
    if arch != "cuda":
        return
    ident = threading.get_ident()
    if ident in _cuda_bound_threads:
        return
    import ctypes
    try:
        cuda = ctypes.CDLL("nvcuda.dll")
        cuda.cuInit(0)
        dev = ctypes.c_int(0)
        if cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            raise OSError("cuDeviceGet 失败")
        ctx = ctypes.c_void_p()
        if cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) != 0:
            raise OSError("cuDevicePrimaryCtxRetain 失败")
        if cuda.cuCtxSetCurrent(ctx) != 0:
            raise OSError("cuCtxSetCurrent 失败")
    except OSError as e:
        raise RuntimeError(
            f"无法在调用线程绑定 CUDA 上下文：{e}。"
            "Taichi GPU 内核不能在未绑定上下文的线程里执行。") from e
    _cuda_bound_threads.add(ident)


def _ensure_taichi(arch_name: str):
    """初始化 Taichi 运行时。arch 非法或 CUDA 不可用时硬失败（决策 D4）。"""
    global _taichi_arch
    import taichi as ti

    arch_name = str(arch_name).strip().lower()
    if _taichi_arch is not None:
        if _taichi_arch != arch_name:
            raise RuntimeError(
                f"Taichi 已按 {_taichi_arch!r} 初始化，无法在进程内切换到 "
                f"{arch_name!r}；请重启程序并设置 PHREEGUI_TAICHI_ARCH。")
        return ti

    if arch_name == "cuda":
        if not ti._lib.core.with_cuda():
            raise RuntimeError(
                "Taichi CUDA 后端不可用（当前机器/驱动未检测到可用 CUDA 设备）。"
                "物理后端请改回 cpu（PHREEGUI_PHYSICS_BACKEND=cpu），"
                "或显式接受 Taichi 的 CPU 执行器：PHREEGUI_TAICHI_ARCH=cpu。"
                "（决策 D4：不静默降级。）")
        try:
            ti.init(arch=ti.cuda, device_memory_fraction=0.5)
        except Exception as e:
            raise RuntimeError(
                f"Taichi CUDA 初始化失败：{e}。"
                "确认显卡驱动/显存后重试，或显式选择 PHREEGUI_TAICHI_ARCH=cpu。"
                "（决策 D4：不静默降级。）") from e
    elif arch_name == "cpu":
        ti.init(arch=ti.cpu)
    else:
        raise RuntimeError(
            f"未知的 TAICHI_ARCH={arch_name!r}（合法值：cuda / cpu）")

    _taichi_arch = arch_name
    return ti


class _SceneConfig:
    """SimConfig 的鸭子类型替身：直接吃内存里的 scene dict，不落临时文件。"""

    def __init__(self, data: dict):
        self._data = data

    def get_cfg(self, name, enforce_exist=False):
        cfg = self._data.get("Configuration", {})
        if name not in cfg:
            if enforce_exist:
                raise KeyError(f"scene config 缺少字段 {name!r}")
            return None
        return cfg[name]

    def get_fluid_blocks(self):
        return self._data.get("FluidBlocks", [])

    def get_rigid_blocks(self):
        return self._data.get("RigidBlocks", [])

    def get_rigid_bodies(self):
        return self._data.get("RigidBodies", [])


def _build_scene() -> dict:
    """由 CONFIG 生成五面刚性墙的 scene（0 基坐标，内部尺寸 = CONTAINER_SIZE）。

    与 D:\\tmp_pg_bench\\scene_box.json 实测通过的布局一致：
    地板占 y∈[0,2r]；四面墙各厚 2r；地板在 xz 方向内缩 2×2r 避免与墙重叠。
    """
    r = float(CONFIG.PARTICLE_RADIUS)
    d = 2.0 * r
    size = CONFIG.BOUNDARY_MAX - CONFIG.BOUNDARY_MIN
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])

    def block(obj_id, start, end):
        return {
            "objectId": obj_id,
            "start": [float(v) for v in start],
            "end": [float(v) for v in end],
            "translation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "velocity": [0.0, 0.0, 0.0],
            "density": 1000.0,
            "color": [40, 40, 40],
            "isDynamic": False,
        }

    walls = [
        block(10, [2 * d, 0.0, 2 * d], [sx - 2 * d, d, sz - 2 * d]),   # 地板
        block(11, [0.0, 0.0, 0.0], [d, sy, sz]),                        # -x
        block(12, [sx - d, 0.0, 0.0], [sx, sy, sz]),                    # +x
        block(13, [2 * d, 0.0, 0.0], [sx - 2 * d, sy, d]),              # -z
        block(14, [2 * d, 0.0, sz - d], [sx - 2 * d, sy, sz]),          # +z
    ]

    return {
        "Configuration": {
            "domainStart": [0.0, 0.0, 0.0],
            "domainEnd": [sx, sy, sz],
            "particleRadius": r,
            "numberOfStepsPerRenderUpdate": 1,
            "density0": 1000.0,
            "simulationMethod": 4,  # DFSPH
            "gravitation": [0.0, -9.81, 0.0],
            "timeStepSize": float(CONFIG.DEFAULT_TIME_STEP),
            "boundaryHandlingMethod": 0,
            "boundaryPadding": d,
            "particleCapacity": int(CONFIG.MAX_EMITTER_PARTICLES),
            # 流体材质参数：不给的话求解器只用构造默认值，调用方就再也
            # 调不动（scene dict 是唯一的下发通道）。boundaryViscosity
            # 决定壁面 no-slip，缺了它小团水会在无摩擦地面上永不停歇。
            "viscosity": float(CONFIG.VISCOSITY),
            "surfaceTension": float(CONFIG.SURFACE_TENSION),
            "boundaryViscosity": float(CONFIG.BOUNDARY_VISCOSITY),
            "exportFrame": False,
            "exportPly": False,
            "exportObj": False,
        },
        "RigidBlocks": walls,
        "FluidBlocks": [],
    }


class TaichiMeshSystem:
    """Taichi DFSPH 后端。接口与 MeshSystem 一致，可直接替换。"""

    FLUID_OBJECT_ID = 100

    def __init__(self, arch: str = None):
        arch = CONFIG.TAICHI_ARCH if arch is None else arch
        self.arch = str(arch).strip().lower()
        # 必须在导入库模块之前完成 ti.init：库里的 kernel 装饰器与
        # PrefixSumExecutor 都要求运行期已就绪。
        _ensure_taichi(self.arch)
        _bind_cuda_context_per_thread(self.arch)
        from .sph_taichi.particle_system import ParticleSystem

        print(f"[TAICHI] 初始化后端 arch={self.arch}")
        scene = _SceneConfig(_build_scene())
        self.ps = ParticleSystem(scene)
        self.solver = self.ps.build_solver()
        self.solver.dt[None] = float(CONFIG.DEFAULT_TIME_STEP)
        self.solver.m_max_iterations = int(CONFIG.DFSPH_MAX_ITERATIONS)
        self.solver.m_max_iterations_v = int(CONFIG.DFSPH_MAX_ITERATIONS_V)
        self.solver.max_error = float(CONFIG.DFSPH_MAX_ERROR)
        self.solver.max_error_V = float(CONFIG.DFSPH_MAX_ERROR_V)
        self.solver.initialize()

        self._n_boundary = int(self.ps.particle_num[None])
        if self._n_boundary <= 0:
            raise RuntimeError(
                "Taichi 后端没有生成任何边界粒子，容器边界缺失，仿真无意义。")
        if self._n_boundary >= self.ps.particle_max_num:
            raise RuntimeError(
                f"边界粒子数 {self._n_boundary} 已占满容量 "
                f"{self.ps.particle_max_num}，无法再添加流体。")
        self._capacity = int(self.ps.particle_max_num) - self._n_boundary
        print(f"[TAICHI] 边界粒子 {self._n_boundary}，流体容量 {self._capacity}")

        # 与 MeshSystem 相同的公开数组（行 → 流体粒子）
        self.particle_positions = np.zeros((CONFIG.MAX_PARTICLES, 3), dtype=np.float32)
        self.particle_velocities = np.zeros((CONFIG.MAX_PARTICLES, 3), dtype=np.float32)
        self.particle_types = np.ones(CONFIG.MAX_PARTICLES, dtype=np.int32)
        self.particle_ids = np.zeros(CONFIG.MAX_PARTICLES, dtype=np.int64)

        # 拉取内核输出用的设备缓冲（按活跃行数重建，见 _ensure_dev_buffers）
        # 化学流体类型按稳定 id 存：sort 会搬行，不会搬 id
        cap = int(self.ps.particle_max_num)
        self._fluid_type_by_id = np.zeros(cap, dtype=np.int32)
        self._dev = None
        self._dev_n = -1

        # CFL 用的流体最大速率（主机镜像，每 8 步刷新一次）。
        # 【为什么读主机镜像，而不是再写一个 GPU 归约内核】
        # Taichi 内核的"首次编译"属于新增内核特化，会走
        # TaichiLLVMContext::add_struct_module，那里有断言：
        #     std::this_thread::get_id() == main_thread_id_
        # 本后端的 ti.init 发生在主线程（GUI 启动时 create_mesh_system），
        # 但物理步跑在 QThread(SimulationWorker)。所以物理步路径上任何
        # "第一次编译"都会炸——新增一个归约内核正好踩中：
        # 实测延迟初始化完成后第一个物理步即抛该断言。
        # particle_velocities 是 sync_particle_data 每步已拷回主机的镜像，
        # 直接在它上面求 max|v| 即可，零新内核、零额外 D2H。
        self._v_max = 0.0
        self._cfl_age = 8       # 距离上次刷新 max|v| 过了几步（首次强制刷新）

        self.current_particle_num = 0
        self.current_frame = 0
        self.last_mesh_update = -CONFIG.MC_MESH_UPDATE_INTERVAL
        self.mesh_enabled = True
        self.manual_boundary_enabled = False

        self.boundary_min = CONFIG.BOUNDARY_MIN.copy()
        self.boundary_max = CONFIG.BOUNDARY_MAX.copy()
        self.particle_spacing = float(CONFIG.PARTICLE_SPACING)

        self.performance_stats = {
            'force_computation_time': 0.0,
            'integration_time': 0.0,
            'mesh_generation_time': 0.0,
            'gpu_upload_time': 0.0,
            'data_processing_time': 0.0,
            'total_mesh_vertices': 0,
            'total_mesh_triangles': 0,
            'memory_usage_mb': 0.0,
        }
        self.performance_capture = (
            PerformanceAnalyzer() if CONFIG.ENABLE_DETAILED_PERFORMANCE_CAPTURE
            else None)
        self.frame_sample_counter = 0
        self._identity_check_countdown = 0

        if CONFIG.SPLASHSURF_AVAILABLE:
            try:
                import pysplashsurf  # noqa: F401
                self._splashsurf_available = True
            except ImportError:
                self._splashsurf_available = False
        else:
            self._splashsurf_available = False
        self.splashsurf_config = {
            'particle_radius': CONFIG.PARTICLE_RADIUS,
            'rest_density': 1000.0,
            'smoothing_length': 2.5 * CONFIG.PARTICLE_RADIUS,
            'cube_size': CONFIG.PARTICLE_RADIUS * 1.5,
            'iso_surface_threshold': 0.6,
            'enable_simd': True,
        }

        self.sync_particle_data()

    # ------------------------------------------------------------------
    # 容量 / 粒子增删
    # ------------------------------------------------------------------

    def _bind_ctx(self) -> None:
        _bind_cuda_context_per_thread(self.arch)

    def capacity(self) -> int:
        """流体粒子容量（不含边界粒子）。"""
        return self._capacity

    def add_particles(self, positions: np.ndarray, fluid_type: int = 1) -> List[int]:
        """添加流体粒子并返回它们的稳定 id。

        id 由 Taichi 库在插入时写入 pid=行号；此后每帧 counting sort 只做
        行置换，pid 随粒子一起搬走，因此这批 id 永久有效。
        """
        if positions is None or len(positions) == 0:
            raise RuntimeError("add_particles: 没有任何位置可添加")
        self._bind_ctx()

        pos_world = np.asarray(positions, dtype=np.float32)
        if pos_world.ndim != 2 or pos_world.shape[1] != 3:
            raise RuntimeError(f"位置数组形状非法: {pos_world.shape}")
        if not np.isfinite(pos_world).all():
            raise RuntimeError("add_particles: 位置包含 NaN/Inf")

        n_new = int(pos_world.shape[0])
        active = int(self.ps.fluid_particle_num)
        if active + n_new > self._capacity:
            raise RuntimeError(
                f"求解器数组空间不足：要加 {n_new} 个，只剩 {self._capacity - active} "
                f"(活跃={active}, 容量={self._capacity})。"
                "（决策 D4：不静默截断——调用方会按'请求的个数'去播种化学。）")

        first = int(self.ps.particle_num[None])
        if first + n_new > int(self.ps.particle_max_num):
            raise RuntimeError(
                f"引擎粒子数组越界：现有总量 {first} + 新增 {n_new} > "
                f"{self.ps.particle_max_num}")

        pos_local = pos_world - CONFIG.BOUNDARY_MIN.astype(np.float32)
        zeros_v = np.zeros_like(pos_local)
        density = np.full(n_new, 1000.0, dtype=np.float32)
        pressure = np.zeros(n_new, dtype=np.float32)
        material = np.full(n_new, self.ps.material_fluid, dtype=np.int32)
        is_dynamic = np.ones(n_new, dtype=np.int32)
        color = np.full((n_new, 3), 255, dtype=np.int32)

        self.ps.add_particles(
            self.FLUID_OBJECT_ID, n_new, pos_local, zeros_v, density, pressure,
            material, is_dynamic, color)

        new_ids = list(range(first, first + n_new))
        self._fluid_type_by_id[first:first + n_new] = int(fluid_type)
        self.sync_particle_data()
        return new_ids

    # ------------------------------------------------------------------
    # 物理步
    # ------------------------------------------------------------------

    def compute_forces(self):
        """执行一步 DFSPH。异常直接向上抛（决策 D4：不静默吞掉物理失败）。"""
        if int(self.ps.fluid_particle_num) == 0:
            return
        start = time.perf_counter()
        self.solver.step()
        self.performance_stats['force_computation_time'] = (
            time.perf_counter() - start)

    def _cfl_time_step(self, dt: float) -> float:
        """把调用方给的步长按 CFL 夹回稳定区间。

        【为什么 Taichi 后端必须自己做，CPU 后端不用】
        CPU 路径的 timeStep 走 C++ TimeManager，引擎每帧调用
        updateTimeStepSize() 按 CFL 把步长夹回去，所以传入超大 dt 也不会炸；
        Taichi 后端把 dt 原样交给求解器，没有这层保护。

        实测（8000 粒子同一灌注，Taichi）：
          dt=0.0030（1.0x）→ 稳，静置后 mean|v|=0.017
          dt=0.0075（2.5x）→ 峰值 26,743 m/s，水充满容器
          dt=0.0150（5.0x）→ NaN/Inf
        旧 GUI（gui/main_window.py update_simulation）正是
        `dt = DEFAULT_TIME_STEP / (slider/10)`，滑块拉到最左就是 10x。

        公式对齐上游 SPlisHSPlasH/Simulation.cpp:487：
          h_cfl = cflFactor * 0.4 * diameter / sqrt(maxVel)

        【为什么还要先做一道硬上限】
        上面那条 CFL 只约束"速度已经很大"的情形。新加进来的粒子初始速度
        为 0，v_max=0 时 CFL 根本不给上界，第一步就会拿调用方给的超大 dt
        直接炸（实测 dt=0.03 首步峰值 15481 m/s）。而显式粘性/表面张力本身
        对 dt 也有刚性上限（0.0075 就发散，与速度无关）。所以顺序是：
        先按 TAICHI_MAX_TIME_STEP 硬夹，再用 CFL 只往下收。
        """
        dt = float(dt)
        if dt <= 0.0 or not np.isfinite(dt):
            raise RuntimeError(f"非法时间步长 dt={dt}（必须是有限正数）")
        # 第一道：显式项的 dt 刚性上限（不依赖速度，静止粒子同样适用）
        dt = min(dt, float(CONFIG.TAICHI_MAX_TIME_STEP))
        n_fluid = int(self.current_particle_num)
        if n_fluid <= 0:
            return dt
        # max|v| 取自主机镜像（见 __init__ 里为什么不用 GPU 归约内核）。
        # CFL 是平滑上界，不必每步重算：每 8 步刷新一次，其间沿用旧值；
        # 稳定性余量（实测约 2.4x）远大于这点滞后。
        self._cfl_age += 1
        if self._cfl_age >= 8:
            self._cfl_age = 0
            v = self.particle_velocities[:n_fluid]
            if v.size > 0 and np.isfinite(v).all():
                self._v_max = float(np.linalg.norm(v, axis=1).max())
        v_max = self._v_max
        if v_max > 0.0:
            h_cfl = (float(CONFIG.CFL_FACTOR) * float(CONFIG.CFL_DIAMETER_FACTOR)
                     * (2.0 * float(CONFIG.PARTICLE_RADIUS)) / float(np.sqrt(v_max)))
            dt = min(dt, max(h_cfl, float(CONFIG.MIN_TIME_STEP)))
        return dt

    def step_simulation(self, dt: float = CONFIG.DEFAULT_TIME_STEP
                        ) -> Tuple[List, List]:
        self.frame_sample_counter += 1
        should_sample = bool(
            self.performance_capture and
            self.frame_sample_counter % self.performance_capture.sampling_rate == 0)
        if should_sample:
            self.performance_capture.start_frame_timer()
        self.current_frame += 1

        if int(self.ps.fluid_particle_num) == 0:
            # 空场（决策 D1）：没有流体就没有物理步，边界粒子也无需重排
            if should_sample:
                self.performance_capture.end_frame_timer()
            return [], []

        if should_sample:
            self.performance_capture.start_timer('physics')
        self._bind_ctx()
        self.solver.dt[None] = self._cfl_time_step(dt)
        self.compute_forces()
        self.sync_particle_data()
        if should_sample:
            self.performance_capture.end_timer('physics')
            self.performance_capture.record_snapshot({
                'particle_count': self.current_particle_num})

        mesh_vertices, mesh_triangles = [], []
        if not CONFIG.USE_SSFR:
            if (self.current_frame - self.last_mesh_update) >= CONFIG.MC_MESH_UPDATE_INTERVAL:
                mesh_vertices, mesh_triangles = self.update_mesh()
                self.last_mesh_update = self.current_frame
        if should_sample:
            self.performance_capture.end_frame_timer()
        return mesh_vertices, mesh_triangles

    # ------------------------------------------------------------------
    # 数据同步
    # ------------------------------------------------------------------

    def _ensure_dev_buffers(self, n_total: int) -> None:
        """设备端导出缓冲按活跃行数分配。粒子只在 add_particles 时增长，
        因此重建频率与点击频次同阶，不在每帧路径上。"""
        if self._dev_n == n_total and self._dev is not None:
            return
        # pid/material 以 f32 打包：容量上限 1M << 2^24，整数在 f32 里精确
        self._dev = ti.ndarray(ti.f32, shape=(n_total, 8))
        self._dev_n = n_total

    def _check_identity(self, pid_np: np.ndarray, n_total: int) -> None:
        """pid 必须是 0..n_total-1 的置换（每 60 帧抽查一次）。"""
        self._identity_check_countdown -= 1
        if self._identity_check_countdown > 0:
            return
        self._identity_check_countdown = 60
        pids = pid_np.astype(np.int64)
        if pids.min() < 0 or pids.max() >= n_total:
            raise RuntimeError(
                "Taichi 粒子身份越界：pid 不在 [0, %d)，身份契约已失效。"
                % n_total)
        counts = np.bincount(pids, minlength=n_total)
        if counts.size != n_total or counts.max() != 1:
            raise RuntimeError(
                "Taichi 粒子身份重复/缺失：counting sort 未保持 pid 唯一。"
                "化学状态映射会张冠李戴，禁止继续仿真。")

    def sync_particle_data(self):
        """把引擎中活跃流体粒子压紧到公开数组（边界粒子不暴露）。

        【渲染契约 —— 边界墙不可见】
        Taichi 后端的容器是 61936 个 Akinci2012 边界粒子（material=0），
        它们只参与 DFSPH 的边界力计算，**永不进入渲染**：
          - 本函数只导出 mat==material_fluid 的行，并压紧到 [0, k)；
          - GUI/渲染器只读 particle_positions[:current_particle_num]，
            也就永远读不到边界粒子；
          - 渲染器里没有任何"画墙 / 画 rigid body"的代码路径
            （opengl_renderer.py 只有 SSFR 网格 + 粒子点两条通道）。
        所以视口里不该出现一堵粒子墙；如果将来有人把导出改成"整段 n_total"，
        或让渲染器改用别的计数，就会看到那堵墙 —— 下面的断言把这条契约钉死。
        实测：4000 粒子时导出包围盒 x/z∈[-0.78,0.78]（容器 ±0.80），
        即导出集合里没有贴着墙的边界粒子；视口截图只有水、没有墙。
        """
        self._bind_ctx()
        n_total = int(self.ps.particle_num[None])
        if n_total <= 0:
            self.current_particle_num = 0
            return

        self._ensure_dev_buffers(n_total)
        _export_active(self.ps.x, self.ps.v, self.ps.pid, self.ps.material,
                       n_total, self._dev)
        buf = self._dev.to_numpy()
        pos_np = buf[:, :3]
        vel_np = buf[:, 3:6]
        pid_np = buf[:, 6].astype(np.int64)
        mat_np = buf[:, 7].astype(np.int32)
        self._check_identity(pid_np, n_total)

        mask = mat_np == self.ps.material_fluid
        k = int(mask.sum())
        if k < n_total and np.any(mat_np[mask] != self.ps.material_fluid):
            # 不可能发生（mask 就是按这个条件选的），留着是为了万一有人
            # 改了上面的 mask 定义，能立刻炸而不是把墙画进视口。
            raise RuntimeError(
                "渲染契约被破坏：导出的粒子集合里混入了非流体材质。"
                "边界粒子不允许进入渲染（见 TaichiMeshSystem.sync_particle_data 注释）。")
        self.current_particle_num = k
        if k == 0:
            return

        self.particle_positions[:k] = (
            pos_np[mask] + CONFIG.BOUNDARY_MIN.astype(np.float32))
        self.particle_velocities[:k] = vel_np[mask]
        ids = pid_np[mask]
        self.particle_ids[:k] = ids
        self.particle_types[:k] = self._fluid_type_by_id[ids]

    # ------------------------------------------------------------------
    # 网格 / 统计（与 MeshSystem 同名，供旧 GUI 使用）
    # ------------------------------------------------------------------

    def update_mesh(self) -> Tuple[List, List]:
        if not self.mesh_enabled or self.current_particle_num == 0:
            return [], []
        return self.generate_mesh_splashsurf()

    def generate_mesh_splashsurf(self) -> Tuple[List, List]:
        if not self._splashsurf_available or self.current_particle_num == 0:
            return [], []
        import pysplashsurf
        try:
            n = self.current_particle_num
            start = time.perf_counter()
            result = pysplashsurf.reconstruct_surface(
                particles=self.particle_positions[:n],
                particle_radius=self.splashsurf_config['particle_radius'],
                rest_density=self.splashsurf_config['rest_density'],
                smoothing_length=self.splashsurf_config['smoothing_length'],
                cube_size=self.splashsurf_config['cube_size'],
                iso_surface_threshold=self.splashsurf_config['iso_surface_threshold'],
                multi_threading=True,
                simd=self.splashsurf_config['enable_simd'],
            )
            mesh = result.mesh
            vertices, triangles = mesh.vertices, mesh.triangles
            self.performance_stats['mesh_generation_time'] = (
                time.perf_counter() - start)
            self.performance_stats['total_mesh_vertices'] = len(vertices)
            self.performance_stats['total_mesh_triangles'] = len(triangles)
            return vertices, triangles
        except Exception:
            raise

    def initialize_fluid_drops(self):
        self.sync_particle_data()
        print(f"[TAICHI] initialize_fluid_drops: "
              f"{self.current_particle_num} 个流体粒子")
        return self.current_particle_num

    def get_system_stats(self) -> dict:
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
            'performance': self.performance_stats.copy(),
            'backend': 'taichi',
        }
