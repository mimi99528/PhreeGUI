import taichi as ti
from .sph_base import SPHBase


class DFSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        
        # 表面张力系数：优先取 scene 的 "surfaceTension"（_build_scene 会
        # 从 CONFIG.SURFACE_TENSION 下发），缺省保留上游默认 0.01。
        # 它是内聚力项，决定水"抱不抱团"。注意：边界粘性修好后（见下），
        # 0.01/0.05/0.08 对静置收敛的影响已可忽略，0.01 反而更快，故保持
        # 上游默认（实测见 config.py 的 SURFACE_TENSION 注释）。
        st_cfg = self.ps.cfg.get_cfg("surfaceTension")
        self.surface_tension = 0.01 if st_cfg is None else float(st_cfg)

        # 边界粘性（Akinci2012 的 no-slip 壁面项）。
        # 【为什么必须有】这是 Taichi 后端"粒子很轻、一直到处窜"的直接原因：
        # 该分支原本硬编码 boundary_viscosity = 0.0，壁面是完全无粘的
        # （free-slip）；而流体-流体粘性只在支撑半径内有邻居时才起作用——
        # 默认点击量是 8 个粒子（2x2x2），彼此很快散开失去耦合，
        # 于是整团水在地上无摩擦滑行/弹跳，永远不停。
        # 实测（8 粒子、面板默认 3 子步/帧，跑 200 帧后）：
        # boundaryViscosity=0 时 mean|v|=0.695 m/s、足迹 1.52 m、随后 100 帧
        # 仍漂移 0.82 m（一直在滑）；加上边界粘性后 mean|v|=0.0066、
        # 足迹 0.078 m、漂移 0.0007 m（静止）。8k 粒子灌注同样：t=2s 时
        # >0.3 m/s 的粒子从 31.3% 降到 0.0%。见 config.py 的实测表。
        # 上游 SPlisHSPlasH/Viscosity/Viscosity_Standard.cpp 同样提供
        # "viscosityBoundary" 参数，只是默认 0（它的示例场景用 Bender2019）。
        bv_cfg = self.ps.cfg.get_cfg("boundaryViscosity")
        self.boundary_viscosity = 0.0 if bv_cfg is None else float(bv_cfg)

        self.dt[None] = self.ps.cfg.get_cfg("timeStepSize")

        self.enable_divergence_solver = True

        self.m_max_iterations_v = 100
        self.m_max_iterations = 100

        self.m_eps = 1e-5

        self.max_error_V = 0.1
        self.max_error = 0.05
    

    @ti.func
    def compute_densities_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())


    @ti.kernel
    def compute_densities(self):
        # for p_i in range(self.ps.particle_num[None]):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.density[p_i] += den
            self.ps.density[p_i] *= self.density_0
    

    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        
        ############## Surface Tension ###############
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
            x_j = self.ps.x[p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            if r2 > diameter2:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())
            
        
        ############### Viscosoty Force ###############
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        
        if self.ps.material[p_j] == self.ps.material_fluid:
            f_v = d * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (
                r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            ret += f_v
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012 —— 壁面 no-slip 项，系数由 scene 的
            ## "boundaryViscosity" 下发（默认 0 = free-slip，见构造函数注释）。
            f_v = d * self.boundary_viscosity * (self.density_0 * self.ps.m_V[p_j] / (self.ps.density[p_i])) * v_xy / (
                r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            ret += f_v
            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[p_j] += -f_v * self.ps.density[p_i] / self.ps.density[p_j]


    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            self.ps.acceleration[p_i] = d_v
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
                self.ps.acceleration[p_i] = d_v
    

    @ti.kernel
    def advect(self):
        # Update position
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic[p_i]:
                if self.ps.is_dynamic_rigid_body(p_i):
                    self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]
    

    @ti.kernel
    def compute_DFSPH_factor(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0 for _ in range(self.ps.dim)])
            
            # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
            ret = ti.Vector([0.0 for _ in range(self.ps.dim + 1)])
            
            self.ps.for_all_neighbors(p_i, self.compute_DFSPH_factor_task, ret)
            
            sum_grad_p_k = ret[3]
            for i in ti.static(range(3)):
                grad_p_i[i] = ret[i]
            sum_grad_p_k += grad_p_i.norm_sqr()

            # Compute pressure stiffness denominator
            factor = 0.0
            if sum_grad_p_k > 1e-6:
                factor = -1.0 / sum_grad_p_k
            else:
                factor = 0.0
            self.ps.dfsph_factor[p_i] = factor
            

    @ti.func
    def compute_DFSPH_factor_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
            ret[3] += grad_p_j.norm_sqr() # sum_grad_p_k
            for i in ti.static(range(3)): # grad_p_i
                ret[i] -= grad_p_j[i]
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
            for i in ti.static(range(3)): # grad_p_i
                ret[i] -= grad_p_j[i]
    

    @ti.kernel
    def compute_density_change(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            ret = ti.Struct(density_adv=0.0, num_neighbors=0)
            self.ps.for_all_neighbors(p_i, self.compute_density_change_task, ret)

            # only correct positive divergence
            density_adv = ti.max(ret.density_adv, 0.0)
            num_neighbors = ret.num_neighbors

            # Do not perform divergence solve when paritlce deficiency happens
            if self.ps.dim == 3:
                if num_neighbors < 20:
                    density_adv = 0.0
            else:
                if num_neighbors < 7:
                    density_adv = 0.0
     
            self.ps.density_adv[p_i] = density_adv


    @ti.func
    def compute_density_change_task(self, p_i, p_j, ret: ti.template()):
        v_i = self.ps.v[p_i]
        v_j = self.ps.v[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            ret.density_adv += self.ps.m_V[p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            ret.density_adv += self.ps.m_V[p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
        
        # Compute the number of neighbors
        ret.num_neighbors += 1
    

    @ti.kernel
    def compute_density_adv(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            delta = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_density_adv_task, delta)
            density_adv = self.ps.density[p_i] /self.density_0 + self.dt[None] * delta
            self.ps.density_adv[p_i] = ti.max(density_adv, 1.0)


    @ti.func
    def compute_density_adv_task(self, p_i, p_j, ret: ti.template()):
        v_i = self.ps.v[p_i]
        v_j = self.ps.v[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            ret += self.ps.m_V[p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            ret += self.ps.m_V[p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))


    @ti.kernel
    def compute_density_error(self, offset: float) -> float:
        density_error = 0.0
        for I in range(self.ps.particle_num[None]):
            if self.ps.material[I] == self.ps.material_fluid:
                density_error += self.density_0 * self.ps.density_adv[I] - offset
        return density_error

    @ti.kernel
    def multiply_time_step(self, field: ti.template(), time_step: float):
        for I in range(self.ps.particle_num[None]):
            if self.ps.material[I] == self.ps.material_fluid:
                field[I] *= time_step


    def divergence_solve(self):
        # TODO: warm start 
        # Compute velocity of density change
        self.compute_density_change()
        inv_dt = 1 / self.dt[None]
        self.multiply_time_step(self.ps.dfsph_factor, inv_dt)

        m_iterations_v = 0
        
        # Start solver
        avg_density_err = 0.0

        while m_iterations_v < 1 or m_iterations_v < self.m_max_iterations_v:
            
            avg_density_err = self.divergence_solver_iteration()
            # Max allowed density fluctuation
            # use max density error divided by time step size
            eta = 1.0 / self.dt[None] * self.max_error_V * 0.01 * self.density_0
            # print("eta ", eta)
            if avg_density_err <= eta:
                break
            m_iterations_v += 1
        self.last_iterations_v = m_iterations_v
        self.last_err_v = avg_density_err

        # Multiply by h, the time step size has to be removed 
        # to make the stiffness value independent 
        # of the time step size

        # TODO: if warm start
        # also remove for kappa v

        self.multiply_time_step(self.ps.dfsph_factor, self.dt[None])


    def divergence_solver_iteration(self):
        self.divergence_solver_iteration_kernel()
        self.compute_density_change()
        density_err = self.compute_density_error(0.0)
        return density_err / self.ps.fluid_particle_num


    @ti.kernel
    def divergence_solver_iteration_kernel(self):
        # Perform Jacobi iteration
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            # evaluate rhs
            b_i = self.ps.density_adv[p_i]
            k_i = b_i*self.ps.dfsph_factor[p_i]
            ret = ti.Struct(dv=ti.Vector([0.0 for _ in range(self.ps.dim)]), k_i=k_i)
            # TODO: if warm start
            # get_kappa_V += k_i
            self.ps.for_all_neighbors(p_i, self.divergence_solver_iteration_task, ret)
            self.ps.v[p_i] += ret.dv
        
    
    @ti.func
    def divergence_solver_iteration_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            b_j = self.ps.density_adv[p_j]
            k_j = b_j * self.ps.dfsph_factor[p_j]
            k_sum = ret.k_i + self.density_0 / self.density_0 * k_j  # TODO: make the neighbor density0 different for multiphase fluid
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                ret.dv -= self.dt[None] * k_sum * grad_p_j
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(ret.k_i) > self.m_eps:
                grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                vel_change =  -self.dt[None] * 1.0 * ret.k_i * grad_p_j
                ret.dv += vel_change
                if self.ps.is_dynamic_rigid_body(p_j):
                    self.ps.acceleration[p_j] += -vel_change * (1 / self.dt[None]) * self.ps.density[p_i] / self.ps.density[p_j]


    def pressure_solve(self):
        inv_dt = 1 / self.dt[None]
        inv_dt2 = 1 / (self.dt[None] * self.dt[None])

        # TODO: warm start
        
        # Compute rho_adv
        self.compute_density_adv()

        self.multiply_time_step(self.ps.dfsph_factor, inv_dt2)

        m_iterations = 0

        # Start solver
        avg_density_err = 0.0

        while m_iterations < 1 or m_iterations < self.m_max_iterations:
            
            avg_density_err = self.pressure_solve_iteration()
            # Max allowed density fluctuation
            eta = self.max_error * 0.01 * self.density_0
            if avg_density_err <= eta:
                break
            m_iterations += 1
        self.last_iterations = m_iterations
        self.last_err = avg_density_err
        # Multiply by h, the time step size has to be removed 
        # to make the stiffness value independent 
        # of the time step size

        # TODO: if warm start
        # also remove for kappa v
    
    def pressure_solve_iteration(self):
        self.pressure_solve_iteration_kernel()
        self.compute_density_adv()
        density_err = self.compute_density_error(self.density_0)
        return density_err / self.ps.fluid_particle_num

    
    @ti.kernel
    def pressure_solve_iteration_kernel(self):
        # Compute pressure forces
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            # Evaluate rhs
            b_i = self.ps.density_adv[p_i] - 1.0
            k_i = b_i * self.ps.dfsph_factor[p_i]

            # TODO: if warmstart
            # get kappa V
            self.ps.for_all_neighbors(p_i, self.pressure_solve_iteration_task, k_i)
    

    @ti.func
    def pressure_solve_iteration_task(self, p_i, p_j, k_i: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            b_j = self.ps.density_adv[p_j] - 1.0
            k_j = b_j * self.ps.dfsph_factor[p_j]
            k_sum = k_i + self.density_0 / self.density_0 * k_j # TODO: make the neighbor density0 different for multiphase fluid
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                # Directly update velocities instead of storing pressure accelerations
                self.ps.v[p_i] -= self.dt[None] * k_sum * grad_p_j  # ki, kj already contain inverse density
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(k_i) > self.m_eps:
                grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])

                # Directly update velocities instead of storing pressure accelerations
                vel_change = - self.dt[None] * 1.0 * k_i * grad_p_j  # kj already contains inverse density
                self.ps.v[p_i] += vel_change
                if self.ps.is_dynamic_rigid_body(p_j):
                    self.ps.acceleration[p_j] += -vel_change * 1.0 / self.dt[None] * self.ps.density[p_i] / self.ps.density[p_j] 


    @ti.kernel
    def predict_velocity(self):
        # compute new velocities only considering non-pressure forces
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic[p_i] and self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]


    def substep(self):
        self.compute_densities()
        self.compute_DFSPH_factor()
        if self.enable_divergence_solver:
            self.divergence_solve()
        self.compute_non_pressure_forces()
        self.predict_velocity()
        self.pressure_solve()
        self.advect()
