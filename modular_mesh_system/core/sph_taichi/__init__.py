"""erizmr/SPH_Taichi 的 vendored 副本（DFSPH 路径）。

来源：https://github.com/erizmr/SPH_Taichi（MIT，见 LICENSE.txt）
本地补丁见 particle_system.py / DFSPH.py / sph_base.py 的注释：
  - 稳定 pid 字段随 counting sort 迁移（粒子身份契约）
  - 边缘网格索引钳制 + 邻居区间用 inclusive 前缀和
  - 内核按活跃粒子数循环（不再遍历容量）
  - fluid_particle_num 在动态添加后正确累加
  - 求解器收敛统计改为属性，不再每步 print
"""
