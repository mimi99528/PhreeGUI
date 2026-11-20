"""物理后端工厂：按 CONFIG.PHYSICS_BACKEND 选择 CPU(SPlisHSPlasH) 或 Taichi。

两个后端对外暴露同一套鸭子类型接口（见 taichi_mesh_system.py 顶部契约）。
选择失败必须硬失败（决策 D4），不做静默降级：
  - backend 值非法 → 直接抛错
  - backend=taichi 但 taichi 未安装/CUDA 不可用 → 直接抛错
"""
from ..utils.config import CONFIG


def create_mesh_system():
    backend = str(CONFIG.PHYSICS_BACKEND).strip().lower()
    if backend == "cpu":
        from .mesh_system import MeshSystem
        return MeshSystem()
    if backend == "taichi":
        try:
            from .taichi_mesh_system import TaichiMeshSystem
        except ImportError as e:
            raise RuntimeError(
                f"PHYSICS_BACKEND=taichi 但 Taichi 后端无法导入：{e}。"
                "请安装 taichi（pip install 'taichi>=1.7'）或改回 "
                "PHREEGUI_PHYSICS_BACKEND=cpu。（决策 D4：不静默降级。）") from e
        return TaichiMeshSystem()
    raise RuntimeError(
        f"未知的 PHYSICS_BACKEND={CONFIG.PHYSICS_BACKEND!r}"
        "（合法值：cpu / taichi）")
