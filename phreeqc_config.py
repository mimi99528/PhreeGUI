"""
PHREEQCRM 配置和初始化模块
提供更稳健的化学系统初始化和错误处理
"""

import os
import logging
import weakref
from typing import List, Dict, Optional
import numpy as np

try:
    from phreeqcrm import PhreeqcRM
    PHREEQC_AVAILABLE = True
except ImportError:
    PHREEQC_AVAILABLE = False
    print("警告: phreeqcrm 模块未安装")

# 配置日志
_chem_logger = logging.getLogger("phreeqc_config")

# ==================== PhreeqcRM 实例唯一性守卫 ====================
# 依据（本轮实测，非推断）：
#   D:\tmp_pg_verify\impl\rm_m*.log      m1–m4：两个存活实例、先 del 再造、
#                                        函数返回后再造，全部 exit 0；
#   D:\tmp_pg_verify\impl\rm2_*.log      c1–c6：1000 单元 / 8 线程 /
#                                        SELECTED_OUTPUT / PrintChemistry(True)
#                                        / CreateMapping 全开，两实例共存 exit 0；
#   D:\tmp_pg_verify\impl\rm3_*.log      线程数不一致（1 与 8）也 exit 0；
#   D:\tmp_pg_verify\impl\raw_raw.log    单实例、未 InitialPhreeqcCell2Module
#                                        且未 CreateMapping，直接 GetConcentrations()
#                                        —— exit 139 段错误；
#   D:\tmp_pg_verify\impl\raw_safe.log    同一步骤但先分配溶液+CreateMapping
#                                        —— exit 0。
# 结论：崩溃与"实例个数"无关，真因是"在未映射的 rm 上读写浓度"。
# 因此这里守两条：
#   (1) 同进程只允许存在一个存活 PhreeqcRM —— 决策 D7 要求化学初始化只有一条
#       路径，多实例本身就是配置错误，必须硬失败而不是继续跑；
#   (2) 任何 GetConcentrations/SetConcentrations/RunCells 之前，必须已经由
#       本模块登记过 CreateMapping —— 否则就是上面那个 139，Python 层抓不到。
_LIVE_RM_REGISTRY = set()  # 元素是 weakref，不延长被管理对象的生命周期

# 【发现 1 实测台账】记录本进程每一次"通过守卫新建的 PhreeqcRM"。
# 用途：端到端跑完后可以如实回答"生产链路到底建了几个 rm、分别由谁建的"，
# 而不是靠读代码猜。台账只增不删（死亡数量由 _LIVE_RM_REGISTRY 反映）。
_RM_CREATIONS = []  # [(where, nxyz, nthreads)]


def rm_creation_ledger():
    """返回本进程历次 rm 创建记录 [(where, nxyz, nthreads), ...]（只读副本）。"""
    return list(_RM_CREATIONS)


def live_rm_count() -> int:
    """当前仍然存活的 PhreeqcRM 实例数（不延长任何对象的生命周期）。"""
    stale = [w for w in _LIVE_RM_REGISTRY if w() is None]
    for w in stale:
        _LIVE_RM_REGISTRY.discard(w)
    return len(_LIVE_RM_REGISTRY)


def _register_rm(rm, where: str, nxyz=None, nthreads=None) -> None:
    """登记一个新建的 PhreeqcRM；若已有存活实例则硬失败。"""
    stale = [w for w in _LIVE_RM_REGISTRY if w() is None]
    for w in stale:
        _LIVE_RM_REGISTRY.discard(w)
    if _LIVE_RM_REGISTRY:
        # 【编号消歧，本轮主动报备】"rm 只能有一条初始化路径"是上一轮旧编号体系
        # 里的 D7，与用户第三轮裁决的 **D7「中性水也必须进化学步」同号不同义**。
        # 原文直接写"决策 D7"会让验收的人以为去改化学步判据能解决它，
        # 所以改成引用领队的收尾条目编号，并显式说明与裁决 D7 无关。
        raise RuntimeError(
            "拒绝创建第二个 PhreeqcRM 实例：本进程已有一个存活的实例，"
            "本次请求来自 %r（nxyz=%r nthreads=%r），已登记 %d 个。"
            "本进程只允许一条化学初始化路径（领队 seq 26 收尾 2 的 rm 台账；"
            "注意：这与用户裁决 D7「中性水也必须进化学步」同号但**无关**）——"
            "两条路径并存就会出现两个"
            "组分集合和两套网格布局。请复用已有的 rm，不要再造一个。"
            "（历次创建记录：%s）"
            % (where, nxyz, nthreads, len(_LIVE_RM_REGISTRY),
               " -> ".join(c[0] for c in _RM_CREATIONS)))
    _LIVE_RM_REGISTRY.add(weakref.ref(rm))
    _RM_CREATIONS.append((where, nxyz, nthreads))


def mark_rm_mapped(rm, where: str) -> None:
    """CreateMapping 成功后打标记，供 require_rm_mapped 校验。"""
    rm._phreeqcrm_mapped_by = where


def require_rm_mapped(rm, where: str) -> None:
    """在读/写浓度或 RunCells 之前确认映射已建立，否则硬失败。

    未映射就调用 GetConcentrations() 是进程级段错误（实测 exit 139，
    raw_raw.log），try/except 拦不住，所以必须在 Python 层提前拦住。

    为什么用自记属性而不是问库：实测 rm.GetBackwardMapping() /
    GetForwardMapping() 在 CreateMapping 之前就已经返回长度 1000、
    distinct=1000 的数组，与映射之后完全同形（D:\\tmp_pg_verify\\impl\\rm_disc.log
    的 BEFORE/AFTER 两行），库里没有可区分的状态可查。
    所以只能由"我们自己调用 CreateMapping 的那一行"来登记。
    对测试里的 Mock rm 而言 getattr 返回真值，本守卫自动退化为通过，
    不会干扰单元测试。
    """
    if rm is None:
        raise RuntimeError("%s 需要 rm，但传进来是 None" % where)
    if not getattr(rm, "_phreeqcrm_mapped_by", None):
        raise RuntimeError(
            "%s 要在未 CreateMapping 的 rm 上读写浓度/跑反应 —— 这会让 "
            "PhreeqcRM 直接段错误退出进程（实测 exit 139，"
            "D:\\tmp_pg_verify\\impl\\raw_raw.log），Python 层捕获不到。"
            "正确顺序：LoadDatabase -> RunString -> FindComponents -> "
            "对每个单元 InitialPhreeqcCell2Module -> Set* 物理参数 -> "
            "CreateMapping -> 之后才允许 GetConcentrations/RunCells。"
            % where)


# 默认配置参数
# 【c1 注记】DEFAULT_NX/NY/NZ=6 只作为本文件内三个函数的形参默认值存在，
# 生产链路并不使用它们：GUI 构造期 :584 -> initialize_chemistry_system() ->
# create_fully_initialized_phreeqcrm(nx=10, ny=10, nz=10) 显式传 10，
# 与 CoupledSimulation 侧 CHEM_GRID_DIM=10（10^3=1000 单元）对齐。
# 探针实测（D:\tmp_pg_verify\impl\b7a.log）n_cells=1000 也证实了这一点。
# 因此这里填 6 不代表运行时的网格规模；若有人改这几行想调网格，
# 必须同时改 coupled_simulation.CHEM_GRID_DIM，否则 rm 单元数与化学场单元数
# 不一致会直接触发 _initialize_particle_chemistry 的长度校验硬失败。
DEFAULT_NX = 6
DEFAULT_NY = 6
DEFAULT_NZ = 6
DEFAULT_NXYZ = DEFAULT_NX * DEFAULT_NY * DEFAULT_NZ
# PhreeqcRM 的 OpenMP 线程数。【上限 4，不要上调】
# 前提：必须是打过线程补丁的 wheel。原版 0.0.17/0.0.20 在 nt>1 时 RunCells 内
#   SIGSEGV(139)：CreateMapping 改 count_chemistry 后 SetEndCells 只重算均分区间、
#   不搬单元，worker 拿到自己不持有的 cell，Get_solution(j) 返回 NULL，而 release
#   构建把 assert 编译掉了。补丁另修：搬迁归属判据、nthreads>count_chemistry 的
#   非法硬失败、以及 RunCellsThread 全程持 GIL 导致化学串行。
#   补丁 D:\tmp_pg_verify\rm_build\phreeqcrm-thread-fix.v3.patch（3 文件 9 hunk）；
#   原版 0.0.17 备份 D:\tmp_pg_verify\venv_backup_017\（换回原版必须同时改回 1）。
# 本机 16 逻辑核实测（两条路径分开报，别混用）：
#   恒等映射 1000 单元 45.93→14.42ms(3.19×)，216 单元 9.81→4.05ms(2.42×)；
#   GUI 真实稀疏路径（每帧 n_sig 2~97）3.34→2.24ms(1.49×)，nt=2 为 2.20ms，
#   两者在噪声内持平；nt=8 稀疏反而退化到 2.84ms 且 CPU 翻倍。
#   两条路径各 nt 化学末态与 nt=1 均逐位相同（ndiff=0）。
DEFAULT_NTHREADS = 4

# ==================== 数据库路径（决策 D2：唯一真源 = 仓库根 phreeqc.dat）====================
# 原状态（已实测确证）：phreeqc_config.DATABASE_PATHS 与 utils.DATABASE_PATHS 是两份
# 互不相同的候选列表，且都按 cwd 解析；运行时两边都命中
# C:\phreeqc\database\phreeqc.dat，而 test_water_activity.py 用的是仓库根的 phreeqc.dat。
# 两个文件差 132 行化学定义（NaF 差 2.14 个 log_k ≈138×、氟石 delta_h/analytic 不同、
# 仓库版多 K+ + F- = KF、启用硝酸根→铵还原），于是"pytest 绿"根本不能代表"仿真用的库对"。
#
# 现在：解析一律 __file__ 基，仓库根优先；系统安装路径只是 fallback。
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_DATABASE_PATH = os.path.join(PROJECT_ROOT, "phreeqc.dat")

# 系统安装位置（仅 fallback；不再包含 "../phreeqc.dat" —— 它会随 cwd 漂到上一级目录）
# 注：PITZER.DAT 路线本轮不启用，保留注释以免丢失作者意图。
# "PITZER.DAT",
SYSTEM_DATABASE_PATHS = [
    r"C:\phreeqc\database\phreeqc.dat",
    r"C:\Program Files\USGS\phreeqc\database\phreeqc.dat",
    r"/usr/local/share/phreeqc/phreeqc.dat",
    r"/usr/share/phreeqc/phreeqc.dat",
]


def find_phreeqc_database() -> Optional[str]:
    """返回唯一真源数据库的绝对路径。

    解析顺序（决策 D2）：仓库根 phreeqc.dat → 系统安装路径。
    找不到时返回 None 并记 error；调用方必须硬失败，不得静默换成别的库。
    """
    if os.path.isfile(REPO_DATABASE_PATH):
        path = os.path.abspath(REPO_DATABASE_PATH)
        _chem_logger.info("使用仓库真源 PHREEQC 数据库: %s", path)
        return path

    for db_path in SYSTEM_DATABASE_PATHS:
        if os.path.isfile(db_path):
            path = os.path.abspath(db_path)
            _chem_logger.error(
                "仓库根 %s 缺失，退回系统数据库 %s —— "
                "此时测试结果与运行结果不同源，请尽快补回仓库文件",
                REPO_DATABASE_PATH, path)
            return path

    _chem_logger.error(
        "未找到任何 PHREEQC 数据库（仓库根 %s 与系统路径都不存在）", REPO_DATABASE_PATH)
    return None


# ==================== 化学组分的唯一权威（决策 D7）====================
# 实测确证：SetComponentH2O(False) 下 phreeqcrm 0.0.17 + 本数据库
#   rm.GetComponentCount() == 5
#   rm.GetComponents()    == ['H', 'O', 'Charge', 'Cl', 'Na']
# 而旧代码到处硬写 ['H+', 'OH-', 'Na+', 'Cl-']（那是"物种名"，不是"组分名"）。
# 后果：所有 dict.get(comp_name, ...) 全部落到默认分支 → 传给 SetConcentrations
# 的是全零数组（连水都没有）→ PhreeqcRM 报 "Oxygen Mass of oxygen has not
# converged" 后 RunCells 失败并让整个进程 exit(127)。
def require_components(rm) -> List[str]:
    """组分名的唯一入口。拿不到就硬失败，绝不返回猜出来的列表。"""
    if rm is None:
        raise RuntimeError(
            "化学组分必须来自 PhreeqcRM.GetComponents()，但 rm 为 None；"
            "禁止使用任何硬编码组分名。")
    n_comp = int(rm.GetComponentCount())
    names = [str(c) for c in rm.GetComponents()]
    if n_comp == 0 or len(names) != n_comp:
        raise RuntimeError(
            f"PhreeqcRM 组分信息异常: GetComponentCount()={n_comp}, "
            f"len(GetComponents())={len(names)}")
    return names


# 决策 D6：中性水的元素总量（mol/L），键必须是"真实组分名"。
# 55.35 mol/L = 1000/18.015（水的摩尔浓度）；H 元素 = 2 × 55.34 ≈ 110.68。
#
# 【实测补充，本轮新增】H=110.68 比 2×O=110.70 少 0.02 mol/L，这 0.02 在
# PHREEQC 里以溶解氧的形式出现：实测 pH 6.997 / pe 13.94 / O2 5.0e-3 mol/L
# （探针 _probe_chem_species10）。也就是说本表描述的是"含溶解氧的水"。
# 这个值不能随手改成 H=2×O：无氧水实测 pe≈-2 ~ -8，碳酸水会有一半碳被还原
# 成 CH4（_probe_chem_species11），碳酸盐平衡示例直接失真。
# 因此保留 110.68 作为氧化性基线，新液体都在这个基线上叠加；只有"亚铁水"
# 故意用 H=2×O 造无氧环境（见 FERROUS_WATER_COMPOSITION 的注释）。
NEUTRAL_WATER_COMPOSITION = {
    'H': 110.68,
    'O': 55.35,
    'Charge': 0.0,
    'C': 0.0,
    'Cl': 0.0,
    'Fe': 0.0,
    'Na': 0.0,
}

# 源溶液按"元素总量"表达（不是物种名）：PHREEQC 自己从元素总量解物种分布。
# 旧写法把 H+/OH-/Cl-/Na+（物种）和 H/O/Charge（组分）混在一张表里，
# 真实组分查表时 H+/OH-/Cl-/Na+ 三个键永远不命中，只剩 1e-6 微量默认值，
# 于是"pH=3 的盐酸"实际拿到的是 1e-6 的 Cl —— 酸源等于没加。
#
# HCl 1e-3 mol/L ：H +1e-3、Cl +1e-3（电荷由体系自平衡，Charge 记 0）
ACIDIC_WATER_COMPOSITION = {
    'H': 110.68 + 1.0e-3,
    'O': 55.35,
    'Charge': 0.0,
    'C': 0.0,
    'Cl': 1.0e-3,
    'Fe': 0.0,
    'Na': 0.0,
}

# NaOH 1e-3 mol/L ：Na +1e-3、O +1e-3、H +1e-3
ALKALINE_WATER_COMPOSITION = {
    'H': 110.68 + 1.0e-3,
    'O': 55.35 + 1.0e-3,
    'Charge': 0.0,
    'C': 0.0,
    'Cl': 0.0,
    'Fe': 0.0,
    'Na': 1.0e-3,
}

# ── 示例场景用液体（4..7）───────────────────────────────────────────────
# 以下数值全部由 _probe_chem_species10 按生产路径（CreateMapping + RunCells）
# 实测核对过，括号里是实测结果，不是理论估算。

# 4 碳酸水：CO2 1e-3 mol/L。
#   实测 pH 4.68 / pe 16.24；碳形态 CO2 9.82e-4、HCO3- 2.1e-5、CO3-2 5e-11。
#   即"纯 CO2 溶于水"，酸性来自碳酸自身解离，不是外加酸。
CARBONIC_WATER_COMPOSITION = {
    'H': 110.68,
    'O': 55.35,
    'Charge': 0.0,
    'C': 1.0e-3,
    'Cl': 0.0,
    'Fe': 0.0,
    'Na': 0.0,
}

# 5 碳酸氢钠水：NaHCO3 1e-3 mol/L（Na 与 C 各 1e-3）。
#   实测 pH 8.27 / pe 12.63；碳形态 HCO3- 9.82e-4、CO3-2 9.5e-6、CO2 1.1e-5。
#   H 额外 +1e-3 是 HCO3- 自带的那一个氢，不能省（省掉会变成 Na2CO3 的碱性水）。
BICARBONATE_WATER_COMPOSITION = {
    'H': 110.68 + 1.0e-3,
    'O': 55.35,
    'Charge': 0.0,
    'C': 1.0e-3,
    'Cl': 0.0,
    'Fe': 0.0,
    'Na': 1.0e-3,
}

# 6 亚铁水：FeCl2 1e-3 mol/L，故意做成无氧。
#   H 取恰好 2×O=110.70（不留氧），实测 pH 6.29 / pe -1.96 / O2 0，
#   Fe 全部以 Fe2+ 存在（1.0e-3），没有 Fe(OH)3 沉淀。
#   这就是"地下水里的二价铁"：一旦遇到含氧水（液体 7），pe 升到 +17，
#   Fe2+ 被氧化成 Fe3+ 并水解沉淀，溶解态铁降到 ~1e-5（见 6+7 混合实测）。
FERROUS_WATER_COMPOSITION = {
    'H': 110.70,
    'O': 55.35,
    'Charge': 0.0,
    'C': 0.0,
    'Cl': 2.0e-3,
    'Fe': 1.0e-3,
    'Na': 0.0,
}

# 7 曝气水：在中性水基线上把 O 再加 1e-3，多出来的氧以溶解氧存在。
#   实测 pH 7.00 / pe 13.95 / O2 5.5e-3（中性水本身已有 5.0e-3）。
AERATED_WATER_COMPOSITION = {
    'H': 110.68,
    'O': 55.35 + 1.0e-3,
    'Charge': 0.0,
    'C': 0.0,
    'Cl': 0.0,
    'Fe': 0.0,
    'Na': 0.0,
}

WATER_COMPOSITIONS = {
    1: NEUTRAL_WATER_COMPOSITION,
    2: ACIDIC_WATER_COMPOSITION,
    3: ALKALINE_WATER_COMPOSITION,
    4: CARBONIC_WATER_COMPOSITION,
    5: BICARBONATE_WATER_COMPOSITION,
    6: FERROUS_WATER_COMPOSITION,
    7: AERATED_WATER_COMPOSITION,
}

# 液体类型 → 界面显示名。GUI 的 Ctrl+滚轮切换、液体标签都用这一张表，
# 不再在三个地方各写一份 ["水", "酸性溶液", "碱性溶液"]。
FLUID_TYPE_NAMES = {
    1: "水",
    2: "酸性溶液 (HCl)",
    3: "碱性溶液 (NaOH)",
    4: "碳酸水 (CO₂)",
    5: "碳酸氢钠水 (NaHCO₃)",
    6: "亚铁水 (Fe²⁺)",
    7: "曝气水 (O₂)",
}


def composition_values(comp_names: List[str],
                       table: Dict[str, float],
                       logger=None) -> List[float]:
    """按 comp_names 顺序从任意组分表取值。

    未命中：0.0 + error（不再静默用 1e-6 污染体系，也不静默跳过粒子）。
    """
    values = []
    unknown = []
    for name in comp_names:
        if name in table:
            values.append(float(table[name]))
        else:
            values.append(0.0)
            unknown.append(name)
    if unknown:
        msg = ("组分 %s 不在给定的初始成分表中，已按 0.0 处理；"
               "如果它应当有投入，请补全成分表。" % (sorted(set(unknown))))
        (logger or _chem_logger).error(msg)
    return values


def neutral_water_values(comp_names: List[str], logger=None) -> List[float]:
    """按 comp_names 顺序取中性水组分含量（决策 D6）。"""
    return composition_values(comp_names, NEUTRAL_WATER_COMPOSITION, logger)


def build_component_major_concentrations(comp_names: List[str],
                                         n_cells: int,
                                         logger=None) -> np.ndarray:
    """构造 PhreeqcRM 输入用的 component-major 一维数组。

    PhreeqcRM 约定（Concentrations2Utility 文档："Array contains n values
    for each component in sequence"）：
        [comp0_cell0, comp0_cell1, ..., comp0_cellN-1, comp1_cell0, ...]
    即组分在外层、单元在内层 —— 长度 = n_comp * n_cells。
    旧实现注释写着 component-major，实际循环是 cell-major（单元在外层），
    两者对同一 API 用了相反的布局；本轮统一到 component-major。
    """
    values = np.asarray(neutral_water_values(comp_names, logger), dtype=np.float64)
    return np.repeat(values, int(n_cells))


def create_robust_phreeqcrm_instance(nxyz: int = DEFAULT_NXYZ, 
                                   nthreads: int = DEFAULT_NTHREADS) -> Optional[PhreeqcRM]:
    """
    创建健壮的PHREEQCRM实例，包含错误处理和配置优化
    
    Args:
        nxyz: 网格单元总数
        nthreads: 线程数
        
    Returns:
        PhreeqcRM实例或None（如果失败）
    """
    if not PHREEQC_AVAILABLE:
        _chem_logger.error("PHREEQCRM不可用")
        return None
        
    try:
        # 创建实例
        rm = PhreeqcRM(nxyz, nthreads)
        # 【发现 1 修复】登记实例唯一性：同进程第二个存活 PhreeqcRM 直接硬失败。
        _register_rm(rm, "phreeqc_config.create_robust_phreeqcrm_instance",
                     nxyz=nxyz, nthreads=nthreads)

        rm.SetComponentH2O(False)  # <-- 移到这里，最优先级
        
        # 基本配置
        rm.SetErrorOn(True)
        rm.SetErrorHandlerMode(1)  # 详细错误模式
        
        # 性能优化设置
        rm.SetRebalanceByCell(True)
        rm.SetRebalanceFraction(0.1)
        rm.SetSpeciesSaveOn(False)
        rm.SetPartitionUZSolids(False)
        
        # 单位设置
        rm.SetUnitsSolution(2)  # mol/L
        rm.SetUnitsPPassemblage(1)
        rm.SetUnitsExchange(1)
        
        _chem_logger.info("PHREEQCRM实例创建成功")
        return rm

    except RuntimeError:
        # 单例守卫（_register_rm）抛的是 RuntimeError。若被下面这个
        # 宽 except 吞掉并 return None，调用方会以为"PHREEQC 不可用"，
        # 进而去走第二条初始化路径 —— 那正是决策 D4 禁止的静默降级。
        _chem_logger.error("PHREEQCRM 实例登记被拒绝（进程内已有存活实例）")
        raise
    except Exception as e:
        _chem_logger.error(f"创建PHREEQCRM实例失败: {e}")
        return None

def load_phreeqc_database(rm: PhreeqcRM) -> bool:
    """
    加载PHREEQC数据库
    
    Args:
        rm: PhreeqcRM实例
        
    Returns:
        bool: 是否成功加载
    """
    db_path = find_phreeqc_database()
    if not db_path:
        _chem_logger.error("找不到PHREEQC数据库文件")
        return False
    rm.SetPrintChemistryOn(True, True, True)
    rm.SetFilePrefix("phreeqc")
    try:
        rm.LoadDatabase(db_path)
        _chem_logger.info(f"成功加载数据库: {db_path}")
        return True
    except Exception as e:
        _chem_logger.error(f"加载数据库失败 {db_path}: {e}")
        return False

def setup_default_chemical_system(rm: PhreeqcRM) -> bool:
    """
    设置默认的化学系统（酸碱中和反应）
    
    Args:
        rm: PhreeqcRM实例
        
    Returns:
        bool: 是否成功设置
    """
    # 1. 必须在 FindComponents 之前设置！
    rm.SetComponentH2O(False)
    try:
        # 定义简化的化学系统（使用mol/L单位，避免水组分计算）
        input_string = """
        SOLUTION 1 
            units mol/L
            temp 25.0 
            pH 7.0
            density 1.0
        END
        
        SOLUTION 2 Simple Acidic
            pH 3.0 charge
            temp 25.0
            units mol/L
            Cl 0.01 as HCl
        END
        
        SOLUTION 3 Simple Basic
            pH 11.0 charge
            temp 25.0
            units mol/L
            Na 0.01 as NaOH
        END

        # 4..7 只用来让 FindComponents 把 C / Fe 登记成组分（粒子播种需要
        # SetConcentrations 里有这两列），同时作为液体 4..7 的可读定义。
        # 启动时 1000 个单元仍然全部铺 SOLUTION 1（决策 D8），这几条不会被
        # 分配给任何单元。
        SOLUTION 4 Carbonic
            units mol/L
            temp 25.0
            pH 5.6 charge
            C 1.0e-3
        END

        SOLUTION 5 Bicarbonate
            units mol/L
            temp 25.0
            pH 8.3 charge
            C 1.0e-3
            Na 1.0e-3
        END

        SOLUTION 6 Ferrous
            units mol/L
            temp 25.0
            pH 5.0 charge
            Fe(2) 1.0e-3
        END

        SOLUTION 7 Aerated
            units mol/L
            temp 25.0
            pH 7.0 charge
            O(0) 1.0e-3
        END
        
        SELECTED_OUTPUT 1
            -reset false
            -pH
            -pe
            -molalities CO2 HCO3- CO3-2 Fe+2 Fe+3 O2
        END
        """
        
        # 运行输入字符串
        rm.RunString(True, True, True, input_string)
        # 识别化学组分
        rm.FindComponents()
        
        # 【修复 b1/D7】组分名只能来自 rm；0 个组分是配置错误，不是"用默认值"的理由。
        comps = require_components(rm)
        ncomps = len(comps)
        _chem_logger.info(f"检测到 {ncomps} 个化学组分: {comps}")

        return True
        
    except Exception as e:
        _chem_logger.error(f"设置化学系统失败: {e}")
        return False

def initialize_grid_distribution(rm: PhreeqcRM, nx: int = DEFAULT_NX, 
                               ny: int = DEFAULT_NY, nz: int = DEFAULT_NZ) -> bool:
    """
    初始化网格分布和初始条件
    
    Args:
        rm: PhreeqcRM实例
        nx, ny, nz: 网格维度
        
    Returns:
        bool: 是否成功初始化
    """
    try:
        nxyz = nx * ny * nz

        # 【决策 D8 —— 用户裁决：启动化学场改为全中性】
        # 原来的"空间分布策略"在启动时就把 180 个单元铺成酸性(溶液2)、
        # 80 个铺成碱性(溶液3)，于是 0 粒子时 pH 图层就有 2.04/11.95 的热区
        # （上一轮实测：run11 的"网格化学初值布局完成 … 酸性 180、碱性 80"）。
        # 现在整个场只铺背景溶液（中性），酸/碱只在用户点击生成粒子时
        # 按该粒子的 fluid_type 赋予 —— 唯一入口是
        # CoupledSimulation.seed_particle_chemistry(ids, fluid_type)
        # （coupled_simulation.py:315，组分表 = phreeqc_config.WATER_COMPOSITIONS）。
        # 溶液 2/3 的定义仍保留在 setup_default_chemical_system 的 RunString 里，
        # 这里只是不再预铺；未改 phreeqc.dat 的任何化学内容。
        for i in range(nxyz):
            rm.InitialPhreeqcCell2Module(1, [i])

        # 【清理 c3 同类】原先这里是逐单元 print(x, y, z, " acid")，
        # 每次启动往控制台倒 260 行，既不说明结果也拦不住问题。
        # 换成一条汇总 INFO：单元化学初值的布局是"发现 2"的核心事实，
        # 需要用一条可比对的日志记录，而不是 260 行噪音。
        # （只改日志，不改任何 InitialPhreeqcCell2Module 的分配策略。）
        _chem_logger.info(
            "网格化学初值布局完成: %d×%d×%d=%d 单元 —— 全部为背景溶液(溶液1，中性)。"
            "决策 D8：酸/碱不再预铺，只有点击生成粒子时才按 fluid_type 赋酸/碱性。",
            nx, ny, nz, nxyz)
        
        # 设置物理参数
        rm.SetTimeStep(0.017)  # 17ms
        rm.SetTemperature([25.0] * nxyz)
        rm.SetPressure([1.0] * nxyz)
        rm.SetPorosity([0.3] * nxyz)
        rm.SetSaturationUser([1.0] * nxyz)
        rm.SetRepresentativeVolume([1.0] * nxyz)
        rm.SetComponentH2O(False)

        # 创建映射
        grid2chem = list(range(nxyz))
        # 【c3】原来这里 print(grid2chem) 把 1000 个整数整个打进 stdout，
        # 既不是诊断信息也不可核对，只把日志冲掉。要核对映射请读
        # rm.GetForwardMapping()，不要把上千个元素当日志打印。
        rm.CreateMapping(grid2chem)
        # 【发现 1 修复】登记"此 rm 已建立网格↔化学映射"。
        # 未 CreateMapping 就读 GetConcentrations() 是进程级段错误
        # （实测 raw_raw.log exit 139），Python 层拦不住，只能靠这个标记前置拦截。
        mark_rm_mapped(rm, "phreeqc_config.initialize_grid_distribution")

        # 运行初始平衡
        _chem_logger.info("运行初始化学平衡...")
        status = rm.RunCells()
        if status != 0:
            _chem_logger.warning(f"初始平衡返回状态: {status}")
        
        _chem_logger.info("网格分布初始化完成")
        return True

    except RuntimeError:
        # 守卫类错误必须原样上抛，不能降级成 return False：
        # create_fully_initialized_phreeqcrm 见 False 会 return None，
        # GUI 见 None 就去建第二个 PhreeqcRM（决策 D7 禁止的第三条路）。
        _chem_logger.error("网格分布初始化被守卫拦下，硬失败上抛")
        raise
    except Exception as e:
        _chem_logger.error(f"网格分布初始化失败: {e}")
        return False

def get_chemical_status(rm: PhreeqcRM) -> Dict:
    """
    获取化学系统的当前状态
    
    Args:
        rm: PhreeqcRM实例
        
    Returns:
        dict: 包含系统状态信息的字典
    """
    # 【发现 1 修复】GetConcentrations() 在未 CreateMapping 的 rm 上是
    # 进程级段错误（实测 raw_raw.log exit 139），必须先过守卫。
    require_rm_mapped(rm, "phreeqc_config.get_chemical_status")
    try:
        status = {
            'component_count': rm.GetComponentCount(),
            'components': [str(c) for c in rm.GetComponents()],
            'concentrations': rm.GetConcentrations(),
            'time': rm.GetTime(),
            'time_step': rm.GetTimeStep()
        }

        # 【c3 更正】原来这段是 `selected_out[0].get('pH', [])` 再套一个裸 except，
        # 而实测 GetSelectedOutput() 返回的是 ndarray（形状 (单元数, 列数)），
        # [0] 是标量，根本没有 .get —— 于是每次都抛 AttributeError 被裸 except
        # 吞掉，ph_data 恒为 []，看起来像"PHREEQC 没给 pH"，实际是读法写错。
        # 现在按实测结构取列，并把真实原因记下来，不再用空列表掩盖。
        try:
            n_sel = int(rm.GetSelectedOutputCount())
            headings = [str(h) for h in rm.GetSelectedOutputHeadings()]
            status['selected_output_count'] = n_sel
            status['selected_output_headings'] = headings
            if n_sel > 0 and 'pH' in headings:
                so = np.asarray(rm.GetSelectedOutput(), dtype=np.float64)
                if so.ndim == 1:
                    so = so.reshape((-1, 1))
                status['ph_data'] = so[:, headings.index('pH')]
            else:
                status['ph_data'] = np.zeros(0, dtype=np.float64)
        except Exception as e:
            _chem_logger.error("读取 SelectedOutput 失败（不是'没有 pH'，是读失败）: %r", e)
            status['ph_data'] = np.zeros(0, dtype=np.float64)

        return status

    except RuntimeError:
        raise
    except Exception as e:
        _chem_logger.error(f"获取化学状态失败: {e}")
        return {}

def create_fully_initialized_phreeqcrm(nx: int = DEFAULT_NX, 
                                    ny: int = DEFAULT_NY, 
                                    nz: int = DEFAULT_NZ,
                                    nthreads: int = DEFAULT_NTHREADS) -> Optional[PhreeqcRM]:
    """
    创建完全初始化的PHREEQCRM实例

    这是一个便利函数，执行完整的初始化流程

    Returns:
        PhreeqcRM实例；任何一步失败都硬失败上抛（决策 D4），不再返回 None。

    【决策 D4 更正】原来这里有三个 `return None` 分支。调用方
    （coupled_chemistry_physics_gui.initialize_chemistry_system）拿到 None 后
    只是把 chem_initialized 置 False 并打一条 error，仿真照旧继续；
    之后 worker 发现 coupled_sim.rm is None，就去 :299 再造第二个 PhreeqcRM。
    一个"返回 None"就同时违反 D4（静默降级）和 D7（只留一条初始化路径），
    所以改成上抛：让失败在启动时就以异常形式暴露出来。
    """
    _chem_logger.info("开始PHREEQCRM完整初始化流程...")

    # 1. 创建实例
    rm = create_robust_phreeqcrm_instance(nx * ny * nz, nthreads)
    if rm is None:
        raise RuntimeError(
            "PHREEQCRM 实例创建失败（nx=%d ny=%d nz=%d nthreads=%d）。"
            "决策 D4：初始化失败必须显式硬失败，不允许退回"
            "\"返回 None 让上层继续跑无化学仿真\"。" % (nx, ny, nz, nthreads))

    # 2. 加载数据库
    if not load_phreeqc_database(rm):
        raise RuntimeError(
            "数据库加载失败，没有任何组分定义，后面的 FindComponents / "
            "RunCells 全部无意义。决策 D2：唯一真源是仓库根 phreeqc.dat "
            "（解析结果=%r），不允许再退回到 InitialPhreeqc 自行定义组件。"
            % (find_phreeqc_database(),))

    # 3. 设置化学系统
    if not setup_default_chemical_system(rm):
        raise RuntimeError("化学系统（SOLUTION / SELECTED_OUTPUT 定义）设置失败")

    # 4. 初始化网格分布
    if not initialize_grid_distribution(rm, nx, ny, nz):
        raise RuntimeError("网格分布初始化失败")

    # 5. 验证状态：读实际值，不读猜测值
    status = get_chemical_status(rm)
    if not status:
        raise RuntimeError("get_chemical_status 返回空字典，说明状态读取本身就失败了")
    n_comps = int(status['component_count'])
    ph = status.get('ph_data')
    _chem_logger.info(
        "最终状态 - 组分数: %d 组分: %s 时间: %r 时间步: %r "
        "SelectedOutput 列数: %d pH 序列长度: %d",
        n_comps, status['components'], status['time'], status['time_step'],
        len(status.get('selected_output_headings', [])),
        0 if ph is None else int(np.asarray(ph).size))

    _chem_logger.info("PHREEQCRM完整初始化成功!")
    return rm

# 便利函数别名
initialize_phreeqcrm = create_fully_initialized_phreeqcrm