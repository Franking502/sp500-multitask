# -*- coding: utf-8 -*-
"""波动率建模模块的全局配置：路径、切分点、模型规格、回测参数。

所有相对路径都基于本文件位置解析，因此在任何工作目录下运行都成立。
模块内不修改仓库其他目录，输出统一写到 ``Outcome/``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_CSV = DATA_DIR / "processed" / "sp500_clean.csv"
RAW_CSV = DATA_DIR / "SP500_daily.csv"
OUTCOME_DIR = MODULE_DIR / "Outcome"
REQUIREMENT_FILE = MODULE_DIR / "requirement.txt"

# ---- 时间划分（本模块按日历划分，与模块 0 的 70/15/15 比例划分不同） ----
TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"

# ---- 建模与回测常量 ----
ANNUALIZE = 252                 # 年化交易日
HORIZONS: tuple[int, ...] = (1, 5, 10)
REFIT_EVERY = 20                # 每 20 个交易日重新估计一次参数
TARGET_VOL = 0.15               # 目标年化波动率 15%
W_MAX = 1.0                     # 仓位上限（不加杠杆）
COST = 0.001                    # 单边交易成本 0.1%
RNG_SEED = 20240101             # EGARCH 多步模拟预测的随机种子（保证可复现）

# ---- 图表配色（与模块 0 保持一致） ----
NAVY = "#1f3b57"
TEAL = "#2a9d8f"
CRIMSON = "#d1495b"
AMBER = "#c9962b"
GREY = "#6b7280"
GRID = "#d7dde5"
MODEL_COLORS = {"garch": NAVY, "egarch": CRIMSON, "gjr": TEAL}


@dataclass(frozen=True)
class ModelSpec:
    """一个 GARCH 族模型的设定。"""

    key: str
    label: str
    vol: str
    p: int
    o: int
    q: int
    dist: str
    has_gamma: bool
    forecast_method: str = "analytic"
    simulations: int = 0

    @property
    def uses_t(self) -> bool:
        return self.dist != "normal"


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("garch", "GARCH(1,1)", "Garch", 1, 0, 1, "normal", False),
    # EGARCH 的对数方差结构没有多步解析解，arch 要求用模拟；固定种子保证可复现
    ModelSpec("egarch", "EGARCH(1,1)", "EGARCH", 1, 1, 1, "studentst", True,
              forecast_method="simulation", simulations=2000),
    ModelSpec("gjr", "GJR-GARCH(1,1)", "Garch", 1, 1, 1, "studentst", True),
)

SPEC_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}
MODEL_KEYS: tuple[str, ...] = tuple(spec.key for spec in MODEL_SPECS)


def ensure_arch_available() -> None:
    """检查 arch 是否可导入；缺失时给出明确的安装提示。

    依赖清单见同目录 ``requirement.txt``：

        pip install -r src/5_VolatilityModelingAndForecasting/requirement.txt
    """
    try:
        import arch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "缺少 arch 依赖（GARCH 族模型需要）。请执行：\n"
            "    pip install -r "
            f'"{REQUIREMENT_FILE}"\n'
            "或：\n"
            "    pip install arch==9.0.0"
        ) from exc
