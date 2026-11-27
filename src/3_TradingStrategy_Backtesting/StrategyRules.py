"""模块 D 的策略层：五种仓位规则。

对外暴露：
    PARAMS                      -> dict                成本、融资、目标波动、缓冲带
    load_market()               -> (close, fwd, lr)   收盘价、次日收益、日对数收益
    load_signal(close)          -> (s, mask)          模块 C 信号、评估区间掩码
    band(x, w)                  -> np.ndarray         缓冲带（迟滞）后的仓位
    build_positions(close, lr, s) -> dict[str, Series] 五个策略的仓位

设计约束：
    1. 仓位在 t 日收盘确定，收益取 t -> t+1，因此用 fwd = shift(-1)；
    2. 波动率、均线只用 <= t 的数据，rolling 不 center、不 shift 正向；
    3. D 与 E 的波动率窗口与缓冲带完全相同，只有杠杆上限不同，
       两者的差异因此可以单独归因于趋势约束。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
            if (p / "data" / "SP500_daily.csv").exists())
PRICE_FILE = ROOT / "data/SP500_daily.csv"

# ---------------------------------------------------------------- 参数

PARAMS = {
    "cost": 0.001,   # 单边成本，千分之一
    "rf": 0.03,      # 融资年化利率，超过 1 倍仓位的部分
    "target_vol": 0.20,   # 目标年化波动
    "band": 0.30,    # 换手缓冲带：目标与当前仓位差超过该值才调仓
    "vol_window": 40,
    "trend_window": 200,
    "cap_fixed": 1.75,    # D 的固定杠杆上限
    "cap_up": 1.40,       # E 在趋势线上方的上限
    "cap_down": 0.50,     # E 在趋势线下方的上限
}

NAMES = {
    "A": "A 买入持有",
    "B": "B 信号 多/空仓",
    "C": "C 信号 多/空",
    "D": "D 波动率目标",
    "E": "E 波动率+趋势",
}

SIGNAL_FILE = ROOT / "reports/2_RiseFallForecast/signals/test_signals.csv"

# ---------------------------------------------------------------- 数据

def load_market():
    """读取收盘价，构造次日收益 fwd 与日对数收益 lr。

    直接读 data/SP500_daily.csv，不依赖 src/features.py，因此本模块可单独拷进团队仓库运行。
    数据本身已按日期升序且无重复日期，这里只做 sort_index 兜底。
    """
    close = (pd.read_csv(PRICE_FILE, parse_dates=["Date"])
             .set_index("Date")["Close"].sort_index())
    fwd = close.shift(-1) / close - 1
    lr = np.log(close).diff()
    return close, fwd, lr


def read_signal() -> pd.DataFrame:
    """读取模块 C 输出的测试期信号表（列：signal_date, target_date, p_up, pred_up）。"""
    return pd.read_csv(SIGNAL_FILE, parse_dates=["signal_date"])


def signal_summary() -> dict:
    """信号本身的覆盖信息：天数、持仓率、起止日期。"""
    sig = read_signal()
    return {"days": int(len(sig)),
            "holding_rate": float(sig["pred_up"].mean()),
            "start": str(sig["signal_date"].min().date()),
            "end": str(sig["signal_date"].max().date())}


def load_signal(close: pd.Series):
    """把信号按交易日对齐，没有信号的日子补 0（空仓），并返回评估区间掩码。"""
    sig = read_signal()
    s = sig.set_index("signal_date")["pred_up"].reindex(close.index).fillna(0)
    mask = close.index >= sig["signal_date"].min()
    return s, mask

# ---------------------------------------------------------------- 规则

def band(x, w: float = None) -> np.ndarray:
    """缓冲带：目标仓位与当前仓位之差超过 w 才调仓，否则维持不动。"""
    w = PARAMS["band"] if w is None else w
    out, cur = np.zeros(len(x)), 0.0
    for i, v in enumerate(x):
        cur = v if abs(v - cur) > w else cur
        out[i] = cur
    return out


def build_positions(close: pd.Series, lr: pd.Series, s: pd.Series) -> dict[str, np.ndarray]:
    """五个策略的逐日目标仓位（未按评估区间切片）。"""
    p = PARAMS
    vol = (lr.rolling(p["vol_window"]).std() * np.sqrt(252)).fillna(p["target_vol"]).values
    trend = close.values > close.rolling(p["trend_window"]).mean().values
    cap = np.where(trend, p["cap_up"], p["cap_down"])
    return {
        "A": np.ones(len(close)),
        "B": s.values,
        "C": np.where(s.values == 1, 1.0, -1.0),
        "D": band(np.minimum(p["target_vol"] / vol, p["cap_fixed"])),
        "E": band(np.minimum(p["target_vol"] / vol, cap)),
    }


def position_frame(close: pd.Series, pos: dict[str, np.ndarray]) -> pd.DataFrame:
    """把仓位字典整理成按日期的 DataFrame。"""
    return pd.DataFrame(pos, index=close.index)
