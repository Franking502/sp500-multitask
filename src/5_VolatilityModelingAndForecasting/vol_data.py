# -*- coding: utf-8 -*-
"""数据层：读取清洗后的收盘价 → 对数收益率 → 日历切分 → 已实现波动率。

约定
----
* 收益率统一用 **百分数点位**：``100 * ln(Close_t / Close_{t-1})``，
  即 1.2 表示 1.2%。arch 对数值尺度敏感，用 % 比用小数更稳定。
* 年化波动率统一为 **年化 %**：``sqrt(方差) * sqrt(252)``。
* 预测目标 RV 为**前瞻**已实现波动率：在 t 日计算的预测，对应未来
  t, t+1, ..., t+h-1 这 h 天的平均日方差开方再年化。

数据来源
--------
优先读取模块 0 的导出产物 ``data/processed/sp500_clean.csv``（由
``DataPreprocessor.export_processed()`` 生成）；若不存在，退回原始
``data/SP500_daily.csv`` 并做最小清洗。本模块只读数据，不写其他目录。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vol_config import ANNUALIZE, CLEAN_CSV, RAW_CSV, TRAIN_END, VAL_END


def _read_close(path: Path) -> pd.Series:
    """读取含 Date、Close 的 CSV，返回按日期升序、去重后的收盘价。"""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    lookup = {c.lower(): c for c in df.columns}
    date_col = lookup.get("date")
    close_col = lookup.get("close")
    if date_col is None or close_col is None:
        raise ValueError(f"{path.name} 需要包含 Date 与 Close 列，实际列: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[date_col], errors="coerce"),
            "Close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["Date", "Close"])
    out = out.loc[out["Close"] > 0]
    out = out.set_index("Date").sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    return out["Close"].astype(float)


def _resolve_path(csv_path: str | Path | None) -> Path:
    if csv_path is not None:
        return Path(csv_path)
    if CLEAN_CSV.exists():
        return CLEAN_CSV
    if RAW_CSV.exists():
        return RAW_CSV
    raise FileNotFoundError(
        f"找不到数据：{CLEAN_CSV} 与 {RAW_CSV} 均不存在；"
        "请先运行 `python src/0_DataPreprocessing/DataPreprocessing.py`"
    )


def load_returns(csv_path: str | Path | None = None) -> pd.Series:
    """读取收盘价并计算对数收益率（百分数，列名 ``ret_pct``）。"""
    close = _read_close(_resolve_path(csv_path))
    ret = (100.0 * np.log(close / close.shift(1))).dropna()
    ret.name = "ret_pct"
    ret.index.name = "Date"
    return ret


def load_simple_returns(csv_path: str | Path | None = None) -> pd.Series:
    """读取收盘价并计算简单收益率（百分数，列名 ``ret_pct``）。

    回测必须用简单收益率计算净值：``prod(1 + r_t) = Close_T / Close_0``；
    若把对数收益率当作简单收益率复利，会系统性低估上涨行情。
    """
    close = _read_close(_resolve_path(csv_path))
    ret = (100.0 * (close / close.shift(1) - 1.0)).dropna()
    ret.name = "ret_pct"
    ret.index.name = "Date"
    return ret


def split_calendar(
    ret: pd.Series,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> dict[str, pd.Series]:
    """按日历切分：train ≤ train_end < val ≤ val_end < test。"""
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    splits = {
        "train": ret.loc[ret.index <= train_end_ts],
        "val": ret.loc[(ret.index > train_end_ts) & (ret.index <= val_end_ts)],
        "test": ret.loc[ret.index > val_end_ts],
    }
    if any(len(part) == 0 for part in splits.values()):
        raise ValueError("日历切分后存在空子集，请检查数据范围与切分点")
    return splits


def split_table(splits: dict[str, pd.Series]) -> pd.DataFrame:
    """切分概览：起止日期、样本量、占比、年化波动率。"""
    rows = []
    total = sum(len(part) for part in splits.values())
    for name in ("train", "val", "test"):
        part = splits[name]
        rows.append(
            {
                "segment": name,
                "start": part.index.min().date(),
                "end": part.index.max().date(),
                "n_days": len(part),
                "share": round(len(part) / total, 4),
                "ann_vol_pct": round(float(part.std(ddof=1) * np.sqrt(ANNUALIZE)), 2),
            }
        )
    return pd.DataFrame(rows)


def realized_vol_forward(ret: pd.Series, horizon: int) -> pd.Series:
    """前瞻已实现波动率（年化 %），索引对齐到预测起点 t。

    ``RV_t = sqrt(mean(r_t^2, ..., r_{t+h-1}^2)) * sqrt(252)``
    与 GARCH 的 h 步预测（t 到 t+h-1 的平均日方差）口径一致。
    """
    if horizon < 1:
        raise ValueError("horizon 必须 >= 1")
    sq = ret.pow(2)
    rolling_mean = sq.rolling(horizon, min_periods=horizon).mean()
    rv = np.sqrt(rolling_mean.shift(-(horizon - 1))) * np.sqrt(ANNUALIZE)
    rv.name = f"rv_{horizon}"
    return rv


def trailing_vol(ret: pd.Series, window: int) -> pd.Series:
    """历史滚动波动率（年化 %），t 日只使用 t-window+1..t。"""
    sq = ret.pow(2)
    vol = np.sqrt(sq.rolling(window, min_periods=window).mean()) * np.sqrt(ANNUALIZE)
    vol.name = f"rv_trail_{window}"
    return vol


def naive_forecast(ret: pd.Series, window: int = 20) -> pd.Series:
    """naive 基准：用 t-1 收盘时的历史窗口波动率预测 t 日起的未来波动。"""
    vol = trailing_vol(ret, window).shift(1)
    vol.name = f"rv{window}_naive"
    return vol
