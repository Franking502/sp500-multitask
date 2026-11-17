# -*- coding: utf-8 -*-
"""评价层：RMSE / MAE / QLIKE、Mincer-Zarnowitz 回归、Diebold-Mariano 检验。

口径
----
* 预测值与真值都是 **年化 %**（例如 18.5 表示 18.5%）；
* 预测在 t 日给出，覆盖 t..t+h-1 共 h 天，真值由 ``realized_vol_forward`` 同口径构造；
* QLIKE 在**方差**上计算：``mean(RV²/σ̂² - ln(RV²/σ̂²) - 1)``，对低估波动惩罚更重；
* DM 检验用平方误差损失，方差按 Newey-West 修正（滞后 h-1）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from vol_config import HORIZONS


def _paired(forecast: pd.Series, target: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """取交集、去重、剔除 NaN/inf 后返回 (预测, 真值) 数组。"""
    frame = pd.concat([forecast.rename("f"), target.rename("t")], axis=1, join="inner").dropna()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame["f"].to_numpy(dtype=float), frame["t"].to_numpy(dtype=float)


def rmse(forecast: pd.Series, target: pd.Series) -> float:
    f, t = _paired(forecast, target)
    return float(np.sqrt(np.mean((f - t) ** 2))) if len(f) else float("nan")


def mae(forecast: pd.Series, target: pd.Series) -> float:
    f, t = _paired(forecast, target)
    return float(np.mean(np.abs(f - t))) if len(f) else float("nan")


def qlike(forecast: pd.Series, target: pd.Series) -> float:
    """方差口径的 QLIKE 损失（越小越好）。

    对 ``RV²/σ̂²`` 做上下截断，避免个别 0 收益日导致 log(0) 发散。
    """
    f, t = _paired(forecast, target)
    if len(f) == 0:
        return float("nan")
    ratio = np.clip((t**2) / np.maximum(f**2, 1e-12), 1e-10, 1e10)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mz_regression(forecast: pd.Series, target: pd.Series) -> dict[str, float]:
    """Mincer-Zarnowitz：target = a + b * forecast + e，理想情况 a≈0、b≈1、R² 高。"""
    f, t = _paired(forecast, target)
    if len(f) < 3 or np.std(f) == 0:
        return {"mz_a": np.nan, "mz_b": np.nan, "mz_r2": np.nan}
    slope, intercept = np.polyfit(f, t, 1)
    fitted = intercept + slope * f
    ss_res = float(np.sum((t - fitted) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"mz_a": float(intercept), "mz_b": float(slope), "mz_r2": float(r2)}


def _nw_variance(values: np.ndarray, lag: int) -> float:
    """Newey-West 长期方差估计。"""
    centered = values - values.mean()
    n = len(centered)
    var = float(centered @ centered) / n
    for i in range(1, lag + 1):
        weight = 1.0 - i / (lag + 1.0)
        gamma = float(centered[i:] @ centered[:-i]) / n
        var += 2.0 * weight * gamma
    return max(var, 1e-18)


def dm_test(
    forecast_a: pd.Series, forecast_b: pd.Series, target: pd.Series, horizon: int = 1
) -> dict[str, float]:
    """Diebold-Mariano 检验：A 的平方误差是否显著小于 B。

    返回统计量与双侧 p 值；统计量为负表示 A 的损失更小（A 更优）。
    """
    common = pd.concat(
        [forecast_a.rename("a"), forecast_b.rename("b"), target.rename("t")], axis=1, join="inner"
    ).dropna()
    if common.empty:
        return {"dm_stat": np.nan, "dm_pvalue": np.nan, "n": 0}

    diff = (common["a"] - common["t"]) ** 2 - (common["b"] - common["t"]) ** 2
    values = diff.to_numpy(dtype=float)
    n = len(values)
    mean_diff = float(values.mean())
    var = _nw_variance(values, lag=max(horizon - 1, 0))
    stat = mean_diff / np.sqrt(var / n)
    pvalue = float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))
    return {"dm_stat": float(stat), "dm_pvalue": pvalue, "n": int(n)}


def evaluate(
    forecasts: dict[str, pd.DataFrame],
    rv: dict[int, pd.Series],
    *,
    segment: str,
    naive: pd.Series | None = None,
    naive_name: str = "rv20_naive",
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """计算 ``模型 × 步长`` 的全部精度指标。

    ``forecasts``: {模型名: 含 sigma_h 列的 DataFrame}（已切到目标区间）；
    ``rv``: {h: 前瞻已实现波动率}；
    ``naive``: 可选的基准预测（同一序列用于所有步长）。
    """
    rows: list[dict[str, object]] = []
    horizons = tuple(h for h in horizons if h in rv)
    for name, frame in forecasts.items():
        for h in horizons:
            target = rv[h]
            sigma = frame[f"sigma_{h}"]
            f, t = _paired(sigma, target)
            if len(f) == 0:
                continue
            row = {
                "segment": segment,
                "model": name,
                "horizon": h,
                "n": len(f),
                "rmse": rmse(sigma, target),
                "mae": mae(sigma, target),
                "qlike": qlike(sigma, target),
            }
            row.update(mz_regression(sigma, target))
            rows.append(row)

    if naive is not None:
        if forecasts:
            common_index = next(iter(forecasts.values())).index
            naive = naive.loc[naive.index.intersection(common_index)]
        for h in horizons:
            target = rv[h]
            f, t = _paired(naive, target)
            if len(f) == 0:
                continue
            row = {
                "segment": segment,
                "model": naive_name,
                "horizon": h,
                "n": len(f),
                "rmse": rmse(naive, target),
                "mae": mae(naive, target),
                "qlike": qlike(naive, target),
            }
            row.update(mz_regression(naive, target))
            rows.append(row)

    return pd.DataFrame(rows)


def dm_table(
    forecasts: dict[str, pd.DataFrame],
    rv: dict[int, pd.Series],
    *,
    segment: str,
    baseline: str = "garch",
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """以某个模型为基准，两两比较各步长的预测损失。"""
    rows = []
    if baseline not in forecasts:
        return pd.DataFrame(rows)
    horizons = tuple(h for h in horizons if h in rv)
    for name, frame in forecasts.items():
        if name == baseline:
            continue
        for h in horizons:
            res = dm_test(frame[f"sigma_{h}"], forecasts[baseline][f"sigma_{h}"], rv[h], horizon=h)
            rows.append(
                {
                    "segment": segment,
                    "model": name,
                    "baseline": baseline,
                    "horizon": h,
                    "dm_stat": res["dm_stat"],
                    "dm_pvalue": res["dm_pvalue"],
                    "n": res["n"],
                }
            )
    return pd.DataFrame(rows)
