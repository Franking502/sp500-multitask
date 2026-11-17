# -*- coding: utf-8 -*-
"""回测层：波动率目标仓位 + 交易成本 + 绩效指标。

仓位公式
--------
``w_t = min(σ_target / σ̂_{t|t-1}, w_max)``，σ 均为年化值（σ̂ 为 %，σ_target 为小数）。

无前视
------
``vol_forecast.rolling_forecast`` 返回的 σ̂ 以**目标日 t** 为索引，且只用到 t-1
的收盘信息，因此 ``w[t]`` 可以直接乘以 ``r[t]``，不需要再 shift。回测前请保持
``ret`` 与 ``weights`` 的索引一致（本模块的 ``run_backtest`` 会强制 reindex）。

成本
----
单边 ``cost``（默认 0.1%），按 ``|Δw|`` 计费：``cost_t = cost * |w_t - w_{t-1}|``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vol_config import ANNUALIZE, COST, TARGET_VOL, W_MAX


def target_weights(
    sigma_ann_pct: pd.Series,
    *,
    target_vol: float = TARGET_VOL,
    w_max: float = W_MAX,
) -> pd.Series:
    """由年化波动率预测（%）生成仓位，截断到 [0, w_max]。"""
    if target_vol <= 0 or w_max <= 0:
        raise ValueError("target_vol 与 w_max 必须为正")
    sigma = sigma_ann_pct.astype(float) / 100.0
    weights = target_vol / sigma.replace(0.0, np.nan)
    weights = weights.clip(lower=0.0, upper=w_max)
    weights.name = "weight"
    return weights


def run_backtest(
    ret: pd.Series,
    weights: pd.Series,
    *,
    cost: float = COST,
    name: str = "strategy",
) -> pd.DataFrame:
    """按给定仓位回测，返回逐日明细。

    列：``ret``（标的）、``weight``、``dw``、``cost``、``strat_ret``、``equity``。
    ``ret/strat_ret/cost`` 的单位都是 %（与输入的 ``ret`` 一致）。
    """
    ret = ret.astype(float)
    w = weights.reindex(ret.index).astype(float).fillna(0.0).clip(0.0, None)

    dw = w.diff()
    dw = dw.fillna(w)                      # 首日从 0 建仓
    costs = dw.abs() * cost * 100.0        # 转成 % 点位
    strat_ret = w * ret - costs

    equity = (1.0 + strat_ret / 100.0).cumprod()
    out = pd.DataFrame(
        {
            "ret": ret,
            "weight": w,
            "dw": dw,
            "cost": costs,
            "strat_ret": strat_ret,
            "equity": equity,
        }
    )
    out.index.name = "Date"
    out.attrs["name"] = name
    return out


def buy_and_hold(ret: pd.Series, *, cost: float = COST) -> pd.DataFrame:
    """满仓基准（只在第一天建仓，之后不再交易）。"""
    weights = pd.Series(1.0, index=ret.index, name="weight")
    return run_backtest(ret, weights, cost=cost, name="Buy & Hold")


def drawdown(equity: pd.Series) -> pd.Series:
    """相对历史高点的回撤（小数，≤0）。"""
    return equity / equity.cummax() - 1.0


def performance(backtest: pd.DataFrame, name: str | None = None) -> dict[str, object]:
    """绩效指标：年化收益、年化波动、Sharpe、最大回撤、Calmar、仓位与换手。"""
    strat = backtest["strat_ret"].astype(float) / 100.0
    equity = backtest["equity"].astype(float)
    years = len(backtest) / ANNUALIZE
    if years <= 0 or len(strat) < 2:
        raise ValueError("回测样本过短")

    final = float(equity.iloc[-1])
    cagr = final ** (1.0 / years) - 1.0
    vol = float(strat.std(ddof=1) * np.sqrt(ANNUALIZE))
    sharpe = float(strat.mean() / strat.std(ddof=1) * np.sqrt(ANNUALIZE))
    dd = drawdown(equity)
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")

    return {
        "strategy": name or backtest.attrs.get("name", "strategy"),
        "start": backtest.index.min().date(),
        "end": backtest.index.max().date(),
        "n_days": len(backtest),
        "total_return": round(final - 1.0, 4),
        "cagr": round(cagr, 4),
        "ann_vol": round(vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 3),
        "avg_weight": round(float(backtest["weight"].mean()), 3),
        "ann_turnover": round(float(backtest["dw"].abs().sum() / years), 2),
        "ann_cost_drag": round(float(backtest["cost"].sum() / years) / 100.0, 4),
    }


def cost_sensitivity(
    ret: pd.Series, weights: pd.Series, costs: tuple[float, ...] = (0.0, 0.001, 0.002)
) -> pd.DataFrame:
    """同一仓位在不同单边成本下的绩效对比。"""
    rows = []
    for cost in costs:
        bt = run_backtest(ret, weights, cost=cost, name=f"cost={cost:.3%}")
        row = performance(bt)
        row["cost"] = cost
        rows.append(row)
    return pd.DataFrame(rows)
