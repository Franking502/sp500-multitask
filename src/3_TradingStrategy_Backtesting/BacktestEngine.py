"""模块 D 的回测引擎：净收益、净值、回撤与逐笔交易。

对外暴露：
    net_returns(p, fwd, mask, cost, rf) -> (pos, net)   扣成本后的日净收益
    equity(net)                         -> np.ndarray   净值
    drawdown(eq)                        -> np.ndarray   相对历史高点的回撤
    perf(p, fwd, mask, ...)             -> dict         单策略指标
    perf_table(pos, fwd, mask)          -> DataFrame    五策略指标表
    trades_table(pos, fwd, mask)        -> DataFrame    逐笔交易明细

口径：
    净收益 = 仓位 × 次日收益 − |Δ仓位| × 单边成本 − max(仓位−1, 0) × 融资年化 / 252
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from StrategyRules import PARAMS, NAMES


def net_returns(p, fwd: pd.Series, mask, cost: float = None, rf: float = None):
    """按评估区间切片并扣掉换手成本与融资成本。"""
    cost = PARAMS["cost"] if cost is None else cost
    rf = PARAMS["rf"] if rf is None else rf
    pos = pd.Series(p, index=fwd.index)[mask].fillna(0).values
    ret = fwd[mask].fillna(0).values
    turnover = np.abs(np.diff(pos, prepend=0)) * cost
    financing = np.maximum(pos - 1, 0) * rf / 252
    return pos, pos * ret - turnover - financing


def equity(net) -> np.ndarray:
    return np.cumprod(1 + net)


def drawdown(eq) -> np.ndarray:
    return eq / np.maximum.accumulate(eq) - 1


def trade_log(pos, net) -> pd.DataFrame:
    """把仓位不变的一段视为一笔交易，返回每笔的区间与收益。"""
    pos = pd.Series(pos)
    blk = pos.ne(pos.shift()).cumsum()
    rows = []
    for _, idx in pos.groupby(blk).groups.items():
        seg = np.asarray(net)[idx]
        rows.append({"position": pos.iloc[idx[0]], "days": len(idx),
                     "return": float(np.prod(1 + seg) - 1)})
    return pd.DataFrame(rows)


def perf(p, fwd: pd.Series, mask, cost: float = None, rf: float = None) -> dict:
    pos, net = net_returns(p, fwd, mask, cost, rf)
    eq = equity(net)
    dd = drawdown(eq)
    tr = trade_log(pos, net)
    tr = tr[tr["position"] != 0]
    cagr = eq[-1] ** (252 / len(net)) - 1
    gross = pos * fwd[mask].fillna(0).values
    return {
        "累计净值": float(eq[-1]),
        "年化": float(cagr),
        "年化波动": float(net.std() * np.sqrt(252)),
        "夏普": float(net.mean() / net.std() * np.sqrt(252)),
        "最大回撤": float(dd.min()),
        "卡玛": float(cagr / abs(dd.min())),
        "胜率": float((tr["return"] > 0).mean()),
        "盈亏比": float(tr.loc[tr["return"] > 0, "return"].mean()
                    / abs(tr.loc[tr["return"] < 0, "return"].mean())),
        "笔数": int(len(tr)),
        "平均仓位": float(pos.mean()),
        "持仓比例": float((pos != 0).mean()),
        "累计换手": float(np.abs(np.diff(pos, prepend=0)).sum()),
        "成本拖累(年化)": float((gross.mean() - net.mean()) * 252),
    }


def perf_table(pos: dict, fwd: pd.Series, mask) -> pd.DataFrame:
    tab = pd.DataFrame({NAMES[k]: perf(v, fwd, mask) for k, v in pos.items()}).T
    tab["回撤改善(pp)"] = (tab["最大回撤"] - tab.loc[NAMES["A"], "最大回撤"]) * 100
    return tab


def trades_table(pos: dict, fwd: pd.Series, mask) -> pd.DataFrame:
    """五策略逐笔交易明细（只含持仓段，空仓/平仓区间不计为一笔）。"""
    dates = fwd.index[mask]
    out = []
    for k, v in pos.items():
        pos_k, net_k = net_returns(v, fwd, mask)
        series = pd.Series(pos_k)
        blk = series.ne(series.shift()).cumsum()
        for _, idx in series.groupby(blk).groups.items():
            i0, i1 = idx[0], idx[-1]
            if abs(float(np.mean(pos_k[idx]))) < 1e-12:
                continue
            out.append({
                "strategy": k, "start": dates[i0].date(), "end": dates[i1].date(),
                "days": len(idx), "avg_position": float(np.mean(pos_k[idx])),
                "return": float(np.prod(1 + net_k[idx]) - 1),
            })
    return pd.DataFrame(out)
