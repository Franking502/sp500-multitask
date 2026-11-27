"""模块 D 的评估与产物导出：生成 metrics/*.json 与 positions/*.csv。

用法：
    python src/3_TradingStrategy_Backtesting/EvaluateStrategyMetrics.py
    python src/3_TradingStrategy_Backtesting/EvaluateStrategyMetrics.py --output-dir reports/3_TradingStrategy_Backtesting

产物：
    metrics/test_metrics.json          五策略全部指标
    metrics/benchmark_comparison.json  相对买入持有的超额（pp）
    metrics/year_by_year.json          分年度收益
    metrics/cost_sensitivity.json      融资成本 3% / 4.5% / 6% 下的稳健性
    metrics/drawdown_window_2025.json  2025-02-19 -> 2025-04-07 最大回撤窗口
    positions/positions_test.csv       逐日目标仓位
    positions/equity_test.csv          逐日净值与回撤
    positions/trades_test.csv          逐笔交易明细
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from BacktestEngine import net_returns, equity, drawdown, perf, perf_table, trades_table
from StrategyRules import (PARAMS, NAMES, SIGNAL_FILE, ROOT, build_positions,
                           load_market, load_signal, read_signal, signal_summary)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(obj):
    """把 numpy 标量 / NaN 转成可序列化的值。"""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else round(float(obj), 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def yearly_returns(pos: dict, fwd: pd.Series, mask) -> dict:
    out = {}
    for k, v in pos.items():
        _, net = net_returns(v, fwd, mask)
        s = pd.Series(net, index=fwd.index[mask])
        out[NAMES[k]] = {str(y): round(float(np.prod(1 + g) - 1), 6)
                         for y, g in s.groupby(s.index.year)}
    return out


def cost_sensitivity(pos: dict, fwd: pd.Series, mask, levels=(0.03, 0.045, 0.06)) -> dict:
    out = {}
    for rf in levels:
        row = {NAMES[k]: {"年化": round(perf(v, fwd, mask, rf=rf)["年化"], 6),
                          "夏普": round(perf(v, fwd, mask, rf=rf)["夏普"], 6),
                          "最大回撤": round(perf(v, fwd, mask, rf=rf)["最大回撤"], 6)}
               for k, v in pos.items()}
        row["超额收益(pp)"] = round((row[NAMES["E"]]["年化"] - row[NAMES["A"]]["年化"]) * 100, 3)
        out[f"{rf:.1%}"] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(ROOT / "reports/3_TradingStrategy_Backtesting"))
    args = ap.parse_args()
    out = Path(args.output_dir)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "positions").mkdir(parents=True, exist_ok=True)

    close, fwd, lr = load_market()
    s, mask = load_signal(close)
    pos = build_positions(close, lr, s)
    tab = perf_table(pos, fwd, mask)

    meta = {
        "module": "3_TradingStrategy_Backtesting",
        "split": "test",
        "window": {"start": str(close.index[mask][0].date()),
                   "end": str(close.index[mask][-1].date()),
                   "trading_days": int(mask.sum())},
        "params": PARAMS,
        "signal": {**{k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in signal_summary().items()},
                   "source": str(SIGNAL_FILE.relative_to(ROOT)),
                   "no_signal_days": [str(d.date()) for d in
                                      close.index[mask].difference(pd.Index(read_signal()["signal_date"]))],
                   "note": "无信号的交易日按空仓处理；持仓率按有信号的交易日统计"},
        "data_sha256": {str(Path("data/SP500_daily.csv")): sha256(ROOT / "data/SP500_daily.csv"),
                        str(SIGNAL_FILE.relative_to(ROOT)): sha256(SIGNAL_FILE)},
    }

    strategies = {}
    for k, name in NAMES.items():
        row = tab.loc[name].to_dict()
        strategies[k] = {"name": name, **row}
    (out / "metrics/test_metrics.json").write_text(
        json.dumps(clean({**meta, "strategies": strategies}), ensure_ascii=False, indent=2),
        encoding="utf-8")

    base = tab.loc[NAMES["A"]]
    comp = {NAMES[k]: {
        "年化超额(pp)": round((tab.loc[NAMES[k], "年化"] - base["年化"]) * 100, 3),
        "夏普差": round(tab.loc[NAMES[k], "夏普"] - base["夏普"], 4),
        "回撤改善(pp)": round(tab.loc[NAMES[k], "回撤改善(pp)"], 3),
        "卡玛差": round(tab.loc[NAMES[k], "卡玛"] - base["卡玛"], 4),
    } for k in "BCDE"}
    (out / "metrics/benchmark_comparison.json").write_text(
        json.dumps(clean({"benchmark": NAMES["A"], "comparison": comp}),
                   ensure_ascii=False, indent=2), encoding="utf-8")

    (out / "metrics/year_by_year.json").write_text(
        json.dumps(clean(yearly_returns(pos, fwd, mask)), ensure_ascii=False, indent=2),
        encoding="utf-8")

    (out / "metrics/cost_sensitivity.json").write_text(
        json.dumps(clean({"strategy_set": [NAMES["A"], NAMES["E"]],
                          "results": cost_sensitivity(pos, fwd, mask)}),
                   ensure_ascii=False, indent=2), encoding="utf-8")

    m = (close.index >= "2025-02-19") & (close.index <= "2025-04-07")
    window = {NAMES[k]: {"平均仓位": round(float(net_returns(v, fwd, m)[0].mean()), 4),
                         "区间收益": round(float(np.prod(1 + net_returns(v, fwd, m)[1]) - 1), 6)}
              for k, v in pos.items()}
    (out / "metrics/drawdown_window_2025.json").write_text(
        json.dumps(clean({"start": "2025-02-19", "end": "2025-04-07",
                          "trading_days": int(m.sum()), "strategies": window}),
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- CSV
    pf = pd.DataFrame(pos, index=close.index)[mask]
    pf.index.name = "date"
    pf.to_csv(out / "positions/positions_test.csv", float_format="%.4f")

    eq = pd.DataFrame(index=close.index[mask])
    for k, v in pos.items():
        _, net = net_returns(v, fwd, mask)
        e = equity(net)
        eq[f"eq_{k}"] = np.round(e, 6)
        eq[f"dd_{k}"] = np.round(drawdown(e), 6)
    eq.index.name = "date"
    eq.to_csv(out / "positions/equity_test.csv", float_format="%.6f")

    trades_table(pos, fwd, mask).to_csv(out / "positions/trades_test.csv",
                                        index=False, float_format="%.6f")

    print(f"评估区间 {meta['window']['start']} -> {meta['window']['end']}"
          f"（{meta['window']['trading_days']} 个交易日）")
    print(tab.round(4).to_string())
    print(f"\n产物已写入 {out}")


if __name__ == "__main__":
    main()
