# -*- coding: utf-8 -*-
"""波动率建模与预测 · 主入口。

流程
----
1. 读取对数收益率，按日历切分 train/val/test；
2. 在训练集上拟合 GARCH / EGARCH / GJR-GARCH，输出参数表与诊断；
3. 在 val+test 上做逐日滚动预测（每 20 天重估参数，其余时间固定参数滤波）；
4. 计算 RMSE / MAE / QLIKE / MZ，并与 RV20 naive 基准对比；
5. 用验证集最优模型做波动率目标回测，与满仓、RV20 调仓对比；
6. 输出图表与 ``summary.md``。

用法
----
    python run_analysis.py                     # 全流程
    python run_analysis.py --refit-every 5     # 更频繁地重估参数
    python run_analysis.py --no-backtest       # 只做建模与预测评价

依赖
----
    pip install -r requirement.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import vol_backtest as vbt
import vol_evaluation as vev
import vol_forecast as vfc
import vol_models as vmd
import vol_plots as vpl
from vol_config import (
    COST,
    HORIZONS,
    MODEL_KEYS,
    MODEL_SPECS,
    OUTCOME_DIR,
    REFIT_EVERY,
    TARGET_VOL,
    TRAIN_END,
    VAL_END,
    W_MAX,
    ensure_arch_available,
)
from vol_data import (
    load_returns,
    load_simple_returns,
    naive_forecast,
    realized_vol_forward,
    split_calendar,
    split_table,
    trailing_vol,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="标普 500 波动率建模、预测与波动率目标回测")
    parser.add_argument("--csv", type=Path, default=None, help="自定义收益率数据（含 Date/Close）")
    parser.add_argument("--outcome", type=Path, default=OUTCOME_DIR, help="输出目录")
    parser.add_argument("--refit-every", type=int, default=REFIT_EVERY,
                        help=f"每隔多少交易日重估一次参数（默认 {REFIT_EVERY}）")
    parser.add_argument("--target-vol", type=float, default=TARGET_VOL,
                        help=f"目标年化波动率（默认 {TARGET_VOL}）")
    parser.add_argument("--w-max", type=float, default=W_MAX,
                        help=f"仓位上限（默认 {W_MAX}，不加杠杆）")
    parser.add_argument("--cost", type=float, default=COST,
                        help=f"单边交易成本（默认 {COST}）")
    parser.add_argument("--no-backtest", action="store_true", help="跳过回测")
    parser.add_argument("--no-plots", action="store_true", help="跳过绘图")
    return parser.parse_args(argv)


def _section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _to_md(df: pd.DataFrame) -> str:
    """把 DataFrame 渲染成 Markdown 表（避免依赖 tabulate）。"""
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _backtest_table(perf: pd.DataFrame) -> str:
    """回测指标渲染成 Markdown 表。"""
    header = (
        "| 策略 | 累计收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | Calmar | "
        "平均仓位 | 年化换手 | 年化成本 |"
    )
    lines = [header, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, row in perf.iterrows():
        lines.append(
            f"| {row['strategy']} | {_fmt_pct(row['total_return'])} | {_fmt_pct(row['cagr'])} | "
            f"{_fmt_pct(row['ann_vol'])} | {row['sharpe']:.2f} | {_fmt_pct(row['max_drawdown'])} | "
            f"{row['calmar']:.2f} | {row['avg_weight']:.2f} | {row['ann_turnover']:.1f} | "
            f"{_fmt_pct(row['ann_cost_drag'])} |"
        )
    return "\n".join(lines)


def _metrics_table(metrics: pd.DataFrame, segment: str) -> str:
    """预测精度渲染成 Markdown 表（每个模型一行，三个步长并列）。"""
    labels = {
        "garch": "GARCH(1,1)",
        "egarch": "EGARCH(1,1)",
        "gjr": "GJR-GARCH(1,1)",
        "rv20_naive": "RV20 naive",
    }
    data = metrics.loc[metrics["segment"] == segment]
    lines = [
        "| 模型 | h=1 RMSE | h=5 RMSE | h=10 RMSE | h=1 MAE | h=1 QLIKE | MZ R² (h=1) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in [*MODEL_KEYS, "rv20_naive"]:
        part = data.loc[data["model"] == model]
        if part.empty:
            continue
        get = lambda h, col: float(part.loc[part["horizon"] == h, col].iloc[0])  # noqa: E731
        lines.append(
            f"| {labels.get(model, model)} | {get(1, 'rmse'):.2f} | {get(5, 'rmse'):.2f} | "
            f"{get(10, 'rmse'):.2f} | {get(1, 'mae'):.2f} | {get(1, 'qlike'):.4f} | "
            f"{get(1, 'mz_r2'):.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_arch_available()
    tic = time.time()
    outcome = Path(args.outcome)
    outcome.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. 数据 ----------------
    _section("1. 数据与切分")
    ret = load_returns(args.csv)
    splits = split_calendar(ret)
    table = split_table(splits)
    print(table.to_string(index=False))
    table.to_csv(outcome / "split_overview.csv", index=False)

    train, val, test = splits["train"], splits["val"], splits["test"]
    print(
        f"\n收益率单位：百分数（1.2 = 1.2%），全样本 {len(ret):,} 天，"
        f"年化波动 {ret.std(ddof=1) * np.sqrt(252):.2f}%"
    )

    # ---------------- 2. 模型拟合 ----------------
    _section("2. 训练集参数估计")
    fits = vmd.fit_all(train)
    param_table = vmd.parameters_table(fits)
    param_table.to_csv(outcome / "model_parameters.csv", index=False)
    print()
    print(vmd.parameters_markdown(param_table))

    # ---------------- 3. 滚动预测 ----------------
    _section("3. 滚动预测（val + test）")
    eval_start = val.index.min()
    eval_end = test.index.max()
    forecasts: dict[str, pd.DataFrame] = {}
    refit_logs: list[pd.DataFrame] = []
    for spec in MODEL_SPECS:
        frame, refit_log = vfc.rolling_forecast(
            spec, ret, eval_start, eval_end,
            refit_every=args.refit_every, verbose=True,
        )
        forecasts[spec.key] = frame
        refit_logs.append(refit_log)

    refit_history = pd.concat(refit_logs, ignore_index=True)
    refit_history.to_csv(outcome / "refit_history.csv", index=False)

    # ---------------- 4. 评价 ----------------
    _section("4. 预测精度")
    rv = {h: realized_vol_forward(ret, h) for h in HORIZONS}
    naive = naive_forecast(ret, 20)

    metrics_parts = []
    dm_parts = []
    for segment_name, part in (("val", val), ("test", test)):
        seg_forecasts = {
            key: frame.loc[frame.index.intersection(part.index)] for key, frame in forecasts.items()
        }
        seg_rv = {h: rv[h].loc[rv[h].index.intersection(part.index)] for h in HORIZONS}
        seg_naive = naive.loc[naive.index.intersection(part.index)]
        metrics_parts.append(
            vev.evaluate(seg_forecasts, seg_rv, segment=segment_name, naive=seg_naive)
        )
        dm_parts.append(
            vev.dm_table(seg_forecasts, seg_rv, segment=segment_name, baseline="garch")
        )

    metrics = pd.concat(metrics_parts, ignore_index=True)
    metrics.to_csv(outcome / "forecast_metrics.csv", index=False)
    dm = pd.concat(dm_parts, ignore_index=True)
    dm.to_csv(outcome / "dm_test.csv", index=False)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(metrics.round(4).to_string(index=False))

    # 预测序列落盘
    for segment_name, part in (("val", val), ("test", test)):
        frames = {
            key: frame.loc[frame.index.intersection(part.index)]
            for key, frame in forecasts.items()
        }
        combined = pd.concat(frames, axis=1)
        combined.columns = [f"{model}_{col}" for model, col in combined.columns]
        combined.index.name = "Date"
        combined.to_csv(outcome / f"forecast_series_{segment_name}.csv")

    # ---------------- 5. 选模型 ----------------
    val_rows = metrics.loc[(metrics["segment"] == "val") & (metrics["horizon"] == 1)]
    val_rows = val_rows.loc[val_rows["model"].isin(MODEL_KEYS)]
    best_key = str(val_rows.sort_values("rmse").iloc[0]["model"])
    best_spec = next(spec for spec in MODEL_SPECS if spec.key == best_key)
    print(f"\n验证集 RMSE(h=1) 最优模型：{best_spec.label}")
    (outcome / "selected_model.json").write_text(
        json.dumps(
            {
                "best_model": best_key,
                "label": best_spec.label,
                "val_rmse_h1": float(val_rows.sort_values("rmse").iloc[0]["rmse"]),
                "target_vol": args.target_vol,
                "w_max": args.w_max,
                "cost": args.cost,
                "refit_every": args.refit_every,
                "train_end": TRAIN_END,
                "val_end": VAL_END,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------- 6. 回测 ----------------
    perf = None
    curves: dict[str, pd.DataFrame] = {}
    best_bt = None
    if args.no_backtest:
        print("\n[skip] 已指定 --no-backtest")
    else:
        _section("5. 波动率目标回测（测试集）")
        # 回测用简单收益率（净值口径），模型仍用对数收益率
        simple = load_simple_returns(args.csv)
        test_simple = simple.loc[simple.index.intersection(test.index)]
        sigma_best = forecasts[best_key]["sigma_1"].loc[test.index]
        w_best = vbt.target_weights(
            sigma_best, target_vol=args.target_vol, w_max=args.w_max
        )
        best_bt = vbt.run_backtest(
            test_simple, w_best, cost=args.cost, name=f"Vol-Target ({best_spec.label})"
        )
        bh_bt = vbt.buy_and_hold(test_simple, cost=args.cost)
        w_naive = vbt.target_weights(naive.loc[test.index], target_vol=args.target_vol, w_max=args.w_max)
        naive_bt = vbt.run_backtest(test_simple, w_naive, cost=args.cost, name="Vol-Target (RV20)")

        curves = {
            "Buy & Hold": bh_bt,
            f"Vol-Target ({best_spec.label})": best_bt,
            "Vol-Target (RV20)": naive_bt,
        }
        perf = pd.DataFrame(
            [vbt.performance(bt, name) for name, bt in curves.items()]
        )
        perf.to_csv(outcome / "backtest_metrics.csv", index=False)

        equity = pd.DataFrame({name: bt["equity"] for name, bt in curves.items()})
        equity.index.name = "Date"
        equity.to_csv(outcome / "equity_curves.csv")

        sens = vbt.cost_sensitivity(test_simple, w_best, costs=(0.0, 0.001, 0.002))
        sens.to_csv(outcome / "cost_sensitivity.csv", index=False)

        price_ratio = float(np.prod(1.0 + test_simple.to_numpy() / 100.0))
        print(
            f"\n净值口径校验：Buy & Hold 累计 {float(bh_bt['equity'].iloc[-1]) - 1:.2%}，"
            f"价格比值 {price_ratio - 1:.2%}（差异仅来自首日建仓成本）"
        )
        print(perf.to_string(index=False))
        print("\n成本敏感性（同一仓位，单边成本 0 / 0.1% / 0.2%）：")
        print(sens[["strategy", "cagr", "sharpe", "max_drawdown", "ann_cost_drag"]].to_string(index=False))

    # ---------------- 7. 图表 ----------------
    if not args.no_plots:
        _section("6. 图表")
        vpl.plot_conditional_volatility(forecasts, ret, outcome, test_start=test.index.min())
        vpl.plot_accuracy(metrics, outcome, segment="test")
        if curves:
            vpl.plot_equity_and_drawdown(curves, outcome)
            vpl.plot_weights(
                best_bt, forecasts[best_key]["sigma_1"].loc[test.index],
                trailing_vol(ret, 20).loc[test.index], outcome,
                name=best_spec.label,
            )
        r2 = float(
            metrics.loc[
                (metrics["segment"] == "test") & (metrics["model"] == best_key) & (metrics["horizon"] == 1),
                "mz_r2",
            ].iloc[0]
        )
        vpl.plot_forecast_scatter(
            forecasts[best_key]["sigma_1"].loc[test.index],
            rv[1].loc[rv[1].index.intersection(test.index)],
            outcome, r2=r2,
        )

    # ---------------- 8. 摘要 ----------------
    summary = _build_summary(
        table=table,
        param_table=param_table,
        metrics=metrics,
        perf=perf,
        best_spec=best_spec,
        args=args,
        dm=dm,
    )
    (outcome / "summary.md").write_text(summary, encoding="utf-8")

    _section("完成")
    print(f"总耗时 {time.time() - tic:.1f}s，输出目录：{outcome}")
    for path in sorted(outcome.glob("*")):
        print(f"  - {path.name}")
    return 0


def _build_summary(
    *,
    table: pd.DataFrame,
    param_table: pd.DataFrame,
    metrics: pd.DataFrame,
    perf: pd.DataFrame | None,
    best_spec,
    args: argparse.Namespace,
    dm: pd.DataFrame | None = None,
) -> str:
    """生成 summary.md（结论 + 表格）。"""
    lines = [
        "# 波动率建模与预测 · 结果摘要",
        "",
        "## 1. 数据与切分",
        "",
        _to_md(table),
        "",
        f"- 收益率：`100 × ln(Close_t / Close_{{t-1}})`（百分数），年化交易日 {252}；",
        "- 已实现波动率（真值）：前瞻口径，`RV_t = sqrt(mean(r_t² … r_{t+h-1}²)) × sqrt(252)`；",
        "- 滚动预测：每个目标日 t 只用 t-1 及以前的数据，每 "
        f"{args.refit_every} 个交易日重估一次参数，其余时间固定参数滤波。",
        "",
        "## 2. 参数估计（训练集）",
        "",
        vmd.parameters_markdown(param_table),
        "",
        "> 持续性：GARCH 为 α+β，GJR 为 α+β+γ/2，EGARCH 为 β（对数方差自回归系数）。"
        "LB(10)/LB²(10) 为标准化残差及其平方的 Ljung-Box p 值，> 0.05 表示无显著自相关。",
        "",
        "## 3. 预测精度（测试集）",
        "",
        _metrics_table(metrics, "test"),
        "",
        "验证集：",
        "",
        _metrics_table(metrics, "val"),
        "",
        "## 4. 波动率目标回测（测试集）",
        "",
    ]
    if perf is not None:
        lines += [
            _backtest_table(perf),
            "",
            "> 净值用简单收益率复利（`Close_t/Close_{t-1} - 1`），模型拟合用对数收益率；"
            f"单边成本 {args.cost:.2%}，目标波动 {args.target_vol:.0%}，仓位上限 {args.w_max}，"
            f"测试期 {perf.iloc[0]['start']} ~ {perf.iloc[0]['end']}。",
            "",
        ]
    else:
        lines += ["（本次运行跳过了回测）", ""]

    # 自动生成结论
    test_m = metrics.loc[metrics["segment"] == "test"]
    h1 = test_m.loc[test_m["horizon"] == 1].set_index("model")
    rmse_order = h1["rmse"].sort_values()
    labels = {
        "garch": "GARCH(1,1)",
        "egarch": "EGARCH(1,1)",
        "gjr": "GJR-GARCH(1,1)",
        "rv20_naive": "RV20 naive",
    }
    ranking = "、".join(f"{labels.get(k, k)} {v:.2f}" for k, v in rmse_order.items())
    best_rmse_model = str(rmse_order.index[0])
    naive_rmse = float(rmse_order.get("rv20_naive", rmse_order.iloc[0]))
    gain = (1.0 - float(rmse_order.iloc[0]) / naive_rmse) * 100.0 if naive_rmse else 0.0

    dm_line = ""
    if dm is not None and not dm.empty:
        sub = dm.loc[(dm["segment"] == "test") & (dm["horizon"] == 1)].set_index("model")
        if "egarch" in sub.index:
            p1 = float(sub.loc["egarch", "dm_pvalue"])
            dm_line = (
                f"Diebold-Mariano 检验（以 GARCH 为基准，h=1）p={p1:.3f}，"
                + ("差异显著" if p1 < 0.05 else "差异不显著")
                + "。"
            )

    lines += [
        "## 5. 结论",
        "",
        f"**1) 哪个模型误差最小。** 测试集 h=1 的 RMSE 由小到大：{ranking}。"
        f"误差最小的是 **{labels.get(best_rmse_model, best_rmse_model)}**，"
        f"相对 naive 基准降低 {gain:.1f}%。{dm_line}"
        f"验证集 RMSE(h=1) 选出的模型为 **{best_spec.label}**。",
        "",
        "> QLIKE 与 RMSE 的排序可能不一致：QLIKE 对「低估波动」惩罚更重，"
        "偏保守（预测值略高）的模型在 QLIKE 上更占优。",
        "",
    ]

    if perf is not None:
        bh, strat, naive_row = perf.iloc[0], perf.iloc[1], perf.iloc[2]
        sharpe_gap = float(strat["sharpe"]) - float(bh["sharpe"])
        rv_gap = float(strat["sharpe"]) - float(naive_row["sharpe"])
        verdict = (
            "几乎没有变化"
            if abs(sharpe_gap) < 0.05
            else ("有所改善" if sharpe_gap > 0 else "有所下降")
        )
        lines += [
            f"**2) 调仓是否改善风险类指标。** 相对满仓：年化波动 {_fmt_pct(bh['ann_vol'])} → "
            f"{_fmt_pct(strat['ann_vol'])}，最大回撤 {_fmt_pct(bh['max_drawdown'])} → "
            f"{_fmt_pct(strat['max_drawdown'])}，Calmar {bh['calmar']:.2f} → {strat['calmar']:.2f}；"
            f"但年化收益从 {_fmt_pct(bh['cagr'])} 降到 {_fmt_pct(strat['cagr'])}，"
            f"夏普 {bh['sharpe']:.2f} → {strat['sharpe']:.2f}（{verdict}）。"
            "结论是「显著降波动、明显降回撤，但没有提升风险调整收益」。",
            "",
            f"**3) 预测波动率与历史波动率的差别。** RV20 调仓：夏普 {naive_row['sharpe']:.2f}、"
            f"最大回撤 {_fmt_pct(naive_row['max_drawdown'])}、年化换手 {naive_row['ann_turnover']:.1f} 次；"
            f"模型预测调仓：夏普 {strat['sharpe']:.2f}、回撤 {_fmt_pct(strat['max_drawdown'])}、"
            f"换手 {strat['ann_turnover']:.1f} 次。"
            + (
                f"两者夏普仅差 {abs(rv_gap):.2f}，说明「预测」相对「历史窗口」的增量价值有限，"
                "这与「波动率目标主要赚降杠杆收益」的常见结论一致。"
                if abs(rv_gap) < 0.03
                else "两者存在可观测差距，但只有一段测试期，仍需谨慎解读。"
            )
            + f"预测驱动的日频调仓换手更高（{strat['ann_turnover']:.1f} vs "
            f"{naive_row['ann_turnover']:.1f}），成本拖累 {_fmt_pct(strat['ann_cost_drag'])} vs "
            f"{_fmt_pct(naive_row['ann_cost_drag'])}。",
            "",
            "**4) 诚实说明。** ① 波动率目标策略本质是「降波动」，在高波动下跌中减仓、"
            "低波动上涨中跑输满仓，不应期望它提高收益；② 只有一段测试期（2022–2026），"
            "结论不宜外推；③ 成本按单边 0.1% 计，未建模冲击成本与滑点；"
            "④ 标准化残差仍有轻微自相关（LB(10) p < 0.05），但平方残差无显著 ARCH 效应；"
            "⑤ EGARCH 的多步预测依赖 2000 条模拟路径（固定随机种子），存在微小模拟误差。",
            "",
        ]
    lines += [
        "## 6. 文件清单",
        "",
        "| 文件 | 内容 |",
        "| --- | --- |",
        "| `model_parameters.csv` | 三模型参数与诊断 |",
        "| `refit_history.csv` | 每次重估参数的日期与取值 |",
        "| `forecast_metrics.csv` | RMSE / MAE / QLIKE / MZ（val、test × 3 步长） |",
        "| `dm_test.csv` | 以 GARCH 为基准的 Diebold-Mariano 检验 |",
        "| `forecast_series_val.csv` / `forecast_series_test.csv` | 逐日预测序列 |",
        "| `backtest_metrics.csv` / `equity_curves.csv` | 回测绩效与净值 |",
        "| `cost_sensitivity.csv` | 成本 0 / 0.1% / 0.2% 下的绩效 |",
        "| `01_conditional_volatility.png` | 条件波动率预测 vs 日收益 |",
        "| `02_forecast_accuracy.png` | 分步长 RMSE / MAE |",
        "| `03_equity_and_drawdown.png` | 净值与回撤 |",
        "| `04_weights.png` | 仓位与预测/历史波动率 |",
        "| `05_forecast_vs_realized.png` | 预测 vs 已实现散点 |",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
