"""模块 D 的配图：生成汇报用 slide1–slide3、结果对比与回撤解剖图。

用法：
    python src/3_TradingStrategy_Backtesting/PlotPresentationFigures.py

产物（默认写到 reports/3_TradingStrategy_Backtesting/）：
    slide1_data_and_signal.png     数据与模块 C 信号覆盖
    slide2_position_rules.png      五个策略的仓位规则
    slide3_backtest_protocol.png   回测时点与成本口径
    overall_return_improvement.png 五策略相对买入持有的改善
    drawdown_anatomy_2025.png      2025 年急跌窗口的仓位与净值
    equity_drawdown.png            净值与回撤曲线
"""

from __future__ import annotations

import argparse
from pathlib import Path

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

from BacktestEngine import equity, net_returns
from StrategyRules import (NAMES, ROOT, build_positions, load_market, load_signal,
                           read_signal)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 中文字体无 bold 字重时不刷屏

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "Heiti SC"],
    "axes.unicode_minus": False,
    "axes.edgecolor": "#B9C2CF",
    "axes.labelcolor": "#333333",
    "figure.facecolor": "#FFFFFF",
})

C = {"A": "#7f8c8d", "B": "#5B8FF9", "C": "#F6BD16", "D": "#C0392B", "E": "#1E4FA8"}
SHORT = {"A": "A 买入持有", "B": "B 信号 多/空仓", "C": "C 信号 多/空",
         "D": "D 波动率（上限 1.75）", "E": "E 波动率 + 趋势上限"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ---------------------------------------------------------------- slide 1

def slide1(close, s, mask, out: Path) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.2),
                           gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.35})
    ax[0].plot(close.index, close.values, color="#3B4A5A", lw=0.9)
    ax[0].axvspan(close.index[mask][0], close.index[mask][-1], color="#1E4FA8", alpha=0.12)
    ax[0].annotate("评估区间 2022-05 → 2026-11\n1133 个交易日",
                   xy=(close.index[mask][0], close.max() * 0.55),
                   xytext=(-150, 0), textcoords="offset points", fontsize=10,
                   color="#1E4FA8", va="center",
                   arrowprops=dict(arrowstyle="->", color="#1E4FA8", lw=1))
    ax[0].set_title("标普 500 收盘点位（1996–2026）与模块 D 的评估区间", fontsize=13, pad=10)
    ax[0].set_ylabel("点位")

    # 只画有信号的交易日：2023-05-24 数据缺口、2026-11-13 信号末端按空仓处理，不参与着色
    valid = close.index[mask].isin(read_signal()["signal_date"])
    c, sig = close[mask][valid], s[mask][valid]
    ax[1].plot(c.index, c.values, color="#3B4A5A", lw=1.0, label="收盘点位")
    ax[1].fill_between(c.index, c.min(), c.max(), where=sig.values == 1,
                       color="#5B8FF9", alpha=0.18, label="模型 C 判为上涨（持仓）")
    ax[1].fill_between(c.index, c.min(), c.max(), where=sig.values == 0,
                       color="#C0392B", alpha=0.18, label="判为下跌（空仓/做空）")
    ax[1].set_ylim(c.min() * 0.97, c.max() * 1.02)
    ax[1].legend(loc="upper left", fontsize=9, frameon=False)
    ax[1].set_title(f"评估区间内的信号覆盖：持仓率 {sig.mean():.1%}（{int(sig.sum())} 天持仓），"
                    f"空仓仅 {int((sig == 0).sum())} 天",
                    fontsize=12, pad=8)
    ax[1].set_ylabel("点位")
    _save(fig, out / "slide1_data_and_signal.png")


# ---------------------------------------------------------------- slide 2

def slide2(close, pos, mask, out: Path) -> None:
    idx = close.index[mask]
    fig, axes = plt.subplots(5, 1, figsize=(11, 7.4), sharex=True,
                             gridspec_kw={"hspace": 0.42})
    for ax, k in zip(axes, "ABCDE"):
        p = pos[k][mask]
        ax.fill_between(idx, 0, p, color=C[k], alpha=0.30, step="post")
        ax.step(idx, p, color=C[k], lw=1.1, where="post")
        ax.axhline(1, color="#9AA5B1", lw=0.8, ls=":")
        ax.set_ylim(-1.25 if k == "C" else -0.15, 1.85)
        ax.set_yticks([-1, 0, 1] if k == "C" else [0, 1])
        ax.set_ylabel("仓位", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(SHORT[k], loc="left", fontsize=10, color=C[k],
                     fontweight="bold", pad=4)
    fig.suptitle("五个策略的逐日目标仓位：B/C 来自模块 C 信号，D/E 由波动率与趋势决定",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.90, hspace=0.52)
    _save(fig, out / "slide2_position_rules.png")


# ---------------------------------------------------------------- slide 3

def slide3(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    ax.plot([1.0, 9.0], [3.55, 3.55], color="#C7D0DB", lw=2, zorder=1)
    for x, lab in [(2.8, "t 日收盘：定仓位"), (6.6, "t+1 日收盘：结算收益")]:
        ax.plot([x], [3.55], "o", ms=9, color="#1E4FA8", zorder=3)
        ax.text(x, 3.85, lab, ha="center", fontsize=10.5, color="#1E4FA8")
    ax.annotate("", xy=(6.45, 3.55), xytext=(2.95, 3.55),
                arrowprops=dict(arrowstyle="<->", color="#1E4FA8", lw=1.6))
    ax.text(4.7, 3.18, "持有一天的收益  fwd = close(t+1) / close(t) − 1",
            ha="center", fontsize=10)
    ax.text(2.8, 4.35, "① 只用 t 日及以前的信息定仓位", ha="center", fontsize=9.5, color="#6b7280")
    ax.text(6.6, 4.35, "② 收益落在下一交易日", ha="center", fontsize=9.5, color="#6b7280")

    box = dict(boxstyle="round,pad=0.5", facecolor="#F4F6FA", edgecolor="#CDD6E4")
    ax.text(5.0, 2.05, "净收益 = 仓位 × 次日收益 − |Δ仓位| × 0.1% − max(仓位 − 1, 0) × 3% / 252",
            ha="center", va="center", fontsize=11.5, bbox=box)
    ax.text(3.0, 1.15, "换手成本：每次调仓都收，B/C 每天反手所以最贵",
            ha="center", fontsize=9.5, color="#4b5563")
    ax.text(7.2, 1.15, "融资成本：只有超过 1 倍的部分才付",
            ha="center", fontsize=9.5, color="#4b5563")
    ax.text(5.0, 0.55, "满仓记为 1.0；D 最高 1.75 倍，E 在 200 日线上方 1.4 倍、下方 0.5 倍",
            ha="center", fontsize=9.5, color="#6b7280")
    ax.set_title("回测口径：t 日收盘定仓，赚 t → t+1 的收益，并逐日扣掉换手与融资",
                 fontsize=12.5, pad=10)
    _save(fig, out / "slide3_backtest_protocol.png")


# ---------------------------------------------------------------- 结果对比

def overall(close, fwd, pos, mask, tab, out: Path) -> None:
    names = [NAMES[k] for k in "ABCDE"]
    base = tab.loc[NAMES["A"]]
    metrics = [
        ("年化超额 (pp)", [(tab.loc[n, "年化"] - base["年化"]) * 100 for n in names], "%.1f"),
        ("夏普差", [tab.loc[n, "夏普"] - base["夏普"] for n in names], "%.2f"),
        ("回撤改善 (pp)", [tab.loc[n, "回撤改善(pp)"] for n in names], "%.1f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    for ax, (title, vals, fmt) in zip(axes, metrics):
        colors = [C[k] for k in "ABCDE"]
        ax.barh(range(5), vals, color=colors, alpha=0.85, height=0.62)
        ax.axvline(0, color="#6b7280", lw=1)
        ax.set_yticks(range(5))
        ax.set_yticklabels(["A 基准", "B", "C", "D", "E"], fontsize=10)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11.5)
        ax.tick_params(labelsize=9)
        for i, v in enumerate(vals):
            ax.text(v + (0.12 if v >= 0 else -0.12), i, fmt % v, va="center",
                    ha="left" if v >= 0 else "right", fontsize=9)
        lo, hi = min(vals + [0]), max(vals + [0])
        pad = (hi - lo) * 0.28 + 0.1
        ax.set_xlim(lo - pad, hi + pad)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("相对买入持有的改善：B/C（作业要求项）三项全负或为零，E（扩展项）三项同时为正",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    _save(fig, out / "overall_return_improvement.png")


# ---------------------------------------------------------------- 2025 窗口

def window_2025(close, fwd, pos, out: Path) -> None:
    m = (close.index >= "2025-02-19") & (close.index <= "2025-04-07")
    idx = close.index[m]
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=True,
                           gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.12})
    for k in "ADE":
        _, net = net_returns(pos[k], fwd, m)
        ax[0].plot(idx, equity(net), color=C[k], lw=1.9,
                   label=f"{NAMES[k]}  {np.prod(1 + net) - 1:.2%}，均仓 {pos[k][m].mean():.2f}x")
    ax[0].axhline(1, color="#9AA5B1", lw=0.8, ls=":")
    ax[0].legend(fontsize=9.5, frameon=False, loc="lower left")
    ax[0].set_ylabel("区间净值")
    ax[0].set_title("2025-02-19 → 2025-04-07：一个月跌 18.98%，D 反而放大亏损，E 比满仓少亏",
                    fontsize=12, pad=10)
    for k in "ADE":
        ax[1].step(idx, pos[k][m], color=C[k], lw=1.5, where="post")
    ax[1].axhline(1, color="#9AA5B1", lw=0.8, ls=":")
    ax[1].set_ylabel("仓位")
    ax[1].set_ylim(0.3, 1.75)
    ax[1].set_xlim(idx[0], idx[-1])
    _save(fig, out / "drawdown_anatomy_2025.png")


# ---------------------------------------------------------------- 净值 + 回撤

def equity_drawdown(close, fwd, pos, mask, tab, out: Path) -> None:
    idx = close.index[mask]
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.08})
    for k in "AE":
        _, net = net_returns(pos[k], fwd, mask)
        e = equity(net)
        ax[0].plot(idx, e, color=C[k], lw=1.9 if k == "E" else 1.2,
                   label=f"{NAMES[k]}  {e[-1]:.2f}x / 回撤 {tab.loc[NAMES[k], '最大回撤']:.2%}")
        ax[1].fill_between(idx, (e / np.maximum.accumulate(e) - 1) * 100, 0,
                           color=C[k], alpha=0.35)
    ax[0].set_yscale("log")
    ax[0].set_yticks([1, 1.5, 2], ["1.0x", "1.5x", "2.0x"])
    ax[0].minorticks_off()
    ax[0].set_ylim(0.92, 2.12)
    ax[0].legend(fontsize=9.5, frameon=False, loc="upper left")
    ax[0].set_title("2022–2026：净值（上，对数坐标）与回撤（下）", fontsize=12.5, pad=10)
    ax[1].set_ylabel("回撤 (%)")
    _save(fig, out / "equity_drawdown.png")


# ---------------------------------------------------------------- 分年度

def year_by_year(close, fwd, pos, mask, out: Path) -> None:
    years = sorted({d.year for d in close.index[mask]})
    ret = {}
    for k in "AE":
        _, net = net_returns(pos[k], fwd, mask)
        ser = pd.Series(net, index=close.index[mask])
        ret[k] = ser.groupby(ser.index.year).apply(lambda x: np.prod(1 + x) - 1)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(years))
    ax.bar(x - 0.19, [ret["A"].get(y, np.nan) * 100 for y in years], 0.36,
           color=C["A"], alpha=0.85, label="A 买入持有")
    ax.bar(x + 0.19, [ret["E"].get(y, np.nan) * 100 for y in years], 0.36,
           color=C["E"], alpha=0.9, label="E 波动率+趋势")
    for i, y in enumerate(years):
        a, e = ret["A"].get(y, np.nan) * 100, ret["E"].get(y, np.nan) * 100
        ax.text(i, max(a, e) + 1.2, f"{e - a:+.1f}pp", ha="center", fontsize=9,
                color="#16A34A" if e - a > 0 else "#B91C1C")
    ax.axhline(0, color="#6b7280", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("年度收益 (%)")
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("分年度收益：超额几乎全部来自 2024 年，其余年份小幅落后",
                 fontsize=12, pad=10)
    _save(fig, out / "year_by_year_excess.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(ROOT / "reports/3_TradingStrategy_Backtesting"))
    out = Path(ap.parse_args().output_dir)
    close, fwd, lr = load_market()
    s, mask = load_signal(close)
    pos = build_positions(close, lr, s)
    from BacktestEngine import perf_table
    tab = perf_table(pos, fwd, mask)

    print("生成配图：")
    slide1(close, s, mask, out)
    slide2(close, pos, mask, out)
    slide3(out)
    overall(close, fwd, pos, mask, tab, out)
    window_2025(close, fwd, pos, out)
    equity_drawdown(close, fwd, pos, mask, tab, out)
    year_by_year(close, fwd, pos, mask, out)


if __name__ == "__main__":
    main()
