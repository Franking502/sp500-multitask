# -*- coding: utf-8 -*-
"""绘图层：5 张英文标注图，输出到 ``Outcome/``。

图 1 条件波动率预测（1 步）与日收益绝对值
图 2 分步长的 RMSE / MAE 对比（含 naive 基准）
图 3 三条净值曲线与回撤
图 4 仓位时序与预测/已实现波动率
图 5 预测 vs 已实现散点（h=1，测试集）
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullLocator

from vol_config import AMBER, CRIMSON, GREY, GRID, MODEL_COLORS, NAVY, TEAL

DPI = 150
LABEL_COLORS = {"garch": "GARCH(1,1)", "egarch": "EGARCH(1,1)", "gjr": "GJR-GARCH(1,1)"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#4b5563",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.autolayout": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] {path.name}")
    return path


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7, color=GREY)


def _year_axis(ax: plt.Axes, start: pd.Timestamp, end: pd.Timestamp, years: int = 1) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(years))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(start, end)


def plot_conditional_volatility(
    forecasts: dict[str, pd.DataFrame],
    ret: pd.Series,
    outcome_dir: Path,
    *,
    test_start: pd.Timestamp,
) -> Path:
    """图 1：三个模型的 1 步条件波动率 + 当日 |收益|。"""
    _style()
    frame = pd.DataFrame({name: f["sigma_1"] for name, f in forecasts.items()}).dropna()
    start, end = frame.index.min(), frame.index.max()

    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    ax.scatter(
        ret.index, ret.abs().to_numpy(),
        s=3, color=GREY, alpha=0.18, linewidths=0, label="|daily return|",
    )
    for key in frame.columns:
        ax.plot(
            frame.index, frame[key].to_numpy(),
            color=MODEL_COLORS.get(key, NAVY), linewidth=1.1,
            label=LABEL_COLORS.get(key, key),
        )
    ax.axvline(test_start, color=CRIMSON, linestyle="--", linewidth=0.9, alpha=0.8)
    ax.annotate(
        "test start", xy=(test_start, 1.0), xycoords=("data", "axes fraction"),
        xytext=(3, -6), textcoords="offset points", ha="left", va="top",
        fontsize=7.5, color=CRIMSON,
    )
    ax.set_title("Conditional Volatility Forecasts (1-Day Ahead, Annualized)")
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_xlabel("Date")
    ymax = float(np.nanmax(frame.to_numpy())) * 1.06
    ax.set_ylim(0, min(160.0, ymax))
    ax.legend(loc="upper right", ncol=2)
    _year_axis(ax, start, end)
    _caption(fig, f"Evaluation sample: {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    return _save(fig, outcome_dir / "01_conditional_volatility.png")


def plot_accuracy(metrics: pd.DataFrame, outcome_dir: Path, *, segment: str = "test") -> Path:
    """图 2：分步长的 RMSE / MAE 分组柱状图。"""
    _style()
    data = metrics.loc[metrics["segment"] == segment].copy()
    horizons = sorted(data["horizon"].unique())
    models = [k for k in ("garch", "egarch", "gjr") if k in set(data["model"])]
    if "rv20_naive" in set(data["model"]):
        models.append("rv20_naive")
    labels = [LABEL_COLORS.get(m, "RV20 naive") for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    width = 0.8 / max(len(models), 1)
    for ax, metric, title in zip(axes, ("rmse", "mae"), ("RMSE", "MAE")):
        for i, model in enumerate(models):
            values = [
                float(data.loc[(data["model"] == model) & (data["horizon"] == h), metric].iloc[0])
                if not data.loc[(data["model"] == model) & (data["horizon"] == h)].empty
                else np.nan
                for h in horizons
            ]
            positions = np.arange(len(horizons)) + (i - (len(models) - 1) / 2) * width
            ax.bar(
                positions, values, width=width * 0.92,
                color=MODEL_COLORS.get(model, "#9ca3af"),
                alpha=0.9 if model != "rv20_naive" else 0.55,
                label=labels[i],
            )
        ax.set_xticks(np.arange(len(horizons)), [f"h={h}" for h in horizons])
        ax.set_ylabel(f"{title} (annualized vol, %)")
        ax.set_title(f"{title} by Forecast Horizon ({segment})")
        ax.grid(True, axis="y")
    axes[0].legend(loc="upper right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _caption(fig, "Lower is better; RV20 naive uses trailing 20-day realized volatility")
    return _save(fig, outcome_dir / "02_forecast_accuracy.png")


def plot_equity_and_drawdown(curves: dict[str, pd.DataFrame], outcome_dir: Path) -> Path:
    """图 3：净值曲线（对数轴）+ 回撤。"""
    _style()
    palette = [GREY, CRIMSON, TEAL, NAVY, AMBER]
    color_by_name = {name: palette[i % len(palette)] for i, name in enumerate(curves)}
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.0, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.08},
    )
    for name, bt in curves.items():
        equity = bt["equity"]
        color = color_by_name[name]
        ax1.plot(equity.index, equity.to_numpy(), linewidth=1.2, color=color, label=name)
        dd = (equity / equity.cummax() - 1.0) * 100.0
        ax2.plot(dd.index, dd.to_numpy(), linewidth=0.9, color=color, alpha=0.9)

    ax1.set_yscale("log")
    ax1.set_yticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}x"))
    ax1.set_ylim(0.74, 1.62)
    ax1.set_ylabel("Equity (log scale)")
    ax1.set_title("Volatility Targeting vs Buy & Hold (out-of-sample)")
    ax1.legend(loc="upper left")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.axhline(0, color=GREY, linewidth=0.8)
    _year_axis(ax2, next(iter(curves.values())).index.min(), next(iter(curves.values())).index.max())
    _caption(fig, "Net of trading costs (one-way 0.1%)")
    return _save(fig, outcome_dir / "03_equity_and_drawdown.png")


def plot_weights(
    backtest: pd.DataFrame,
    sigma_forecast: pd.Series,
    trailing_vol: pd.Series,
    outcome_dir: Path,
    *,
    name: str,
) -> Path:
    """图 4：仓位时序 + 预测波动率 vs 历史波动率。"""
    _style()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.0, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.6], "hspace": 0.1},
    )
    w = backtest["weight"]
    ax1.fill_between(w.index, 0, w.to_numpy(), color=NAVY, alpha=0.18, linewidth=0)
    ax1.plot(w.index, w.to_numpy(), color=NAVY, linewidth=0.9)
    ax1.axhline(1.0, color=GREY, linestyle="--", linewidth=0.8)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Target weight")
    ax1.set_title(f"Volatility-Target Position ({name})")

    ax2.plot(sigma_forecast.index, sigma_forecast.to_numpy(), color=CRIMSON,
             linewidth=1.0, label="Forecast (1-day ahead)")
    ax2.plot(trailing_vol.index, trailing_vol.to_numpy(), color=TEAL,
             linewidth=0.9, alpha=0.85, label="Trailing 20-day realized")
    ax2.set_ylabel("Annualized volatility (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, float(np.nanpercentile(sigma_forecast.to_numpy(), 99)) * 1.35)
    ax2.legend(loc="upper left")
    _year_axis(ax2, w.index.min(), w.index.max())
    _caption(fig, "Weight is applied to the same-day return; forecast uses data up to t-1")
    return _save(fig, outcome_dir / "04_weights.png")


def plot_forecast_scatter(
    sigma: pd.Series, realized: pd.Series, outcome_dir: Path, *, r2: float
) -> Path:
    """图 5：h=1 预测 vs 已实现（对数坐标 + 45° 线）。"""
    _style()
    frame = pd.concat([sigma.rename("f"), realized.rename("r")], axis=1, join="inner").dropna()
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.scatter(frame["f"], frame["r"], s=8, color=NAVY, alpha=0.35, linewidths=0)
    ax.plot([2, 120], [2, 120], color=CRIMSON, linestyle="--", linewidth=1.0, label="45° line")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2, 120)
    ax.set_ylim(2, 120)
    ax.set_xlabel("Forecast volatility, h=1 (annualized %)")
    ax.set_ylabel("Realized volatility |r| (annualized %)")
    ax.set_title("Forecast vs Realized Volatility")
    ax.text(
        0.03, 0.96, f"n = {len(frame):,}\nR² (MZ) = {r2:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=GRID, alpha=0.95),
    )
    ax.legend(loc="lower right")
    _caption(fig, "Points above the 45° line were under-forecast")
    return _save(fig, outcome_dir / "05_forecast_vs_realized.png")
