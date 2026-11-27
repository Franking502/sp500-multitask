"""Create the three explanatory figures used in the time-series presentation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import pandas as pd

from ForecastModels import FEATURES, LOOKBACK, LSTM_SETTINGS


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/1_TimeSerialAnalyze"
BLUE, ORANGE, GREEN = "#376a9f", "#bf762b", "#087f75"
INK, GRAY = "#26384b", "#dce4ec"
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 13,
    "figure.facecolor": "white",
})


def save(fig, filename):
    fig.savefig(OUTPUT / filename, dpi=180, facecolor="white")
    plt.close(fig)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)


def plot_data(clean, features):
    fig = plt.figure(figsize=(16, 9), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=(1.25, 1))
    history = fig.add_subplot(grid[0])
    recent_price = fig.add_subplot(grid[1])
    fig.suptitle("数据准备｜长期走势与近期趋势", fontsize=25, weight="bold")

    phase_colors = {"train": "#eaf0f8", "val": "#fcf1e3", "test": "#e4f2ed"}
    phase_names = {"train": "训练集", "val": "验证集", "test": "测试集"}
    for phase in ("train", "val", "test"):
        part = clean[clean.segment == phase]
        history.axvspan(part.Date.iloc[0], part.Date.iloc[-1],
                        color=phase_colors[phase], zorder=0)
    history.plot(clean.Date, clean.Close, color=BLUE, linewidth=1.35, label="收盘点位")
    history.set(title=f"清洗后的日线 · {clean.Date.iloc[0]:%Y-%m-%d} 至 {clean.Date.iloc[-1]:%Y-%m-%d}",
                ylabel="收盘点位（指数点）")
    history.xaxis.set_major_locator(mdates.YearLocator(5))
    history.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Line2D([], [], color=BLUE, label="收盘点位")]
    handles += [Patch(facecolor=phase_colors[p], label=phase_names[p]) for p in phase_names]
    history.legend(handles=handles, loc="upper left", ncols=4, frameon=False)

    recent = features.tail(126)
    for column, label, color in (
        ("Close", "收盘点位", BLUE), ("ma_5", "5 日均线", ORANGE), ("ma_20", "20 日均线", GREEN),
    ):
        recent_price.plot(recent.Date, recent[column], label=label, color=color, linewidth=1.6)
    recent_price.set(title="最近 126 条记录 · 趋势信息", ylabel="指数点")
    recent_price.legend(loc="upper left", ncols=3, fontsize=13, frameon=False)
    recent_price.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    recent_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for ax in (history, recent_price):
        style_axis(ax)
    fig.supxlabel("均线用于展示趋势；实际入模的是收盘点位相对均线的偏离度。", fontsize=13)
    save(fig, "slide1_data_preparation.png")


def box(ax, x, y, width, height, text, color=BLUE, fontsize=15):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=color, edgecolor="none", alpha=0.10,
    ))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            color=INK, fontsize=fontsize, linespacing=1.6)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                color="#6c7f91", linewidth=1.35))


def plot_methods():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.025, right=0.98, top=0.90, bottom=0.05)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.suptitle("方法｜在同一截止日直接预测未来 H 个交易日", fontsize=25, weight="bold", y=0.97)
    ax.text(0.5, 0.94, "H = 5 或 20，所有预测均以截止日收盘点位为起点",
            ha="center", fontsize=16, color="#596c7d")

    ax.text(0.015, 0.835, "Naive\n朴素预测", fontsize=19, va="center", color=INK)
    box(ax, 0.18, 0.79, 0.18, 0.09, "截止日收盘点位 P(t)")
    box(ax, 0.47, 0.79, 0.18, 0.09, "沿用同一个点位")
    box(ax, 0.76, 0.79, 0.21, 0.09, "未来 H 天均预测为 P(t)")
    arrow(ax, (0.37, 0.835), (0.46, 0.835))
    arrow(ax, (0.66, 0.835), (0.75, 0.835))

    ax.text(0.015, 0.53, "XGBoost\n独立步长", fontsize=19, va="center", color=ORANGE)
    box(ax, 0.18, 0.465, 0.18, 0.13, f"截止日的 {len(FEATURES)} 个特征", ORANGE)
    for y, label in ((0.635, "第 1 步回归模型"), (0.565, "第 2 步回归模型"),
                     (0.495, "..."), (0.425, "第 H 步回归模型")):
        box(ax, 0.47, y, 0.18, 0.05, label, ORANGE, fontsize=14)
        arrow(ax, (0.37, 0.53), (0.46, y + 0.025))
        arrow(ax, (0.66, y + 0.025), (0.75, 0.53))
    box(ax, 0.76, 0.465, 0.21, 0.13, "分别预测 H 个点位变化\n再加回 P(t)", ORANGE)
    ax.text(0.56, 0.395, "每个回归模型含 100 棵树", ha="center", fontsize=12, color="#596c7d")

    ax.text(0.015, 0.25, "LSTM\n序列输入", fontsize=19, va="center", color=GREEN)
    box(ax, 0.18, 0.19, 0.18, 0.12, f"最近 {LOOKBACK} 条记录\n{LOOKBACK} 天 × {len(FEATURES)} 个特征", GREEN)
    box(ax, 0.43, 0.19, 0.13, 0.12, f"单层 LSTM\n{LSTM_SETTINGS['hidden_size']} 个隐藏单元", GREEN)
    box(ax, 0.61, 0.19, 0.13, 0.12, "线性输出层\n一次输出 H 个值", GREEN, fontsize=14)
    box(ax, 0.79, 0.19, 0.18, 0.12, "未来 H 天点位变化\n再加回 P(t)", GREEN)
    for start, end in ((0.37, 0.42), (0.57, 0.60), (0.75, 0.78)):
        arrow(ax, (start, 0.25), (end, 0.25))
    for y in (0.73, 0.36):
        ax.axhline(y, xmin=0.015, xmax=0.98, color=GRAY, linewidth=1)
    ax.text(0.5, 0.075, "训练标签 = 目标日点位 − 截止日点位    ｜    预测点位 = 截止日点位 + 预测变化",
            ha="center", fontsize=15, color=INK)
    ax.text(0.5, 0.025, "XGBoost 与 LSTM 都以绝对误差为训练目标；每个未来步长直接使用截止日已知的输入。",
            ha="center", fontsize=13, color="#596c7d")
    save(fig, "slide2_direct_forecasting.png")


def plot_protocol():
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.035, right=0.975, top=0.76, bottom=0.18, wspace=0.12)
    fig.suptitle("评估设计｜5 日固定训练，20 日逐窗口重训", fontsize=25, weight="bold", y=0.96)
    fig.text(0.5, 0.88, "每次仅用截止日及以前的信息，一次预测整个窗口",
             ha="center", fontsize=16, color="#596c7d")
    for ax, horizon in zip(axes, (5, 20)):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        title = "5 日：模型参数固定" if horizon == 5 else "20 日：每次重新训练"
        ax.set_title(title, fontsize=21, pad=25, color=INK)
        for number, label in enumerate(("本次", "下次")):
            y = 0.72 - number * 0.34
            cutoff = 0.49 + number * 0.20
            train_end = 0.49 if horizon == 5 else cutoff
            ax.text(0.01, y, label, va="center", fontsize=15, color=INK)
            ax.barh(y, train_end - 0.12, left=0.12, height=0.13, color=BLUE, alpha=0.85)
            if horizon == 5 and number:
                ax.barh(y, cutoff - train_end, left=train_end, height=0.13, color=GRAY)
            ax.barh(y, 0.20, left=cutoff, height=0.13, color=GREEN, alpha=0.85)
            ax.plot([cutoff, cutoff], [y - 0.09, y + 0.09], color=ORANGE, linewidth=2.5)
            ax.text(cutoff + 0.10, y, f"{horizon} 天", ha="center", va="center",
                    fontsize=15, color="white")
        ax.annotate("时间推进", xy=(0.93, 0.10), xytext=(0.12, 0.10), va="center",
                    arrowprops=dict(arrowstyle="->", color="#6c7f91"), color="#596c7d", fontsize=14)
    handles = [
        Patch(facecolor=BLUE, label="训练历史"),
        Patch(facecolor=GRAY, label="仅更新预测输入"),
        Patch(facecolor=GREEN, label="预测窗口"),
        Line2D([], [], color=ORANGE, linewidth=2, label="截止日"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.085),
               ncols=4, frameon=False, fontsize=14)
    fig.text(0.5, 0.04, "时间轴为示意；LSTM 标准化只使用当次训练历史的均值和标准差。",
             ha="center", fontsize=13, color="#596c7d")
    save(fig, "slide3_evaluation_protocol.png")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "data/processed"
    clean = pd.read_csv(folder / "sp500_clean.csv", parse_dates=["Date"])
    features = pd.read_csv(folder / "sp500_features.csv", parse_dates=["Date"])
    plot_data(clean, features)
    plot_methods()
    plot_protocol()


if __name__ == "__main__":
    main()
