# -*- coding: utf-8 -*-
"""模块 0：数据加载、清洗、交易日对齐、收益率计算、时间序列划分与 EDA。

职责
----
1. 读取 ``data/SP500_daily.csv``，处理缺失值、重复日期、异常 OHLC、零成交量行；
2. 对齐交易日：以工作日日历为基准，区分「节假日休市」与「数据断档」，
   只对真正的断档做前向填充并打上 ``is_filled`` 标记；
3. 计算对数收益率、简单收益率、累计对数收益、回撤等派生列；
4. 按时间顺序划分训练 / 验证 / 测试集（严禁随机打乱）；
5. 输出 5 张英文标注的 EDA 图到 ``src/0_DataPreprocessing/Outcome``。

对外接口
--------
>>> from DataPreprocessing import DataPreprocessor
>>> prep = DataPreprocessor()                 # 默认读取 data/SP500_daily.csv
>>> df = prep.prepare()                       # 清洗后的 DataFrame（含 log_ret 等）
>>> splits = prep.split()                     # {"train": df, "val": df, "test": df}
>>> prep.split_table()                        # 各集合的起止日期与样本量
>>> prep.report.to_frame()                    # 清洗质量报告
>>> prep.export_processed()                   # data/processed/sp500_clean.csv

命令行
------
python DataPreprocessing.py                   # 清洗 + 划分 + EDA + 落盘
python DataPreprocessing.py --no-eda          # 只做清洗与划分
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullLocator

TRADING_DAYS_PER_YEAR = 252

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "SP500_daily.csv"
DEFAULT_OUTCOME = PACKAGE_DIR / "Outcome"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"

PRICE_COLS = ["Open", "High", "Low", "Close"]
BASE_COLS = PRICE_COLS + ["Volume"]
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEGMENT_ORDER = ("train", "val", "test")

NAVY = "#1f3b57"
TEAL = "#2a9d8f"
CRIMSON = "#d1495b"
AMBER = "#e9c46a"
GREY = "#6b7280"
GRID = "#d7dde5"

__all__ = [
    "DataPreprocessor",
    "EDAReporter",
    "CleaningReport",
    "DEFAULT_CSV",
    "DEFAULT_OUTCOME",
    "SPLIT_RATIOS",
]


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _apply_style() -> None:
    """统一的图表风格；字体固定为 DejaVu Sans，保证图中不出现中文。"""
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


def _consecutive_runs(index: pd.DatetimeIndex, missing: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """把 ``missing`` 中在 ``index`` 上位置连续的元素合并成闭区间 [(start, end), ...]。"""
    if len(missing) == 0:
        return []
    pos = np.sort(np.asarray(index.get_indexer(missing), dtype=int))
    runs: list[tuple[int, int]] = []
    start = prev = int(pos[0])
    for raw in pos[1:]:
        cur = int(raw)
        if cur == prev + 1:
            prev = cur
        else:
            runs.append((start, prev))
            start = prev = cur
    runs.append((start, prev))
    return runs


def _splits_to_frame(splits: dict[str, pd.DataFrame] | None) -> pd.DataFrame | None:
    """把切分结果汇总成「集合 / 起止日期 / 样本量 / 占比」表。"""
    if not splits:
        return None
    total = sum(len(part) for part in splits.values())
    rows = []
    for name in SEGMENT_ORDER:
        if name not in splits:
            continue
        part = splits[name]
        rows.append(
            {
                "segment": name,
                "start": part.index.min().date(),
                "end": part.index.max().date(),
                "n_days": len(part),
                "share": round(len(part) / total, 4),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 清洗报告
# --------------------------------------------------------------------------- #
@dataclass
class CleaningReport:
    """记录每一步清洗动作，便于在报告里说明「数据完整性」。"""

    rows_raw: int = 0
    rows_clean: int = 0
    duplicates_removed: int = 0
    invalid_rows_removed: int = 0
    high_low_clamped: int = 0
    zero_volume_removed: int = 0
    off_calendar_rows: int = 0
    holiday_rows: int = 0
    filled_rows: int = 0
    rows_dropped_nan: int = 0
    first_date: str = ""
    last_date: str = ""
    missing_ratio_raw: float = 0.0
    missing_ratio_clean: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        items = [
            ("raw rows", self.rows_raw),
            ("clean rows", self.rows_clean),
            ("duplicate dates removed", self.duplicates_removed),
            ("invalid OHLC rows removed", self.invalid_rows_removed),
            ("high/low clamped", self.high_low_clamped),
            ("zero-volume rows removed", self.zero_volume_removed),
            ("off-calendar (weekend) rows dropped", self.off_calendar_rows),
            ("holiday/weekday gaps kept as non-trading", self.holiday_rows),
            ("data-dropout rows forward filled", self.filled_rows),
            ("rows dropped for remaining NaN", self.rows_dropped_nan),
            ("first date", self.first_date),
            ("last date", self.last_date),
            ("missing ratio (raw)", round(self.missing_ratio_raw, 6)),
            ("missing ratio (clean)", round(self.missing_ratio_clean, 6)),
        ]
        return pd.DataFrame(items, columns=["item", "value"])

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# 主类：数据预处理
# --------------------------------------------------------------------------- #
class DataPreprocessor:
    """S&P 500 日频数据的加载、清洗、交易日对齐、收益率计算与时间序列划分。

    Parameters
    ----------
    csv_path:
        原始 CSV 路径，需包含 ``Date, Open, High, Low, Close, Volume`` 六列。
    outcome_dir:
        EDA 图片输出目录，默认 ``src/0_DataPreprocessing/Outcome``。
    max_holiday_gap:
        连续缺失的工作日数量不超过该阈值时视为节假日休市（不填充）；
        超过则视为数据断档，做前向填充。
    verbose:
        是否打印清洗过程。
    """

    def __init__(
        self,
        csv_path: str | Path = DEFAULT_CSV,
        outcome_dir: str | Path = DEFAULT_OUTCOME,
        *,
        max_holiday_gap: int = 4,
        verbose: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.outcome_dir = Path(outcome_dir)
        self.max_holiday_gap = int(max_holiday_gap)
        self.verbose = verbose

        self.raw: pd.DataFrame | None = None
        self.data: pd.DataFrame | None = None
        self.report: CleaningReport | None = None
        self.calendar: pd.DatetimeIndex | None = None
        self._splits: dict[str, pd.DataFrame] | None = None

    # ------------------------------ 日志 -------------------------------- #
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[DataPreprocessor] {message}")

    # ---------------------------- 数据加载 ------------------------------ #
    def load_raw(self) -> pd.DataFrame:
        """读取 CSV，解析日期并排序；不做任何删除，保留原始形态以便报告缺失率。"""
        if self.raw is not None:
            return self.raw
        if not self.csv_path.exists():
            raise FileNotFoundError(f"找不到数据文件: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        df.columns = [str(col).strip() for col in df.columns]
        missing_cols = [col for col in BASE_COLS + ["Date"] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}；实际列: {list(df.columns)}")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        bad_date = df["Date"].isna()
        if bad_date.any():
            self._log(f"丢弃 {int(bad_date.sum())} 行无法解析的日期")
            df = df.loc[~bad_date]

        df = df.set_index("Date").sort_index()
        df.index.name = "Date"
        self.raw = df
        self._log(f"读取 {len(df)} 行原始数据: {df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}")
        return df

    # ---------------------------- 交易日对齐 ---------------------------- #
    def _align_calendar(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """以工作日日历为基准对齐；短缺口视为节假日，长缺口视为断档并前向填充。"""
        calendar = pd.bdate_range(df.index.min(), df.index.max(), freq="B")
        off_calendar = df.index.difference(calendar)
        report.off_calendar_rows = int(len(off_calendar))
        if len(off_calendar):
            df = df.loc[df.index.intersection(calendar)]

        self.calendar = calendar
        missing = calendar.difference(df.index)
        runs = _consecutive_runs(calendar, missing)

        full = df.reindex(calendar)
        full["is_filled"] = False

        fill_positions: list[int] = []
        for start, end in runs:
            span = end - start + 1
            if span <= self.max_holiday_gap:
                report.holiday_rows += span
            else:
                fill_positions.extend(range(start, end + 1))

        if fill_positions:
            idx = calendar[fill_positions]
            for col in BASE_COLS:
                full.loc[idx, col] = full[col].ffill().loc[idx].to_numpy()
            full.loc[idx, "is_filled"] = True
            report.filled_rows = len(idx)
            self._log(
                f"检测到 {len(fill_positions)} 个交易日的数据断档，已前向填充并标记 is_filled"
            )

        full = full.dropna(subset=["Close"])
        full["is_filled"] = full["is_filled"].fillna(False).astype(bool)
        return full

    # ------------------------------ 清洗 -------------------------------- #
    def clean(self) -> pd.DataFrame:
        """完整清洗流程，返回含派生列的清洗后数据。"""
        raw = self.load_raw()
        report = CleaningReport(
            rows_raw=len(raw),
            missing_ratio_raw=float(raw[BASE_COLS].isna().to_numpy().mean()),
        )

        df = raw.copy()

        duplicated = df.index.duplicated(keep="first")
        report.duplicates_removed = int(duplicated.sum())
        df = df.loc[~duplicated]

        for col in BASE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        bad_price = (
            df[PRICE_COLS].isna().any(axis=1)
            | (df[PRICE_COLS] <= 0).any(axis=1)
            | (df["High"] < df["Low"])
        )
        report.invalid_rows_removed = int(bad_price.sum())
        df = df.loc[~bad_price]

        # 保证 High / Low 包住 Open 与 Close，避免异常刻度
        high = df[PRICE_COLS].max(axis=1)
        low = df[PRICE_COLS].min(axis=1)
        report.high_low_clamped = int(((df["High"] < high) | (df["Low"] > low)).sum())
        df["High"] = high
        df["Low"] = low

        df["Volume"] = df["Volume"].fillna(0).clip(lower=0).round()

        zero_volume = df["Volume"] <= 0
        report.zero_volume_removed = int(zero_volume.sum())
        if report.zero_volume_removed:
            self._log(f"移除 {report.zero_volume_removed} 行零成交量记录")
        df = df.loc[~zero_volume]

        df = self._align_calendar(df, report)

        residual_nan = df[BASE_COLS].isna().any(axis=1)
        report.rows_dropped_nan = int(residual_nan.sum())
        df = df.loc[~residual_nan]

        df["Volume"] = df["Volume"].astype("int64")
        df.index.name = "Date"
        df = df.sort_index()

        report.rows_clean = len(df)
        report.first_date = f"{df.index.min():%Y-%m-%d}"
        report.last_date = f"{df.index.max():%Y-%m-%d}"
        report.missing_ratio_clean = float(df[BASE_COLS].isna().to_numpy().mean())

        self.report = report
        self.data = self.compute_returns(df)
        self._splits = None
        self._log(
            f"清洗完成: {report.rows_clean} 行 ({report.first_date} ~ {report.last_date})，"
            f"缺失率 {report.missing_ratio_clean:.4%}"
        )
        return self.data

    # ---------------------------- 派生指标 ------------------------------ #
    @staticmethod
    def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
        """计算收益率与回撤等派生列（不修改入参）。"""
        out = df.copy()
        close = out["Close"].astype(float)

        out["ret"] = close / close.shift(1) - 1.0
        out["log_ret"] = np.log(close).diff()
        out["cum_log_ret"] = out["log_ret"].fillna(0.0).cumsum()
        out["dollar_volume"] = close * out["Volume"].astype(float)
        out["range_pct"] = (out["High"] - out["Low"]) / close
        out["gap_pct"] = out["Open"] / close.shift(1) - 1.0

        running_max = close.cummax()
        out["running_max"] = running_max
        out["drawdown"] = close / running_max - 1.0
        return out

    @staticmethod
    def rolling_volatility(
        returns: pd.Series, window: int = 21, *, annualize: bool = True
    ) -> pd.Series:
        """滚动波动率：收益率标准差，默认年化（×√252）。"""
        vol = returns.rolling(window).std(ddof=1)
        if annualize:
            vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        vol.name = f"vol_{window}" + ("_ann" if annualize else "")
        return vol

    @staticmethod
    def drawdown_series(close: pd.Series) -> pd.Series:
        """相对历史最高点的回撤序列（负数）。"""
        dd = close / close.cummax() - 1.0
        dd.name = "drawdown"
        return dd

    def prepare(self) -> pd.DataFrame:
        """load → clean，返回可直接使用的 DataFrame。"""
        return self.clean()

    # ---------------------------- 数据集划分 ---------------------------- #
    def split(self, ratios: Sequence[float] = SPLIT_RATIOS) -> dict[str, pd.DataFrame]:
        """按时间顺序切分为训练 / 验证 / 测试集（不洗牌、不回看未来）。"""
        if self.data is None:
            raise RuntimeError("请先调用 prepare() 或 clean() 再划分数据集")

        ratios_arr = np.asarray(ratios, dtype=float)
        if ratios_arr.shape != (3,):
            raise ValueError("ratios 需为三个数: (train, val, test)")
        if np.any(ratios_arr <= 0) or abs(ratios_arr.sum() - 1.0) > 1e-6:
            raise ValueError(f"ratios 需为正且和为 1，当前: {tuple(ratios_arr)}")

        df = self.data
        n = len(df)
        if n < 30:
            raise ValueError(f"样本量过小（{n} 行），无法划分训练/验证/测试集")

        n_train = int(round(n * ratios_arr[0]))
        n_val = int(round(n * ratios_arr[1]))
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))

        self._splits = {
            "train": df.iloc[:n_train].copy(),
            "val": df.iloc[n_train : n_train + n_val].copy(),
            "test": df.iloc[n_train + n_val :].copy(),
        }
        self._check_split_integrity()
        for name, part in self._splits.items():
            self._log(f"{name:5s}: {len(part):5d} 行  {part.index.min():%Y-%m-%d} ~ {part.index.max():%Y-%m-%d}")
        return dict(self._splits)

    def _check_split_integrity(self) -> None:
        """校验：三段首尾相接、无重叠、严格按时间递增。"""
        train, val, test = (self._splits[name] for name in SEGMENT_ORDER)  # type: ignore[index]
        assert train.index.max() < val.index.min(), "训练集与验证集存在重叠"
        assert val.index.max() < test.index.min(), "验证集与测试集存在重叠"
        assert len(train) + len(val) + len(test) == len(self.data)  # type: ignore[arg-type]

    def split_table(self) -> pd.DataFrame:
        """划分结果概览表。"""
        if self._splits is None:
            self.split()
        return _splits_to_frame(self._splits)  # type: ignore[return-value]

    @property
    def splits(self) -> dict[str, pd.DataFrame]:
        if self._splits is None:
            self.split()
        return dict(self._splits)  # type: ignore[arg-type]

    # ------------------------------ 导出 -------------------------------- #
    def export_processed(
        self,
        path: str | Path = DEFAULT_PROCESSED / "sp500_clean.csv",
        *,
        with_segment: bool = True,
    ) -> Path:
        """导出清洗后数据（可选写入 segment 列，方便其他成员直接取用）。"""
        if self.data is None:
            raise RuntimeError("请先调用 prepare()")

        out = self.data.copy()
        if with_segment:
            segment = pd.Series("", index=out.index, dtype=object)
            for name, part in self.splits.items():
                segment.loc[part.index] = name
            out["segment"] = segment

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index_label="Date", float_format="%.10g")
        self._log(f"清洗后数据已导出: {path}")
        return path


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
class EDAReporter:
    """生成 5 张 EDA 图（全部英文标注）并汇总关键统计指标。

    图 1 ``01_price_volume.png``        价格与成交量时序图
    图 2 ``02_return_distribution.png`` 日对数收益率分布 + 正态拟合 + 5% VaR
    图 3 ``03_rolling_volatility.png``  21/63 日滚动年化波动率
    图 4 ``04_annual_return_heatmap.png`` 年 × 月收益率热力图（附年度收益）
    图 5 ``05_max_drawdown.png``        最大回撤曲线
    """

    def __init__(
        self,
        data: pd.DataFrame,
        output_dir: str | Path = DEFAULT_OUTCOME,
        *,
        splits: dict[str, pd.DataFrame] | None = None,
        dpi: int = 150,
    ) -> None:
        if "Close" not in data.columns:
            raise ValueError("data 必须包含 Close 列")
        _apply_style()

        self.df = data.copy()
        self.df.index = pd.DatetimeIndex(self.df.index)
        self.df = self.df.sort_index()
        if "log_ret" not in self.df.columns:
            self.df = DataPreprocessor.compute_returns(self.df)

        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.split_table = _splits_to_frame(splits)
        self.dpi = int(dpi)
        self.figures: list[Path] = []

    # ------------------------------ 公共 -------------------------------- #
    def _save(self, fig: plt.Figure, filename: str) -> Path:
        path = self.out_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.figures.append(path)
        print(f"[EDA] 已输出: {path.name}")
        return path

    def _caption(self, fig: plt.Figure) -> None:
        first = self.df.index.min()
        last = self.df.index.max()
        fig.text(
            0.995,
            0.005,
            f"S&P 500 daily data: {first:%Y-%m-%d} to {last:%Y-%m-%d}  |  n = {len(self.df):,} trading days",
            ha="right",
            va="bottom",
            fontsize=7,
            color=GREY,
        )

    def _shade_splits(self, axes: Sequence[plt.Axes], *, label: bool = False) -> None:
        """在时间轴上给训练/验证/测试区段淡色底纹并画分界线。"""
        if self.split_table is None:
            return
        colors = {"train": NAVY, "val": AMBER, "test": CRIMSON}
        for ax in axes:
            for _, row in self.split_table.iterrows():
                ax.axvspan(
                    row["start"],
                    row["end"],
                    color=colors.get(row["segment"], GREY),
                    alpha=0.045,
                    linewidth=0,
                    zorder=0,
                )
            for _, row in self.split_table.loc[self.split_table["segment"] != "train"].iterrows():
                ax.axvline(
                    row["start"],
                    color=colors.get(row["segment"], GREY),
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.75,
                    zorder=1,
                )
        if label:
            axes[0].legend(loc="upper left", frameon=True)
            names = {"val": "validation", "test": "test"}
            top = self.split_table.loc[self.split_table["segment"] != "train"]
            for _, row in top.iterrows():
                axes[0].annotate(
                    f"{names.get(row['segment'], row['segment'])} start",
                    xy=(row["start"], 0.03),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=7.2,
                    color=colors.get(row["segment"], GREY),
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.85),
                )

    def _year_axis(self, ax: plt.Axes, years: int = 5) -> None:
        ax.xaxis.set_major_locator(mdates.YearLocator(years))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(self.df.index.min(), self.df.index.max())

    # --------------------------- 图 1：价量 ------------------------------ #
    def plot_price_volume(self) -> Path:
        df = self.df
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(13.5, 7.4),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06},
        )

        ax1.plot(df.index, df["Close"], color=NAVY, linewidth=1.0, label="Close")
        ma200 = df["Close"].rolling(200, min_periods=200).mean()
        ax1.plot(df.index, ma200, color=GREY, linewidth=0.9, linestyle="--", label="200-day SMA")
        ax1.set_yscale("log")
        ax1.set_yticks([700, 1000, 1500, 2000, 3000, 5000, 7000, 10000])
        ax1.yaxis.set_minor_locator(NullLocator())
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax1.set_ylim(600, 10500)
        ax1.set_ylabel("Index level (log scale)")
        ax1.set_title("S&P 500 Daily Close and Trading Volume")
        ax1.legend(loc="upper left")

        monthly_volume = df["Volume"].resample("ME").mean() / 1e9
        monthly_close = df["Close"].resample("ME").last()
        monthly_ret = monthly_close.pct_change().reindex(monthly_volume.index)
        colors = np.where(monthly_ret.fillna(0.0).to_numpy() >= 0, TEAL, CRIMSON)
        ax2.bar(
            monthly_volume.index,
            monthly_volume.to_numpy(),
            width=24,
            color=colors,
            alpha=0.75,
            linewidth=0,
        )
        ax2.set_ylabel("Avg daily volume\n(billion shares)")
        ax2.set_xlabel("Date")

        self._shade_splits([ax1, ax2], label=True)
        self._year_axis(ax2)
        self._caption(fig)
        return self._save(fig, "01_price_volume.png")

    # -------------------------- 图 2：收益率分布 -------------------------- #
    def plot_return_distribution(self) -> Path:
        ret = self.df["log_ret"].dropna() * 100.0
        mu, sigma = float(ret.mean()), float(ret.std(ddof=1))

        fig, ax = plt.subplots(figsize=(11.0, 6.2))
        ax.hist(
            ret,
            bins=140,
            density=True,
            color=TEAL,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.3,
            label="Empirical",
        )

        grid = np.linspace(ret.min(), ret.max(), 800)
        pdf = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(grid, pdf, color=CRIMSON, linewidth=1.7,
                label=f"Normal fit  (mu={mu:.3f}%, sigma={sigma:.2f}%)")

        var95 = float(np.percentile(ret, 5))
        ax.axvline(var95, color="#b7791f", linestyle="--", linewidth=1.2,
                   label=f"5% VaR = {var95:.2f}%")
        ax.axvline(0.0, color=GREY, linestyle=":", linewidth=0.9)

        skew = float(ret.skew())
        kurt = float(ret.kurtosis())
        stats_text = (
            f"mean      {mu:+.3f}%\n"
            f"std dev   {sigma:.3f}%\n"
            f"skewness  {skew:+.2f}\n"
            f"ex. kurt  {kurt:.2f}\n"
            f"up days   {float((ret > 0).mean()):.2%}\n"
            f"min / max {ret.min():+.2f}% / {ret.max():+.2f}%"
        )
        ax.text(
            0.015,
            0.97,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor=GRID, alpha=0.95),
        )

        ax.set_title(f"Distribution of Daily Log Returns (n = {len(ret):,})")
        ax.set_xlabel("Daily log return (%)")
        ax.set_ylabel("Probability density (log scale)")
        ax.set_xlim(float(ret.min()) - 0.4, float(ret.max()) + 0.4)
        # 对数纵轴：既能看清 0 附近的分布形状，又能显示 ±10% 的极端尾部（厚尾）
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.0)
        ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.6)
        ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.3, alpha=0.6)
        ax.legend(loc="upper right")
        self._caption(fig)
        return self._save(fig, "02_return_distribution.png")

    # -------------------------- 图 3：滚动波动率 -------------------------- #
    def plot_rolling_volatility(self) -> Path:
        ret = self.df["log_ret"]
        vol21 = DataPreprocessor.rolling_volatility(ret, 21) * 100
        vol63 = DataPreprocessor.rolling_volatility(ret, 63) * 100
        full_vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)

        fig, ax = plt.subplots(figsize=(13.0, 6.2))
        ax.fill_between(vol63.index, 0, vol63.to_numpy(), color=NAVY, alpha=0.10, linewidth=0)
        ax.plot(vol21.index, vol21.to_numpy(), color=TEAL, linewidth=0.9, label="21-day")
        ax.plot(vol63.index, vol63.to_numpy(), color=NAVY, linewidth=1.4, label="63-day")
        ax.axhline(full_vol, color=CRIMSON, linestyle="--", linewidth=1.0,
                   label=f"Full-sample volatility ({full_vol:.1f}%)")

        peak_idx = vol21.idxmax()
        ax.annotate(
            f"Peak {vol21.max():.1f}%\n{peak_idx:%Y-%m-%d}",
            xy=(peak_idx, float(vol21.max())),
            xytext=(18, -26),
            textcoords="offset points",
            fontsize=8.5,
            color=CRIMSON,
            arrowprops=dict(arrowstyle="->", color=CRIMSON, linewidth=0.9),
        )

        ax.set_title("Rolling Annualized Volatility of Daily Log Returns")
        ax.set_ylabel("Annualized volatility (%)")
        ax.set_xlabel("Date")
        ax.set_ylim(0, 105)
        ax.set_xlim(self.df.index.min(), self.df.index.max())
        ax.legend(loc="upper left")
        self._shade_splits([ax])
        self._year_axis(ax)
        self._caption(fig)
        return self._save(fig, "03_rolling_volatility.png")

    # -------------------------- 图 4：收益热力图 -------------------------- #
    def plot_annual_return_heatmap(self) -> Path:
        df = self.df
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        month_end = df["Close"].resample("ME").last()
        monthly_ret = month_end.pct_change() * 100.0
        first_month = month_end.index.min()
        if len(month_end) > 1 and np.isnan(monthly_ret.iloc[0]):
            first_close = df.loc[:first_month, "Close"].iloc[0]
            monthly_ret.iloc[0] = (month_end.iloc[0] / first_close - 1.0) * 100.0

        table = pd.DataFrame(
            {
                "year": monthly_ret.index.year,
                "month": monthly_ret.index.month,
                "ret": monthly_ret.to_numpy(),
            }
        )
        pivot = table.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))
        annual = (1.0 + pivot.fillna(0.0) / 100.0).prod(axis=1) - 1.0
        annual = annual * 100.0

        years = pivot.index.to_numpy()
        n_years = len(years)
        fig = plt.figure(figsize=(14.0, max(6.0, 0.30 * n_years + 2.2)))
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=[12.0, 1.5, 0.28],
            wspace=0.05,
            left=0.065,
            right=0.93,
            top=0.925,
            bottom=0.105,
        )
        ax_m = fig.add_subplot(gs[0, 0])
        ax_y = fig.add_subplot(gs[0, 1], sharey=ax_m)
        cax_m = fig.add_subplot(gs[0, 2])

        values = pivot.to_numpy(dtype=float)
        vmax = max(1.0, float(np.nanpercentile(np.abs(values), 98)))
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#e5e7eb")

        im_m = ax_m.imshow(
            np.ma.masked_invalid(values),
            aspect="auto",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        for row in range(n_years):
            for col in range(12):
                val = values[row, col]
                if np.isnan(val):
                    continue
                shade = "white" if abs(val) > 0.55 * vmax else "#1f2937"
                ax_m.text(col, row, f"{val:.1f}", ha="center", va="center",
                          fontsize=6.0, color=shade)

        annual_values = annual.to_numpy(dtype=float)
        vmax_a = max(1.0, float(np.nanpercentile(np.abs(annual_values), 98)))
        ax_y.imshow(
            np.ma.masked_invalid(annual_values).reshape(-1, 1),
            aspect="auto",
            cmap=cmap,
            vmin=-vmax_a,
            vmax=vmax_a,
            interpolation="nearest",
        )
        for row in range(n_years):
            val = annual_values[row]
            if np.isnan(val):
                continue
            shade = "white" if abs(val) > 0.55 * vmax_a else "#1f2937"
            ax_y.text(0, row, f"{val:.1f}", ha="center", va="center", fontsize=6.4,
                      color=shade, fontweight="bold")

        ax_m.set_xticks(range(12), month_names, fontsize=8)
        ax_m.set_yticks(range(n_years), [str(y) for y in years], fontsize=7.5)
        ax_m.set_xlabel("Month")
        ax_m.set_ylabel("Year")
        ax_m.set_title("Monthly Returns Heatmap (%) with Annual Return Column")
        ax_m.grid(False)
        ax_m.tick_params(length=0)

        ax_y.set_xticks([0], ["Year"], fontsize=8)
        ax_y.tick_params(length=0, labelleft=False)
        ax_y.grid(False)
        for spine in ax_y.spines.values():
            spine.set_visible(False)

        cb_m = fig.colorbar(im_m, cax=cax_m)
        cb_m.set_label("Monthly return (%)", fontsize=7.5, labelpad=4)
        cb_m.ax.tick_params(labelsize=6.5)

        fig.text(
            0.065,
            0.028,
            "Grey cells: no data (series starts 1996-11-14, ends 2026-11-13). "
            "The annual column is colored on its own scale.",
            fontsize=7,
            color=GREY,
            ha="left",
        )
        self._caption(fig)
        return self._save(fig, "04_annual_return_heatmap.png")

    # -------------------------- 图 5：最大回撤 ---------------------------- #
    def plot_drawdown(self) -> Path:
        df = self.df
        close = df["Close"]
        dd = close / close.cummax() - 1.0

        fig, ax = plt.subplots(figsize=(13.0, 6.0))
        ax.fill_between(dd.index, dd.to_numpy() * 100, 0, color=CRIMSON, alpha=0.32, linewidth=0)
        ax.plot(dd.index, dd.to_numpy() * 100, color=CRIMSON, linewidth=0.9)
        ax.axhline(0.0, color=GREY, linewidth=0.8)

        trough = dd.idxmin()
        trough_val = float(dd.min()) * 100
        peak_date = close.loc[:trough].idxmax()
        peak_val = float(close.loc[peak_date])

        recovered_date = None
        after = close.loc[trough:]
        recovered = after[after >= peak_val]
        if len(recovered):
            recovered_date = recovered.index.min()

        ax.scatter([trough], [trough_val], color="#7f1d1d", s=28, zorder=5)
        label = f"Max drawdown {trough_val:.2f}%\n{trough:%Y-%m-%d}"
        if recovered_date is not None:
            label += f"\nrecovered {recovered_date:%Y-%m-%d}"
        ax.annotate(
            label,
            xy=(trough, trough_val),
            xytext=(-140, -34),
            textcoords="offset points",
            fontsize=9,
            color="#7f1d1d",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#f0b3b8", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#7f1d1d", linewidth=1.0),
        )
        ax.annotate(
            f"Peak {peak_val:,.0f}\n{peak_date:%Y-%m-%d}",
            xy=(peak_date, 0.0),
            xytext=(16, -18),
            textcoords="offset points",
            fontsize=8.5,
            color=NAVY,
            arrowprops=dict(arrowstyle="->", color=NAVY, linewidth=0.9),
        )

        ax.set_title("S&P 500 Drawdown from Running Peak")
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Date")
        ax.set_ylim(trough_val * 1.22, 2.0)
        self._shade_splits([ax])
        self._year_axis(ax)
        self._caption(fig)
        return self._save(fig, "05_max_drawdown.png")

    # ------------------------------ 汇总 -------------------------------- #
    def run_all(self) -> list[Path]:
        """按顺序生成全部图，返回文件路径列表。"""
        self.plot_price_volume()
        self.plot_return_distribution()
        self.plot_rolling_volatility()
        self.plot_annual_return_heatmap()
        self.plot_drawdown()
        print(f"[EDA] 共输出 {len(self.figures)} 张图 -> {self.out_dir}")
        return list(self.figures)

    def metrics(self) -> dict[str, object]:
        """关键统计指标（供报告引用）。"""
        df = self.df
        ret = df["log_ret"].dropna()
        close = df["Close"].astype(float)

        years = len(ret) / TRADING_DAYS_PER_YEAR
        total_return = close.iloc[-1] / close.iloc[0] - 1.0
        cagr = (close.iloc[-1] / close.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
        ann_vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

        downside = ret[ret < 0]
        sortino = float(ret.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

        dd = close / close.cummax() - 1.0
        max_dd = float(dd.min())
        mdd_date = dd.idxmin()

        underwater = dd < 0
        longest, run = 0, 0
        for flag in underwater.to_numpy():
            run = run + 1 if flag else 0
            longest = max(longest, run)

        yearly_ret = df["ret"].groupby(df.index.year).apply(
            lambda s: (1.0 + s.fillna(0.0)).prod() - 1.0
        )

        return {
            "n_obs": int(len(df)),
            "start": f"{df.index.min():%Y-%m-%d}",
            "end": f"{df.index.max():%Y-%m-%d}",
            "years": round(years, 2),
            "total_return": round(float(total_return), 4),
            "cagr": round(float(cagr), 4),
            "annual_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown": round(max_dd, 4),
            "max_drawdown_date": f"{mdd_date:%Y-%m-%d}",
            "longest_underwater_days": int(longest),
            "up_day_ratio": round(float((ret > 0).mean()), 4),
            "skewness": round(float(ret.skew()), 3),
            "excess_kurtosis": round(float(ret.kurtosis()), 3),
            "var_95_daily": round(float(np.percentile(ret, 5)), 4),
            "cvar_95_daily": round(float(ret[ret <= np.percentile(ret, 5)].mean()), 4),
            "best_day": round(float(ret.max()), 4),
            "worst_day": round(float(ret.min()), 4),
            "best_year": int(yearly_ret.idxmax()),
            "best_year_return": round(float(yearly_ret.max()), 4),
            "worst_year": int(yearly_ret.idxmin()),
            "worst_year_return": round(float(yearly_ret.min()), 4),
        }

    def save_summary(self, filename: str = "eda_summary.md") -> Path:
        """把关键指标写入 Markdown + CSV，便于直接引用到报告。"""
        metrics = self.metrics()
        rows = pd.DataFrame(sorted(metrics.items()), columns=["metric", "value"])
        csv_path = self.out_dir / "eda_metrics.csv"
        rows.to_csv(csv_path, index=False)

        def fmt(key: str) -> str:
            value = metrics[key]
            if key in {"total_return", "cagr", "annual_volatility", "max_drawdown",
                       "best_year_return", "worst_year_return"}:
                return f"{value * 100:.2f}%"
            if key in {"var_95_daily", "cvar_95_daily", "best_day", "worst_day"}:
                return f"{value * 100:.2f}%"
            return f"{value}"

        md = [
            "# 0_DataPreprocessing EDA 摘要",
            "",
            f"- 样本区间：{metrics['start']} ~ {metrics['end']}"
            f"（{metrics['n_obs']:,} 个交易日，约 {metrics['years']} 年）",
            "- 图表均在 `Outcome/` 目录，图中文字为英文，可直接放入报告。",
            "",
            "| 指标 | 数值 |",
            "| --- | --- |",
            f"| 累计收益 Total return | {fmt('total_return')} |",
            f"| 年化收益 CAGR | {fmt('cagr')} |",
            f"| 年化波动率 Annual volatility | {fmt('annual_volatility')} |",
            "| 夏普比率 Sharpe ratio | " + fmt("sharpe_ratio") + " |",
            "| 索提诺比率 Sortino ratio | " + fmt("sortino_ratio") + " |",
            f"| 最大回撤 Max drawdown | {fmt('max_drawdown')}（{metrics['max_drawdown_date']}） |",
            f"| 最长水下天数 Longest underwater | {metrics['longest_underwater_days']} 个交易日 |",
            f"| 上涨日占比 Up-day ratio | {fmt('up_day_ratio')} |",
            "| 偏度 Skewness | " + fmt("skewness") + " |",
            "| 超额峰度 Excess kurtosis | " + fmt("excess_kurtosis") + " |",
            f"| 日 VaR (95%) | {fmt('var_95_daily')} |",
            f"| 日 CVaR (95%) | {fmt('cvar_95_daily')} |",
            f"| 最佳 / 最差单日 | {fmt('best_day')} / {fmt('worst_day')} |",
            f"| 最佳年份 | {metrics['best_year']}（{fmt('best_year_return')}） |",
            f"| 最差年份 | {metrics['worst_year']}（{fmt('worst_year_return')}） |",
            "",
            "## 图表清单",
            "",
            "1. `01_price_volume.png` — 收盘价（对数轴）、200 日均线与月度平均成交量",
            "2. `02_return_distribution.png` — 日对数收益率分布、正态拟合与 5% VaR",
            "3. `03_rolling_volatility.png` — 21/63 日滚动年化波动率",
            "4. `04_annual_return_heatmap.png` — 年 × 月收益率热力图与年度收益列",
            "5. `05_max_drawdown.png` — 回撤曲线与最大回撤区间标注",
            "",
            "> 图中虚线为训练/验证/测试集的分界（若传入 splits）。",
        ]
        md_path = self.out_dir / filename
        md_path.write_text("\n".join(md), encoding="utf-8")
        print(f"[EDA] 指标摘要: {md_path.name}, {csv_path.name}")
        return md_path


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S&P 500 数据预处理 + EDA")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="原始 CSV 路径")
    parser.add_argument("--outcome", type=Path, default=DEFAULT_OUTCOME, help="EDA 图片输出目录")
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED, help="清洗后数据输出目录")
    parser.add_argument("--ratios", type=float, nargs=3, default=list(SPLIT_RATIOS),
                        metavar=("TRAIN", "VAL", "TEST"), help="训练/验证/测试比例，默认 0.7 0.15 0.15")
    parser.add_argument("--dpi", type=int, default=150, help="图片分辨率")
    parser.add_argument("--no-eda", action="store_true", help="不生成 EDA 图")
    parser.add_argument("--no-export", action="store_true", help="不导出清洗后 CSV")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    prep = DataPreprocessor(csv_path=args.csv, outcome_dir=args.outcome, verbose=True)
    prep.prepare()

    print("\n=== 清洗质量报告 ===")
    print(prep.report.to_frame().to_string(index=False))  # type: ignore[union-attr]

    print("\n=== 时间序列划分（不打乱） ===")
    prep.split(args.ratios)
    table = prep.split_table()
    print(table.to_string(index=False))

    if not args.no_export:
        prep.export_processed(args.processed / "sp500_clean.csv")

    if not args.no_eda:
        print("\n=== EDA ===")
        eda = EDAReporter(prep.data, args.outcome, splits=prep.splits, dpi=args.dpi)  # type: ignore[arg-type]
        eda.run_all()
        eda.save_summary()
        print("\n=== 关键指标 ===")
        for key, value in eda.metrics().items():
            print(f"  {key:24s} {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
