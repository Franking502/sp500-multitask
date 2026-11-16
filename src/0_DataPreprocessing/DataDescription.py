# -*- coding: utf-8 -*-
"""模块 0：特征工程（技术指标）—— 统一类接口，供分类 / 预测 / 策略成员直接调用。

设计目标
--------
* **一个类、一个入口**：``FeatureEngineer(df)`` 之后，调用任意方法即返回对应指标；
  每个方法都支持传入 ``series`` 参数，因此也可以脱离 ``self.data`` 单独计算。
* **不产生未来信息**：全部指标只使用当日及之前的收盘价（滞后收益率显式 shift）。
* **命名稳定**：返回的 Series / DataFrame 列名固定（如 ``ma_20``、``rsi_14``、
  ``macd_hist``、``bb_upper_20``、``vol_21``、``log_ret_lag_3``），便于下游直接引用。

示例
----
>>> from DataDescription import FeatureEngineer
>>> fe = FeatureEngineer.from_csv("data/processed/sp500_clean.csv")
>>> fe.ma(20).tail(3)                      # 20 日均线
>>> fe.rsi(14).tail(3)                     # RSI(14)
>>> fe.macd()                              # DataFrame: macd / macd_signal / macd_hist
>>> fe.bollinger(20, 2.0)                  # DataFrame: 中轨 / 上轨 / 下轨 / %B / 带宽
>>> fe.rolling_volatility(21)              # 21 日年化滚动波动率
>>> fe.lagged_returns((1, 2, 3))           # 滞后 1/2/3 日收益率
>>> features = fe.build()                  # 一次性拼出全部特征，可直接喂给模型
>>> fe.feature_columns()                   # 特征列清单（不含 OHLCV 与当期收益率）

兼容函数接口（readme 中约定的 ``build_features(df)``）::

    from DataDescription import build_features
    features = build_features(clean_df)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "SP500_daily.csv"
DEFAULT_CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "sp500_clean.csv"
DEFAULT_FEATURE_CSV = PROJECT_ROOT / "data" / "processed" / "sp500_features.csv"

BASE_FIELDS = ("open", "high", "low", "close", "volume")

DEFAULT_CONFIG: dict[str, object] = {
    "ma_windows": (5, 10, 20, 50, 200),
    "ema_spans": (12, 26),
    "rsi_window": 14,
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"window": 20, "num_std": 2.0},
    "vol_windows": (21, 63),
    "lags": (1, 2, 3, 4, 5),
    "atr_window": 14,
}

__all__ = ["FeatureEngineer", "build_features", "DEFAULT_CONFIG"]


class FeatureEngineer:
    """技术指标计算器。

    Parameters
    ----------
    data:
        含 ``Open/High/Low/Close/Volume``（大小写不敏感）的日频行情表，
        索引为交易日；为 ``None`` 时可在调用方法时直接传入 ``series``。
    price_col:
        价格列名，默认 ``Close``；若数据列名不同（如 ``close``）会自动匹配。
    annualize:
        年化交易日数，默认 252，用于滚动波动率年化。
    verbose:
        是否打印构建信息。

    Notes
    -----
    所有方法都不会修改 ``self.data``，返回的 Series / DataFrame 与输入等长、
    索引对齐（预热期为正 NaN），方便下游 ``pd.concat``。
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        *,
        price_col: str = "Close",
        volume_col: str = "Volume",
        annualize: int = TRADING_DAYS_PER_YEAR,
        verbose: bool = False,
    ) -> None:
        self.price_col = price_col
        self.volume_col = volume_col
        self.annualize = int(annualize)
        self.verbose = verbose
        self._data: pd.DataFrame | None = None
        self._cols: dict[str, str] = {}
        self._last_config: dict[str, object] = {}
        self._last_output: pd.DataFrame | None = None
        if data is not None:
            self.set_data(data)

    # ------------------------------------------------------------------ #
    # 数据装载
    # ------------------------------------------------------------------ #
    def set_data(self, data: pd.DataFrame) -> "FeatureEngineer":
        """绑定行情数据；列名匹配大小写，索引统一为交易日。"""
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("data 必须是非空 DataFrame")

        df = data.copy()
        df.columns = [str(col).strip() for col in df.columns]
        lookup = {col.lower(): col for col in df.columns}

        cols: dict[str, str] = {}
        for field in BASE_FIELDS:
            if field in lookup:
                cols[field] = lookup[field]
        if "close" not in cols:
            raise ValueError(
                f"数据缺少收盘价列（期望 {self.price_col!r} 或 'close'），实际列: {list(df.columns)}"
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except (TypeError, ValueError) as exc:  # pragma: no cover - 防御性分支
                raise ValueError("索引无法解析为日期") from exc

        df = df.sort_index()
        df = df.loc[~df.index.duplicated(keep="last")]

        for field, col in cols.items():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        self._data = df
        self._cols = cols
        if self.verbose:
            print(f"[FeatureEngineer] 绑定 {len(df)} 行数据，列映射: {cols}")
        return self

    @classmethod
    def from_csv(
        cls,
        path: str | Path = DEFAULT_CLEAN_CSV,
        *,
        date_col: str = "Date",
        **kwargs: object,
    ) -> "FeatureEngineer":
        """从 CSV 构建：默认读取清洗后的 ``data/processed/sp500_clean.csv``。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到数据文件: {path}")
        df = pd.read_csv(path)
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)
        return cls(df, **kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # 属性与内部工具
    # ------------------------------------------------------------------ #
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise RuntimeError("尚未绑定数据：请先调用 set_data() 或构造时传入 data")
        return self._data

    @property
    def close(self) -> pd.Series:
        return self.data[self._cols["close"]].astype(float)

    def _column(self, field: str) -> pd.Series:
        if field not in self._cols:
            raise ValueError(f"数据缺少 {field!r} 列，无法计算该指标")
        return self.data[self._cols[field]].astype(float)

    def _price(self, series: pd.Series | pd.DataFrame | None) -> pd.Series:
        """解析价格序列：优先使用传入值，其次用 self.data 的收盘价。"""
        if series is None:
            return self.close
        if isinstance(series, pd.DataFrame):
            lookup = {col.lower(): col for col in series.columns}
            if "close" in lookup:
                return series[lookup["close"]].astype(float)
            raise ValueError("传入的 DataFrame 需包含 Close/close 列")
        if isinstance(series, pd.Series):
            return series.astype(float)
        raise TypeError(f"series 需为 Series / DataFrame / None，实际为 {type(series)}")

    @staticmethod
    def _check_window(window: int, name: str, *, minimum: int = 1) -> int:
        window = int(window)
        if window < minimum:
            raise ValueError(f"{name} 必须 >= {minimum}，当前 {window}")
        return window

    def _check_length(self, series: pd.Series, required: int, indicator: str) -> None:
        if len(series) < required:
            raise ValueError(
                f"数据长度 {len(series)} 不足，{indicator} 至少需要 {required} 个观测"
            )

    # ------------------------------------------------------------------ #
    # 1. 移动平均 MA
    # ------------------------------------------------------------------ #
    def ma(self, window: int = 20, series: pd.Series | pd.DataFrame | None = None) -> pd.Series:
        """简单移动平均：``MA_t = mean(Close_{t-w+1..t})``。

        Returns
        -------
        pd.Series
            列名 ``ma_{window}``，前 ``window-1`` 个值为 NaN。
        """
        window = self._check_window(window, "window")
        close = self._price(series)
        self._check_length(close, window, "MA")
        out = close.rolling(window=window, min_periods=window).mean()
        out.name = f"ma_{window}"
        return out

    def ma_ratio(
        self, window: int = 20, series: pd.Series | pd.DataFrame | None = None
    ) -> pd.Series:
        """价格相对均线的偏离度 ``Close / MA - 1``，量纲统一、更适合做特征。"""
        close = self._price(series)
        out = close / self.ma(window, close) - 1.0
        out.name = f"ma_ratio_{window}"
        return out

    # ------------------------------------------------------------------ #
    # 2. 指数移动平均 EMA
    # ------------------------------------------------------------------ #
    def ema(self, span: int = 20, series: pd.Series | pd.DataFrame | None = None) -> pd.Series:
        """指数移动平均：``alpha = 2 / (span + 1)``，与 TA-Lib 口径一致（adjust=False）。"""
        span = self._check_window(span, "span")
        close = self._price(series)
        self._check_length(close, span, "EMA")
        out = close.ewm(span=span, adjust=False, min_periods=span).mean()
        out.name = f"ema_{span}"
        return out

    # ------------------------------------------------------------------ #
    # 3. 相对强弱指标 RSI
    # ------------------------------------------------------------------ #
    def rsi(self, window: int = 14, series: pd.Series | pd.DataFrame | None = None) -> pd.Series:
        """Wilder RSI：用 ``alpha = 1/window`` 的指数平滑近似 Wilder 平均。

        取值范围 0~100；无下跌时取 100，完全无波动时取 50。
        """
        window = self._check_window(window, "window", minimum=2)
        close = self._price(series)
        self._check_length(close, window + 1, "RSI")

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        out = 100.0 - 100.0 / (1.0 + rs)
        out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
        out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
        out.name = f"rsi_{window}"
        return out

    # ------------------------------------------------------------------ #
    # 4. MACD
    # ------------------------------------------------------------------ #
    def macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        series: pd.Series | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """MACD：快慢 EMA 差、信号线（MACD 的 EMA）与柱状图。

        Returns
        -------
        pd.DataFrame
            列：``macd``（DIF）、``macd_signal``（DEA）、``macd_hist``（DIF-DEA）。
        """
        fast = self._check_window(fast, "fast")
        slow = self._check_window(slow, "slow")
        signal = self._check_window(signal, "signal")
        if fast >= slow:
            raise ValueError(f"fast({fast}) 必须小于 slow({slow})")

        close = self._price(series)
        self._check_length(close, slow + signal, "MACD")

        fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()

        return pd.DataFrame(
            {
                "macd": macd_line,
                "macd_signal": signal_line,
                "macd_hist": macd_line - signal_line,
            }
        )

    # ------------------------------------------------------------------ #
    # 5. 布林带
    # ------------------------------------------------------------------ #
    def bollinger(
        self,
        window: int = 20,
        num_std: float = 2.0,
        *,
        ddof: int = 0,
        series: pd.Series | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """布林带：中轨为 SMA，上下轨为中轨 ± ``num_std`` 倍滚动标准差。

        Returns
        -------
        pd.DataFrame
            列：``bb_mid_{w}``、``bb_upper_{w}``、``bb_lower_{w}``、
            ``bb_pctb_{w}``（%B）、``bb_width_{w}``（带宽 / 中轨）。
        """
        window = self._check_window(window, "window", minimum=2)
        if num_std <= 0:
            raise ValueError("num_std 必须为正数")

        close = self._price(series)
        self._check_length(close, window, "Bollinger")

        mid = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std(ddof=ddof)
        upper = mid + num_std * std
        lower = mid - num_std * std
        width = (upper - lower) / mid
        pct_b = (close - lower) / (upper - lower)

        suffix = f"_{window}"
        return pd.DataFrame(
            {
                f"bb_mid{suffix}": mid,
                f"bb_upper{suffix}": upper,
                f"bb_lower{suffix}": lower,
                f"bb_pctb{suffix}": pct_b,
                f"bb_width{suffix}": width,
            }
        )

    # ------------------------------------------------------------------ #
    # 6. 滚动波动率
    # ------------------------------------------------------------------ #
    def rolling_volatility(
        self,
        window: int = 21,
        *,
        annualize: bool = True,
        series: pd.Series | pd.DataFrame | None = None,
    ) -> pd.Series:
        """滚动波动率：对数收益率的标准差，默认年化（×√252）。

        ``series`` 可以是价格序列（内部先取对数差分），也可以直接是收益率序列。
        """
        window = self._check_window(window, "window", minimum=2)
        values = self._price(series)
        self._check_length(values, window + 1, "滚动波动率")

        returns = np.log(values).diff()
        vol = returns.rolling(window, min_periods=window).std(ddof=1)
        if annualize:
            vol = vol * np.sqrt(self.annualize)
        vol.name = f"vol_{window}" + ("_ann" if annualize else "")
        return vol

    # ------------------------------------------------------------------ #
    # 7. 滞后收益率
    # ------------------------------------------------------------------ #
    def lagged_returns(
        self,
        lags: Iterable[int] = (1, 2, 3, 4, 5),
        *,
        log: bool = True,
        series: pd.Series | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """滞后收益率：``r_{t-k}``，避免使用当日收益造成未来信息泄漏。

        Parameters
        ----------
        lags:
            滞后阶数集合，例如 ``(1, 2, 5)``。
        log:
            True 返回对数收益率，False 返回简单收益率。
        """
        close = self._price(series)
        lag_list = sorted({self._check_window(lag, "lag") for lag in lags})
        if not lag_list:
            raise ValueError("lags 不能为空")
        self._check_length(close, max(lag_list) + 1, "滞后收益率")

        if log:
            base = np.log(close).diff()
            prefix = "log_ret_lag"
        else:
            base = close / close.shift(1) - 1.0
            prefix = "ret_lag"

        return pd.DataFrame({f"{prefix}_{lag}": base.shift(lag) for lag in lag_list})

    # ------------------------------------------------------------------ #
    # 8. 附加：ATR（真实波幅的 Wilder 平滑）
    # ------------------------------------------------------------------ #
    def atr(
        self,
        window: int = 14,
        *,
        normalized: bool = False,
    ) -> pd.Series:
        """平均真实波幅：``max(H-L, |H-C_prev|, |L-C_prev|)`` 的指数平滑。

        Parameters
        ----------
        normalized:
            True 时返回 ``ATR / Close``，便于跨时期比较。
        """
        window = self._check_window(window, "window", minimum=2)
        high, low, close = self._column("high"), self._column("low"), self._column("close")
        self._check_length(close, window + 1, "ATR")

        prev_close = close.shift(1)
        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        out = true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        if normalized:
            out = out / close
            out.name = f"atr_pct_{window}"
        else:
            out.name = f"atr_{window}"
        return out

    # ------------------------------------------------------------------ #
    # 组合构建
    # ------------------------------------------------------------------ #
    def build(
        self,
        *,
        dropna: bool = True,
        config: Mapping[str, object] | None = None,
    ) -> pd.DataFrame:
        """一次性计算全部特征，返回「原始行情 + 收益率 + 指标」的宽表。

        Parameters
        ----------
        dropna:
            True（默认）删除指标预热期（NaN）所在行，得到可直接训练的样本；
            False 保留完整时间轴，方便检查指标曲线。
        config:
            覆盖默认参数，键同 :data:`DEFAULT_CONFIG`，例如
            ``{"ma_windows": (10, 20), "lags": (1, 2, 3)}``。

        Returns
        -------
        pd.DataFrame
            索引为交易日；列包含 OHLCV、``ret``、``log_ret`` 以及全部指标。
        """
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        close = self.close
        base = self.data

        features: list[pd.DataFrame | pd.Series] = []
        features.append((close / close.shift(1) - 1.0).rename("ret"))
        features.append(np.log(close).diff().rename("log_ret"))

        for window in cfg["ma_windows"]:  # type: ignore[union-attr]
            features.append(self.ma(window))
            features.append(self.ma_ratio(window))
        for span in cfg["ema_spans"]:  # type: ignore[union-attr]
            features.append(self.ema(span))

        features.append(self.rsi(int(cfg["rsi_window"])))  # type: ignore[arg-type]
        features.append(self.macd(**cfg["macd"]))  # type: ignore[arg-type]
        features.append(self.bollinger(**cfg["bollinger"]))  # type: ignore[arg-type]

        for window in cfg["vol_windows"]:  # type: ignore[union-attr]
            features.append(self.rolling_volatility(window))
        features.append(self.lagged_returns(tuple(cfg["lags"])))  # type: ignore[arg-type]

        if "high" in self._cols and "low" in self._cols:
            features.append(self.atr(int(cfg["atr_window"])))  # type: ignore[arg-type]

        indicator_df = pd.concat(features, axis=1)
        indicator_df = indicator_df.loc[:, ~indicator_df.columns.duplicated()]

        out = pd.concat([base, indicator_df], axis=1)
        # base 里可能已经带有 ret / log_ret（清洗阶段算过），去重保留第一份
        out = out.loc[:, ~out.columns.duplicated()]
        self._last_config = cfg

        if dropna:
            before = len(out)
            feature_cols = [col for col in indicator_df.columns if col not in {"ret", "log_ret"}]
            out = out.dropna(subset=feature_cols)
            if out.empty:
                raise ValueError(
                    "去除预热期后没有剩余样本：数据太短，或窗口设置过大"
                    f"（最长窗口 {max(cfg['ma_windows'])}）"  # type: ignore[arg-type]
                )
            if self.verbose:
                print(f"[FeatureEngineer] 剔除预热期 {before - len(out)} 行，保留 {len(out)} 行")
        self._last_output = out
        return out

    def feature_columns(self) -> list[str]:
        """全部指标列名（不含 OHLCV 与当期 ``ret`` / ``log_ret``，供模型直接使用）。"""
        if self._data is None:
            raise RuntimeError("尚未绑定数据")
        preview = self.build(dropna=False)
        excluded = {col for col in self.data.columns} | {"ret", "log_ret"}
        return [col for col in preview.columns if col not in excluded]

    @staticmethod
    def _group_of(name: str) -> str:
        """把列名映射到指标类别，便于在报告里归类统计覆盖率。"""
        if name.startswith(("ma_", "ma_ratio")):
            return "ma"
        if name.startswith("ema_"):
            return "ema"
        if name.startswith("rsi_"):
            return "rsi"
        if name.startswith("macd"):
            return "macd"
        if name.startswith("bb_"):
            return "bollinger"
        if name.startswith("vol_"):
            return "volatility"
        if name.startswith(("log_ret_lag", "ret_lag")):
            return "lag_return"
        if name.startswith("atr"):
            return "atr"
        return "other"

    def describe(self) -> pd.DataFrame:
        """特征清单与覆盖率：列名、类别、有效样本数、缺失率、首个有效日期。"""
        preview = self.build(dropna=False)
        rows = []
        for col in preview.columns:
            if col in self.data.columns or col in {"ret", "log_ret"}:
                continue
            series = preview[col]
            valid = series.notna()
            first_valid = series.index[valid][0] if valid.any() else pd.NaT
            rows.append(
                {
                    "feature": col,
                    "group": self._group_of(col),
                    "n_valid": int(valid.sum()),
                    "coverage": round(float(valid.mean()), 4),
                    "first_valid": first_valid.strftime("%Y-%m-%d") if valid.any() else "",
                }
            )
        return pd.DataFrame(rows)

    def save_features(self, path: str | Path = DEFAULT_FEATURE_CSV) -> Path:
        """把 ``build()`` 的结果导出为 CSV（索引列名为 Date）。"""
        features = self.build(dropna=True)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(path, index_label="Date", float_format="%.10g")
        if self.verbose:
            print(f"[FeatureEngineer] 特征已导出: {path}  形状={features.shape}")
        return path


# --------------------------------------------------------------------------- #
# 函数式兼容接口
# --------------------------------------------------------------------------- #
def build_features(
    data: pd.DataFrame,
    *,
    dropna: bool = True,
    config: Mapping[str, object] | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """``FeatureEngineer.build`` 的函数式封装，保持模块 A 约定的调用方式::

        features = build_features(clean_df)

    额外参数（如 ``annualize``）会透传给 :class:`FeatureEngineer`。
    """
    return FeatureEngineer(data, **kwargs).build(dropna=dropna, config=config)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S&P 500 技术指标特征工程")
    parser.add_argument("--csv", type=Path, default=None,
                        help="输入 CSV；缺省优先使用 data/processed/sp500_clean.csv，否则用原始数据")
    parser.add_argument("--out", type=Path, default=DEFAULT_FEATURE_CSV, help="特征输出 CSV")
    parser.add_argument("--no-save", action="store_true", help="只打印，不写文件")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    source = args.csv
    if source is None:
        source = DEFAULT_CLEAN_CSV if DEFAULT_CLEAN_CSV.exists() else DEFAULT_RAW_CSV
    print(f"[FeatureEngineer] 输入数据: {source}")

    fe = FeatureEngineer.from_csv(source, verbose=True)
    features = fe.build()

    print("\n=== 特征覆盖率 ===")
    print(fe.describe().to_string(index=False))

    print(f"\n=== 特征矩阵 === 形状={features.shape}")
    print("前 3 行、前 8 列:")
    print(features.iloc[:3, :8].to_string())

    if not args.no_save:
        fe.save_features(args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
