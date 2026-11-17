# -*- coding: utf-8 -*-
"""滚动预测：逐日生成 1/5/10 步条件波动率预测（无前视）。

算法
----
对评估区间内的每个目标日 t：

1. 取 **t-1 及之前** 的收益率作为历史（``ret.iloc[:pos]``，pos 为 t 的位置）；
2. 若距上次参数估计已达 ``refit_every`` 天，则用历史重新做极大似然估计；
   否则用 ``arch`` 的 ``fix(params)`` 做**固定参数滤波**（不重新估计，毫秒级）；
3. 用 ``forecast(horizon=max_h)`` 得到未来各日条件方差，
   h 步预测取前 h 天方差的均值再开方、年化：
   ``sigma_h = sqrt(mean(var_1..var_h)) * sqrt(252)``。

返回的 DataFrame 以**目标日 t** 为索引，列 ``sigma_1/sigma_5/sigma_10``，
单位是年化 %。t 日仓位可直接使用 ``sigma_1[t]``，无需再 shift。
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from vol_config import ANNUALIZE, HORIZONS, REFIT_EVERY, RNG_SEED, ModelSpec
from vol_models import build_model


def forecast_kwargs(spec: ModelSpec, horizon: int) -> dict[str, object]:
    """按模型规格生成 ``forecast()`` 参数。

    EGARCH 的对数方差结构没有多步解析解，arch 要求 ``method="simulation"``；
    这里固定随机种子并做 2000 条路径平均，保证同一输入每次结果一致。
    """
    kwargs: dict[str, object] = {"horizon": horizon, "reindex": False}
    if spec.forecast_method != "analytic":
        kwargs["method"] = spec.forecast_method
        kwargs["simulations"] = spec.simulations
        kwargs["rng"] = np.random.default_rng(RNG_SEED).random
    return kwargs


def rolling_forecast(
    spec: ModelSpec,
    ret: pd.Series,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    refit_every: int = REFIT_EVERY,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成逐日滚动预测。

    Parameters
    ----------
    spec:
        模型规格（见 ``vol_config.MODEL_SPECS``）。
    ret:
        全样本对数收益率（百分数），必须覆盖训练期。
    start, end:
        评估区间的首末日期（含）。
    horizons:
        预测步长，默认 (1, 5, 10)。
    refit_every:
        每隔多少个交易日重新估计一次参数，默认 20。

    Returns
    -------
    (forecast, refit_log)
        ``forecast``：索引为目标日，列 ``sigma_h``（年化 %）与 ``refit`` 标记；
        ``refit_log``：每次参数估计的日期与参数值，便于复现。
    """
    if refit_every < 1:
        raise ValueError("refit_every 必须 >= 1")
    max_h = max(horizons)

    window = ret.loc[(ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(end))]
    if window.empty:
        raise ValueError("评估区间内没有数据")

    params: pd.Series | None = None
    refit_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    last_fit_step = -(10**9)
    tic = time.time()

    positions = ret.index.get_indexer(window.index)
    for step, (target, pos) in enumerate(zip(window.index, positions)):
        if pos <= 0:
            continue
        history = ret.iloc[:pos]                      # 严格到 t-1
        assert history.index.max() < target, "历史数据包含目标日，存在前视"

        need_refit = params is None or (step - last_fit_step) >= refit_every
        try:
            if need_refit:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = build_model(spec, history).fit(disp="off", show_warning=False)
                params = result.params.copy()
                last_fit_step = step
                refit_rows.append(
                    {"date": target, "model": spec.key, **{k: float(v) for k, v in params.items()}}
                )
            else:
                result = build_model(spec, history).fix(params)
        except Exception:                              # 估计失败则沿用上次参数
            if params is None:
                raise
            result = build_model(spec, history).fix(params)
            need_refit = False

        forecast = result.forecast(**forecast_kwargs(spec, max_h))
        variance = np.asarray(forecast.variance.iloc[-1], dtype=float)

        record: dict[str, object] = {"date": target}
        for h in horizons:
            mean_var = float(np.mean(variance[:h]))
            record[f"sigma_{h}"] = float(np.sqrt(mean_var) * np.sqrt(ANNUALIZE))
        record["refit"] = bool(need_refit)
        rows.append(record)

        if verbose and (step + 1) % 250 == 0:
            print(f"  [{spec.key}] {step + 1}/{len(window)}  ({time.time() - tic:.1f}s)")

    forecast_df = pd.DataFrame(rows).set_index("date")
    forecast_df.index.name = "Date"
    refit_log = pd.DataFrame(refit_rows)
    if verbose:
        print(
            f"[forecast] {spec.label:14s} {len(forecast_df):5d} 天，"
            f"重拟合 {len(refit_log):3d} 次，耗时 {time.time() - tic:.1f}s"
        )
    return forecast_df, refit_log


def forecasts_by_segment(
    forecast: pd.DataFrame, splits: dict[str, pd.Series]
) -> dict[str, pd.DataFrame]:
    """按 train/val/test 的日期索引切分预测结果（train 通常为空）。"""
    out: dict[str, pd.DataFrame] = {}
    for name, part in splits.items():
        sliced = forecast.loc[forecast.index.intersection(part.index)]
        out[name] = sliced
    return out
