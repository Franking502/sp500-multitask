# -*- coding: utf-8 -*-
"""模型层：三个 GARCH 族模型的拟合、参数表与残差诊断。

模型
----
============  ==============================  ============
key           设定                             分布
============  ==============================  ============
garch         GARCH(1,1)                       normal
egarch        EGARCH(1,1)，o=1                 学生 t
gjr           GJR-GARCH(1,1)，o=1              学生 t
============  ==============================  ============

说明
----
* 传入的收益率单位是 **百分数**（1.2 表示 1.2%），因此 arch 报出的方差单位是 %²。
* 持续性：GARCH 为 ``alpha + beta``；GJR 为 ``alpha + beta + gamma/2``；
  EGARCH 为对数方差的自回归系数 ``beta``。
* 诊断：对标准化残差 ``z = resid / conditional_volatility`` 及其平方做
  Ljung-Box 检验（lag=10），p 值 > 0.05 表示残差无显著自相关。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from vol_config import MODEL_SPECS, ModelSpec, ensure_arch_available


@dataclass
class FitResult:
    """一次拟合的轻量封装，避免下游直接依赖 arch 的结果对象。"""

    spec: ModelSpec
    result: object                 # arch ARCHModelResult
    nobs: int
    converged: bool

    @property
    def params(self) -> pd.Series:
        return self.result.params


def build_model(spec: ModelSpec, ret: pd.Series):
    """按规格构造未拟合的 arch 模型。"""
    ensure_arch_available()
    from arch import arch_model

    return arch_model(
        ret,
        mean="constant",
        vol=spec.vol,
        p=spec.p,
        o=spec.o,
        q=spec.q,
        dist=spec.dist,
    )


def fit_model(spec: ModelSpec, ret: pd.Series, *, show_warning: bool = False) -> FitResult:
    """在给定收益率序列上做极大似然估计。"""
    model = build_model(spec, ret)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp="off", show_warning=show_warning)
    converged = int(getattr(result, "convergence_flag", 0)) == 0
    return FitResult(spec=spec, result=result, nobs=int(result.nobs), converged=converged)


def fit_all(ret: pd.Series, *, verbose: bool = True) -> dict[str, FitResult]:
    """依次拟合三个模型。"""
    fits: dict[str, FitResult] = {}
    for spec in MODEL_SPECS:
        fit = fit_model(spec, ret)
        fits[spec.key] = fit
        if verbose:
            print(
                f"[fit] {spec.label:14s} nobs={fit.nobs:5d} "
                f"logL={fit.result.loglikelihood:10.2f} "
                f"AIC={fit.result.aic:9.2f} converged={fit.converged}"
            )
    return fits


def _get(params: pd.Series, key: str) -> float:
    return float(params[key]) if key in params.index else float("nan")


def persistence(spec: ModelSpec, params: pd.Series) -> float:
    """波动持续性指标（定义见模块 docstring）。"""
    alpha = _get(params, "alpha[1]")
    beta = _get(params, "beta[1]")
    gamma = _get(params, "gamma[1]")
    if spec.key == "egarch":
        return beta
    if spec.key == "gjr":
        gamma = 0.0 if np.isnan(gamma) else gamma
        return alpha + beta + 0.5 * gamma
    return alpha + beta


def residual_diagnostics(fit: FitResult, lags: int = 10) -> dict[str, float]:
    """标准化残差与其平方的 Ljung-Box 统计量。"""
    result = fit.result
    z = pd.Series(result.resid / result.conditional_volatility).dropna()
    lb = acorr_ljungbox(z, lags=[lags], return_df=True)
    lb2 = acorr_ljungbox(z.pow(2), lags=[lags], return_df=True)
    return {
        "lb_stat": float(lb["lb_stat"].iloc[0]),
        "lb_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "lb2_stat": float(lb2["lb_stat"].iloc[0]),
        "lb2_pvalue": float(lb2["lb_pvalue"].iloc[0]),
    }


def parameters_table(fits: dict[str, FitResult]) -> pd.DataFrame:
    """把拟合结果汇总成参数表（一行一个模型）。"""
    rows = []
    for key, fit in fits.items():
        spec = fit.spec
        params = fit.params
        diag = residual_diagnostics(fit)
        rows.append(
            {
                "model": key,
                "spec": spec.label,
                "dist": spec.dist,
                "mu": _get(params, "mu"),
                "omega": _get(params, "omega"),
                "alpha": _get(params, "alpha[1]"),
                "gamma": _get(params, "gamma[1]") if spec.has_gamma else np.nan,
                "beta": _get(params, "beta[1]"),
                "nu": _get(params, "nu"),
                "persistence": persistence(spec, params),
                "loglik": float(fit.result.loglikelihood),
                "aic": float(fit.result.aic),
                "bic": float(fit.result.bic),
                "lb10_stat": diag["lb_stat"],
                "lb10_pvalue": diag["lb_pvalue"],
                "lb10_sq_pvalue": diag["lb2_pvalue"],
                "nobs": fit.nobs,
                "converged": fit.converged,
            }
        )
    return pd.DataFrame(rows)


def parameters_markdown(table: pd.DataFrame) -> str:
    """把参数表渲染成 Markdown（供 summary.md 使用）。"""

    def fmt(value: float, digits: int = 4) -> str:
        if pd.isna(value):
            return "—"
        return f"{value:.{digits}f}"

    lines = [
        "| 模型 | 分布 | ω | α | γ | β | ν | 持续性 | logL | AIC | LB(10) p | LB²(10) p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| {spec} | {dist} | {omega} | {alpha} | {gamma} | {beta} | {nu} | {pers} | "
            "{loglik} | {aic} | {lb} | {lb2} |".format(
                spec=row["spec"],
                dist=row["dist"],
                omega=fmt(row["omega"], 5),
                alpha=fmt(row["alpha"]),
                gamma=fmt(row["gamma"]),
                beta=fmt(row["beta"]),
                nu=fmt(row["nu"], 2),
                pers=fmt(row["persistence"], 3),
                loglik=f"{row['loglik']:.1f}",
                aic=f"{row['aic']:.1f}",
                lb=fmt(row["lb10_pvalue"], 3),
                lb2=fmt(row["lb10_sq_pvalue"], 3),
            )
        )
    return "\n".join(lines)
