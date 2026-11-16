# 0_DataPreprocessing EDA 摘要

- 样本区间：1996-11-14 ~ 2026-11-13（7,546 个交易日，约 29.94 年）
- 图表均在 `Outcome/` 目录，图中文字为英文，可直接放入报告。

| 指标 | 数值 |
| --- | --- |
| 累计收益 Total return | 927.57% |
| 年化收益 CAGR | 8.09% |
| 年化波动率 Annual volatility | 19.18% |
| 夏普比率 Sharpe ratio | 0.406 |
| 索提诺比率 Sortino ratio | 0.51 |
| 最大回撤 Max drawdown | -56.78%（2009-03-09） |
| 最长水下天数 Longest underwater | 1802 个交易日 |
| 上涨日占比 Up-day ratio | 0.5366 |
| 偏度 Skewness | -0.361 |
| 超额峰度 Excess kurtosis | 10.046 |
| 日 VaR (95%) | -1.84% |
| 日 CVaR (95%) | -2.92% |
| 最佳 / 最差单日 | 10.96% / -12.77% |
| 最佳年份 | 1997（31.01%） |
| 最差年份 | 2008（-38.49%） |

## 图表清单

1. `01_price_volume.png` — 收盘价（对数轴）、200 日均线与月度平均成交量
2. `02_return_distribution.png` — 日对数收益率分布、正态拟合与 5% VaR
3. `03_rolling_volatility.png` — 21/63 日滚动年化波动率
4. `04_annual_return_heatmap.png` — 年 × 月收益率热力图与年度收益列
5. `05_max_drawdown.png` — 回撤曲线与最大回撤区间标注

> 图中虚线为训练/验证/测试集的分界（若传入 splits）。