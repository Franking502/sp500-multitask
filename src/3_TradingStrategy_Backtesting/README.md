# 交易策略与回测模块

使用模块 C 输出的涨跌分类信号和标普 500 日线，回测五个仓位规则：买入持有（基准）、信号多/空仓（B）、信号多/空（C）、波动率目标（D）、波动率加趋势上限（E）。评估区间为 2022-05-11 至 2026-11-13，共 1,133 个交易日，扣 0.1% 单边成本与 3% 融资。

完整实验说明见[交易策略与回测报告](../../reports/3_TradingStrategy_Backtesting/交易策略与回测报告.md)，从头运行可使用 [TradingStrategy_Backtesting.ipynb](TradingStrategy_Backtesting.ipynb)。

## 数据与策略

需要项目中的两个文件：

- [SP500_daily.csv](../../data/SP500_daily.csv)：收盘价，7,547 条记录，1996-11-14 至 2026-11-13。
- [test_signals.csv](../../reports/2_RiseFallForecast/signals/test_signals.csv)：模块 C 的测试期信号，1,131 行，`pred_up` 为 1 表示判涨。

评估区间由信号覆盖范围决定，不是全样本。1,133 个交易日中有 2 天无信号（2023-05-24 信号表缺失、2026-11-13 信号末端），按空仓处理；这两天只影响 B、C，D、E 的仓位与信号无关。

- A：恒为 1.0，基准。
- B：信号为 1 满仓，为 0 空仓。
- C：信号为 1 做多，为 0 做空（−1）。
- D：`min(20% ÷ 40 日年化波动率, 1.75)`，经 0.30 缓冲带后执行。
- E：同上，杠杆上限改为 `1.4`（收盘在 200 日均线上方）或 `0.5`（下方）。

D 与 E 的波动率窗口、缓冲带完全相同，只差杠杆上限，因此两者差异可直接归因于趋势约束。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| [TradingStrategy_Backtesting.ipynb](TradingStrategy_Backtesting.ipynb) | 主入口：7 个代码单元、48 行，读完即得指标表与净值图；同时写出 `equity_drawdown.png`，与脚本版本内容一致、样式略简。 |
| [StrategyRules.py](StrategyRules.py) | 参数、行情与信号读取、缓冲带、五个策略的仓位。 |
| [BacktestEngine.py](BacktestEngine.py) | 净收益、净值、回撤、指标与逐笔交易。 |
| [EvaluateStrategyMetrics.py](EvaluateStrategyMetrics.py) | 导出 `metrics/*.json` 与 `positions/*.csv`。 |
| [PlotPresentationFigures.py](PlotPresentationFigures.py) | 生成 slide1–slide3 与结果配图。 |

## 运行方法

环境为 Python 3.13，依赖 NumPy、pandas、Matplotlib，无需额外安装。所有命令在**项目根目录**执行，即 `data/`、`src/`、`reports/` 所在目录。

```powershell
# 生成指标与持仓明细
python src/3_TradingStrategy_Backtesting/EvaluateStrategyMetrics.py

# 生成全部配图
python src/3_TradingStrategy_Backtesting/PlotPresentationFigures.py
```

两个脚本都支持 `--output-dir`，默认写入 `reports/3_TradingStrategy_Backtesting`。项目根由脚本自身位置推出，数据按绝对路径读取，因此在任何目录下执行都可以，不要求先 `cd` 到项目根。

## 口径要点

1. 仓位在 t 日收盘确定，收益取 t → t+1，`fwd = close.shift(-1) / close - 1`；写成 `shift(1)` 会引入未来信息。
2. 换手成本为 `|Δ仓位| × 0.1%`，建仓、平仓、反手都收。
3. 融资成本为 `max(仓位 − 1, 0) × 3% / 252`，只有加杠杆的部分付息。
4. 波动率与均线只用 t 日及以前的数据，`rolling` 不 center、不正向 shift。
5. 标普 500 是价格指数，不可直接交易，本回测为信号级回测；实盘对应 SPY（融资约 6%）或 ES 期货（约 3%）。
