# 项目说明
## 各部分说明
- 0_DataPreprocessing 数据预处理
- 1_TimeSerialAnalyze 时间序列分析
- 2_RiseFallForecast 涨跌预测
- 3_TradingStrategy_Backtesting 交易策略与回测
- 4_Long-termBenefits 长期收益


## 一、仓库与环境搭建

**仓库结构**：

```
sp500-multitask/
├─data
├─reports
│  ├─0_DataPreprocessing
│  ├─1_TimeSerialAnalyze
│  ├─2_RiseFallForecast
│  ├─3_TradingStrategy_Backtesting
│  └─4_Long-termBenefits
└─src
    ├─0_DataPreprocessing
    ├─1_TimeSerialAnalyze
    ├─2_RiseFallForecast
    ├─3_TradingStrategy_Backtesting
    └─4_Long-termBenefits
```

---

## 二、任务模块拆分与分工

### 模块 A：数据工程 + EDA + 可视化

**产出**：清洗后的日频数据（`data/processed/sp500_clean.csv`）、EDA 报告 notebook、特征工程函数库（`src/features.py`）。

**具体内容**：
- 数据加载与清洗：处理缺失值、对齐交易日、计算对数收益率。
- EDA：价格与成交量时序图、收益率分布、滚动波动率、年度收益热力图、最大回撤曲线。
- 特征工程：MA、EMA、RSI、MACD、布林带、滚动波动率、滞后收益率等，输出为统一函数供其他成员调用。
- 划分训练/验证/测试集（时间序列划分，严禁随机打乱）。

**交付给其他成员的接口**：`features.py` 中暴露 `build_features(df) -> DataFrame`，分类和策略成员直接调用。

**评价指标**：数据完整性（缺失率）、特征覆盖率、EDA 图表数量与信息密度。

---

### 模块 B：时间序列预测

**产出**：预测 notebook + 最佳模型保存 + 指定窗口预测结果。

**具体内容**：
- 基准模型：ARIMA 或朴素预测（naive）。
- 机器学习模型：LightGBM 或 XGBoost（用滞后特征）。
- 深度学习模型（可选）：LSTM 或 Transformer 简化版。
- 评价指标：**RMSE、MAE、R²**。股票预测中 R² 往往偏低，需如实报告并讨论滞后性问题。
- **最终交付**：对助教/课程指定的时间窗口（例如最后 20 个交易日）输出价格预测值，保存为 `reports/forecast_window.csv`。

**注意**：预测目标用“收益率”还是“价格”需统一。建议同时输出两者，但以价格预测作为最终交付格式。

---

### 模块 C：分类（涨跌方向预测）

**产出**：分类 notebook + 最佳模型 + 分类指标报告。

**具体内容**：
- 标签构造：次日涨/跌二分类（或三分类：涨/平/跌）。
- 模型：逻辑回归、随机森林、XGBoost，可做简单 ensemble。
- 特征：模块 A 的技术指标 + 滞后收益率。
- 评价指标：**Accuracy、Precision、Recall、F1、AUC**。注意股票方向预测准确率若在 50%~55% 区间需如实讨论，不要过度宣称“有效”。
- 输出每个模型的分类报告和混淆矩阵。

---

### 模块 D：交易策略与回测

**产出**：策略回测 notebook + 绩效指标表 + 净值曲线图。

**具体内容**：
- 基于模块 C 的分类信号构建简单策略：预测涨则持有，预测跌则空仓/做空（视标的可操作性而定）。
- 基准策略：Buy & Hold。
- 回测指标：**年化收益率、夏普比率、最大回撤、胜率、盈亏比**。
- 对比策略与基准的表现，尤其关注最大回撤是否改善。
- 注意交易成本：至少设置 0.1% 单边成本，否则回测结果不可信。

**依赖**：需要模块 C 的输出信号。若模块 C 进度延迟，可先用简单 MA 交叉信号做临时占位，但最终报告须替换为分类模型信号。

---

### 模块 E：强化学习 + 长期收益

**产出**：RL notebook + 训练曲线 + 长期收益对比。

**具体内容**：
- 环境构建：基于标普500日线，状态为近期收益率 + 技术指标窗口，动作为 {空仓, 满仓} 或 {0%, 50%, 100%}。
- 算法：DQN 或 PPO（使用 stable-baselines3 降低实现成本）。
- 奖励设计：日收益率，或夏普比率的滚动近似。若时间有限，采用日收益率即可。
- 评价指标：**累计收益、夏普比率、最大回撤**，与 Buy & Hold 及模块 D 的策略对比。
- 诚实报告：RL 在单一股票指数上的训练容易过拟合，需在验证集上早停，并报告测试集结果而非训练集结果。

**简化建议**：若 RL 实现困难，可降级为“基于 Q-learning 的离散状态-动作表格方法”，状态离散化为（趋势 up/down × 波动率 high/low），动作 {持有/空仓}，保证可运行、可解释。

---

## 三、时间线与协作节点

| 阶段 | 时间 | 关键动作 |
|------|------|----------|
| Day 1 | 环境与数据 | 组长建仓库、pyproject.toml、uv sync 验证；成员 1 下载数据并推送 raw 数据 |
| Day 2–3 | 模块并行启动 | 成员 1 完成 EDA 和 features.py；成员 2/3 开始建模；成员 4/5 阅读数据与设计 |
| Day 4–5 | 中期检查 | 各模块提交初版 notebook；统一特征接口；确定最终预测窗口 |
| Day 6–7 | 整合与调优 | 成员 2 输出指定窗口预测；成员 4 用分类信号跑回测；成员 5 完成 RL 训练 |
| Day 8 | 汇总报告 | 全员审阅 notebook；组长整合 README 和最终报告；录制演示 |

---

## 四、评价指标汇总

| 模块 | 指标 |
|------|------|
| 预测 | RMSE、MAE、R² |
| 分类 | Accuracy、Precision、Recall、F1、AUC |
| 策略 | 年化收益、夏普比率、最大回撤、胜率 |
| 强化学习 | 累计收益、夏普比率、最大回撤 |
| 最终预测 | 指定窗口的 RMSE / MAE（若有真实值）/ 方向正确率 |

---

## 五、风险与应对

- **数据不足 30 年**：若免费源仅覆盖较短区间，用 SPY 替代并说明；或使用多源拼接（需对齐复权方式）。
- **预测 R² 过低**：股票价格预测 R² 低于 0.5 是常见的，不必强行美化；重点讨论模型是否优于 naive 基准。
- **RL 不收敛**：降级为 Q-table 或直接以“策略模块”的监督学习回测作为长期收益分析，RL 作为扩展尝试报告。
- **GitHub 协作冲突**：每人只在个人分支修改自己负责的 notebook 和 `src/` 中自己的文件；共享文件（如 `features.py`）由成员 1 统一维护，其他人通过 PR 提修改。

---

## 六、模块 A 接口说明（`src/0_DataPreprocessing/`）

### 1. 数据预处理 `DataPreprocessing.py`

```python
from DataPreprocessing import DataPreprocessor

prep = DataPreprocessor()        # 默认读取 data/SP500_daily.csv
prep.prepare()                   # 去重、缺失值、异常 OHLC、零成交量、交易日对齐
df = prep.data                   # Date 索引 + OHLCV + ret / log_ret / cum_log_ret / drawdown ...
splits = prep.split()            # {"train": …, "val": …, "test": …}，按时间顺序切分，不打乱
prep.split_table()               # 各集合起止日期、样本量与占比
prep.export_processed()          # 导出 data/processed/sp500_clean.csv（附 segment 列）
```

- 交易日对齐：以工作日日历为基准，≤4 个工作日的缺口视为节假日休市（不填充），
  更长的缺口视为数据断档，前向填充并打 `is_filled` 标记。
- 生成 5 张英文标注的 EDA 图到 `src/0_DataPreprocessing/Outcome/`：
  价量时序、收益率分布、滚动波动率、年月收益热力图、最大回撤。
- 命令行：`python src/0_DataPreprocessing/DataPreprocessing.py`。

### 2. 特征工程 `DataDescription.py`

```python
from DataDescription import FeatureEngineer

fe = FeatureEngineer.from_csv("data/processed/sp500_clean.csv")
fe.ma(20)                   # ma_20 / ma_ratio_20
fe.ema(12)                  # ema_12
fe.rsi(14)                  # rsi_14
fe.macd()                   # macd / macd_signal / macd_hist
fe.bollinger(20, 2.0)       # bb_mid_20 / bb_upper_20 / bb_lower_20 / bb_pctb_20 / bb_width_20
fe.rolling_volatility(21)   # vol_21_ann
fe.lagged_returns((1, 2, 3))  # log_ret_lag_1 … log_ret_lag_3
features = fe.build()       # 一体化特征矩阵（默认 29 个特征，已剔除预热期）
```

- 也兼容函数式调用：`build_features(df)`（即 README 中约定的入口）。
- 命令行：`python src/0_DataPreprocessing/DataDescription.py`，输出 `data/processed/sp500_features.csv`。
- 每个方法都接受 `series` 参数，可脱离 `self.data` 单独计算，例如 `fe.rsi(14, other_close)`。

