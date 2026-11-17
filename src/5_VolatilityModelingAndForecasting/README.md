# 5 · 波动率建模与预测

在原有四个维度（时序分析、涨跌预测、策略回测、长期收益）之外，新增的**独立分析模块**：
用 GARCH 族模型预测标普 500 的条件波动率，并把预测波动率用于仓位调整回测。

## 产出

1. 三个模型（GARCH / EGARCH / GJR-GARCH）的参数表与残差诊断；
2. 验证集、测试集上逐日的 1/5/10 步波动率预测序列；
3. RMSE / MAE / QLIKE 对比表（模型 × 步长 × 数据集，含 RV20 naive 基准）；
4. 波动率目标仓位回测：满仓、模型预测调仓、历史 RV20 调仓三条净值曲线；
5. `Outcome/summary.md`：哪个模型误差最小、调仓是否改善风险指标、诚实结论。

## 快速开始

```bash
# 1) 安装依赖（arch 是新增依赖，其余已随项目安装）
pip install -r src/5_VolatilityModelingAndForecasting/requirement.txt

# 2) 全流程（约 35 秒）
python src/5_VolatilityModelingAndForecasting/run_analysis.py

# 常用参数
python src/5_VolatilityModelingAndForecasting/run_analysis.py \
    --refit-every 5 --target-vol 0.12 --w-max 1.0 --cost 0.001
```

```bash
python .../run_analysis.py --no-backtest   # 只做建模、预测与评价
python .../run_analysis.py --no-plots      # 不画图
python .../run_analysis.py --csv path.csv  # 换数据（需含 Date、Close）
```

## 目录

```
5_VolatilityModelingAndForecasting/
├─ vol_config.py       # 路径、切分点、模型规格、回测参数（唯一配置源）
├─ vol_data.py         # 对数/简单收益率、日历切分、前瞻 RV
├─ vol_models.py       # 三模型拟合、参数表、Ljung-Box 诊断
├─ vol_forecast.py     # 滚动预测（重拟合 + 固定参数滤波）
├─ vol_evaluation.py   # RMSE/MAE/QLIKE、MZ 回归、DM 检验
├─ vol_backtest.py     # 波动率目标仓位、成本、绩效指标
├─ vol_plots.py        # 5 张英文标注图
├─ run_analysis.py     # CLI 主入口
├─ requirement.txt     # 依赖清单
└─ Outcome/            # 图表、CSV、summary.md
```

## 数据

- 来源：`data/processed/sp500_clean.csv`（模块 0 `DataPreprocessor.export_processed()` 的产物）；
  若不存在则退回 `data/SP500_daily.csv`。本模块只读数据。
- 收益率：`100 × ln(Close_t / Close_{t-1})`（百分数）。用 % 而不是小数，是因为 arch 的
  优化器在 % 尺度下更稳定，参数也更好读。
- 切分（按日历，与模块 0 的 70/15/15 比例切分不同）：

| 集合 | 区间 | 样本 |
| --- | --- | --- |
| train | 1996-11-15 ~ 2018-12-31 | 5,567 |
| val | 2019-01-02 ~ 2021-12-31 | 757 |
| test | 2022-01-03 ~ 2026-11-13 | 1,221 |

## 模型与滚动预测

| key | 设定 | 分布 | 多步预测 |
| --- | --- | --- | --- |
| `garch` | GARCH(1,1) | normal | 解析解 |
| `egarch` | EGARCH(1,1)，o=1 | studentst | 模拟（2000 路径，固定种子） |
| `gjr` | GJR-GARCH(1,1)，o=1 | studentst | 解析解 |

- **为什么滚动**：GARCH 是条件模型，新数据会更新条件方差，不能一次预测整段。
- **实现**：每个目标日 t 只用 t-1 及以前的数据；每 `refit_every`（默认 20）个交易日重新
  做极大似然估计，其余时间用 `arch` 的 `fix(params)` 只做滤波（毫秒级）。
- **EGARCH 的特殊性**：对数方差结构没有多步解析解，`arch` 会报
  "Analytic forecasts not available for horizon > 1"，因此改用
  `method="simulation"` + 2000 条路径；通过固定 `numpy` 随机种子保证可复现。

## 评价口径

- 真值是**前瞻**已实现波动率：在 t 日预测 t..t+h-1 共 h 天的平均日方差，
  `RV_t = sqrt(mean(r_t² … r_{t+h-1}²)) × sqrt(252)`。
  若用后视 5 日波动率当真值，衡量的是「预测当前水平」而不是「预测未来」。
- 年化 %：`sqrt(方差) × sqrt(252)`；QLIKE 在方差上计算，
  `mean(RV²/σ̂² − ln(RV²/σ̂²) − 1)`，对低估波动惩罚更重。
- 另报 Mincer-Zarnowitz 回归（`RV = a + b·σ̂ + e`）与 Diebold-Mariano 检验
  （以 GARCH 为基准，Newey-West 修正）。

## 回测

```
w_t = min(σ_target / σ̂_{t|t-1}, w_max)      # σ_target = 15%，w_max = 1.0
r_t^strat = w_t · r_t − 0.001 · |w_t − w_{t-1}|
```

- 无前视：σ̂ 以**目标日 t** 为索引且只用 t-1 的信息，因此 `w[t]` 直接乘 `r[t]`，不再 shift。
- 净值用**简单收益率**（`Close_t/Close_{t-1}-1`）复利，模型拟合用对数收益率。
- 交易成本单边 0.1%，另附 0 / 0.2% 的成本敏感性。

## 结果速览（当前数据）

- 测试集 h=1 RMSE：EGARCH 11.96 < GJR 12.12 < GARCH 12.34 < RV20 naive 12.50；
  EGARCH 相对 GARCH 的 DM 检验 p≈0.001（显著）。
- 回测（2022–2026）：满仓年化 9.97%、波动 17.17%、最大回撤 −25.43%、Sharpe 0.64；
  波动率目标（EGARCH）年化 7.47%、波动 12.99%、回撤 −19.42%、Sharpe 0.62。
  → **降波动、降回撤有效，风险调整收益没有提升**；RV20 调仓夏普 0.64，与模型预测几乎相同，
  说明预测的增量价值有限（且日频调仓换手 6.7 次/年，成本拖累 0.67%/年）。

## 依赖

依赖清单见 `requirement.txt`，一条命令安装：

```bash
pip install -r src/5_VolatilityModelingAndForecasting/requirement.txt
```

- `arch==9.0.0`：GARCH / EGARCH / GJR-GARCH 的拟合与预测（本项目新增依赖）；
- `numpy` / `pandas` / `scipy` / `statsmodels` / `matplotlib`：项目已使用；
- 若 arch 未安装，运行时会给出明确提示（`ensure_arch_available()`），不会静默失败。

## 限制

- 只有一段测试期（2022–2026），结论不宜外推到其他市场或时期；
- 交易成本只含手续费，未建模冲击成本与滑点；
- 标准化残差仍有轻微自相关（LB(10) p < 0.05），但平方残差无显著 ARCH 效应；
- EGARCH 多步预测依赖模拟，存在微小模拟误差（固定种子保证可复现）。
