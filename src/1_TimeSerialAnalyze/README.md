# 时间序列预测模块

使用团队已有的标普 500 日线和特征表，在截止日收盘后，一次预测未来 5 个或 20 个交易日的每日收盘点位。比较 Naive、XGBoost 和 LSTM，课堂预测采用 LSTM。

完整实验说明见[时间序列分析报告](../../reports/1_TimeSerialAnalyze/时间序列分析报告.md)，从头运行可使用 [TimeSeriesAnalysis.ipynb](TimeSeriesAnalysis.ipynb)。

## 数据与预测方式

需要保留项目中的以下两个文件，由数据预处理模块提供：

- [sp500_clean.csv](../../data/processed/sp500_clean.csv)：核对交易日期、真实点位和数据集划分。
- [sp500_features.csv](../../data/processed/sp500_features.csv)：提供模型使用的 10 个特征，包括均线偏离度、RSI、布林带位置、波动率和滞后收益率。

特征表有 7,347 条记录，最新日期为 **2026-11-13**。本模块直接读取现成数据，不重新执行预处理。

- Naive：未来各天均使用截止日的收盘点位。
- XGBoost：输入截止日的 10 个特征，每个步长训练一个回归模型。
- LSTM：输入连续 30 个交易日的 10 个特征，一次输出整个窗口。

两种学习模型的训练目标均为**目标日点位减去截止日点位**，预测后加回截止日点位。当前版本使用点位变化目标。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| [TimeSeriesAnalysis.ipynb](TimeSeriesAnalysis.ipynb) | 主入口：读取数据、训练、验证测试、绘图和课堂演示预测。 |
| [ForecastModels.py](ForecastModels.py) | 特征清单、XGBoost/LSTM、模型保存与加载。 |
| [EvaluateForecastMetrics.py](EvaluateForecastMetrics.py) | 安排验证或测试窗口，比较三种方法的 MAE、RMSE、R²。 |
| [EvaluatePercentageErrors.py](EvaluatePercentageErrors.py) | 在原测试窗口上计算 Naive 与 LSTM 的整体和逐步长 MAPE。 |
| [ClassroomForecast.py](ClassroomForecast.py) | 输入日期后训练或加载 LSTM，保存模型并导出逐日预测 CSV。 |
| [PlotForecastMAE.py](PlotForecastMAE.py) | 使用已记录的 MAE 绘制改善比例柱状图和逐步长热力图。 |
| [PlotPresentationFigures.py](PlotPresentationFigures.py) | 生成数据趋势、模型结构和评估时间轴三张配图。 |
| [tests/test_calendar_alignment.py](tests/test_calendar_alignment.py) | 六项日期对齐与模型版本回归检查。 |
| [models/](models/) | 保存课堂预测模型；当前已有一个 5 日演示模型。 |

## 运行方法

本次实验环境为 Python 3.12，主要依赖 NumPy、pandas、PyTorch、XGBoost、Matplotlib 和 exchange-calendars。运行 Notebook 还需要 Jupyter 内核。请选择已经安装这些依赖的 Python 解释器或内核；脚本使用 CPU 训练。

所有终端命令均在**项目根目录**执行，即 `data/`、`src/`、`reports/` 所在目录。推荐先打开 Notebook，从上到下运行；20 日实验需要逐窗口重训，耗时较长。

### 验证与测试

```powershell
python src/1_TimeSerialAnalyze/EvaluateForecastMetrics.py --horizon 5 --split test
python src/1_TimeSerialAnalyze/EvaluateForecastMetrics.py --horizon 20 --split test
```

将 `--split test` 改为 `--split val` 可运行验证集。增加 `--output reports/1_TimeSerialAnalyze/metrics.json` 可保存整体及逐步长指标。

计算原测试窗口的百分比误差：

```powershell
python src/1_TimeSerialAnalyze/EvaluatePercentageErrors.py
```

结果保存到 [test_percentage_errors.json](../../reports/1_TimeSerialAnalyze/test_percentage_errors.json)。该脚本按日期对齐后的有效测试窗口计算，当前为 219 个五日窗口和 54 个二十日窗口。中间预测缓存写入项目同级的 `tmp/sp500-percentage-errors/`，相同数据、模型代码和 PyTorch 版本下可复用。

### 生成图片

```powershell
python src/1_TimeSerialAnalyze/PlotForecastMAE.py
python src/1_TimeSerialAnalyze/PlotPresentationFigures.py
```

图片保存到 [reports/1_TimeSerialAnalyze/](../../reports/1_TimeSerialAnalyze/)。Notebook 中的误差图使用当次运行结果，独立误差绘图脚本使用代码中保存的历史 MAE。中文图使用微软雅黑或黑体。

### 课堂预测

```powershell
python src/1_TimeSerialAnalyze/ClassroomForecast.py 2026-11-13 2026-11-16 2026-11-20
```

三个日期依次为截止日、窗口开始日、窗口结束日。不填写日期参数时，程序在控制台逐项询问。日期按 NYSE 交易日历安排，休市日截止日期回退至此前最近的交易日。

默认使用截止日及以前的全部历史重新训练，并保存：

- 模型：`src/1_TimeSerialAnalyze/models/lstm_截止日_步长day.pt`，含权重、标准化参数、特征清单和训练配置。
- [预测 CSV](../../reports/1_TimeSerialAnalyze/forecast_window.csv)：`date`、`step`、`predicted_close` 三列；步长从截止日后的首个交易日起算。

使用 `--model 模型路径` 可加载已有模型，`--output CSV路径` 可另存预测文件。当前日期示例及 CSV 均为 **5 日演示**；正式使用时替换为老师指定的日期，并确保特征表包含截止日记录。

## 已有实验结果

训练、验证、测试按时间顺序划分，评估窗口的目标日期互不重叠：

- **5 日**：原训练集拟合后固定模型，验证 226 个窗口、测试 219 个窗口。
- **20 日**：每个截止日用已有历史从头训练，模型设置固定，验证 56 个窗口、测试 54 个窗口。

下表为测试期的整体指标，均对所有窗口、全部步长计算平均误差。

| 预测长度 | Naive MAE（点） | XGBoost MAE（点） | LSTM MAE（点） | Naive MAPE | LSTM MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 日 | 68.31 | 68.01 | **67.74** | 1.300% | **1.291%** |
| 20 日 | 119.56 | 117.62 | **116.97** | 2.282% | **2.237%** |

MAPE 是每条绝对误差除以对应真实点位后取平均。LSTM 在当前实验中的整体 MAE 最低，相对 Naive 有小幅改善，部分步长仍由其他方法领先。完整 MAE、RMSE、R² 和图表见实验报告。

## 日期对齐修正与待完成项

- 已修正缺失日对训练标签和 LSTM 输入的影响：按 NYSE 交易日历排除不连续跨度，三种方法使用同一批有效评估窗口。原始数据不补填，参数保持不变，模型与结果已重新生成。
- 本地尚无 `pyproject.toml` 和 `uv.lock`，课程要求的 `uv sync` 环境需要在团队仓库中确认或补齐。
- 老师指定的正式窗口尚未公布，最终预测 CSV 需在窗口确定后生成；现有测试只覆盖 5 日和 20 日预测。

上传 GitHub 时保留代码、Notebook、报告、图片、结果及演示模型，按团队约定共享输入数据。`__pycache__/`、虚拟环境和临时缓存不纳入版本控制。

日期对齐检查：

```powershell
python -m unittest discover -s src/1_TimeSerialAnalyze/tests -v
```
