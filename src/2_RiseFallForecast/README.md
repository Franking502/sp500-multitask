# Next-day S&P 500 direction

This module predicts whether the index will close higher on the next trading day using three features available after the current close: the lagged daily log return, the deviation from the 20-day moving average, and 21-day annualized volatility. Run the Python script for the group project or open the notebook to follow each step.

## Run the full workflow

From the repository root, run:

```sh
uv run --locked --project src/2_RiseFallForecast python src/2_RiseFallForecast/RiseFallForecast.py
```

The script finds `data/processed/sp500_features.csv` relative to its own location. It writes the metrics, figures, selected model and signals under `reports/2_RiseFallForecast/`. The final handoff file is `reports/2_RiseFallForecast/signals/test_signals.csv`. To supply another feature CSV or results directory, pass `--input` or `--output-dir` after the script name.

For a group runner stored at the repository root, call the same command from Python. `check=True` stops the runner if this module fails, before the backtest reads the signals.

```python
from pathlib import Path
import subprocess

repo_root = Path(__file__).resolve().parent
module = "src/2_RiseFallForecast"
subprocess.run(
    ["uv", "run", "--locked", "--project", module, "python", f"{module}/RiseFallForecast.py"],
    cwd=repo_root,
    check=True,
)
signals_file = repo_root / "reports/2_RiseFallForecast/signals/test_signals.csv"
```

## Run the notebook

From the repository root, run the following commands with `uv` installed. The module uses Python 3.12.

```sh
cd src/2_RiseFallForecast
uv sync --locked
uv run jupyter lab rise_fall_forecast.ipynb
```

In JupyterLab, choose **Restart Kernel and Run All Cells**. The notebook reads [the shared feature table](../../data/processed/sp500_features.csv) and uses its chronological `segment` assignments. Within each segment, the label is 1 if the next close is higher and 0 if it is equal or lower. The last row of each segment is dropped because its next close lies outside that segment.

## Model selection and validation

The majority-class baseline always predicts the most common training label. Logistic regression (with a standard scaler), random forest, and XGBoost use the same training rows and features. Each model is evaluated on the validation segment with a classification report, including precision, recall and F1 for each class, and a confusion matrix. The model with the highest validation ROC AUC, calculated from its predicted probabilities of an up day, is selected. For logistic regression, the fitted scaler and classifier stay together in one pipeline.

## Held-out results

Validation ROC AUC selected logistic regression (0.5009, compared with 0.5000 for the constant baseline). On the test period the notebook applies the models as fitted on the training period, keeping the same features and classification rule. The [test metrics](../../reports/2_RiseFallForecast/metrics/test_metrics.csv) cover 1,131 forecasts.

| Model | Test accuracy | Precision for up | ROC AUC |
| --- | ---: | ---: | ---: |
| Always up | 53.58% | 53.58% | 0.5000 |
| Logistic regression (selected) | 53.58% | 53.75% | 0.5125 |
| Random forest | 54.38% | 54.24% | 0.5255 |
| XGBoost | 53.23% | 53.70% | 0.5142 |

Logistic regression correctly identified 26 flat or down days and missed 26 up days. It therefore matched the baseline at 606 correct forecasts. These results show limited ability to distinguish next-day directions with the three historical indicators. See the [class reports](../../reports/2_RiseFallForecast/metrics/) and [test confusion matrices](../../reports/2_RiseFallForecast/figures/test_confusion_matrices.png).

## Signals for the backtesting team

Use [test_signals.csv](../../reports/2_RiseFallForecast/signals/test_signals.csv) for the final backtest. It contains one signal per trading day from 2022-05-11 through 2026-11-12, targeting trading days through 2026-11-13. The [selected model](../../reports/2_RiseFallForecast/models/validation_selected.joblib) includes the scaler and accepts the three feature columns in notebook order. The [10-row validation sample](../../reports/2_RiseFallForecast/signals/validation_sample.csv) illustrates the same schema.

The close on the signal date is already known when the signal is generated. A next-open backtest measures performance from that execution price, while the classification label compares consecutive closes.

| Column | Meaning |
| --- | --- |
| `signal_date` | Trading day whose closing data produced the signal. The signal is available after this close. |
| `target_date` | Next trading day. Its close will determine the actual direction. |
| `p_up` | Selected model's estimated probability of a higher close. |
| `pred_up` | Predicted direction: 1 for up, 0 for flat or down. |
