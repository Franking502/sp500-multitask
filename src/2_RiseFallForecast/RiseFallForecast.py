"""Train and evaluate the S&P 500 next-day direction classifiers."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "sp500_features.csv"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "2_RiseFallForecast"
FEATURES = ["log_ret_lag_1", "ma_ratio_20", "vol_21_ann"]


def load_labeled_data(input_file):
    daily = pd.read_csv(input_file, parse_dates=["Date"]).sort_values("Date")
    # Pair dates inside each split so the final training label cannot use validation prices.
    segments = daily.groupby("segment")
    daily["target_date"] = segments["Date"].shift(-1)
    daily["next_close"] = segments["Close"].shift(-1)
    labeled = daily.dropna(subset=["next_close"]).copy()
    labeled["up"] = (labeled["next_close"] > labeled["Close"]).astype(int)
    return labeled


def evaluate(split, data, models, training_labels, output_dir):
    """Score the fitted models on one period and write its metrics and figures."""
    actual = data["up"]
    predictions = pd.DataFrame(index=data.index)
    probabilities = pd.DataFrame(index=data.index)
    predictions["baseline"] = training_labels.value_counts().idxmax()
    probabilities["baseline"] = training_labels.mean()

    figure = Figure(figsize=(11, 3.5), constrained_layout=True)
    axes = figure.subplots(1, len(models))
    for ax, (name, model) in zip(axes, models.items()):
        predictions[name] = model.predict(data[FEATURES])
        probabilities[name] = model.predict_proba(data[FEATURES])[:, 1]
        report = classification_report(
            actual, predictions[name], labels=[0, 1],
            target_names=["not up", "up"], zero_division=0,
        )
        report_path = output_dir / "metrics" / f"{split}_{name}_report.txt"
        report_path.write_text(report, encoding="utf-8")
        ConfusionMatrixDisplay.from_predictions(
            actual, predictions[name], labels=[0, 1], display_labels=["not up", "up"],
            ax=ax, colorbar=False, cmap="Blues",
        )
        ax.set_title(name)
    figure.savefig(output_dir / "figures" / f"{split}_confusion_matrices.png", dpi=150)

    rows = []
    for name in predictions:
        predicted = predictions[name]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(actual, predicted),
            "precision": precision_score(actual, predicted, zero_division=0),
            "recall": recall_score(actual, predicted, zero_division=0),
            "f1": f1_score(actual, predicted, zero_division=0),
            "roc_auc": roc_auc_score(actual, probabilities[name]),
        })
    metrics = pd.DataFrame(rows).set_index("model")
    metrics.to_csv(output_dir / "metrics" / f"{split}_metrics.csv")
    return metrics, predictions, probabilities


def make_signals(data, predictions, probabilities):
    return pd.DataFrame({
        "signal_date": data["Date"],
        "target_date": data["target_date"],
        "p_up": probabilities,
        "pred_up": predictions,
    })


def main(input_file=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT) -> Path:
    """Run the module and return the path to its held-out backtesting signals."""
    output_dir = Path(output_dir)
    labeled = load_labeled_data(input_file)
    train = labeled.loc[labeled["segment"] == "train"]
    val = labeled.loc[labeled["segment"] == "val"]
    test = labeled.loc[labeled["segment"] == "test"]
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression()),
        "forest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        "xgboost": XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42),
    }
    for model in models.values():
        model.fit(train[FEATURES], train["up"])

    for folder in ["metrics", "figures", "models", "signals"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    val_metrics, val_predictions, val_probabilities = evaluate(
        "validation", val, models, train["up"], output_dir
    )
    best_name = val_metrics.loc[list(models), "roc_auc"].idxmax()
    joblib.dump(models[best_name], output_dir / "models" / "validation_selected.joblib")
    sample = make_signals(val, val_predictions[best_name], val_probabilities[best_name])
    sample.head(10).to_csv(output_dir / "signals" / "validation_sample.csv", index=False)

    # Keep the validation choice fixed when evaluating on the held-out period.
    _, test_predictions, test_probabilities = evaluate(
        "test", test, models, train["up"], output_dir
    )
    signals = make_signals(test, test_predictions[best_name], test_probabilities[best_name])
    signal_file = output_dir / "signals" / "test_signals.csv"
    signals.to_csv(signal_file, index=False)
    print(f"Selected by validation ROC AUC: {best_name}")
    print(f"Saved {len(signals)} test signals to {signal_file}")
    return signal_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Shared feature CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Module reports directory")
    args = parser.parse_args()
    main(args.input, args.output_dir)
