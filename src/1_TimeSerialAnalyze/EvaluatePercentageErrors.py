"""Compare LSTM and Naive percentage errors on calendar-aligned test windows."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from EvaluateForecastMetrics import evaluation_windows, load_data, metrics
from ForecastModels import atomic_output, fit_model, predict_closes


ROOT = Path(__file__).resolve().parents[2]


def load_worker_data():
    global worker_data
    worker_data, _ = load_data()


def predict_cutoff(cutoff):
    history = worker_data.iloc[:int(cutoff) + 1]
    fit = fit_model(history, 20)
    return predict_closes(*fit, history, [int(cutoff)])[0]


def save_predictions(path, windows, predictions):
    with atomic_output(path) as temporary:
        with temporary.open("wb") as file:
            np.savez_compressed(file, windows=windows, predictions=np.asarray(predictions))


def main():
    digest = hashlib.sha256()
    for name in ("data/processed/sp500_features.csv", "data/processed/sp500_clean.csv",
                 "src/1_TimeSerialAnalyze/ForecastModels.py",
                 "src/1_TimeSerialAnalyze/EvaluateForecastMetrics.py"):
        digest.update((ROOT / name).read_bytes())
    digest.update(str(torch.__version__).encode())
    fingerprint = digest.hexdigest()
    cache = ROOT.parent / "tmp/sp500-percentage-errors" / fingerprint[:12]
    cache.mkdir(parents=True, exist_ok=True)
    data, clean = load_data()
    close = data.Close.to_numpy()
    output = ROOT / "reports/1_TimeSerialAnalyze/test_percentage_errors.json"
    summary = {"definition": "MAPE = mean(abs(predicted - actual) / abs(actual)) * 100",
               "split": "test", "data_and_source_sha256": fingerprint, "experiments": {}}

    for horizon in (5, 20):
        windows = evaluation_windows(data, clean, horizon, "test")
        count = len(windows)
        actual = close[windows[:, 1:]]
        assert np.isfinite(actual).all() and (actual > 0).all()
        path = cache / f"{horizon}_test_predictions.npz"
        predictions = []
        if path.exists():
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["windows"], windows)
                predictions = list(saved["predictions"])
        print(f"{horizon} days: {len(predictions)}/{count} cached", flush=True)
        started = time.monotonic()
        if horizon == 5 and not predictions:
            fit = fit_model(data[data.segment == "train"], horizon)
            predictions = list(predict_closes(*fit, data, windows[:, 0]))
            save_predictions(path, windows, predictions)
        elif horizon == 20 and len(predictions) < count:
            # Independent processes preserve each fit's original seed and settings.
            with ProcessPoolExecutor(max_workers=4, initializer=load_worker_data) as pool:
                pending = windows[len(predictions):, 0].tolist()
                for prediction in pool.map(predict_cutoff, pending):
                    predictions.append(prediction)
                    save_predictions(path, windows, predictions)
                    print(f"20 days: {len(predictions)}/{count}; {time.monotonic() - started:.0f}s", flush=True)

        predicted = np.asarray(predictions)
        assert predicted.shape == actual.shape
        models = {"Naive": np.broadcast_to(close[windows[:, 0], None], actual.shape), "LSTM": predicted}
        model_results = {}
        for name, values in models.items():
            ape = np.abs(values - actual) / np.abs(actual) * 100
            model_results[name] = {
                **metrics(actual, values, axis=None), "mape_percent": float(ape.mean()),
                "median_ape_percent": float(np.median(ape)),
                "p90_ape_percent": float(np.percentile(ape, 90)),
                "by_step_mape_percent": ape.mean(axis=0).tolist(),
            }
            print(horizon, name, {k: v for k, v in model_results[name].items()
                                  if k != "by_step_mape_percent"}, flush=True)
        naive = model_results["Naive"]["mape_percent"]
        lstm = model_results["LSTM"]["mape_percent"]
        summary["experiments"][str(horizon)] = {
            "windows": count, "prediction_count": int(actual.size),
            "training": "original train only" if horizon == 5 else "expanding history refitted at every cutoff",
            "models": model_results, "mape_reduction_percentage_points": naive - lstm,
            "mape_relative_reduction_percent": (naive - lstm) / naive * 100,
        }
        with atomic_output(output) as temporary:
            temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
