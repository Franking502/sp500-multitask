"""Shared data loading, forecast windows and evaluation for the Notebook and CLI."""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ForecastModels import (
    FEATURES, FIELDS, LONG_HORIZON, SHORT_HORIZON, atomic_output,
    LOOKBACK, complete_cutoffs, fit_model, predict_closes, predict_xgboost,
)


def load_data(folder=None):
    folder = Path(folder) if folder is not None else Path(__file__).resolve().parents[2] / "data/processed"
    data = pd.read_csv(folder / "sp500_features.csv", usecols=FIELDS + FEATURES, parse_dates=["Date"])
    clean = pd.read_csv(folder / "sp500_clean.csv", usecols=FIELDS, parse_dates=["Date"])
    if not data.Date.is_monotonic_increasing or not data.Date.is_unique:
        raise ValueError("Feature dates must be unique and increasing.")
    return data, clean


def evaluation_windows(data, clean, horizon, split):
    previous = "train" if split == "val" else "val"
    first = data.index[data.segment == previous][-1]
    last = data.index[data.segment == split][-1]
    clean_rows = pd.concat([clean[clean.segment == previous].tail(1), clean[clean.segment == split]])
    if not data.loc[first:last, FIELDS].reset_index(drop=True).equals(
        clean_rows[FIELDS].reset_index(drop=True)
    ):
        raise ValueError("Feature and clean tables must agree on evaluation dates and closes.")

    dates = pd.DatetimeIndex(data.Date)
    start, end = dates[first], dates[last]
    buffer = pd.Timedelta(days=14)
    calendar = xcals.get_calendar("XNYS", start=start - buffer, end=end + buffer)
    sessions = calendar.sessions_in_range(start, end)
    starts = np.arange(0, len(sessions) - horizon, horizon)
    schedule = sessions.to_numpy()[starts[:, None] + np.arange(horizon + 1)]
    rows = dates.get_indexer(schedule.ravel()).reshape(schedule.shape)
    windows = rows[(rows >= 0).all(axis=1)]
    windows = windows[np.isin(windows[:, 0], complete_cutoffs(data, LOOKBACK))]
    if not len(windows) or not (data.segment.to_numpy()[windows[:, 1:]] == split).all():
        raise ValueError(f"Need complete {horizon}-day windows inside {split}.")
    return windows


def load_windows(horizon, split):
    data, clean = load_data()
    windows = evaluation_windows(data, clean, horizon, split)
    return data.loc[:data.index[data.segment == split][-1]], windows


def forecast_cutoff(data, cutoff, horizon):
    history = data.iloc[:int(cutoff) + 1]
    xgb = predict_xgboost(history, history.tail(1), horizon)[0]
    fit = fit_model(history, horizon)
    lstm = predict_closes(*fit, history, [int(cutoff)])[0]
    return xgb, lstm


def evaluate(data, clean, horizon, split, lstm_5=None, workers=4):
    windows = evaluation_windows(data, clean, horizon, split)
    cutoffs = windows[:, 0]
    close = data.Close.to_numpy()
    print(f"{horizon} 日 {split}：{len(windows)} 个窗口", file=sys.stderr, flush=True)
    if horizon == SHORT_HORIZON:
        train = data[data.segment == "train"]
        fit = fit_model(train) if lstm_5 is None else lstm_5
        forecasts = {
            "XGBoost": predict_xgboost(train, data.iloc[cutoffs], horizon),
            "LSTM": predict_closes(*fit, data, cutoffs),
        }
    else:
        forecasts = {"XGBoost": [], "LSTM": []}
        # Each worker independently fits the same models with the same fixed seeds.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = pool.map(forecast_cutoff, repeat(data), cutoffs, repeat(horizon))
            for number, (xgb, lstm) in enumerate(jobs, 1):
                forecasts["XGBoost"].append(xgb)
                forecasts["LSTM"].append(lstm)
                if number % 10 == 0 or number == len(windows):
                    print(f"  已完成 {number}/{len(windows)} 个窗口", file=sys.stderr, flush=True)
        forecasts = {name: np.stack(rows) for name, rows in forecasts.items()}
    return close[windows[:, 1:]], {"Naive": close[cutoffs, None], **forecasts}


def metrics(actual, predicted, axis):
    error = actual - predicted
    square_error = np.square(error).mean(axis=axis)
    variance = np.square(actual - actual.mean(axis=axis, keepdims=True)).mean(axis=axis)
    return {
        "mae": np.abs(error).mean(axis=axis).tolist(),
        "rmse": np.sqrt(square_error).tolist(),
        "r2": (1 - square_error / variance).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=(SHORT_HORIZON, LONG_HORIZON), required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, help="Optional metrics JSON path.")
    args = parser.parse_args()
    data, clean = load_data()
    actual, predictions = evaluate(data, clean, args.horizon, args.split)
    result = {
        "horizon": args.horizon, "split": args.split, "windows": len(actual),
        "models": {
            name: {"overall": metrics(actual, pred, axis=None),
                   "by_step": metrics(actual, pred, axis=0)}
            for name, pred in predictions.items()
        },
    }
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with atomic_output(args.output) as temporary:
            temporary.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
