"""Train or load a classroom LSTM, save its model, and export daily forecasts."""

import argparse
from datetime import date
from pathlib import Path
import sys

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ForecastModels import (
    FEATURES, LONG_HORIZON, SHORT_HORIZON, atomic_output,
    fit_model, load_model, predict_closes, save_model,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_OUTPUT = ROOT / "reports/1_TimeSerialAnalyze/forecast_window.csv"


def forecast_dates(cutoff, start, end):
    if not cutoff < start <= end:
        raise ValueError("Dates must satisfy cutoff < start <= end.")
    buffer = pd.Timedelta(days=14)
    calendar = xcals.get_calendar("XNYS", start=cutoff - buffer, end=end + buffer)
    sessions = calendar.sessions
    origin = sessions[sessions <= cutoff][-1]
    future = sessions[(sessions > origin) & (sessions <= end)]
    targets = [(step, day) for step, day in enumerate(future, 1) if day >= start]
    if not targets:
        raise ValueError("The requested window has no NYSE trading sessions.")
    return origin, targets


def forecast_window(cutoff, start, end, *, model_path=None, output=DEFAULT_OUTPUT):
    origin, targets = forecast_dates(*map(pd.Timestamp, (cutoff, start, end)))
    data = pd.read_csv(ROOT / "data/processed/sp500_features.csv",
                       usecols=["Date", "Close", *FEATURES], parse_dates=["Date"])
    history = data[data.Date <= origin].sort_values("Date").reset_index(drop=True)
    if history.empty or history.Date.iloc[-1] != origin:
        raise ValueError(f"Features for cutoff {origin.date()} are missing. Latest date is {data.Date.max().date()}.")
    horizon = max(SHORT_HORIZON, targets[-1][0])
    checkpoint = Path(model_path) if model_path is not None else MODEL_DIR / f"lstm_{origin:%Y%m%d}_{horizon}day.pt"
    output = Path(output)
    if output.resolve() == checkpoint.resolve():
        raise ValueError("The model and CSV output paths must differ.")

    if model_path is None:
        print(f"按 {origin.date()} 截止的历史训练 {horizon} 日 LSTM。", file=sys.stderr, flush=True)
        if horizon not in (SHORT_HORIZON, LONG_HORIZON):
            print(f"{horizon} 步模型尚无同长度的独立测试结果。", file=sys.stderr)
        fit = fit_model(history, horizon)
        save_model(fit, history, checkpoint)
    else:
        fit, metadata = load_model(checkpoint)
        if pd.Timestamp(metadata["cutoff"]) > origin:
            raise ValueError("The saved model was trained beyond the requested cutoff.")
        if metadata["horizon"] < targets[-1][0]:
            raise ValueError("The saved model has too few forecast steps for this window.")

    prices = predict_closes(*fit, history, [len(history) - 1])[0]
    steps = np.array([step for step, day in targets], dtype=np.int64)
    forecast = pd.DataFrame({
        "date": [str(day.date()) for step, day in targets],
        "step": steps, "predicted_close": prices[steps - 1].round(2),
    })
    with atomic_output(output) as temporary:
        forecast.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.2f")
    return forecast, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cutoff", "start", "end"):
        parser.add_argument(name, nargs="?", metavar=name.upper(), help="YYYY-MM-DD")
    parser.add_argument("--model", type=Path, help="Load an existing model instead of training.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Forecast CSV path.")
    args = parser.parse_args()
    try:
        dates = [date.fromisoformat(value or input(f"{name} (YYYY-MM-DD): ").strip())
                 for name, value in (("cutoff", args.cutoff), ("start", args.start), ("end", args.end))]
        forecast, checkpoint = forecast_window(*dates, model_path=args.model, output=args.output)
    except (ValueError, EOFError, OSError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(forecast.to_csv(index=False, float_format="%.2f", lineterminator="\n"), end="")
    print(f"模型文件 {checkpoint}\n预测 CSV {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
