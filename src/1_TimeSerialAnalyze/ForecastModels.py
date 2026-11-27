"""Train XGBoost and LSTM models and predict S&P 500 closing prices."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import exchange_calendars as xcals
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


SHORT_HORIZON, LONG_HORIZON = 5, 20
FIELDS = ["Date", "Close", "segment"]
FEATURES = [
    "ma_ratio_5", "ma_ratio_20", "rsi_14", "bb_pctb_20", "vol_21_ann",
    "log_ret_lag_1", "log_ret_lag_2", "log_ret_lag_3", "log_ret_lag_4", "log_ret_lag_5",
]
LOOKBACK = 30
POINT_SCALE = 100  # Scale training targets; multiply back before scoring.
ALIGNMENT_VERSION = "XNYS-contiguous-v1"
LSTM_SETTINGS = dict(hidden_size=16, epochs=25, batch_size=128, learning_rate=0.003, seed=42)


def complete_cutoffs(data, lookback=1, horizon=0):
    """Return cutoffs whose input and target rows are consecutive NYSE sessions."""
    dates = pd.DatetimeIndex(data.Date)
    if dates.empty or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be nonempty, unique and increasing.")
    calendar = xcals.get_calendar("XNYS", start=dates[0] - pd.Timedelta(days=14),
                                 end=dates[-1] + pd.Timedelta(days=14))
    positions = calendar.sessions_in_range(dates[0], dates[-1]).get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("Data dates must be NYSE trading sessions.")
    cutoffs = np.arange(lookback - 1, len(data) - horizon)
    span = positions[cutoffs + horizon] - positions[cutoffs - lookback + 1]
    return cutoffs[span == lookback - 1 + horizon]


def predict_xgboost(train, cutoffs, horizon=SHORT_HORIZON):
    """Fit one model per step and return closing prices at all cutoffs."""
    forecasts = []
    for day in range(1, horizon + 1):
        rows = complete_cutoffs(train, horizon=day)
        if not len(rows):
            raise ValueError(f"No complete {day}-day training targets.")
        current = train.iloc[rows]
        target = train.Close.iloc[rows + day].to_numpy() - current.Close.to_numpy()
        model = XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=100, max_depth=2, learning_rate=0.03,
            tree_method="hist", random_state=42, n_jobs=2,
        )
        model.fit(current[FEATURES], target)
        forecasts.append(model.predict(cutoffs[FEATURES]))
    return cutoffs.Close.to_numpy()[:, None] + np.column_stack(forecasts)


class ForecastModel(nn.Module):
    def __init__(self, horizon=SHORT_HORIZON, hidden_size=LSTM_SETTINGS["hidden_size"]):
        super().__init__()
        self.lstm = nn.LSTM(len(FEATURES), hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, horizon)

    def forward(self, inputs):
        sequence, _ = self.lstm(inputs)
        return self.output(sequence[:, -1])


def make_windows(features, cutoffs):
    return torch.from_numpy(np.stack([
        features[i - LOOKBACK + 1 : i + 1] for i in cutoffs
    ]))


def fit_model(train, horizon=SHORT_HORIZON):
    """Fit the LSTM using inputs and targets within the given history."""
    torch.manual_seed(LSTM_SETTINGS["seed"])
    torch.set_num_threads(2)
    train_cutoffs = complete_cutoffs(train, LOOKBACK, horizon)
    if not len(train_cutoffs):
        raise ValueError(f"Need a complete {LOOKBACK + horizon}-session training span.")
    features = train[FEATURES].to_numpy(dtype=np.float32)
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-8)
    features = (features - mean) / std

    close = train.Close.to_numpy()
    days = np.arange(1, horizon + 1)
    changes = close[train_cutoffs[:, None] + days] - close[train_cutoffs, None]
    labels = torch.tensor(changes / POINT_SCALE, dtype=torch.float32)
    batches = DataLoader(TensorDataset(make_windows(features, train_cutoffs), labels),
                         batch_size=LSTM_SETTINGS["batch_size"])

    model = ForecastModel(horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_SETTINGS["learning_rate"])
    loss = nn.L1Loss()
    for _ in range(LSTM_SETTINGS["epochs"]):
        for inputs, targets in batches:
            optimizer.zero_grad()
            loss(model(inputs), targets).backward()
            optimizer.step()
    return model, mean, std


def predict_closes(model, mean, std, history, cutoffs):
    cutoffs = np.asarray(cutoffs, dtype=int)
    if not np.isin(cutoffs, complete_cutoffs(history, LOOKBACK)).all():
        raise ValueError(f"Each prediction requires {LOOKBACK} consecutive NYSE input sessions.")
    features = (history[FEATURES].to_numpy(dtype=np.float32) - mean) / std
    close = history.Close.to_numpy()
    model.eval()
    with torch.no_grad():
        changes = model(make_windows(features, cutoffs)).numpy() * POINT_SCALE
    return close[cutoffs, None] + changes


@contextmanager
def atomic_output(path):
    """Replace a destination only after its new contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_model(fit, history, path):
    model, mean, std = fit
    checkpoint = {
        "state_dict": model.state_dict(),
        "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
        "features": FEATURES, "lookback": LOOKBACK, "point_scale": POINT_SCALE,
        "horizon": model.output.out_features,
        "train_start": str(history.Date.iloc[0])[:10],
        "cutoff": str(history.Date.iloc[-1])[:10],
        "settings": dict(LSTM_SETTINGS, hidden_size=model.lstm.hidden_size),
        "torch_version": str(torch.__version__),
        "calendar_alignment": ALIGNMENT_VERSION,
    }
    with atomic_output(path) as temporary:
        torch.save(checkpoint, temporary)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("calendar_alignment") != ALIGNMENT_VERSION:
        raise ValueError("Saved model predates trading-day alignment; retrain it.")
    if (checkpoint["features"] != FEATURES or checkpoint["lookback"] != LOOKBACK
            or checkpoint["point_scale"] != POINT_SCALE):
        raise ValueError("Saved features or input scaling differ from the current model.")
    torch.set_num_threads(2)
    model = ForecastModel(checkpoint["horizon"], checkpoint["settings"]["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    fit = model, checkpoint["mean"].numpy(), checkpoint["std"].numpy()
    return fit, checkpoint
