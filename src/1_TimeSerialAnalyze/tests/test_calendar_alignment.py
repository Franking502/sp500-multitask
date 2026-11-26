"""Regression checks for missing sessions, holidays and old checkpoints."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ForecastModels import (
    FEATURES, ForecastModel, complete_cutoffs, load_model, predict_closes,
    predict_xgboost, save_model,
)


class CalendarAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        calendar = xcals.get_calendar("XNYS", start="2023-01-01", end="2023-10-02")
        cls.sessions = calendar.sessions_in_range("2023-01-03", "2023-10-02")
        cls.full = pd.DataFrame({"Date": cls.sessions, "Close": 1000 + np.arange(len(cls.sessions))})
        for name in FEATURES:
            cls.full[name] = 0.1
        cls.gapped = cls.full[cls.full.Date != "2023-05-24"].reset_index(drop=True)

    def test_holidays_and_weekends_are_not_missing_sessions(self):
        np.testing.assert_array_equal(complete_cutoffs(self.full, 30, 20),
                                      np.arange(29, len(self.full) - 20))

    def test_missing_session_is_excluded_from_input_and_targets(self):
        rows = complete_cutoffs(self.gapped, 30, 20)
        self.assertEqual(len(self.gapped) - 29 - 20 - len(rows), 49)
        for cutoff in rows:
            day = self.sessions.get_loc(self.gapped.Date.iloc[cutoff])
            np.testing.assert_array_equal(self.gapped.Date.iloc[cutoff - 29:cutoff + 21],
                                          self.sessions[day - 29:day + 21])

    def test_xgboost_targets_keep_the_requested_calendar_step(self):
        seen = []

        class Regressor:
            def __init__(self, **kwargs):
                pass

            def fit(self, inputs, targets):
                seen.append(targets)

            def predict(self, inputs):
                return np.zeros(len(inputs))

        with patch("ForecastModels.XGBRegressor", Regressor):
            predict_xgboost(self.gapped, self.gapped.tail(1), 5)
        for day, targets in enumerate(seen, 1):
            np.testing.assert_array_equal(targets, np.full(len(targets), day))

    def test_prediction_rejects_an_incomplete_lookback(self):
        history = self.gapped[self.gapped.Date <= "2023-06-02"]
        with self.assertRaisesRegex(ValueError, "consecutive NYSE"):
            predict_closes(None, np.zeros(10), np.ones(10), history, [len(history) - 1])

    def test_training_never_needs_dates_after_its_history(self):
        history = self.full[self.full.Date <= "2023-05-15"]
        rows = complete_cutoffs(history, 30, 5)
        np.testing.assert_array_equal(rows, np.arange(29, len(history) - 5))
        self.assertLess(int(rows[-1]) + 5, len(history))

    def test_checkpoint_alignment_is_required_and_saved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "model.pt"
            torch.save({"features": FEATURES}, path)
            with self.assertRaisesRegex(ValueError, "predates"):
                load_model(path)
            fit = ForecastModel(), np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32)
            save_model(fit, self.full, path)
            loaded, metadata = load_model(path)
            self.assertEqual(metadata["calendar_alignment"], "XNYS-contiguous-v1")
            for name, value in fit[0].state_dict().items():
                torch.testing.assert_close(value, loaded[0].state_dict()[name])


if __name__ == "__main__":
    unittest.main()
