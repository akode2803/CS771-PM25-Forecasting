"""Part 5 submission: next-day 12:00 PM2.5 forecast for all 468 Bihar stations.

The model (a ridge regression on log-scale features) was fitted in the notebook and saved to
model.pkl as plain numpy arrays (scaler mean and scale, ridge coefficients and intercept, mixing and bias-correction weights), so it
does not depend on the grader's scikit-learn version. Here we only load it and compute features.
Nothing is refitted.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                         # so features.py is found however we are imported

from features import neighbour_sets, my_features

# Loaded once, when the grader imports this file
SAVED = joblib.load(HERE / "model.pkl")
STATIONS = pd.read_csv(HERE / "bihar_stations.csv", dtype={"sensor_id": str}).set_index("sensor_id")
NBRS = neighbour_sets(STATIONS)


def forecast(history: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """history: wide hourly table (DatetimeIndex 'timestamp', one string column per sensor_id),
    every hour from 1 Jan 2024 00:00 up to and including issue_time.
    issue_time: 12:00 on some day (IST).
    Returns predicted PM2.5 for issue_time + 24 h, indexed by sensor_id (all 468 stations)."""
    history = history[STATIONS.index]                 # make sure columns are in the training order
    # The data has no gaps, but just in case: carry a missing reading forward from the same station
    # (past values only), and if there is none, use that hour's mean over the other stations.
    A = history.loc[:issue_time].ffill().to_numpy(dtype=float)
    if np.isnan(A).any():
        hour_mean = np.nan_to_num(np.nanmean(A, axis=1, keepdims=True))
        A = np.where(np.isnan(A), hour_mean, A)
    p = len(A) - 1                                    # row of the issue time

    def log_forecast(q, t):
        """Ridge forecast (log scale) for 24 h after row q, using only rows up to q."""
        X = my_features(A, q, t, NBRS)[SAVED["columns"]].to_numpy()
        log_change = (X - SAVED["mean"]) / SAVED["scale"] @ SAVED["coef"] + SAVED["intercept"] + SAVED["smear"]
        return np.maximum(np.log1p(np.clip(A[q], 0, None)) + log_change, 0)

    # State-wide bias correction: yesterday's forecast was for the issue time, so its error is known now
    yesterday_error = np.log1p(np.clip(A[p], 0, None)) - log_forecast(p - 24, issue_time - pd.Timedelta(hours=24))
    log_model = log_forecast(p, issue_time) + SAVED["lambda"] * yesterday_error.mean()

    today = np.clip(A[p], 0, None)
    prediction = SAVED["blend"] * np.expm1(log_model) + (1 - SAVED["blend"]) * today      # mix with persistence

    return pd.Series(np.clip(prediction, 0, None), index=STATIONS.index, name="pm25_forecast")
