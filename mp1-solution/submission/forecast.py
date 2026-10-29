"""Part 5 submission: next-day 12:00 PM2.5 forecast for all 468 Bihar stations.

The model (a ridge regression on log-scale features) was fitted in the notebook and saved to
model.pkl. Here we only load it and compute features. Nothing is refitted.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import neighbour_sets, my_features

HERE = Path(__file__).resolve().parent

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
    A = history.loc[:issue_time].to_numpy(dtype=float)
    p = len(A) - 1                                    # row of the issue time

    X = my_features(A, p, issue_time, NBRS)
    log_change = SAVED["model"].predict(X[SAVED["columns"]]) + SAVED["smear"]
    prediction = np.expm1(np.log1p(np.clip(A[p], 0, None)) + log_change)

    return pd.Series(np.clip(prediction, 0, None), index=STATIONS.index, name="pm25_forecast")
