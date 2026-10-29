"""Part 5 submission: next-day 12:00 PM2.5 forecast for all 468 Bihar stations.

The model is a ridge regression on log-scale features, fitted in the notebook. Its numbers
(scaler means and scales, coefficients, intercept, bias correction) are stored in model.json.
Here we only load them and compute features. Nothing is refitted.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))         # so features.py is found however the grader imports this file
from features import neighbour_sets, my_features

# Loaded once, when the grader imports this file
MODEL = json.loads((HERE / "model.json").read_text())
MEAN, SCALE, COEF = (np.array(MODEL[k]) for k in ("mean", "scale", "coef"))
STATIONS = pd.read_csv(HERE / "bihar_stations.csv", dtype={"sensor_id": str}).set_index("sensor_id")
NBRS = neighbour_sets(STATIONS)


def forecast(history: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """history: wide hourly table (DatetimeIndex 'timestamp', one string column per sensor_id),
    every hour from 1 Jan 2024 00:00 up to and including issue_time.
    issue_time: 12:00 on some day (IST).
    Returns predicted PM2.5 for issue_time + 24 h, indexed by sensor_id (all 468 stations)."""
    history = history.loc[:issue_time, STATIONS.index]   # training column order; nothing after t
    history = history.ffill()                             # safety: fill any gap with the last reading
    A = history.to_numpy(dtype=float)
    p = len(A) - 1                                        # row of the issue time

    X = my_features(A, p, issue_time, NBRS)[MODEL["columns"]].to_numpy()
    log_change = ((X - MEAN) / SCALE) @ COEF + MODEL["intercept"] + MODEL["smear"]
    prediction = np.expm1(np.log1p(np.clip(A[p], 0, None)) + log_change)

    prediction = np.nan_to_num(prediction, nan=np.nanmedian(A[p]))   # safety: never return NaN
    return pd.Series(np.clip(prediction, 0, None), index=STATIONS.index, name="pm25_forecast")
