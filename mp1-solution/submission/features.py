"""Feature code shared by the notebook and forecast.py.

Both functions below build the features for ONE issue time:
    A  : numpy array (hours x 468 stations) with the hourly PM2.5 history
    p  : row number of the issue time in A
    t  : the issue time itself (a pd.Timestamp)
They only ever read rows A[:p + 1], i.e. readings at or before the issue time, so no future data
can leak into a forecast.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------
# Station geometry
# ---------------------------------------------------------------------------------------------
def station_distances_km(stations):
    """468 x 468 matrix of distances, using the km conversion given in the assignment."""
    x = (stations["longitude"].to_numpy() - 85.5) * 100.3
    y = (stations["latitude"].to_numpy() - 25.7) * 110.6
    return np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)


def neighbour_sets(stations):
    """For each station: its 10 nearest other stations, and the stations within 15 km and 40 km."""
    d = station_distances_km(stations)
    np.fill_diagonal(d, np.inf)                  # a station is not its own neighbour
    return {
        "nearest10": np.argsort(d, axis=1)[:, :10],     # shape (468, 10), indices
        "within15": d <= 15,                            # shape (468, 468), True/False
        "within40": d <= 40,
    }


def season_terms(t):
    """sin and cos of the day of year, with d = days since 1 Jan 2024."""
    d = (t.normalize() - pd.Timestamp("2024-01-01")).days
    return np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365)


def mean_over(mask, values):
    """For each station, the mean of `values` over the stations marked True in its row of `mask`."""
    return (mask @ values) / np.maximum(mask.sum(axis=1), 1)


# ---------------------------------------------------------------------------------------------
# Part 1: the fixed 29-feature baseline
# ---------------------------------------------------------------------------------------------
def baseline_features(A, p, t, nbrs):
    last24 = A[p - 23:p + 1]                         # rows t-23h ... t
    near = nbrs["nearest10"]
    sin_d, cos_d = season_terms(t)

    X = pd.DataFrame({f"lag_{k}": A[p - k] for k in range(24)})   # lag_0 = reading at t
    X["nbr10_now"] = A[p][near].mean(axis=1)
    X["nbr10_24h"] = last24[:, near].mean(axis=(0, 2))
    X["network_now"] = A[p].mean()
    X["sin_doy"] = sin_d
    X["cos_doy"] = cos_d
    return X


# ---------------------------------------------------------------------------------------------
# Part 2: our features, organised in groups. Values are log(1 + PM2.5), because pollution
# changes multiplicatively (a bad day is "twice as bad", not "+50").
# ---------------------------------------------------------------------------------------------
def recent_hours(L, p, t, nbrs):
    """The station's last 24 hourly readings."""
    return {f"log_lag_{k}": L[p - k] for k in range(24)}


def neighbourhood(L, p, t, nbrs):
    """What the surrounding area looks like now and over the last 24 hours."""
    now, day = L[p], L[p - 23:p + 1].mean(axis=0)
    return {
        "nbr10_now": now[nbrs["nearest10"]].mean(axis=1),
        "nbr10_day": day[nbrs["nearest10"]].mean(axis=1),
        "r15km_now": mean_over(nbrs["within15"], now),
        "r15km_day": mean_over(nbrs["within15"], day),
        "r40km_now": mean_over(nbrs["within40"], now),
        "r40km_day": mean_over(nbrs["within40"], day),
    }


def network(L, p, t, nbrs):
    """State-wide pollution level and how it is changing (same value for every station)."""
    return {
        "net_now": L[p].mean(),
        "net_day": L[p - 23:p + 1].mean(),
        "net_week": L[p - 167:p + 1].mean(),
        "net_change_24h": L[p].mean() - L[p - 24].mean(),
    }


def calendar(L, p, t, nbrs):
    sin_d, cos_d = season_terms(t)
    return {"sin_doy": sin_d, "cos_doy": cos_d, "weekend": float(t.dayofweek >= 5)}


def station_level(L, p, t, nbrs):
    """The station's normal level over the last 30 days (fewer at the very start of the data)."""
    month = L[max(p - 719, 0):p + 1]
    return {"level_30d": month.mean(axis=0), "level_30d_vs_state": month.mean(axis=0) - month.mean()}


# More ideas we tried. The notebook (Part 2) decides on validation which groups to keep.
def past_days(L, p, t, nbrs):
    """Longer memory: the same hour on each of the previous 6 days, and the last 7 daily means."""
    f = {f"same_hour_d-{k}": L[p - 24 * k] for k in range(1, 7)}
    for k in range(7):
        day = L[p - 24 * k - 23:p - 24 * k + 1]      # the 24 hours ending k days before t
        f[f"day_mean_d-{k}"] = day.mean(axis=0)
    return f


def trends(L, p, t, nbrs):
    """Short-term changes at the station."""
    return {
        "change_3h": L[p] - L[p - 3],
        "change_6h": L[p] - L[p - 6],
        "change_24h": L[p] - L[p - 24],
    }


def variability(L, p, t, nbrs):
    """How unsettled the air is: the station's ups and downs over the last 24 hours, and how much
    its 10 nearest neighbours changed over the last 24 hours."""
    near = nbrs["nearest10"]
    return {
        "station_std_24h": L[p - 23:p + 1].std(axis=0),
        "nbr10_change_24h": L[p][near].mean(axis=1) - L[p - 24][near].mean(axis=1),
    }


FEATURE_GROUPS = {
    "recent_hours": recent_hours,
    "neighbourhood": neighbourhood,
    "network": network,
    "calendar": calendar,
    "station_level": station_level,
    "past_days": past_days,
    "trends": trends,
    "variability": variability,
}
FINAL_GROUPS = ["recent_hours", "neighbourhood", "network", "calendar", "station_level", "variability"]
HISTORY_NEEDED = 168          # hours of history the features need (one week)


def my_features(A, p, t, nbrs, groups=FINAL_GROUPS):
    L = np.log1p(np.clip(A[:p + 1], 0, None))        # only rows up to the issue time
    columns = {}
    for name in groups:
        columns.update(FEATURE_GROUPS[name](L, p, t, nbrs))
    return pd.DataFrame(columns, index=range(A.shape[1]))
