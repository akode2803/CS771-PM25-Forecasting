"""Independent reproduction of the team's validation numbers + cache of datasets for experiments."""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]          # repo root (this file is analysis/nov_critique/)
sys.path.insert(0, str(REPO / "mp1-solution/submission"))
from features import neighbour_sets, baseline_features, my_features, FEATURE_GROUPS, FINAL_GROUPS, HISTORY_NEEDED

OUT = Path(__file__).resolve().parent
DATA = REPO / "mp1-package/data"
W = pd.read_csv(DATA / "bihar_pm25_hourly_jan_nov_2024.csv", index_col="timestamp", parse_dates=True)
W.columns = W.columns.astype(str)
stations = pd.read_csv(DATA / "bihar_stations.csv", dtype={"sensor_id": str}).set_index("sensor_id")
A = W.to_numpy(float); times = W.index; nbrs = neighbour_sets(stations)

def build(feature_fn, targets, lead=24):
    parts = []
    for target in targets:
        p = times.get_loc(target - pd.Timedelta(hours=lead))
        X = feature_fn(A, p, times[p], nbrs)
        X["target"] = A[p + lead]; X["persistence"] = A[p]; X["target_time"] = target; X["station"] = np.arange(A.shape[1])
        parts.append(X)
    return pd.concat(parts, ignore_index=True)

def noon_targets(first_issue_row):
    return [times[r] for r in np.where(times.hour == 12)[0] if r - 24 >= first_issue_row]

TRAIN_END = pd.Timestamp("2024-08-31 23:59")
def split(df): return df[df.target_time <= TRAIN_END], df[df.target_time >= "2024-09-01"]

def scores(df, pred):
    e = pred - df["target"].to_numpy(); m = df["target_time"].dt.month.to_numpy()
    out = {"all": (np.sqrt(np.mean(e**2)), np.mean(np.abs(e)))}
    for k, nm in [(9, "Sep"), (10, "Oct"), (11, "Nov")]:
        out[nm] = (np.sqrt(np.mean(e[m == k]**2)), np.mean(np.abs(e[m == k])))
    return pd.DataFrame(out, index=["RMSE", "MAE"]).round(2)

t0 = time.time()
base = build(baseline_features, noon_targets(23)); btr, bva = split(base)
BC = [c for c in base.columns if c not in ("target", "persistence", "target_time", "station")]
rb = Ridge(alpha=1.0).fit(btr[BC], btr["target"])
p_ridge = np.clip(rb.predict(bva[BC]), 0, None)
data = build(lambda A_, p, t, n: my_features(A_, p, t, n, groups=list(FEATURE_GROUPS)), noon_targets(HISTORY_NEEDED - 1))
tr, va = split(data)
print("built in", round(time.time() - t0, 1), "s; base train", len(btr), "ours train", len(tr), "val", len(va))

L_s = np.log1p(A[:HISTORY_NEEDED])
GROUP_COLS = {g: list(fn(L_s, HISTORY_NEEDED - 1, times[HISTORY_NEEDED - 1], nbrs)) for g, fn in FEATURE_GROUPS.items()}
COLS = [c for g in FINAL_GROUPS for c in GROUP_COLS[g]]

def lct(df): return np.log1p(df["target"]) - np.log1p(df["persistence"])
m = make_pipeline(StandardScaler(), Ridge(alpha=1000)).fit(tr[COLS], lct(tr))
smear = (lct(tr) - m.predict(tr[COLS])).var() / 2
def pr(df): return np.clip(np.expm1(np.log1p(df["persistence"]) + m.predict(df[COLS]) + smear).to_numpy(), 0, None)
both = pd.concat([tr, va])
err = np.log1p(both["target"]) - np.log1p(pr(both))
STATE_ERR = err.groupby(both["target_time"]).mean().shift(1, freq="D")
def final(df, lam=0.3, blend=0.9):
    c = STATE_ERR.reindex(df["target_time"]).fillna(0).to_numpy()
    return blend * np.expm1(np.log1p(pr(df)) + lam * c) + (1 - blend) * df["persistence"].to_numpy()

p_final = final(va)
assert (va.target_time.to_numpy() == bva.target_time.to_numpy()).all()
print(pd.concat({"Persistence": scores(bva, bva.persistence.to_numpy()), "Ridge29": scores(bva, p_ridge),
                 "Ours pre-mix": scores(va, pr(va)), "Ours final": scores(va, p_final)}))

pickle.dump({"data": data, "base": base, "GROUP_COLS": GROUP_COLS, "COLS": COLS, "BC": BC,
             "p_ridge": p_ridge, "p_final": p_final, "p_premix": pr(va), "STATE_ERR": STATE_ERR},
            open(OUT / "cache.pkl", "wb"))
print("cached; total", round(time.time() - t0, 1), "s")
