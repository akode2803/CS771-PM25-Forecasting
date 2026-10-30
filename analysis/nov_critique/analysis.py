"""Critique diagnostics: noise floor, November decomposition, pseudo-December test, untried ideas."""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repro as R          # re-runs the reproduction (5 s) and gives us its objects

data, va, tr = R.data, R.va, R.tr
COLS, GC = R.COLS, R.GROUP_COLS
y_va = va["target"].to_numpy(); day_va = va["target_time"].to_numpy()
rng = np.random.default_rng(0)

def rmse(p, y): return float(np.sqrt(np.mean((p - y) ** 2)))

# ------------------------------------------------------------------ 1. noise floor (day-block bootstrap)
def boot_diff(pa, pb, mask=None, n=4000):
    """RMSE(a) - RMSE(b), resampling whole days (errors are correlated across stations within a day)."""
    df = pd.DataFrame({"d": day_va, "a": (pa - y_va) ** 2, "b": (pb - y_va) ** 2})
    if mask is not None: df = df[mask]
    g = df.groupby("d")[["a", "b"]].agg(["sum", "count"])
    sa, sb, c = g[("a", "sum")].to_numpy(), g[("b", "sum")].to_numpy(), g[("a", "count")].to_numpy()
    idx = rng.integers(0, len(c), size=(n, len(c)))
    d = np.sqrt(sa[idx].sum(1) / c[idx].sum(1)) - np.sqrt(sb[idx].sum(1) / c[idx].sum(1))
    point = np.sqrt(sa.sum() / c.sum()) - np.sqrt(sb.sum() / c.sum())
    return round(point, 2), round(np.percentile(d, 2.5), 2), round(np.percentile(d, 97.5), 2)

nov = (va["target_time"].dt.month == 11).to_numpy()
p_pers = va["persistence"].to_numpy(); P_PREMIX = R.pr(va)
p_mix0 = R.final(va, lam=0.0)
p_a300 = None
m300 = make_pipeline(StandardScaler(), Ridge(alpha=300)).fit(tr[COLS], R.lct(tr))
sm300 = (R.lct(tr) - m300.predict(tr[COLS])).var() / 2
p_a300 = np.clip(np.expm1(np.log1p(va["persistence"]) + m300.predict(va[COLS]) + sm300).to_numpy(), 0, None)
def fit_pred(cols, alpha=1000, w=None, trn=tr, tst=va):
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    m.fit(trn[cols], R.lct(trn), ridge__sample_weight=w)
    s = (R.lct(trn) - m.predict(trn[cols])).var() / 2
    return np.clip(np.expm1(np.log1p(tst["persistence"]) + m.predict(tst[cols]) + s).to_numpy(), 0, None), m
p_novar, _ = fit_pred([c for g in R.FINAL_GROUPS if g != "variability" for c in GC[g]])
p_nonbr, _ = fit_pred([c for g in R.FINAL_GROUPS if g != "neighbourhood" for c in GC[g]])

print("\n=== 1. Day-block bootstrap: RMSE difference, point [95% CI] (negative = first is better)")
for name, a, b in [("final - persistence", R.p_final, p_pers), ("final - ridge29", R.p_final, R.p_ridge),
                   ("final - (same, no bias corr)", R.p_final, p_mix0),
                   ("premix a1000 - premix a300", P_PREMIX, p_a300),
                   ("premix - premix w/o variability", P_PREMIX, p_novar),
                   ("premix - premix w/o neighbourhood", P_PREMIX, p_nonbr),
                   ("mix0.9 - premix (blend gain)", p_mix0, P_PREMIX)]:
    print(f"  {name:38s} all {boot_diff(a, b)}   Nov {boot_diff(a, b, nov)}")

# ------------------------------------------------------------------ 2. November decomposition
print("\n=== 2. November error anatomy (final model)")
vn = va[nov].copy(); vn["pred"] = R.p_final[nov]; vn["err"] = vn["pred"] - vn["target"]
vn["se"] = vn["err"] ** 2
tot = vn["se"].sum()
daily = vn.groupby("target_time").agg(se=("se", "sum"), bias=("err", "mean"), obs=("target", "mean"))
print("  Nov share of all validation SE:", round(tot / ((R.p_final - y_va) ** 2).sum(), 3))
top = daily.sort_values("se", ascending=False)
print("  share of Nov SE from worst 1/3/5/10 days:", [round(top.se.iloc[:k].sum() / tot, 3) for k in (1, 3, 5, 10)])
print("  worst 6 days:", [(str(d.date()), round(np.sqrt(r.se / 468), 1), round(r.bias, 1)) for d, r in top.head(6).iterrows()])
print("  Nov RMSE without 29 Nov:", round(np.sqrt(vn.loc[vn.target_time != "2024-11-29 12:00", "se"].mean()), 2))
# state-wide (day-mean) part vs station part
dm = vn.groupby("target_time")["err"].transform("mean")
print("  SE share from the state-wide daily mean error:", round((dm ** 2).sum() / tot, 3))
# station part: persistent station bias over November
sb = vn.groupby("station")["err"].transform("mean")
print("  SE share from each station's November-mean error:", round((sb ** 2).sum() / tot, 3))
# oracles in log space (multiplicative): what if the state-wide daily log change were known?
le = np.log1p(vn["target"]) - np.log1p(vn["pred"])
day_le = le.groupby(vn["target_time"]).transform("mean")
print("  ORACLE: perfect state-wide daily level  -> Nov RMSE",
      round(rmse(np.expm1(np.log1p(vn["pred"]) + day_le).to_numpy(), vn["target"].to_numpy()), 2))
st_le = le.groupby(vn["station"]).transform("mean")
print("  ORACLE: perfect per-station Nov offset  -> Nov RMSE",
      round(rmse(np.expm1(np.log1p(vn["pred"]) + st_le).to_numpy(), vn["target"].to_numpy()), 2))
# skill in predicting the CHANGE
true_lc = np.log1p(vn["target"]) - np.log1p(vn["persistence"])
pred_lc = np.log1p(vn["pred"]) - np.log1p(vn["persistence"])
print("  corr(predicted log change, true log change), station level:", round(np.corrcoef(pred_lc, true_lc)[0, 1], 3))
dt = true_lc.groupby(vn["target_time"]).mean(); dp = pred_lc.groupby(vn["target_time"]).mean()
print("  corr at state level (daily means, 30 days):", round(np.corrcoef(dp, dt)[0, 1], 3),
      "| std true", round(dt.std(), 3), "std pred", round(dp.std(), 3))
# within-day: does the model rank station changes?
wt = true_lc - true_lc.groupby(vn["target_time"]).transform("mean")
wp = pred_lc - pred_lc.groupby(vn["target_time"]).transform("mean")
print("  corr of within-day (station-relative) changes:", round(np.corrcoef(wp, wt)[0, 1], 3))
# conditional bias: conditioning on the forecast (fair) vs on the outcome (artefact)
vn["fbin"] = pd.qcut(vn["pred"], 10, labels=False); vn["obin"] = pd.qcut(vn["target"], 10, labels=False)
print("  mean error by FORECAST decile:", vn.groupby("fbin")["err"].mean().round(1).tolist())
print("  mean error by OUTCOME decile :", vn.groupby("obin")["err"].mean().round(1).tolist())

# ------------------------------------------------------------------ 3. calendar term in the final refit
print("\n=== 3. What the calendar group does in the FINAL (Jan-Nov) refit")
full = pd.concat([tr, va])
mf = make_pipeline(StandardScaler(), Ridge(alpha=1000)).fit(full[COLS], R.lct(full))
sc, rg = mf.named_steps["standardscaler"], mf.named_steps["ridge"]
ci = [COLS.index(c) for c in ("sin_doy", "cos_doy")]
for mo, day0 in [(1, 15), (3, 74), (6, 166), (9, 258), (10, 288), (11, 319), (12, 349), (12, 364)]:
    s, c = np.sin(2 * np.pi * day0 / 365), np.cos(2 * np.pi * day0 / 365)
    contrib = ((np.array([s, c]) - sc.mean_[ci]) / sc.scale_[ci]) @ rg.coef_[ci]
    print(f"  day {day0:3d} (month {mo:2d}): calendar adds {contrib:+.3f} to log change = x{np.exp(contrib):.3f} per day")

# ------------------------------------------------------------------ 4. pseudo-December: fit Jan-Oct, test Nov
print("\n=== 4. Pseudo-December test: fit on targets <= 31 Oct, score November (closest proxy to the real test)")
trO = data[data.target_time <= "2024-10-31 23:59"]; teN = data[data.target_time.dt.month == 11]
bO = R.base[R.base.target_time <= "2024-10-31 23:59"]; bN = R.base[R.base.target_time.dt.month == 11]
rbO = Ridge(alpha=1.0).fit(bO[R.BC], bO["target"]); pr29 = np.clip(rbO.predict(bN[R.BC]), 0, None)
yN = teN["target"].to_numpy()
pO, mO = fit_pred(COLS, trn=trO, tst=teN)
allO = pd.concat([trO, teN]); predall, _ = fit_pred(COLS, trn=trO, tst=allO)
errO = (np.log1p(allO["target"]) - np.log1p(predall)).groupby(allO["target_time"]).mean().shift(1, freq="D")
corrN = errO.reindex(teN["target_time"]).fillna(0).to_numpy()
print("  persistence", round(rmse(teN["persistence"].to_numpy(), yN), 2), "| ridge29", round(rmse(pr29, yN), 2))
for lam in (0, 0.15, 0.3, 0.5):
    for bl in (0.9, 1.0):
        p = bl * np.expm1(np.log1p(pO) + lam * corrN) + (1 - bl) * teN["persistence"].to_numpy()
        print(f"  ours, lambda {lam}, blend {bl}: {rmse(p, yN):.2f}")

# ------------------------------------------------------------------ 5. untried ideas (pre-mix, Jan-Aug fit, val + Nov)
print("\n=== 5. Untried ideas, pre-mix RMSE (Jan-Aug fit). Reference: all 28.50, Nov 41.52")
def rep(name, p): print(f"  {name:55s} all {rmse(p, y_va):.2f}  Nov {rmse(p[nov], y_va[nov]):.2f}")
# 5a. weight rows so log-space loss approximates level-space squared error
for k in (0.5, 1.0, 1.5, 2.0):
    w = (1 + tr["persistence"].to_numpy()) ** k; w = w / w.mean()
    rep(f"level-weighted log LS, w=(1+today)^{k}", fit_pred(COLS, w=w)[0])

# 5b. directional (upwind) neighbourhoods: mean of stations 30-150 km away in 4 compass sectors
st = R.stations
x = (st.longitude.to_numpy() - 85.5) * 100.3; yk = (st.latitude.to_numpy() - 25.7) * 110.6
dx, dy = x[None, :] - x[:, None], yk[None, :] - yk[:, None]
dist = np.hypot(dx, dy); ang = np.degrees(np.arctan2(dy, dx)) % 360
ring = (dist >= 30) & (dist <= 150)
SECT = {"E": (ang < 45) | (ang >= 315), "N": (ang >= 45) & (ang < 135), "W": (ang >= 135) & (ang < 225), "S": (ang >= 225) & (ang < 315)}
SECT = {k: (v & ring).astype(float) for k, v in SECT.items()}
def sector_feats(A_, p, t, nb):
    L = np.log1p(np.clip(A_[:p + 1], 0, None)); now, prev = L[p], L[p - 24]
    f = {}
    for k, M in SECT.items():
        cnt = np.maximum(M.sum(1), 1)
        f[f"sec{k}_now"] = M @ now / cnt
        f[f"sec{k}_chg24"] = (M @ now - M @ prev) / cnt
    return pd.DataFrame(f)
t0 = time.time()
sec = R.build(sector_feats, R.noon_targets(R.HISTORY_NEEDED - 1))
SC = [c for c in sec.columns if c.startswith("sec")]
data2 = pd.concat([data.reset_index(drop=True), sec[SC].reset_index(drop=True)], axis=1)
tr2, va2 = R.split(data2)
p_sec, _ = fit_pred(COLS + SC, trn=tr2, tst=va2)
rep("+ 4 compass-sector (30-150 km) now & 24h-change (8 feats)", p_sec)
print("   sector features built in", round(time.time() - t0, 1), "s;  max |row mismatch|:",
      int((data2.target_time.to_numpy() != data.target_time.to_numpy()).sum()))

# 5c. random Fourier features (kernel ridge approximation, lecture 10)
Xs_tr = StandardScaler().fit(tr[COLS]); Ztr0 = Xs_tr.transform(tr[COLS]); Zva0 = Xs_tr.transform(va[COLS])
for ell, D in [(6.0, 600), (10.0, 600)]:
    Wr = rng.normal(0, 1 / ell, size=(len(COLS), D)); br = rng.uniform(0, 2 * np.pi, D)
    f = lambda Z: np.hstack([Z, np.sqrt(2 / D) * np.cos(Z @ Wr + br)])
    m = Ridge(alpha=1000).fit(f(Ztr0), R.lct(tr)); s = (R.lct(tr) - m.predict(f(Ztr0))).var() / 2
    rep(f"RFF kernel ridge (RBF, ell={ell}, D={D}) + linear", np.clip(np.expm1(np.log1p(va["persistence"]) + m.predict(f(Zva0)) + s).to_numpy(), 0, None))

# 5d. boosted trees on the ridge residual, WITHOUT calendar/absolute-season features
from sklearn.ensemble import HistGradientBoostingRegressor
_, mr = fit_pred(COLS)
res = R.lct(tr) - mr.predict(tr[COLS])
NC = [c for c in COLS if c not in ("sin_doy", "cos_doy", "weekend")]
hgb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=500,
                                    l2_regularization=1.0, random_state=0).fit(tr[NC], res)
s = (res - hgb.predict(tr[NC])).var() / 2
rep("ridge + boosted-tree residual (no calendar feats)",
    np.clip(np.expm1(np.log1p(va["persistence"]) + mr.predict(va[COLS]) + hgb.predict(va[NC]) + s).to_numpy(), 0, None))

pickle.dump({"vn": vn, "daily": daily}, open(HERE / "nov.pkl", "wb"))
