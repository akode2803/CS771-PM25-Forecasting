"""Sector-feature significance and the in-sample monthly drift of the final refit (prints only)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import analysis as X          # reuses everything already computed
import numpy as np, pandas as pd
R = X.R
p_sec = X.p_sec
print("sector premix vs premix   all", X.boot_diff(p_sec, X.P_PREMIX), " Nov", X.boot_diff(p_sec, X.P_PREMIX, X.nov))
# full pipeline with sectors (same blend 0.9, lambda 0.3)
both = pd.concat([X.tr2, X.va2]); pb, _ = X.fit_pred(X.COLS + X.SC, trn=X.tr2, tst=both)
se = (np.log1p(both["target"]) - np.log1p(pb)).groupby(both["target_time"]).mean().shift(1, freq="D")
c = se.reindex(X.va2["target_time"]).fillna(0).to_numpy()
pf = 0.9 * np.expm1(np.log1p(p_sec) + 0.3 * c) + 0.1 * X.va2["persistence"].to_numpy()
print("sector FULL pipeline: all %.2f Nov %.2f (current final 28.04 / 40.91)" % (X.rmse(pf, X.y_va), X.rmse(pf[X.nov], X.y_va[X.nov])))
print("sector full vs final      all", X.boot_diff(pf, R.p_final), " Nov", X.boot_diff(pf, R.p_final, X.nov))
_, m = X.fit_pred(X.COLS + X.SC, trn=X.tr2, tst=X.va2)
co = pd.Series(m.named_steps["ridge"].coef_, index=X.COLS + X.SC)
print("sector coefs (std units):", co[X.SC].round(3).to_dict())
# Is the gain just from 'more neighbourhood'? Same rings without direction:
ringf = X.ring.astype(float); cnt = np.maximum(ringf.sum(1), 1)
def ring_feats(A_, p, t, nb):
    L = np.log1p(np.clip(A_[:p + 1], 0, None)); return pd.DataFrame({"ring_now": ringf @ L[p] / cnt, "ring_chg24": (ringf @ L[p] - ringf @ L[p - 24]) / cnt})
rf = R.build(ring_feats, R.noon_targets(R.HISTORY_NEEDED - 1))
d3 = pd.concat([X.data.reset_index(drop=True), rf[["ring_now", "ring_chg24"]]], axis=1); t3, v3 = R.split(d3)
p_ring, _ = X.fit_pred(X.COLS + ["ring_now", "ring_chg24"], trn=t3, tst=v3)
X.rep("+ 30-150 km ring, no direction (2 feats)", p_ring)
# net predicted vs actual state-mean daily log change by month, Jan-Nov refit (in-sample)
full = pd.concat([X.tr, X.va]); mf = X.mf
pl = pd.Series(mf.predict(full[X.COLS]), index=full.index); tl = R.lct(full)
mo = full["target_time"].dt.month
print("mean daily log change by month (pred | true):")
print(pd.DataFrame({"pred": pl.groupby(mo).mean(), "true": tl.groupby(mo).mean()}).round(3).T.to_string())
