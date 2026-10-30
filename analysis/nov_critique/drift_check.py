"""Out-of-sample monthly drift on validation, and two drift-removal variants (prints only)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import analysis as X
import numpy as np, pandas as pd
R = X.R; va, tr = X.va, X.tr
# out-of-sample (Jan-Aug fit) state-level drift on validation months
_, m = X.fit_pred(X.COLS)
pl = pd.Series(m.predict(va[X.COLS]), index=va.index); tl = R.lct(va); mo = va.target_time.dt.month
print("VAL (Jan-Aug fit) mean daily log change by month, pred | true:")
print(pd.DataFrame({"pred": pl.groupby(mo).mean(), "true": tl.groupby(mo).mean()}).round(3).T.to_string())
# drift-free variant: keep the model's station-relative change, set the state-wide change to 0 (+ smear)
sm = (R.lct(tr) - m.predict(tr[X.COLS])).var() / 2
rel = pl - pl.groupby(va.target_time).transform("mean")
p_rel = np.clip(np.expm1(np.log1p(va.persistence) + rel + sm).to_numpy(), 0, None)
X.rep("drift-free: state-level change := 0, keep station-relative part", p_rel)
for k in (0.5,):
    p_k = np.clip(np.expm1(np.log1p(va.persistence) + rel + k * (pl - rel) + sm).to_numpy(), 0, None)
    X.rep(f"state-level change shrunk x{k}", p_k)
