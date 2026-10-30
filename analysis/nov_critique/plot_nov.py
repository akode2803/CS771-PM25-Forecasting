"""November error diagnostics for the final model (validation fit: Jan-Aug). Writes 3 PNGs + 2 CSVs."""
import sys, io, contextlib
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    import analysis as X                      # final-model predictions, sector experiment, oracles
R = X.R

# ---------------------------------------------------------------- style (reference palette, light surface)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ae"
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1d"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "legend.fontsize": 8.5, "lines.linewidth": 2, "font.family": "DejaVu Sans"})

import textwrap
def note(ax, text, xy=(0.0, -0.2), width=95):
    ax.annotate(textwrap.fill(text, width), xy=xy, xycoords="axes fraction", fontsize=8.3, color=INK2, va="top",
                linespacing=1.35)

# ---------------------------------------------------------------- data
vn = X.vn.copy()                                   # November rows of the validation set, final model
vn["day"] = vn["target_time"].dt.day
vn["pers_err"] = vn["persistence"] - vn["target"]
vn["le"] = np.log1p(vn["pred"]) - np.log1p(vn["target"])          # log error, + = too high
st = R.stations.reset_index()
lon, lat = st["longitude"].to_numpy(), st["latitude"].to_numpy()
outline = pd.read_csv(R.DATA / "bihar_boundary.csv")
tot_se = vn["se"].sum()

def draw_map(ax, values, cmap, norm=None, vmin=None, vmax=None, s=16):
    ax.plot(outline.longitude, outline.latitude, color=INK2, lw=0.8)
    sc = ax.scatter(lon, lat, c=values, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, s=s, linewidths=0.3, edgecolors=SURF)
    ax.set_aspect(1 / np.cos(np.radians(25.7))); ax.grid(False)
    ax.set_xticks([84, 85, 86, 87, 88]); ax.set_yticks([24.5, 25.5, 26.5, 27.5])
    ax.tick_params(labelsize=7.5)
    for sp in ax.spines.values(): sp.set_visible(False)
    return sc

# ================================================================= FIGURE 1: where the November error lives
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
fig.subplots_adjust(left=0.21, right=0.97, top=0.92, bottom=0.11, hspace=0.5, wspace=0.22)
fig.suptitle("November 2024: where the forecast error comes from (final model, fitted on Jan–Aug)",
             fontsize=14, fontweight="bold", x=0.01, ha="left")

# 1a. daily squared error split into the state-wide part and the station scatter
g = vn.groupby("day")
state_part = g["err"].mean() ** 2 * g.size() / tot_se * 100
scatter_part = (g["se"].sum() / tot_se * 100) - state_part
a = ax[0, 0]
a.bar(state_part.index, state_part, color=BLUE, width=0.75, label="whole state off together (daily mean error)",
      edgecolor=SURF, linewidth=1)
a.bar(scatter_part.index, scatter_part, bottom=state_part, color=ORANGE, width=0.75,
      label="stations scattered around that mean", edgecolor=SURF, linewidth=1)
a.set_xticks(range(1, 31)); a.set_xlim(0.3, 30.7); a.tick_params(axis="x", labelsize=7.5)
a.set_xlabel("target day in November (12:00 IST)"); a.set_ylabel("share of November squared error (%)")
a.set_title("1a. Each day's share of November's squared error")
a.legend(loc="upper left")
sp_total = state_part.sum()
a.annotate(f"29 Nov alone: {state_part[29] + scatter_part[29]:.0f}%\n(state fell 159 → 77 µg/m³)",
           xy=(28.6, state_part[29] + scatter_part[29] - 0.5), xytext=(18.5, 11.5), fontsize=8.5, color=INK,
           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
note(a, f"Whole-state part = {sp_total:.0f}% of November's squared error; station scatter = {100 - sp_total:.0f}%. "
        f"The worst 5 days give {X.top.se.iloc[:5].sum() / tot_se * 100:.0f}%, the worst 10 give "
        f"{X.top.se.iloc[:10].sum() / tot_se * 100:.0f}%.", (0.0, -0.12))

# 1b. how concentrated: Lorenz curve over station-days
a = ax[0, 1]
se_sorted = np.sort(vn["se"].to_numpy())[::-1]
cum = np.cumsum(se_sorted) / se_sorted.sum() * 100
frac = np.arange(1, len(cum) + 1) / len(cum) * 100
a.plot(frac, cum, color=BLUE)
a.plot([0, 100], [0, 100], color=MUTED, lw=1, ls=":")
for q in (50, 80):
    k = np.searchsorted(cum, q)
    a.plot([frac[k], frac[k]], [0, q], color=MUTED, lw=0.8); a.plot([0, frac[k]], [q, q], color=MUTED, lw=0.8)
    a.annotate(f"{frac[k]:.1f}% of station-days\n→ {q}% of squared error", xy=(frac[k], q),
               xytext=(frac[k] + 6, q - 9), fontsize=8.5, color=INK)
a.set_xlim(0, 100); a.set_ylim(0, 101)
a.set_xlabel("station-days, worst first (% of 14,040)"); a.set_ylabel("cumulative share of squared error (%)")
a.set_title("1b. A few station-days dominate November's RMSE")

# 1c. headroom: what each idea / oracle does to November RMSE
a = ax[1, 0]
rows = [("Persistence (today's 12:00 value)", 47.81, GRAY),
        ("Ridge baseline (29 features)", 43.06, GRAY),
        ("Our final model (submitted)", round(X.rmse(R.p_final[X.nov], X.y_va[X.nov]), 2), BLUE),
        ("+ compass-sector features (new, not in report)", 40.56, ORANGE),
        ("ORACLE: perfect per-station November offset", 40.65, AQUA),
        ("ORACLE: perfect state-wide daily level", 36.71, AQUA)]
names, vals, cols = zip(*rows)
yy = np.arange(len(rows))[::-1]
# dot plot, not bars: the axis does not start at 0, so bar length would mislead
a.hlines(yy, 30, 50, color=GRID, lw=1)
a.scatter(vals, yy, s=110, color=cols, edgecolors=SURF, linewidths=2, zorder=3)
for y_, v in zip(yy, vals): a.text(v + 0.45, y_, f"{v:.2f}", va="center", fontsize=9, color=INK)
a.set_yticks(yy); a.set_yticklabels(names, fontsize=8.8); a.set_xlim(30, 50.5); a.set_ylim(-0.6, len(rows) - 0.4)
a.set_xlabel("November RMSE (µg/m³)"); a.grid(axis="y", visible=False)
a.set_title("1c. Headroom: even a perfect state-wide forecast leaves 36.7")
note(a, "Oracles use the true November values, so they are ceilings, not methods. Knowing each day's state-wide "
        "level exactly removes only 4.2 µg/m³; the rest is station-to-station day-to-day jumps.", (0.0, -0.12))

# 1d. where the squared error sits by forecast level
a = ax[1, 1]
bands = [0, 50, 100, 150, 200, 300, np.inf]; labels = ["<50", "50–100", "100–150", "150–200", "200–300", ">300"]
vn["band"] = pd.cut(vn["pred"], bands, labels=labels, right=False)
share_rows = vn.groupby("band", observed=False).size() / len(vn) * 100
share_se = vn.groupby("band", observed=False)["se"].sum() / tot_se * 100
xx = np.arange(len(labels))
a.bar(xx - 0.2, share_rows, width=0.38, color=GRAY, label="share of station-days", edgecolor=SURF)
a.bar(xx + 0.2, share_se, width=0.38, color=BLUE, label="share of squared error", edgecolor=SURF)
for x_, v in zip(xx, share_se): a.text(x_ + 0.2, v + 0.6, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK)
a.set_xticks(xx); a.set_xticklabels(labels); a.set_xlabel("our forecast (µg/m³)"); a.set_ylabel("%")
a.set_title("1d. High forecasts carry the squared error"); a.legend(loc="upper right")
rm = vn.groupby("band", observed=False).apply(lambda d: np.sqrt(d["se"].mean()), include_groups=False)
note(a, "RMSE inside each band: " + ", ".join(f"{l} {v:.0f}" for l, v in zip(labels, rm)) + " µg/m³.", (0.0, -0.12), 80)
fig.savefig(OUT / "nov_1_where_the_error_is.png", dpi=130); plt.close(fig)

# ================================================================= FIGURE 2: is any of it predictable?
fig, ax = plt.subplots(2, 3, figsize=(19, 12.5))
fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.11, hspace=0.55, wspace=0.3)
fig.suptitle("November 2024: is the remaining error predictable? (final model, fitted on Jan–Aug)",
             fontsize=14, fontweight="bold", x=0.01, ha="left")
W3 = 72                                            # note width (characters) for the 3-column layout

# 2a. conditional bias: by forecast (actionable) vs by outcome (artefact)
a = ax[0, 0]
q = 12
fb = vn.groupby(pd.qcut(vn["pred"], q), observed=True).agg(x=("pred", "mean"), e=("err", "mean"))
ob = vn.groupby(pd.qcut(vn["target"], q), observed=True).agg(x=("target", "mean"), e=("err", "mean"))
a.axhline(0, color=MUTED, lw=1)
a.plot(fb.x, fb.e, color=BLUE, marker="o", ms=5, label="binned by our FORECAST (fixable bias)")
a.plot(ob.x, ob.e, color=ORANGE, marker="o", ms=5, ls="--", label="binned by the OUTCOME (regression to the mean)")
a.set_xlabel("bin mean (µg/m³)"); a.set_ylabel("mean error, forecast − observed (µg/m³)")
a.set_title("2a. Conditional bias: only the solid line is fixable"); a.legend(loc="lower left")
note(a, "Binning by the outcome always shows 'too low when it was high'. That happens for any honest forecast, so it is "
        "not a pattern you can fix. Binned by the forecast, the bias is ≈0 except the top bin "
        f"({fb.e.iloc[-1]:+.0f} µg/m³: our highest forecasts are too high).",
     (0.0, -0.14), W3)

# 2b. predicted vs true log change, station level
a = ax[0, 1]
tl = np.log1p(vn["target"]) - np.log1p(vn["persistence"]); pl = np.log1p(vn["pred"]) - np.log1p(vn["persistence"])
hb = a.hexbin(pl, tl, gridsize=60, extent=(-1.2, 1.2, -1.6, 1.6), cmap=SEQ, mincnt=1, bins="log", linewidths=0)
a.plot([-1.2, 1.2], [-1.2, 1.2], color=INK2, lw=1, ls=":")
a.set_xlim(-1.2, 1.2); a.set_ylim(-1.6, 1.6)
a.set_xlabel("predicted change, log(1+tomorrow) − log(1+today)"); a.set_ylabel("true change (same units)")
a.set_title("2b. Predicted vs true day-to-day change")
fig.colorbar(hb, ax=a, label="station-days (log scale)", shrink=0.8)
note(a, f"corr = {np.corrcoef(pl, tl)[0, 1]:.2f}. Std of predicted change {pl.std():.2f} vs true {tl.std():.2f}: "
        "the model captures well under half of each station's day-to-day movement.", (0.0, -0.14), W3)

# 2c. state level: daily mean predicted vs true change
a = ax[0, 2]
dp, dt = pl.groupby(vn["day"]).mean(), tl.groupby(vn["day"]).mean()
a.axhline(0, color=MUTED, lw=1); a.axvline(0, color=MUTED, lw=1)
a.plot([-0.8, 0.8], [-0.8, 0.8], color=INK2, lw=1, ls=":")
a.scatter(dp, dt, s=40, color=BLUE, edgecolors=SURF, linewidths=1.5, zorder=3)
for d_ in (dt - dp).abs().sort_values().index[-5:]:
    if True:
        a.annotate(f"{d_} Nov", (dp[d_], dt[d_]), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK2)
a.set_xlim(-0.3, 0.3); a.set_ylim(-0.8, 0.5)
a.set_xlabel("predicted state-mean log change"); a.set_ylabel("true state-mean log change")
a.set_title("2c. State-wide change: some skill, too timid")
note(a, f"30 days, corr = {np.corrcoef(dp, dt)[0, 1]:.2f}. The model never predicts a state-wide move larger than "
        f"{dp.abs().max():.2f} in log units; the true moves reach {dt.abs().max():.2f} (29 Nov).", (0.0, -0.14), W3)

# 2d. do errors persist from one day to the next?
a = ax[1, 0]
E = vn.pivot(index="day", columns="station", values="le")
state_e = E.mean(axis=1); rel = E.sub(state_e, axis=0)
lags = [1, 2, 3, 4, 5]
ac_state = [state_e.autocorr(k) for k in lags]
ac_rel = [np.nanmedian([rel[c].autocorr(k) for c in rel.columns]) for k in lags]
xx = np.arange(len(lags))
a.bar(xx - 0.2, ac_state, width=0.38, color=BLUE, label="state-wide daily error", edgecolor=SURF)
a.bar(xx + 0.2, ac_rel, width=0.38, color=ORANGE, label="station's own error (median over 468)", edgecolor=SURF)
a.axhline(0, color=MUTED, lw=1)
for x_, v in zip(xx, ac_state): a.text(x_ - 0.2, v + (0.02 if v >= 0 else -0.05), f"{v:.2f}", ha="center", fontsize=8)
for x_, v in zip(xx, ac_rel): a.text(x_ + 0.2, v + (0.02 if v >= 0 else -0.05), f"{v:.2f}", ha="center", fontsize=8)
a.set_xticks(xx); a.set_xticklabels([f"{k} day" + ("s" if k > 1 else "") for k in lags])
a.set_xlabel("lag"); a.set_ylabel("autocorrelation of the log error"); a.set_ylim(-0.6, 0.6)
a.set_title("2d. Do errors carry over to the next day?"); a.legend(loc="upper right")
note(a, "Errors here already include the 0.3 × yesterday's-state-error correction. Positive lag-1 values would mean more "
        "correction is possible; values near 0 or negative mean the correction already took what there was.", (0.0, -0.14), W3)

# 2e. spatial coherence of station-relative errors
a = ax[1, 1]
C = np.corrcoef(rel.to_numpy().T)
x_km = (lon - 85.5) * 100.3; y_km = (lat - 25.7) * 110.6
D = np.hypot(x_km[:, None] - x_km[None, :], y_km[:, None] - y_km[None, :])
iu = np.triu_indices(len(lon), 1)
dbins = [0, 5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 250, 500]
cats = pd.cut(D[iu], dbins)
sc_ = pd.Series(C[iu]).groupby(cats, observed=True).agg(["mean", "size"])
mid = [(b.left + b.right) / 2 for b in sc_.index]
a.axhline(0, color=MUTED, lw=1)
a.plot(mid, sc_["mean"], color=BLUE, marker="o", ms=6)
a.set_xscale("log"); a.set_xticks([3, 10, 30, 100, 300]); a.set_xticklabels(["3", "10", "30", "100", "300"])
a.set_xlabel("distance between two stations (km, log scale)")
a.set_ylabel("mean correlation of their November errors")
a.set_title("2e. Station errors are spatially clustered")
note(a, "Each station's daily log error minus that day's state mean, correlated over 30 days for every pair of stations. "
        "High values at short range mean errors hit neighbouring stations together, so local events are shared.", (0.0, -0.14), W3)

# 2f. where does the model add value over persistence?
a = ax[1, 2]
ps = vn.groupby("station").agg(r=("se", lambda s: np.sqrt(s.mean())), rp=("pers_err", lambda s: np.sqrt((s ** 2).mean())))
ratio = (ps["r"] / ps["rp"]).reindex(range(len(lon))).to_numpy()
sc = draw_map(a, ratio, DIV, norm=TwoSlopeNorm(vcenter=1.0, vmin=0.6, vmax=1.4))
fig.colorbar(sc, ax=a, label="station RMSE: ours ÷ persistence", shrink=0.75, pad=0.03)
a.set_title("2f. Beat persistence (blue) or lose (red)")
note(a, f"We beat persistence at {(ratio < 1).sum()} of 468 stations in November; median ratio {np.median(ratio):.2f}.",
     (0.0, -0.06), W3)
fig.savefig(OUT / "nov_2_is_it_predictable.png", dpi=130); plt.close(fig)

# ================================================================= FIGURE 3: the end-of-month event
fig = plt.figure(figsize=(17, 10.5), layout="constrained")
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
fig.suptitle("20–30 November: the build-up and the 29 Nov clean-up, hour by hour",
             fontsize=14, fontweight="bold", x=0.01, ha="left")
a = fig.add_subplot(gs[0, :])
Wn = R.W.loc["2024-11-20":"2024-11-30 23:00"]
terc = pd.qcut(lon, 3, labels=["west", "central", "east"])
for band, colr in zip(["west", "central", "east"], [BLUE, ORANGE, AQUA]):
    s = Wn.iloc[:, np.where(terc == band)[0]].mean(axis=1)
    a.plot(s.index, s.values, color=colr, lw=1.6, label=f"{band} third of the stations (by longitude)")
for d in pd.date_range("2024-11-20 12:00", "2024-11-30 12:00", freq="D"):
    a.axvline(d, color=GRID, lw=1, zorder=0)
a.axvline(pd.Timestamp("2024-11-28 12:00"), color=INK2, lw=1.2, ls="--")
a.annotate("issue time for the 29 Nov target\n(everything to the right is unknown)", xy=(pd.Timestamp("2024-11-28 12:00"), 0.97),
           xycoords=("data", "axes fraction"), xytext=(-8, 0), textcoords="offset points", ha="right", va="top", fontsize=8.5, color=INK2)
a.set_ylabel("mean PM2.5 (µg/m³)"); a.set_xlim(Wn.index[0], Wn.index[-1])
a.set_title("3a. Hourly mean PM2.5 by longitude band (light vertical lines = 12:00 each day)")
a.legend(loc="upper left", ncol=3)

L = np.log1p(R.W)
def chg(t1, t0): return (L.loc[t1] - L.loc[t0]).to_numpy()
known = chg("2024-11-28 12:00", "2024-11-27 12:00")
target = chg("2024-11-29 12:00", "2024-11-28 12:00")
err29 = vn[vn.day == 29].set_index("station")["err"].reindex(range(len(lon))).to_numpy()
lim = 1.2
for k, (vals, title, cmap, norm, lab) in enumerate([
        (known, "3b. Known at issue: 24 h change 27→28 Nov 12:00", DIV, TwoSlopeNorm(0, -lim, lim), "log change"),
        (target, "3c. To forecast: 24 h change 28→29 Nov 12:00", DIV, TwoSlopeNorm(0, -lim, lim), "log change"),
        (err29, "3d. Our error on 29 Nov (forecast − observed)", DIV, TwoSlopeNorm(0, -200, 200), "µg/m³")]):
    a = fig.add_subplot(gs[1, k]); sc = draw_map(a, vals, cmap, norm=norm, s=14)
    a.set_title(title, fontsize=10); fig.colorbar(sc, ax=a, label=lab, shrink=0.7, orientation="horizontal", pad=0.02)
fig.savefig(OUT / "nov_3_end_of_month_event.png", dpi=130); plt.close(fig)

# ================================================================= tables
ids = R.W.columns.to_numpy()
day_tab = vn.groupby("target_time").agg(
    observed_state_mean=("target", "mean"), forecast_state_mean=("pred", "mean"), persistence_state_mean=("persistence", "mean"),
    rmse_ours=("se", lambda s: np.sqrt(s.mean())), rmse_persistence=("pers_err", lambda s: np.sqrt((s ** 2).mean())),
    mean_error=("err", "mean"), share_of_nov_sq_error=("se", lambda s: s.sum() / tot_se)).round(3)
day_tab.to_csv(OUT / "nov_by_day.csv")
st_tab = vn.groupby("station").agg(
    mean_observed=("target", "mean"), rmse_ours=("se", lambda s: np.sqrt(s.mean())),
    rmse_persistence=("pers_err", lambda s: np.sqrt((s ** 2).mean())), mean_error=("err", "mean"),
    share_of_nov_sq_error=("se", lambda s: s.sum() / tot_se))
st_tab["lag1_autocorr_relative_log_error"] = [rel[c].autocorr(1) for c in st_tab.index]
st_tab.insert(0, "sensor_id", ids[st_tab.index]); st_tab.insert(1, "latitude", lat[st_tab.index]); st_tab.insert(2, "longitude", lon[st_tab.index])
st_tab.sort_values("share_of_nov_sq_error", ascending=False).round(4).to_csv(OUT / "nov_by_station.csv", index=False)

print("state-part share %.1f%%" % sp_total, "| lorenz50", frac[np.searchsorted(cum, 50)], "| lorenz80", frac[np.searchsorted(cum, 80)])
print("autocorr state", np.round(ac_state, 2), "| rel", np.round(ac_rel, 2))
print("spatial corr by distance:", sc_["mean"].round(2).to_dict())
print("beat persistence at", (ratio < 1).sum(), "stations; median ratio", np.median(ratio).round(3))
print("top-20 stations share of Nov SE:", st_tab.share_of_nov_sq_error.sort_values(ascending=False).iloc[:20].sum().round(3))
print("forecast-binned bias:", fb.e.round(1).tolist())
