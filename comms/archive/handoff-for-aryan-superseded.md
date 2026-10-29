# CS771 Mini-Project 1 — handoff note for Aryan (29 Oct 2026)

## Where things stand
- **Deadline:** 30 Oct 2026, 11:59 pm. Upload **one** zip to the Dropbox link in the PDF.
  A Dropbox request link can't delete or replace an upload, so upload only once, when final.
- **Submission version = GitHub `main`** of the private repo
  https://github.com/akashiitkanpur/cs771-mini-project-1 (also on Akash's PC at
  `C:\Users\lenovo\Downloads\mp1-project`).
  - `CS771_MP1_submission/` = the exact folder that goes in the zip (both notebooks, report.pdf,
    README, outputs/, submission/ with forecast.py, features.py, model.pkl, bihar_stations.csv).
  - `CS771_MP1_submission.zip` on Akash's PC is the file to upload. A backup of an earlier
    version (28.09, with `net_spread_now`) sits next to it and is tagged `v-28.09-with-net-spread`.
- **In progress (29 Oct evening):** adding a state-wide bias correction (see below). Until that
  is finished and re-checked, the previous zip (validation 28.39 / 15.98) is the fallback.
- Ignore the old cloud copy in `/mnt/project-files/mp1-solution` and `mp1-submission.zip`
  (a `model.json` variant). The group chose the GitHub `main` version.

## The model, in one paragraph
One ridge regression shared by all 468 stations (StandardScaler + Ridge, alpha 1000), on
log(1 + PM2.5) features. It predicts the log change from today's 12:00 reading, adds the
smearing term exp(σ²/2), then mixes 0.9 × model + 0.1 × persistence. Train = targets 1 Jan–31 Aug,
validation = 1 Sep–30 Nov, the scaler is fitted on train only, and the final model is refitted on
everything up to 30 Nov.

Feature groups (41 features): recent_hours (24 hourly lags), neighbourhood (10 nearest / 15 km /
40 km averages, now and over 24 h), network (state mean now / 24 h / 7 days, 24 h change),
calendar (sin/cos day of year, weekend), station_level (30-day mean, and vs state), variability
(station's 24 h std, 10 nearest neighbours' 24 h change).

**Validation RMSE / MAE:** persistence 32.75 / 17.47, ridge baseline 29.58 / 18.16,
**ours 28.39 / 15.98** (Sep 20.23, Oct 17.42, Nov 41.51). We beat both baselines in every month.

## Being added now: state-wide bias correction
forecast += 0.3 × yesterday's state-average error (log scale). `forecast()` works this out from
the history it is given, by running the saved model one day earlier. Nothing is refitted.
- Expected validation: 28.04 / 15.56.
- Fair check: with the 0.3 chosen on Sep–Oct only, unseen November goes 41.51 → 40.91.
- It's the same idea as NOAA's operational PM2.5 bias correction.
Once it's done, confirm the final numbers in the report and that `check_submission.py` passes.

## Decisions already made (don't redo)
- Removed `net_spread_now` (spread across stations right now): it helped validation, but hurt the
  "train up to Oct, forecast Nov" test (41.5 → 42.6), which is the closest stand-in for December.
- Alpha 1000, not 300: they tie on validation, and the notebook's rule is to take the larger one.
- The report's AI-use section is fine as written. Group names are in. Don't mention the raw
  85,762 reading count in the report. The report is exactly 6 pages (the limit), so cut text before
  adding any.

## Tried and didn't help (good viva material)
Random forest (36.4, overfits), gradient boosting (34–43), kernel ridge, KNN, Bayesian ridge and
Huber (same or worse), PCA, lasso and fewer lags, per-station bias correction, separate models for
the dirtiest 2–50% of stations, per-cluster models, splitting by month or season, weighting recent
months, analog forecasting, neighbours chosen by correlation, and latitude/longitude features.
What drives the error: November is 70% of the squared error, and a few late-November days carry most
of it (29 Nov alone is 11%). These are sudden state-wide clean-ups or spikes that need weather data,
which isn't allowed.

## Checks that were run (all passed)
Leakage test (scrambling the future doesn't change the features), the official `check_submission.py`,
a fake December run, shuffled columns and missing values, the notebooks running top to bottom, and an
independent re-implementation of both baselines.

## Left to do
1. Finish and check the bias-correction version, then rebuild the zip.
2. Upload the zip **once** before the deadline.
3. Viva: every member should be able to explain the split, the log-change target, the feature
   groups and ablation, why ridge beat the forest, the lead-time hump, and the maps.

## Working with Claude
In the project thread, Claude has a session on Akash's PC that edits the files and pushes to GitHub.
To change the submission, ask there. The repo's `comms/` folder is for messages between teammates'
AI assistants.
