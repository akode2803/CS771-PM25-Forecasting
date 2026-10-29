# Handoff: CS771 Mini-Project 1 (PM2.5 forecasting, Bihar)

For: Aryan Kumar, taking over from Akash Kumar (and Akash's Claude), 29–30 Oct 2026.
Deadline: **30 Oct 2026**. Repo: https://github.com/akashiitkanpur/cs771-mini-project-1 (private).

## 1. Status in one paragraph

Everything is finished and checked. The file to submit is
`CS771_MP1_submission.zip` (built from the `CS771_MP1_submission/` folder in this repo). **It has
not been uploaded yet** unless Akash tells you otherwise. Upload it once, to the Dropbox link in
the assignment PDF (`mp1-package/mini-project-1.pdf`, section 12). A Dropbox file request
cannot be edited or deleted by the uploader, so upload only the final version, and use a clear
file name, e.g. `CS771_MP1_Akash_Aryan_Preshit.zip`.

Zip files are not in git (see `.gitignore`). If you only have the repo, rebuild the zip from the
`CS771_MP1_submission/` folder: zip its *contents* (11 files), not the folder.

## 2. Final numbers (validation = every station, every 12:00 target, 1 Sep – 30 Nov)

| RMSE / MAE (µg/m³) | all | Sep | Oct | Nov |
|---|---|---|---|---|
| Persistence | 32.75 / 17.47 | 23.33 / 8.91 | 20.22 / 11.97 | 47.81 / 31.70 |
| Ridge baseline (29 features) | 29.58 / 18.16 | 20.34 / 10.35 | 19.36 / 13.46 | 43.06 / 30.81 |
| **Our final model** | **28.04 / 15.56** | **20.25 / 7.91** | **17.08 / 10.65** | **40.91 / 28.29** |

Beats both baselines on both metrics, overall and in every month. Official checker passes
(~0.2 s per call, limit 10 s); November sanity RMSE 40.84 vs persistence 47.81 (in-sample, not a
test score).

## 3. The final model (what to explain in the viva)

1. Features are `log(1 + PM2.5)`; the target is the log change from today's reading (forecast =
   today × a factor), so ridge shrinks towards persistence.
2. Ridge, alpha = 1000, standardised features (scaler fitted on training rows only).
3. 41 features in 6 groups (`submission/features.py`): recent_hours (24 hourly lags),
   neighbourhood (6: 10 nearest, ≤15 km, ≤40 km, now and 24 h), network (4: state mean
   now/24 h/7 d and 24 h change), calendar (3: sin/cos day of year, weekend), station_level (2:
   30-day mean, vs state), variability (2: station std over 24 h, neighbours' 24 h change).
   `past_days` and `trends` were tried and dropped.
4. Groups and alpha chosen together on validation (backward elimination → tune alpha → repeat
   until stable). Alpha 300 and 1000 were within 0.01; we use 1000 (more regularisation).
5. Smearing correction `exp(σ²/2)` for the log → level bias.
6. Mix: 0.9 × model + 0.1 × persistence (weight chosen on validation).
7. State-wide bias correction (from operational NOAA practice): add 0.3 × the state-mean log
   error of *yesterday's* forecast (known at the issue time). 0.3 chosen on Sep–Oct only and
   checked on November (41.51 → 40.91). Nothing is refitted; `forecast()` recomputes yesterday's
   forecast from the history it is given.
8. Final refit on all 153,036 noon targets up to 30 Nov; saved as plain numpy arrays in
   `submission/model.pkl` (no scikit-learn needed at grading time).

## 4. Where things are

| Path | What |
|---|---|
| `mp1-package/` | Course data, starter kit, assignment PDF (unchanged) |
| `mp1-solution/part0_kanpur_cleaning.ipynb` | Part 0 (Kanpur cleaning) |
| `mp1-solution/bihar_forecasting.ipynb` | Parts 1–5; rewrites `submission/model.pkl` and the figures |
| `mp1-solution/submission/` | `forecast.py`, `features.py`, `model.pkl`, `bihar_stations.csv` |
| `mp1-solution/report.md` → `report.pdf` | Report, exactly **6 pages = the limit** |
| `mp1-solution/build_report.py` | Rebuilds the PDF from the Markdown (`pip install markdown`; uses Edge/Chrome) |
| `CS771_MP1_submission/` | Exact copy of what goes into the zip |
| `analysis/` | November error plots and a per-station error table (not part of the submission) |
| `comms/` | Message folders between teammates' assistants (see `comms/README.md`) |

## 5. How to rerun and check

```
pip install -r mp1-package/requirements.txt markdown
cd mp1-solution
jupyter nbconvert --to notebook --execute --inplace part0_kanpur_cleaning.ipynb   # ~20 s
jupyter nbconvert --to notebook --execute --inplace bihar_forecasting.ipynb       # ~2 min
cd ../mp1-package && python starter/check_submission.py ../mp1-solution/submission
cd ../mp1-solution && python build_report.py
```
After any change: copy the changed files into `CS771_MP1_submission/`, rebuild the zip, unzip it
somewhere and run the checker on the unzipped `submission/` folder.

## 6. Older versions (in case you need to go back)

| Version | Validation | Where |
|---|---|---|
| Current (bias correction) | 28.04 / 15.56 | `main` |
| Without bias correction | 28.39 / 15.98 | git tag `v-28.39-no-bias-correction`; `CS771_MP1_submission_backup_28.39.zip` on Akash's PC |
| With `net_spread_now` feature | 28.09 / 15.88 | git tag `v-28.09-with-net-spread`; `CS771_MP1_submission_backup_28.09.zip` on Akash's PC |
| Original | 28.53 / 15.96 | commit `590736f` |

(`net_spread_now` was removed at Akash's request.)

## 7. Things already tried that did NOT help (don't repeat)

Recent-month weighting (incl. Oct ×4, tested with a Jan–Oct fit on November: worse), Huber loss,
random forest, gradient boosting, squared/interaction terms, fewer lags or PCA, heteroscedastic
smearing, lat/lon, K-means station clusters, separate models for the dirtiest 2–50% of stations,
season-only training, analog ensemble, per-station bias correction, blending with the 29-feature
ridge. Details are in the report ("Other things we tried") and the git history.

## 8. Rules to keep in mind

- Split by target date: train ≤ 31 Aug, validation 1 Sep – 30 Nov; final refit ≤ 30 Nov.
- Everything fitted (scalers, models, clusters) on training rows only; no external/weather data;
  no neural networks.
- `forecast()` may compute features from the history but must not refit or update parameters.
- Report ≤ 6 pages, ending with the Use of AI tools section (already there).

## 9. Open items

- [ ] Upload the zip (once) before the deadline.
- [ ] Every member reads the report and the notebook before the 10-minute viva.
- [ ] Optional: nothing else is required.
