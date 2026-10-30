# Brutal critique of the MP1 solution (main @ cfb53d4), with November error analysis

From: Aryan's Claude, 30 Oct 2026, ~19:00 IST. Deadline is **tonight, 23:59**.
Everything below was re-run from scratch. Scripts and figures are in this folder (see the end).

## TL;DR

1. **The zip has not been uploaded (per the handoff). Do that first.** The Dropbox request
   allows one upload with no edits. Every other item here is optional.
2. The solution is sound. It reproduces to the second decimal, I found no leakage, the checker
   passes, and it beats both baselines by more than the noise. But it is
   **"persistence plus a modest learned nudge"**: 28.04 vs ridge 29.58 vs persistence 32.75.
3. **Brute-forcing validation RMSE now is the wrong game.** The rubric rewards protocol, not
   error. Part 5 is pass/fail against persistence, and we already beat it by 4.7 RMSE (95% CI
   [3.4, 6.2]). About half of the recent "improvements" are inside the noise floor.
4. **November is not learnable much further from PM2.5 alone.** 81% of November's squared
   error is single stations jumping, not the state moving together. Even a *perfect* state-wide
   daily forecast only takes November from 40.9 to 36.7. 29 Nov alone is 15% of November's
   squared error, and nothing in the data at the issue time hinted at it.
5. Of six new ideas I tested, only one helped beyond noise: **compass-sector "upwind"
   features**, worth −0.18 overall and −0.34 on November. I would **not** put it in tonight.

## 0. What I checked

| check | result |
|---|---|
| Reproduce Part 1/2 numbers independently | exact: 32.75 / 29.58 / 28.04 all, 47.81 / 43.06 / 40.91 Nov |
| `check_submission.py` on current `submission/` | OK. Median 0.08 s per call; Nov sanity 40.84 (persistence 47.81) |
| `CS771_MP1_submission/` vs `mp1-solution/` | identical (notebooks, report.pdf, outputs, submission); 11 files |
| Leakage | none found. Features read `A[:p+1]` only; scaler is inside the pipeline fitted on train; the bias correction uses the forecast whose target *is* the issue time |
| Allowed methods (course slides L1–16) | ridge is fine. Also allowed and **not** neural: forests and **boosting** (L4), ridge/lasso/absolute loss (L7), kernels incl. **random Fourier features** (L10), probabilistic regression (L13), clustering (L14), PCA/SVD/matrix factorisation/kernel PCA (L15–16). Neural nets start at L19 |
| Stale branch | `origin/finalise-submission` uses `model.json` and an older report. **Do not submit from it**; main is the current state |

## 1. Priorities (the most brutal part)

- **Upload first.** A missing submission scores zero. A 0.2 RMSE gain is worth ~0 marks.
- The rubric says: *"Grading rewards a sound protocol and clear reasoning over a low error."*
  Part 5 gives 5 points for beating persistence on December. On a pseudo-December test
  (fit on targets ≤ 31 Oct, score November), the full pipeline gets **41.02 vs persistence
  47.81**. The margin is ~7, so those 5 points are as safe as they can be.
- **Every change tonight cascades.** The notebook must be re-run. `report.md` has every number
  typed in and is at the 6-page limit (results table, ablation table, Part 3/4 tables). Then
  the PDF is rebuilt, `CS771_MP1_submission/` re-copied, the zip rebuilt, and the checker run on
  the unzipped folder. One mismatch between report and notebook costs more than any RMSE gain.
- "Brute force on validation" means choosing on the same 91 days you then report. The rules
  allow it, but every extra knob makes 28.04 more optimistic and less predictive of December.

## 2. Most recent tuning is inside the noise floor

Validation has **91 effectively independent days, not 42,588 rows**. Errors on the same day
are strongly correlated across stations: state-relative station errors correlate 0.55 under
5 km and still 0.2 at 50 km (Fig. 2e), on top of the shared state-wide error. So I
bootstrapped whole days (4,000 resamples):

| comparison (negative = first is better) | all: Δ RMSE [95% CI] | Nov: Δ RMSE [95% CI] |
|---|---|---|
| final − persistence | −4.71 [−6.17, −3.38] | −6.90 [−9.50, −4.51] |
| final − ridge (29 features) | −1.54 [−2.43, −0.77] | −2.15 [−3.87, −0.57] |
| bias correction on vs off | −0.35 [−0.95, +0.15] | −0.60 [−1.76, +0.41] |
| blend 0.9 vs pure model | −0.10 [−0.24, +0.03] | −0.01 [−0.26, +0.22] |
| alpha 1000 vs 300 | +0.01 [−0.18, +0.22] | −0.02 [−0.41, +0.38] |
| keeping `variability` | −0.03 [−0.08, +0.02] | −0.01 [−0.11, +0.08] |
| keeping `neighbourhood` | −0.09 [−0.24, +0.08] | −0.11 [−0.42, +0.25] |

- The headline claims hold: we beat both baselines, overall and on November.
- `variability` (+0.03) and `neighbourhood` (+0.08) have **not** shown that they "earn their
  place", which is the brief's own test. An examiner who asks "is +0.03 real?" should hear
  "no, it is within noise; we kept it for X". Have that answer ready for the viva.
- The bias correction, the main November fix, has a CI that crosses 0. It is still
  defensible: λ was chosen on Sep–Oct and checked on November, and the pseudo-December test
  also prefers λ 0.3–0.5 (41.51 → 41.02). Just don't oversell it.

## 3. What the model actually learned

- **Station level:** corr(predicted change, true change) = 0.46. The predicted log-change std
  is 0.25; the true std is 0.47 (Fig. 2b).
- **State level:** over the 30 November days the corr is 0.43, but the model never predicts a
  state-wide move larger than 0.21 in log units. True moves reach 0.75 (Fig. 2c). Removing the
  state-level part entirely makes things worse (29.08), so it has real skill; it is just timid.
- **The seasonal drift is badly calibrated in months it has not seen.** Out of sample, the
  Jan–Aug model's mean predicted daily log change in September is **+0.114** against a true
  **0.000**. The sin/cos calendar terms plus the 30-day-level terms learned a drift from
  Jan–Aug that doesn't transfer, and the λ bias correction patches over this. Even in-sample,
  the final Jan–Nov refit is off by ±0.05–0.10 per month (Jan −0.09 vs +0.02, Sep +0.09 vs
  0.00, Nov −0.05 vs +0.01). For December, the calendar term contributes about +0.10 log per
  day (November: +0.12), which other terms offset. If December's trend differs from
  November's, that is where the error will come from. It can't be tested; say so as a
  limitation.

## 4. Errors in our own November analysis

- `analysis/november_errors.png` panel 4a and the report sentence *"Targets above 200 … we
  forecast too low by 79 on average"* **condition on the outcome**. That is regression to the
  mean: any imperfect forecast looks too low on high outcomes and too high on low ones. It is
  not a fixable bias. It was also the motivation for the `variability` group, which added
  0.03, inside noise.
- The fair check conditions on the **forecast** (Fig. 2a). The bias is about 0 in every bin
  except the top one, where our highest forecasts are **+14 µg/m³ too high**. If anything we
  should shrink high forecasts *more*, the opposite of what the outcome-conditioned plot
  suggests.
- Optional, low-risk report fix: reword that sentence to *"…and these days are where most of
  the error is"*, without the "too low by 79" framing. Only do it if `build_report.py` works on
  this machine and the PDF stays at 6 pages.

## 5. Smaller points an examiner could pick at

- The ablation table is run on the pre-mix, pre-correction model. The brief says "starting
  from your final model". The report does say "before mixing", but that is still not the
  final model.
- The forest/boosting comparison is confounded. Trees see calendar and absolute-level features
  whose validation values are outside the training range (Sep–Nov dates never appear in
  training), and trees extrapolate flat. I re-tested boosting on the ridge *residual* without
  the calendar features: still worse (31.77 vs 28.50). The conclusion survives, but "overfits
  Jan–Aug patterns" is a hypothesis, not something we showed.
- `weekend` uses the issue day's weekday, not the target day's. Harmless, but odd if asked.
- `forecast.py` indexes by position (`p - 24`, `p - 167`), which assumes a gap-free hourly
  index. The README promises every hour, so it's fine. `reindex(pd.date_range(...))` would
  make it bulletproof, but **don't touch it tonight** unless you re-run the checker on the
  rebuilt zip.

## 6. Where November's error actually is (figures in this folder)

**Fig. 1: `nov_1_where_the_error_is.png`**
- November is **70%** of all validation squared error.
- **29 Nov = 15%** of November's squared error, the worst 5 days 42%, the worst 10 days 62%.
  November RMSE without 29 Nov is 38.28.
- **5.6% of station-days cause 50%** of November's squared error; 19.7% cause 80%.
- Forecasts of 150–300 µg/m³ are 19% of rows but **47%** of squared error. RMSE inside the
  200–300 band is 82.
- Of November's squared error, **19.5%** is the whole state being off together, **3.5%** is a
  persistent per-station offset, and the rest is station-day scatter. Ceilings: a perfect
  state-wide daily level gives 36.71; a perfect per-station November offset gives 40.65.

**Fig. 2: `nov_2_is_it_predictable.png`**
- 2a is the fair conditional-bias check (see section 4).
- 2d: errors **don't carry over**. The lag-1 autocorrelation of the state-wide error is 0.24
  (n = 30, standard error about 0.18, so not significant). For station-relative errors it is
  0.06. The bias correction has already taken what yesterday's error can give. Don't build
  more lagged-error corrections.
- 2e: state-relative errors are **spatially clustered up to ~60 km**. This is the one structure
  the features don't capture: regional events smaller than the state. It's why directional
  neighbourhoods help a little.
- 2f: we beat persistence at **420 of 468** stations in November (median RMSE ratio 0.86). The
  losing stations are scattered, not clustered.

**Fig. 3: `nov_3_end_of_month_event.png`**
- At the issue time for 29 Nov (28 Nov 12:00), every longitude band looked like the previous
  days. The night of 28→29 Nov peaked as high as ever (~250). The collapse happened **during
  the morning of 29 Nov, statewide** (Fig. 3c is blue almost everywhere). PM2.5 history cannot
  forecast it; only weather data could. **Stop optimising for 29 Nov.**
- In November, 12:00 sits on the steep falling side of the daily cycle: the night peak is
  ~250 and the afternoon trough ~110–150. So the noon value is very sensitive to how early and
  how strongly the morning air mixes, which is weather we don't observe. This is the main
  reason the station-level scatter is so large.

## 7. Ideas I tested (so nobody repeats them)

Pre-mix model fitted on Jan–Aug; reference **28.50 all / 41.52 Nov**.

| idea | all | Nov | verdict |
|---|---|---|---|
| level-weighted log loss, w = (1+today)^k, k = 0.5…2 (to match level RMSE) | 28.80 … 32.15 | 41.92 … 47.09 | worse, gets worse as k grows |
| RFF kernel ridge (RBF, D = 600, ℓ = 6 / 10) + linear (L10) | 28.49 / 28.44 | 41.67 / 41.53 | flat |
| ridge + boosted trees on the residual (no calendar features) | 31.77 | 47.99 | worse |
| drop the state-wide predicted change (keep station-relative part) | 29.08 | 42.61 | worse |
| shrink the state-wide predicted change × 0.5 | 28.59 | 41.73 | worse |
| 30–150 km ring mean, no direction (2 features) | 28.47 | 41.48 | flat |
| **4 compass sectors (E/N/W/S), 30–150 km: mean now + 24 h change (8 features)** | **28.26** | **41.05** | better |
| ↳ same, full pipeline (blend 0.9, λ 0.3) | **27.86** | **40.56** | Δ vs final: −0.18 [−0.32, −0.05] all; −0.34 [−0.62, −0.08] Nov |

The undirected ring of the same stations does nothing, so the *direction* carries the
signal. The largest coefficients are on the E and N sectors' 24 h change, which fits regional
transport. Caveats:
1. This is the best of the ideas I tried on the *same* validation set, so its CI is optimistic.
2. The weights were learned pooled over Jan–Aug, which includes the monsoon months with
   easterly winds, so the direction may not hold in winter.

If the team still wants it in: add a `sectors` group to `features.py`, re-run the notebook,
update the results and ablation tables and the Part 3/4 numbers in `report.md`, rebuild the
PDF (≤ 6 pages), re-copy into `CS771_MP1_submission/`, rebuild the zip, and run the checker on
the **unzipped** folder. Realistically that is 1.5–2 hours with no margin for mistakes.
**My recommendation: ship main as is.**

## 8. Questions to prepare for the viva

- Why is the λ bias correction not leakage? (Its error is for a target at the issue time.)
- What does ridge shrink towards in log-change space? (Persistence, not the global mean.)
- How many independent samples are in validation, and is +0.03 in the ablation real? (~91 days;
  no.)
- Why does persistence dip back at 24 h lead? (Same point in the daily cycle.)
- Why can't the model forecast 29 Nov? (Morning ventilation that day; no weather inputs.)

## Files in this folder

| file | what |
|---|---|
| `repro.py` | independent reproduction of Part 1/2 numbers (~5 s) |
| `analysis.py` | bootstrap CIs, November anatomy, oracles, pseudo-December test, untried ideas (~1 min) |
| `sector_check.py`, `drift_check.py` | sector-feature significance; drift diagnostics |
| `plot_nov.py` | the three figures and two CSVs below |
| `nov_by_day.csv` | per target day: state means (observed / ours / persistence), RMSEs, bias, share of Nov squared error |
| `nov_by_station.csv` | per station, sorted by share of Nov squared error: coordinates, level, RMSEs, bias, lag-1 error autocorrelation |

Run from the repo root: `python analysis/nov_critique/plot_nov.py`. All use the course data in
`mp1-package/data` and the team's own `features.py`; nothing in the submission is modified.
