---
title: "CS771 Mini-Project 1: Next-Day PM2.5 Forecasting and Mapping for Bihar"
subtitle: "Group: Akash Kumar (261110008), Aryan Kumar (261110012), Preshit Tiwari (261110047)"
---

All numbers in this report come from our two notebooks, `part0_kanpur_cleaning.ipynb` and
`bihar_forecasting.ipynb`. Both run from top to bottom and give the same results every time. All
errors are in µg/m³. When we say "validation", we mean every station and every 12:00 target from
1 Sep to 30 Nov. That is 91 days × 468 stations = 42,588 forecasts.

# Part 0: Cleaning the Kanpur data

**What the data looks like.** There are 76 sensors with three kinds of ID: 44 have 10-digit IDs,
31 have 8-character IDs, and 1 has a 12-character ID. The 10-digit and 8-character sensors send at
most one reading per hour, always at minute :00, and in practice about 16 to 19 readings a day. The
12-character sensor sends a reading every 15 minutes. No sensor moves: each one always has the same
latitude and longitude. The 10-digit sensors send whole numbers. The 8-character sensors send long
decimals, which look like averages, and their values are a little lower (median 32 against 37).

**Hourly table.** We take the average of all readings inside each hour and label the hour by its
start time. Error codes are removed before taking the average. This only matters for the 15-minute
sensor. For that sensor, the hourly average differs from the :00 reading by 2.6 on average and by
up to 18.5 in a single hour. The average uses four readings instead of one, so it is less noisy.

**Flags.** Each sensor-hour gets the first rule below that matches it. After that, every hour of a
dropped sensor is marked `dropped_sensor`. We do not delete or fill anything. The table shows how
many hours each rule catches on its own. In the final file the counts are: ok 79,512,
dropped_sensor 1,734, stuck 559, spike 141, error_code 136 and zero 49.

| flag | rule and why we chose it | caught |
|---|---|---|
| error_code | the hour has only the value 999 (always exactly 999, and far above the next highest value, 968) | 136 h, 5 sensors (120 from `1204260025`) |
| zero | pm25 = 0 (outdoor PM2.5 is never exactly zero) | 813 h, 21 sensors, almost all from 2 sensors |
| stuck | the same value for 24 or more reported hours in a row (real PM2.5 goes up and down during the day) | 1,693 h, 7 sensors, e.g. 78 for 126 h, 14.2 for 74 h |
| spike | above 200 *and* more than 5 × that hour's median over all sensors (the city median is about 35 in the monsoon) | 141 h, half from 4 sensors |
| outage | fewer than 10% of sensors report in that hour | 117 h (5–10 Aug) where **no** sensor reported at all. Every other hour has at least 12%, so 10% separates the two clearly. These hours have no readings, so no row has this flag. |
| dropped_sensor | more than 50% of a sensor's hours are flagged | `1203260082` (73%), `1203260068` (58%) |

To find sensors that do not follow the rest of the network, we correlated each sensor's ok
readings with the median of all the other sensors. The typical correlation is 0.76. Two sensors
are much lower: `AC0BFBE34CC7` (0.39) and `U8APQDBX` (0.47, which also has the most spikes). We
report them but do not flag them, because a local pollution source next to a sensor could also
give a low correlation.

**Calibration.** 21 of the 10-digit sensors have an 8-character sensor within 3 km. We fitted a
correction on July and tested it on August, using ok hours only:

| correction (fitted on July) | July MAE / RMSE | August MAE / RMSE |
|---|---|---|
| none | 10.79 / 16.98 | 11.73 / 18.07 |
| constant offset (−0.13) | 10.79 / 16.99 | 11.72 / 18.08 |
| straight line, y = 19.9 + 0.48x | 10.46 / 15.45 | 13.11 / 18.81 |

No correction helps in August. Once the bad readings are flagged, the two families already agree on
average, so the best constant offset is almost zero. The straight-line fit looks better on July
but is worse on August, so it only fits the noise in July. Its slope is below 1 because the sensor
we predict from is itself noisy. The real reason shows up when we look at each pair separately.
Each pair has its own offset, between −10 and +10, and that offset stays the same from July to
August (correlation 0.92). Some pairs are positive and some are negative. So the difference comes
from the place and the individual sensor, not from the device type, and one correction for the
whole family cannot remove it.

**Before and after.** Before: 76 sensors, 82,131 sensor-hours, 27.4% missing. After (ok rows
only): 74 sensors, 79,512 sensor-hours, 29.7% missing. In local time (IST), the daily cycle is
lowest in the afternoon and highest at night, even in the monsoon.

# Part 1: Baselines

The baseline ridge model uses 113,256 training rows: 242 target days (3 Jan to 31 Aug) × 468
stations. These are all the targets that have a full 24 hours of history before them.

| | all | Sep | Oct | Nov |
|---|---|---|---|---|
| Persistence RMSE / MAE | 32.75 / 17.47 | 23.33 / 8.91 | 20.22 / 11.97 | 47.81 / 31.70 |
| Ridge (29 features) RMSE / MAE | 29.58 / 18.16 | 20.34 / 10.35 | 19.36 / 13.46 | 43.06 / 30.81 |

**Does one win on both metrics?** No. Ridge has the lower RMSE, but persistence has the lower MAE.
Ridge is trained to make the squared error small, so the few very large misses on polluted days
matter most to it. To avoid those, it pulls its forecasts towards the average: it lowers very high
values and raises low ones. Because of this, its forecasts are too high on average by 3.5 to 6.9
per month, while persistence is about right on average. Pulling towards the average reduces the
big errors, so RMSE goes down. But on the many normal days, today's value is usually closer to
tomorrow's than the pulled value is, so ridge makes many small errors and its MAE goes up. In
simple terms, squared error is smallest when you predict the *mean*, and absolute error is
smallest when you predict the *median*. PM2.5 has a long tail of very high values, so the mean and
the median are quite different.

# Part 2: Improved forecast

**Main ideas.**
(1) We use `log(1 + PM2.5)` for all features, because pollution tends to change by a factor (for
example "twice as bad") rather than by a fixed amount.
(2) The model predicts the change in log PM2.5 from today's reading, so its forecast is "today's
value × some factor". With this target, regularisation pulls the forecast towards persistence
instead of towards a single overall average.
(3) Going back from the log scale makes the forecast slightly too low on average, so we multiply by
a small correction, `exp(σ²/2)`.
(4) The model is ridge regression on standardised features. The scaler is fitted on the training
rows only, inside a scikit-learn pipeline.

**Training rows.** 110,448: every 12:00 target from 9 Jan to 31 Aug at every station. We skip the
first week because some features need 7 days of history. We only use 12:00 issue times, so training
matches the real task exactly.

**Feature groups and how we chose them.** We started with 7 groups. We then removed groups one at a
time, keeping a removal whenever it made validation better (with alpha = 100). Two groups made
validation *worse* and were dropped: `past_days` (the same hour on each of the previous days, and
7 daily averages; 13 features) and `trends` (changes over 3, 6 and 24 hours; 3 features). Most of
what they say is already in the last 24 hourly readings and the 30-day station level. We then
tuned alpha on validation over values from 1 to 10,000. The best was alpha = 300.

**Ablation of the final model.** We remove one group, retrain with the same alpha and compare with
the final RMSE of 28.53:

| group removed | features | validation RMSE | change |
|---|---|---|---|
| recent_hours (last 24 hourly readings) | 24 | 35.66 | +7.13 |
| calendar (sin/cos of day of year, weekend) | 3 | 29.96 | +1.43 |
| station_level (30-day mean, and the same compared with the state) | 2 | 29.15 | +0.62 |
| network (state mean now, over 24 h and over 7 days, and its 24 h change) | 4 | 28.85 | +0.32 |
| neighbourhood (10 nearest stations, within 15 km, within 40 km; now and over 24 h) | 6 | 28.57 | +0.04 |

`recent_hours` gives most of the accuracy. `calendar` tells the model what time of year it is. The
training data has no dates from September to December, so the model is guessing a little here, but
December is similar to early January, which is in the training data. `neighbourhood` adds very
little, because nearby stations move together with the state average and with the station's own
recent readings, which the model already has.

**A more flexible model.** A random forest with the same features and target fits the training
months much better (train RMSE 19.5 against 26.6) but is much worse on validation (36.4 against
28.5). It learns patterns that only hold in January to August, such as winter smog clearing and a
calm monsoon, and these do not carry over to the smog build-up in October and November, a season it
never saw. A tree also cannot predict values outside the range it saw in training. Ridge learns one
smooth rule, which carries over better, so we kept ridge as our final model.

**Other models we tried** (same features and target, validation RMSE / MAE): Bayesian ridge 28.57 /
15.90, Huber loss 28.65 / 16.04, kernel ridge with an RBF kernel (Nystroem) 33.37 / 18.49, and
KNN with k = 200 34.02 / 18.84. Our ridge model gets 28.53 / 15.96. Probabilistic models did not
help. Bayesian ridge gives almost the same forecast as ridge, and our model is already
probabilistic in a simple way: it assumes the change on the log scale is normally distributed, and
`exp(σ²/2)` turns that into the expected value. The non-linear models overfit January to August,
just like the forest. What limits us is the change of season and the missing weather data, not
how flexible the model is.

**Final results (validation, RMSE / MAE)**

| | all | Sep | Oct | Nov |
|---|---|---|---|---|
| Persistence | 32.75 / 17.47 | 23.33 / 8.91 | 20.22 / 11.97 | 47.81 / 31.70 |
| Ridge (29 features) | 29.58 / 18.16 | 20.34 / 10.35 | 19.36 / 13.46 | 43.06 / 30.81 |
| **Our model** | **28.53 / 15.96** | 20.70 / **8.09** | **17.39 / 10.97** | **41.56 / 28.98** |

Our model beats both baselines overall on both RMSE and MAE, and also in October and November. In
September its RMSE is 0.4 worse than ridge's. The improvement is real, but small.

**What would help most:** weather forecasts (wind, rain, mixing height, temperature inversions),
because weather causes most day-to-day changes in PM2.5. Satellite counts of crop fires would also
help in the October–November burning season.

# Part 3: Lead time

![Validation RMSE against lead time (left) and the average PM2.5 by hour of day in Sep–Nov (right).](outputs/figures/lead_time.png){width=100%}

| lead | Persistence | Ridge (29) | Our model |
|---|---|---|---|
| **1 h (nowcast)** | 23.95 / 10.36 | **18.95** / 8.67 | 19.13 / **8.41** |
| 3 h | 51.43 | 28.37 | 31.83 |
| 6 h | 66.82 | 33.72 | 39.41 |
| 12 h | 72.74 | 31.70 | 34.19 |
| **24 h (day-ahead)** | 32.75 / 17.47 | 29.58 / 18.16 | **28.53 / 15.96** |

(Validation RMSE, plus MAE for 1 h and 24 h.)

**Persistence** does not simply get worse as the lead grows. Its error is lowest at 1 h, highest at
12 h (issued at midnight) and lower again at 24 h. Persistence just copies the reading at the issue
time. Night readings are near the daily peak, about 1.5 to 2 times the noon level in September to
November, so a night value is a poor guess for noon. At 24 h the issue time is noon again, the same
point in the daily cycle as the target.

**The learned models** show a smaller version of the same bump: on validation, 6 h is worse than
24 h. On the *training* data the order is the normal one (our model: 23.6 at 6 h against 26.6 at
24 h). With a night issue time, the model has to turn a night value into a noon value, and the
night-to-noon ratio keeps changing. By month it goes from 1.1 (July–August) to 2.5 (March–April),
and it also changes with the weather each day. A single learned conversion is wrong on many
validation days. Noon-to-noon forecasts need no such conversion.

**For a public warning service** we would use the 24 h lead. The 1-hour nowcast is more accurate
(RMSE about 19 against 29), but it leaves no time to act. A day-ahead warning gives people and
schools time to plan. The cost is about 10 µg/m³ of RMSE. In our results, the in-between leads
(3–12 h) are both less useful and less accurate.

# Part 4: Maps

We interpolate the station values onto the 3,396-point grid with inverse-distance weighting (power
2, all stations). We use the same weights for observed and forecast values, so any difference
between maps comes only from the forecasts.

![Observed, our forecast, persistence and our error for a clean day (15 Sep), a polluted day (27 Nov) and our worst day (29 Nov), all at 12:00 IST.](outputs/figures/maps.png){width=100%}

| average over 91 validation days | station RMSE | map RMSE | pattern correlation |
|---|---|---|---|
| Persistence | 28.57 | 15.37 | 0.580 |
| Our model | 24.96 | 13.96 | 0.602 |

On our worst day (29 Nov), the air became much cleaner overnight, but both forecasts stayed at the
previous day's smog level. So the error map is positive almost everywhere.

**Which RMSE is better, and which should we quote?** Map RMSE is lower, because interpolation
averages nearby stations and smooths out the errors at single stations in both maps. We should
quote the **station RMSE** as our forecast accuracy, because it compares forecasts with real
measurements. Map RMSE only compares two interpolated maps.

**Why the observed map is not ground truth.** Between stations, the map shows a weighted average,
not a measurement. It cannot show a hotspot that lies between sensors. It is least reliable where
stations are far apart, such as near the edges of the state. It also carries any sensor errors and
the gap-filling that was done before we got the data, which is not marked.

**Good pattern correlation but poor RMSE.** Imagine a forecast that gets the order of the districts
right (which are cleaner and which are dirtier) but is 40% too low everywhere on a smog day. Its
correlation is close to 1, but its RMSE is large. For a public warning, getting the level right
matters more, because warnings are issued when PM2.5 crosses fixed limits.

# Part 5: Hidden December test

`submission/forecast.py` loads `model.json`. This file holds our Part 2 ridge model, refitted on
all 153,036 noon targets up to 30 Nov with alpha = 300. It stores the scaler means and scales, the
coefficients, the intercept and the log correction. Plain JSON does not depend on any particular
scikit-learn version, and the notebook checks that the saved numbers give exactly the same
forecasts as the fitted model. `forecast.py` uses the same `features.py` as the notebook, plus
`bihar_stations.csv`, and nothing is refitted inside `forecast()`. It only uses rows up to
`issue_time`, puts the columns in the training order and fills any gap with the last reading.
`check_submission.py` passes: the median call takes 0.16 s and the slowest 0.42 s (the limit is
10 s). The November check score (RMSE 41.3, against 47.8 for persistence) is not a test score,
because November is now part of the training data.

**Other checks.** We changed or removed all data after the issue time for 25 random issue times,
and the features stayed exactly the same, so there is no leakage. A practice December run
(November shifted forward by 30 days) gave 30 valid forecasts. We also wrote the two baselines
again separately and got the same numbers.

# Limitations

Validation is only one autumn, and it contains a season (the October–November smog build-up) that
the training data never shows, so our choices are tuned to that one change. Gap-filled values in
the Bihar data are not marked, so part of our error is in the "true" values themselves. Without
weather data, we cannot predict sudden clean-ups like the one on 29 Nov.

# Use of AI tools

We used Claude (Anthropic) as a coding assistant to draft and debug the analysis code, run the
experiments, make the plots and draft parts of this report. We checked every modelling choice
against the validation results. We also checked the leakage rules ourselves: features only use data
up to the issue time, the scaler is fitted on training rows only, and the split is by time. Every
group member has gone through the code and can explain it.
