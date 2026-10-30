# CS771 Mini-Project 1: solution

Put this folder **next to** `mp1-package/`, because the notebooks read the data from `../mp1-package/data`.

```
pip install -r ../mp1-package/requirements.txt
jupyter notebook
```

| File | What it is |
|---|---|
| `part0_kanpur_cleaning.ipynb` | Part 0: Kanpur cleaning, flags, calibration check |
| `bihar_forecasting.ipynb` | Parts 1–5: baselines, our model, ablation, lead time, maps, final refit |
| `submission/` | Part 5 hand-in: `forecast.py`, `features.py`, `model.pkl`, `bihar_stations.csv` |
| `outputs/kanpur_hourly_clean.csv` | The cleaned Kanpur file |
| `outputs/figures/` | Figures used in the report |
| `report.md`, `report.pdf` | The report (6 pages) |
| `build_report.py` | Rebuilds `report.pdf` from `report.md` (`pip install markdown`, needs Edge or Chrome) |

Run order: `part0_kanpur_cleaning.ipynb` (about 20 s), then `bihar_forecasting.ipynb` (about 1 min).
The second one rewrites `submission/model.pkl`.

Check the submission from inside `mp1-package/`:
```
python starter/check_submission.py ../mp1-solution/submission
```

`submission/features.py` holds all the feature code, so the notebook and `forecast.py` compute the
features in exactly the same way.
