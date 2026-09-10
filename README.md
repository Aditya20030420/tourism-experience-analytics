# Tourism Experience Analytics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-337AB7)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

Classification, prediction, and recommendation on a tourism dataset (9 linked
tables, 52,930 visit transactions). Ships an end-to-end pipeline + Streamlit app.

![Tourism Experience Analytics demo](docs/demo.gif)

<sub>Demo: predicting a trip, then browsing suggestions, trends, and accuracy. Static screenshot: [docs/app.png](docs/app.png).</sub>

## Summary

A tourism analytics tool that turns raw visit records into three decisions a travel
platform or agency actually cares about: **how much a traveller will enjoy a place,
who they'll travel with, and where to send them next.**

The dataset is 52,930 visits by 33,530 users across 30 attractions, spread over 9
linked tables (transactions, users, attractions, and geography/type lookups). The
pipeline cleans and merges them into one master table, then trains three models —
a **rating regressor**, a **visit-mode classifier**, and an **attraction
recommender** — and serves everything through a Streamlit app with four tabs:

- **Predict a trip** — a traveller profile + destination → predicted rating (1–5 stars)
  and most-likely travel style (Business / Couples / Family / Friends / Solo), with a
  probability breakdown.
- **Suggest places** — attractions to visit next, from a place they loved, a specific
  traveller's history, or crowd favourites.
- **Trends** — dashboards on who travels, where they come from, top-rated place types,
  and visits per year.
- **How accurate** — plain-language honesty check on the models' held-out performance.

**Business value:** flag likely low-rated visits before they hurt reviews, target
marketing to the predicted travel style, lift engagement with personalised
recommendations, and read tourism hotspots and trends at a glance.

## Features

- **Three ML tasks in one app** — rating regression, visit-mode classification, and
  attraction recommendation, each with a multi-model comparison and the best kept.
- **Star-rated predictions** — enjoyment shown as a 1–5 star card plus a per-class
  probability chart for the predicted travel style.
- **Cascading, valid inputs** — traveller continent → region → country narrow each
  other, so you can't pick an impossible origin.
- **Hybrid recommender** — item-item collaborative filtering, TF-IDF content-based,
  a per-user blend, and a popularity fallback for cold start.
- **Interactive dashboards** — themed Altair charts (gradient bars, trend area) for
  who travels, where from, top-rated place types, and volume over time.
- **Plain-language accuracy tab** — held-out performance as stat cards with level
  meters; technical tables tucked away for data teams.
- **Leakage-safe modelling** — mean encodings computed on the train split only, so
  reported metrics reflect genuinely unseen visits.
- **Fast, reproducible** — small depth-capped models are lazy-loaded (~1 s cold
  start), and pinned `requirements.txt` + a one-command pipeline reproduce everything.
- **Polished UI** — animated world-map background, skeuomorphic controls, and
  non-technical wording throughout.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Data wrangling | pandas, NumPy, openpyxl (Excel), PyArrow (Parquet) |
| Machine learning | scikit-learn (regression, classification, TF-IDF, cosine similarity), XGBoost, SciPy (sparse matrices), joblib (model persistence) |
| Visualization | Altair (in-app charts), Matplotlib + seaborn (EDA figures) |
| Web app | Streamlit |
| Assets | Natural Earth world map (GeoJSON → SVG) |

All dependencies are pinned in [`requirements.txt`](requirements.txt).

## Run it

```bash
pip install -r requirements.txt
python src/data_prep.py     # clean + merge 9 tables  -> data/master.parquet
python src/eda.py           # visualizations          -> reports/figures/
python src/models.py        # regression + classification, model comparison -> artifacts/
python src/recommend.py     # collaborative + content-based -> artifacts/recsys.joblib
python src/eval_recsys.py   # recsys RMSE + MAP/Precision@10 -> metrics.json
streamlit run app.py        # the app
```

`data/` holds the raw Excel tables (download from the source Drive folder if missing).

## Usage

Once `streamlit run app.py` is running, open the printed URL (default
`http://localhost:8602`) and use the four tabs:

- **Predict a trip** — pick the traveller's origin (continent → region → country,
  which cascade so only valid combinations show), the place they're visiting, and the
  year/month, then click **Show prediction**. You get a star rating for expected
  enjoyment, the most-likely travel style, and a probability chart across all styles.
- **Suggest places** — choose how to recommend: *places like one they loved* (by
  similar travellers or by place type), *for a specific traveller* (enter a traveller
  ID to use their history), or *crowd favourites*. Set how many suggestions and read
  the ranked table.
- **Trends** — browse dashboards: who people travel with, where travellers come from,
  highest-rated place types, and visits per year. Expand **More charts** for the full
  EDA figures.
- **How accurate** — see held-out model performance as stat cards; open **Technical
  details** for the full per-model metrics tables and confusion matrix.

Everything runs locally on the data in `data/master.parquet` and the saved models in
`artifacts/` — no internet or API keys required.

## What each objective does

**1. Regression — predict a user's attraction rating (1–5).**
Features: user geography ids, visit year/month, attraction id/type/city, plus
leakage-safe train-only mean encodings (`attr_avg_rating`, `user_avg_rating`).
Compared LinearRegression / RandomForest / XGBoost.

| model | R² | RMSE | MAE |
|---|---|---|---|
| Linear | -0.069 | 1.003 | 0.734 |
| RandomForest | -0.077 | 1.007 | 0.733 |
| **XGBoost (best)** | **-0.040** | **0.990** | **0.719** |

Ratings are heavily skewed toward 4–5, so they are close to unpredictable from
the available features — R² near zero is the honest result, not a bug. MAE ≈ 0.72
means predictions land within ~0.7 stars.

**2. Classification — predict visit mode (Business / Couples / Family / Friends / Solo).**
Class-imbalanced (Couples 21.6k vs Business 623), so models use balanced class
weights. Compared LogisticRegression / RandomForest / XGBoost.

| model | accuracy | F1 (weighted) |
|---|---|---|
| Logistic | 0.271 | 0.287 |
| RandomForest (depth-capped) | 0.416 | 0.428 |
| **XGBoost (best F1)** | **0.496** | **0.449** |

Best model picked by weighted F1 (XGBoost) since accuracy alone rewards the majority
class. The RandomForest is depth-capped (`max_depth=16`, `min_samples_leaf=5`) so the
saved model is a few MB instead of ~1.1 GB — this made XGBoost the F1 leader and keeps
app cold-start to ~1 s. Confusion matrix in `artifacts/confusion_matrix.png`.

**3. Recommendation — attraction suggestions.**
- Collaborative: item-item cosine similarity on the user×attraction rating matrix.
- Content-based: TF-IDF over attraction name + type + city.
- Per-user hybrid: aggregate CF neighbours of a user's visited attractions.
- Cold-start fallback: most-popular by average rating.

Evaluation ([src/eval_recsys.py](src/eval_recsys.py)): rating **RMSE 1.02** (item-item CF),
ranking **MAP@10 0.34**, **Precision@10 0.07**, **HitRate@10 0.62** — for ~62% of users
a genuinely-visited attraction lands in the top-10 built from their history.

## Project structure

```
tourism-experience-analytics/
├── app.py                     # Streamlit app: Predict a trip / Suggest places / Trends / How accurate
├── src/
│   ├── data_prep.py           # clean + merge the 9 raw tables -> data/master.parquet
│   ├── eda.py                 # exploratory charts -> reports/figures/
│   ├── models.py              # train + compare regression & classification, save best + metrics.json
│   ├── recommend.py           # build recsys artifacts + inference helpers (imported by app)
│   └── eval_recsys.py         # recommender RMSE + MAP/Precision@10 -> metrics.json
├── data/                      # raw Excel tables (master.parquet is generated, git-ignored)
│   └── Additional_Data_for_Attraction_Sites/Updated_Item.xlsx
├── assets/                    # Natural Earth world map (geojson + projected SVG) for the app background
├── artifacts/                 # generated: saved models, metrics.json, confusion matrix (git-ignored)
├── reports/figures/           # generated: EDA plots (git-ignored)
├── docs/                      # README media (demo.gif, app.png)
├── requirements.txt           # pinned dependencies
├── REPORT.md                  # stakeholder-facing findings & insights
├── LICENSE                    # MIT
└── README.md
```

Generated folders (`artifacts/`, `reports/`, `data/*.parquet`) are git-ignored and
rebuilt by the pipeline — see [Run it](#run-it).

## Notes / deliberate simplifications
- Tree models consume raw category ids directly (no one-hot) — fewer moving parts,
  same accuracy on trees.
- Only 30 of 1,698 catalogued attractions actually appear in transactions, so
  collaborative filtering operates over those 30.
