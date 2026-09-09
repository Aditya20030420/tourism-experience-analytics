# Tourism Experience Analytics — Project Report

**Domain:** Tourism · **Data:** 9 linked tables, 52,930 visit transactions from
33,530 users across 30 attractions (2013–2022).

---

## 1. Approach

A single pipeline runs the project end to end:

1. **Clean & merge** ([src/data_prep.py](src/data_prep.py)) — the 9 Excel tables
   are joined into one transaction-level master. Foreign-key ids (continent,
   region, country, city, attraction type, visit mode) are resolved to names;
   the `AttractionTypeId` column, stored as text, is coerced to integer; the
   `0 = "-"` placeholder rows and 4 users with missing cities are labelled
   *Unknown* rather than dropped. Output: `data/master.parquet` (52,930 × 22).
2. **EDA** ([src/eda.py](src/eda.py)) — distributions, popularity, and
   demographic cross-tabs saved to `reports/figures/`.
3. **Modelling** ([src/models.py](src/models.py)) — regression and classification,
   three models each, best kept by the appropriate metric. Category ids are fed
   directly to tree models; mean-encoded aggregates (`attr_avg_rating`,
   `user_avg_rating`) are computed **train-only** to avoid leakage.
4. **Recommendation** ([src/recommend.py](src/recommend.py)) — collaborative
   (item-item cosine), content-based (TF-IDF), and a per-user hybrid, evaluated
   in [src/eval_recsys.py](src/eval_recsys.py).
5. **App** ([app.py](app.py)) — a Streamlit interface for predictions,
   recommendations, and dashboards.

---

## 2. Model results

**Regression — predict a user's rating (1–5).** Best: **XGBoost**, MAE **0.72**,
RMSE 0.99, R² ≈ 0. Ratings are heavily skewed high (mean **4.16**, ~70% are 4–5),
so they are close to unpredictable from demographics and attraction features —
R² near zero is the honest result. In practice the model still lands within
~0.7 stars, which is enough to flag likely low-rated visits.

**Classification — predict visit mode (5 classes).** Best by weighted F1:
**RandomForest**, F1 **0.46**, accuracy 0.47. The classes are very imbalanced
(Couples 41% vs Business 1%), so models use balanced class weights and are judged
on F1, not accuracy. Couples and Family predict well (F1 0.57 / 0.47); Business
and Solo are hard due to scarcity.

**Recommendation.** Rating RMSE **1.02** (item-item CF), ranking **MAP@10 0.34**,
**HitRate@10 0.62** — for ~62% of users a genuinely-visited attraction appears in
the top-10 suggestions built from their history.

---

## 3. Key findings

- **Visit mode is dominated by leisure pairs and families.** Couples (41%) and
  Family (29%) are two-thirds of all trips; Business is negligible (1.2%). Product
  and marketing should assume a leisure audience by default.
- **Demand is concentrated in a handful of Bali attractions.** The Sacred Monkey
  Forest Sanctuary alone accounts for 13,198 visits (25% of all transactions),
  followed by Waterbom Bali and Tegalalang Rice Terrace. **Nature & Wildlife
  Areas** and **Beaches** are the top categories by volume.
- **Satisfaction varies sharply by attraction type.** Water Parks (4.65), National
  Parks (4.42) and Nature & Wildlife Areas (4.27) rate highest; **Historic Sites
  (3.54) and Beaches (3.85)** rate lowest despite Beaches being the #2 volume
  category — a clear service-quality gap on a high-traffic segment.
- **Traffic peaked in 2016 and has since declined** (12.8k visits in 2016 →
  ~0.5k in 2020, reflecting the pandemic). July–September is the seasonal peak.
- **Travel style shifts by region.** Europe skews strongly to Couples (51%);
  Australia & Oceania skews to Family (39%). Same product, different framing per
  market.

---

## 4. Actionable insights

1. **Fix the Beaches experience.** Beaches are the second most-visited type but
   among the lowest rated (3.85). A targeted service/expectation review here lifts
   satisfaction on one of the largest audiences — the highest-leverage single move.
2. **Lead with nature & water.** Water Parks and National Parks both draw volume
   *and* top ratings — feature them in recommendations and promotions to reinforce
   what already delights users.
3. **Segment marketing by region.** Promote couples packages in Europe and
   family packages in Australia & Oceania, matching the observed visit-mode mix.
4. **Reduce dependence on one hotspot.** A quarter of all demand sits on a single
   attraction; use the recommender to steer traffic toward high-rated but
   under-visited sites and spread load.
5. **Use the visit-mode classifier for pre-arrival targeting.** Predicting Couples
   vs Family from user profile and destination lets platforms pre-select
   amenities and offers before booking is complete.

---

## 5. Deliverables

Cleaned dataset (`data/master.parquet`), full source ([src/](src/)), the Streamlit
app ([app.py](app.py)), saved models and metrics (`artifacts/`), EDA figures
(`reports/figures/`), and this report. Reproduce everything with the steps in
[README.md](README.md).
