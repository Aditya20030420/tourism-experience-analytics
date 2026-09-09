"""Train + compare regression (Rating) and classification (VisitMode) models.

Run:  python src/models.py
Outputs: artifacts/reg_model.joblib, artifacts/clf_model.joblib,
         artifacts/metrics.json, artifacts/*.png
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix)
from xgboost import XGBRegressor, XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
RANDOM = 42

# Raw id/count features fed straight to tree models (no one-hot needed).
BASE_FEATS = ["ContinentId", "RegionId", "CountryId", "CityId", "AttractionId",
              "AttractionTypeId", "AttractionCityId", "VisitYear", "VisitMonth"]


def _add_leakfree_aggs(train, test, key, target, name):
    """Mean-encode `target` by `key` using TRAIN only; map onto both. Fills global mean."""
    m = train.groupby(key)[target].mean()
    gm = train[target].mean()
    train[name] = train[key].map(m).fillna(gm)
    test[name] = test[key].map(m).fillna(gm)
    return train, test


def load():
    df = pd.read_parquet(ROOT / "data" / "master.parquet")
    df["AttractionTypeId"] = df["AttractionTypeId"].astype(float)  # was Int64
    df["CityId"] = df["CityId"].fillna(-1)
    return df


# ---------------- Regression: predict Rating ----------------
def run_regression(df, metrics):
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM)
    train, test = _add_leakfree_aggs(train, test, "AttractionId", "Rating", "attr_avg_rating")
    train, test = _add_leakfree_aggs(train, test, "UserId", "Rating", "user_avg_rating")
    feats = BASE_FEATS + ["attr_avg_rating", "user_avg_rating"]
    Xtr, Xte = train[feats], test[feats]
    ytr, yte = train["Rating"], test["Rating"]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                                random_state=RANDOM),
    }
    results, best, best_r2 = {}, None, -1e9
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        r2 = r2_score(yte, pred)
        results[name] = {"R2": round(r2, 4),
                         "RMSE": round(float(np.sqrt(mean_squared_error(yte, pred))), 4),
                         "MAE": round(float(mean_absolute_error(yte, pred)), 4)}
        print(f"[REG] {name}: {results[name]}")
        if r2 > best_r2:
            best_r2, best = r2, (name, mdl)
    metrics["regression"] = {"models": results, "best": best[0], "features": feats}
    joblib.dump({"model": best[1], "features": feats,
                 "attr_avg": train.groupby("AttractionId")["Rating"].mean().to_dict(),
                 "user_avg": train.groupby("UserId")["Rating"].mean().to_dict(),
                 "global_avg": float(train["Rating"].mean())},
                ART / "reg_model.joblib")
    print("Saved best regressor:", best[0])


# ---------------- Classification: predict VisitMode ----------------
def run_classification(df, metrics):
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM,
                                   stratify=df["VisitMode"])
    train, test = _add_leakfree_aggs(train, test, "AttractionId", "Rating", "attr_avg_rating")
    feats = BASE_FEATS + ["Rating", "attr_avg_rating"]
    Xtr, Xte = train[feats], test[feats]
    ytr, yte = train["VisitMode"], test["VisitMode"]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        # depth/leaf caps keep the serialized model tiny (~1.1GB -> a few MB) with ~same accuracy
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=5,
                                               n_jobs=-1, class_weight="balanced", random_state=RANDOM),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                                 random_state=RANDOM),
    }
    # XGBoost needs 0-based labels.
    classes = sorted(ytr.unique())
    remap = {c: i for i, c in enumerate(classes)}
    inv = {i: c for c, i in remap.items()}

    results, best, best_f1 = {}, None, -1
    for name, mdl in models.items():
        if name == "XGBoost":
            mdl.fit(Xtr, ytr.map(remap))
            pred = pd.Series(mdl.predict(Xte)).map(inv).values
        else:
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
        f1 = f1_score(yte, pred, average="weighted")
        results[name] = {"accuracy": round(accuracy_score(yte, pred), 4),
                         "f1_weighted": round(f1, 4),
                         "precision_weighted": round(precision_score(yte, pred, average="weighted", zero_division=0), 4),
                         "recall_weighted": round(recall_score(yte, pred, average="weighted"), 4)}
        print(f"[CLF] {name}: {results[name]}")
        if f1 > best_f1:
            best_f1, best, best_pred = f1, (name, mdl), pred
    metrics["classification"] = {"models": results, "best": best[0], "features": feats}

    # confusion matrix for the best model
    mode_map = df.drop_duplicates("VisitMode").set_index("VisitMode")["VisitModeName"].to_dict()
    labels = sorted(yte.unique())
    cm = confusion_matrix(yte, best_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[mode_map[l] for l in labels],
                yticklabels=[mode_map[l] for l in labels], ax=ax)
    ax.set_title(f"Confusion matrix - {best[0]}")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    fig.tight_layout(); fig.savefig(ART / "confusion_matrix.png", dpi=120); plt.close(fig)

    joblib.dump({"model": best[1], "features": feats, "remap": remap, "inv": inv,
                 "is_xgb": best[0] == "XGBoost",
                 "attr_avg": train.groupby("AttractionId")["Rating"].mean().to_dict(),
                 "global_avg": float(train["Rating"].mean()),
                 "mode_map": mode_map},
                ART / "clf_model.joblib")
    print("Saved best classifier:", best[0])
    print(classification_report(yte, best_pred, target_names=[mode_map[l] for l in labels], zero_division=0))


def main():
    df = load()
    metrics = {}
    run_regression(df, metrics)
    run_classification(df, metrics)
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nWrote artifacts/metrics.json")


if __name__ == "__main__":
    main()
