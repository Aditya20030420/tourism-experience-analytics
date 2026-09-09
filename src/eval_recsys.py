"""Evaluate the recommendation system: RMSE (rating prediction) + MAP/Precision@K (ranking).

Run:  python src/eval_recsys.py   -> appends "recommendation" to artifacts/metrics.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RANDOM = 42
K = 10


def _rating_rmse(train, test):
    """Item-item CF rating prediction: predict a held-out rating as the similarity-
    weighted average of the same user's other TRAIN ratings. RMSE over test rows."""
    train = train.groupby(["UserId", "AttractionId"], as_index=False)["Rating"].mean()  # csr sums dups
    users = train["UserId"].astype("category")
    items = train["AttractionId"].astype("category")
    ucat, icat = users.cat.categories, items.cat.categories
    mat = csr_matrix((train["Rating"], (users.cat.codes, items.cat.codes)),
                     shape=(len(ucat), len(icat)))
    sim = cosine_similarity(mat.T)  # item x item
    ipos = {a: i for i, a in enumerate(icat)}
    upos = {u: i for i, u in enumerate(ucat)}
    global_mean = train["Rating"].mean()
    dense = mat.toarray()

    preds, actuals = [], []
    for u, a, r in zip(test["UserId"], test["AttractionId"], test["Rating"]):
        if u not in upos or a not in ipos:
            preds.append(global_mean); actuals.append(r); continue
        urow = dense[upos[u]]
        rated = np.where(urow > 0)[0]
        if len(rated) == 0:
            preds.append(global_mean); actuals.append(r); continue
        w = np.clip(sim[ipos[a], rated], 0, None)  # negative cosine -> no weight
        preds.append(np.dot(w, urow[rated]) / w.sum() if w.sum() > 1e-8 else global_mean)
        actuals.append(r)
    rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2)))
    return round(rmse, 4)


def _ranking_map(train, test, k=K):
    """For each user, recommend top-k from CF neighbours of their TRAIN attractions;
    score against the attractions they actually visited in TEST (held out)."""
    tr = train.groupby(["UserId", "AttractionId"], as_index=False)["Rating"].mean()
    users = tr["UserId"].astype("category")
    items = tr["AttractionId"].astype("category")
    mat = csr_matrix((tr["Rating"], (users.cat.codes, items.cat.codes)),
                     shape=(len(users.cat.categories), len(items.cat.categories)))
    sim = cosine_similarity(mat.T)
    item_ids = list(items.cat.categories)
    ipos = {a: i for i, a in enumerate(item_ids)}
    train_hist = train.groupby("UserId")["AttractionId"].apply(set).to_dict()
    test_hist = test.groupby("UserId")["AttractionId"].apply(set).to_dict()

    aps, precs, hits, n = [], [], 0, 0
    for u, truth in test_hist.items():
        seed = train_hist.get(u)
        if not seed:
            continue
        scores = {}
        for a in seed:
            if a in ipos:
                for j, s in enumerate(sim[ipos[a]]):
                    cand = item_ids[j]
                    if cand not in seed:
                        scores[cand] = scores.get(cand, 0) + s
        if not scores:
            continue
        ranked = sorted(scores, key=scores.get, reverse=True)[:k]
        rel = [1 if a in truth else 0 for a in ranked]
        n += 1
        precs.append(sum(rel) / k)
        hits += 1 if sum(rel) > 0 else 0
        if sum(rel):  # average precision for this user
            ap, correct = 0.0, 0
            for i, r in enumerate(rel, 1):
                if r:
                    correct += 1
                    ap += correct / i
            aps.append(ap / min(len(truth), k))
        else:
            aps.append(0.0)
    return {"MAP@%d" % k: round(float(np.mean(aps)), 4),
            "Precision@%d" % k: round(float(np.mean(precs)), 4),
            "HitRate@%d" % k: round(hits / n, 4),
            "users_evaluated": n}


def main():
    df = pd.read_parquet(ROOT / "data" / "master.parquet")
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM)
    res = {"RMSE": _rating_rmse(train, test)}
    res.update(_ranking_map(train, test))
    print("Recommendation eval:", res)

    metrics_path = ART / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["recommendation"] = res
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("Updated", metrics_path)


if __name__ == "__main__":
    main()
