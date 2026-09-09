"""Build recommendation artifacts: collaborative (item-item) + content-based.

Run:  python src/recommend.py  -> artifacts/recsys.joblib
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


def build():
    df = pd.read_parquet(ROOT / "data" / "master.parquet")

    # ---- Collaborative filtering: item-item cosine on user x attraction ratings ----
    # Keep attractions with enough signal so the similarity matrix stays meaningful/small.
    pop = df["AttractionId"].value_counts()
    keep = pop[pop >= 20].index
    d = df[df["AttractionId"].isin(keep)]
    dd = d.groupby(["UserId", "AttractionId"], as_index=False)["Rating"].mean()  # csr sums dups
    users = dd["UserId"].astype("category")
    items = dd["AttractionId"].astype("category")
    mat = csr_matrix((dd["Rating"], (users.cat.codes, items.cat.codes)))
    item_sim = cosine_similarity(mat.T)  # (n_items x n_items)
    item_ids = list(items.cat.categories)

    # ---- Content-based: TF-IDF over attraction name + type + city ----
    meta = (df.drop_duplicates("AttractionId")
              .set_index("AttractionId")[["Attraction", "AttractionType", "AttractionCity"]])
    meta["soup"] = (meta["Attraction"] + " " + meta["AttractionType"] + " " + meta["AttractionCity"]).str.lower()
    tfidf = TfidfVectorizer(stop_words="english")
    content_mat = tfidf.fit_transform(meta["soup"])
    content_sim = cosine_similarity(content_mat)
    content_ids = list(meta.index)

    # attraction display table + global popularity ranking (cold-start fallback)
    attr_info = meta[["Attraction", "AttractionType", "AttractionCity"]].copy()
    attr_info["avg_rating"] = df.groupby("AttractionId")["Rating"].mean()
    attr_info["n_visits"] = df.groupby("AttractionId")["Rating"].size()

    joblib.dump({
        "item_sim": item_sim, "item_ids": item_ids,
        "content_sim": content_sim, "content_ids": content_ids,
        "attr_info": attr_info,
        "user_history": d.groupby("UserId")["AttractionId"].apply(list).to_dict(),
    }, ART / "recsys.joblib")
    print(f"Saved recsys.joblib | CF items={len(item_ids)} content items={len(content_ids)}")


# ---- inference helpers (imported by the app) ----
def recommend_collaborative(rec, attraction_id, k=10):
    if attraction_id not in rec["item_ids"]:
        return _popular(rec, k)
    i = rec["item_ids"].index(attraction_id)
    sims = rec["item_sim"][i]
    order = np.argsort(sims)[::-1]
    ids = [rec["item_ids"][j] for j in order if j != i][:k]
    return rec["attr_info"].loc[ids].assign(score=[sims[rec["item_ids"].index(x)] for x in ids])


def recommend_content(rec, attraction_id, k=10):
    if attraction_id not in rec["content_ids"]:
        return _popular(rec, k)
    i = rec["content_ids"].index(attraction_id)
    sims = rec["content_sim"][i]
    order = np.argsort(sims)[::-1]
    ids = [rec["content_ids"][j] for j in order if j != i][:k]
    return rec["attr_info"].loc[ids].assign(score=[sims[rec["content_ids"].index(x)] for x in ids])


def recommend_for_user(rec, user_id, k=10):
    """Hybrid: aggregate CF neighbours of the user's visited attractions."""
    hist = rec["user_history"].get(user_id)
    if not hist:
        return _popular(rec, k)
    scores = {}
    for aid in hist:
        if aid in rec["item_ids"]:
            i = rec["item_ids"].index(aid)
            for j, s in enumerate(rec["item_sim"][i]):
                cand = rec["item_ids"][j]
                if cand not in hist:
                    scores[cand] = scores.get(cand, 0) + s
    if not scores:
        return _popular(rec, k)
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return rec["attr_info"].loc[top].assign(score=[scores[x] for x in top])


def _popular(rec, k):
    a = rec["attr_info"]
    return a[a["n_visits"] >= 20].sort_values("avg_rating", ascending=False).head(k).assign(score=np.nan)


if __name__ == "__main__":
    build()
