"""EDA visualizations for the tourism master dataset.

Run:  python src/eda.py   ->  reports/figures/*.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def main():
    df = pd.read_parquet(ROOT / "data" / "master.parquet")

    # 1. rating distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Rating", data=df, ax=ax, palette="viridis", hue="Rating", legend=False)
    ax.set_title("Rating distribution")
    save(fig, "01_rating_distribution.png")

    # 2. users by continent
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df["Continent"].value_counts().index
    sns.countplot(y="Continent", data=df, order=order, ax=ax, color="steelblue")
    ax.set_title("Transactions by user continent")
    save(fig, "02_continent.png")

    # 3. visit mode share
    fig, ax = plt.subplots(figsize=(6, 4))
    df["VisitModeName"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax, ylabel="")
    ax.set_title("Visit mode share")
    save(fig, "03_visit_mode.png")

    # 4. top 15 attraction types by avg rating (min 100 visits)
    g = df.groupby("AttractionType").agg(avg=("Rating", "mean"), n=("Rating", "size"))
    g = g[g.n >= 100].sort_values("avg", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=g.avg, y=g.index, ax=ax, color="darkorange")
    ax.set_title("Avg rating by attraction type (>=100 visits)")
    ax.set_xlabel("avg rating")
    save(fig, "04_type_rating.png")

    # 5. top 15 attractions by visit count
    top = df["Attraction"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="seagreen")
    ax.set_title("Most visited attractions")
    ax.set_xlabel("visits")
    save(fig, "05_top_attractions.png")

    # 6. visit mode vs continent heatmap
    ct = pd.crosstab(df["Continent"], df["VisitModeName"], normalize="index")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Visit mode mix by continent (row-normalized)")
    save(fig, "06_mode_by_continent.png")

    # 7. visits over time
    ts = df.groupby("VisitYear").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ts.plot(marker="o", ax=ax)
    ax.set_title("Transactions per year")
    ax.set_ylabel("visits")
    save(fig, "07_visits_by_year.png")

    print("EDA done ->", FIG)


if __name__ == "__main__":
    main()
