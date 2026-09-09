"""Load the 9 raw Excel tables, clean them, and merge into one master dataframe.

Run:  python src/data_prep.py
Output: data/master.parquet  (one row per transaction, ids resolved to names)
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ITEM_FILE = DATA / "Additional_Data_for_Attraction_Sites" / "Updated_Item.xlsx"  # 1698 rows, not the 30-row Item.xlsx

# Lookup rows with id 0 are "-" placeholders meaning "unknown".
def _drop_placeholder(df, id_col):
    return df[df[id_col] != 0].copy()


def load_raw():
    """Read every sheet, return a dict of cleaned lookup + fact tables."""
    tx = pd.read_excel(DATA / "Transaction.xlsx")
    user = pd.read_excel(DATA / "User.xlsx")
    item = pd.read_excel(ITEM_FILE)
    city = pd.read_excel(DATA / "City.xlsx")
    country = pd.read_excel(DATA / "Country.xlsx")
    region = pd.read_excel(DATA / "Region.xlsx")
    continent = pd.read_excel(DATA / "Continent.xlsx")
    typ = pd.read_excel(DATA / "Type.xlsx")
    mode = pd.read_excel(DATA / "Mode.xlsx")

    # AttractionTypeId comes in as text ('13') -> make it a clean int to join on Type.
    item["AttractionTypeId"] = pd.to_numeric(
        item["AttractionTypeId"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    ).astype("Int64")

    # Trim stray whitespace in name columns.
    for df, cols in [
        (city, ["CityName"]), (country, ["Country"]), (region, ["Region"]),
        (continent, ["Continent"]), (typ, ["AttractionType"]), (mode, ["VisitMode"]),
        (item, ["Attraction", "AttractionAddress"]),
    ]:
        for c in cols:
            df[c] = df[c].astype(str).str.strip()

    return dict(tx=tx, user=user, item=item, city=city, country=country,
                region=region, continent=continent, typ=typ, mode=mode)


def build_master(save=True):
    t = load_raw()
    tx, user, item = t["tx"], t["user"], t["item"]

    # --- user geography: attach names via the user's own ids ---
    user = user.merge(t["continent"], on="ContinentId", how="left")
    user = user.merge(t["region"][["RegionId", "Region"]], on="RegionId", how="left")
    user = user.merge(t["country"][["CountryId", "Country"]], on="CountryId", how="left")
    user = user.merge(t["city"][["CityId", "CityName"]], on="CityId", how="left")
    user = user.rename(columns={"CityName": "UserCity"})

    # --- attraction location + type names ---
    item = item.merge(t["typ"], on="AttractionTypeId", how="left")
    item = item.merge(
        t["city"][["CityId", "CityName"]].rename(columns={"CityId": "AttractionCityId",
                                                          "CityName": "AttractionCity"}),
        on="AttractionCityId", how="left")

    # --- fact table joins ---
    df = tx.merge(user, on="UserId", how="left")
    df = df.merge(item, on="AttractionId", how="left")
    # VisitMode in Transaction is really VisitModeId -> resolve to its label.
    df = df.merge(t["mode"].rename(columns={"VisitModeId": "VisitMode",
                                            "VisitMode": "VisitModeName"}),
                  on="VisitMode", how="left")

    # 4 users have no city; label them Unknown rather than dropping the transaction.
    for c in ["UserCity", "AttractionCity", "AttractionType", "Region", "Country",
              "Continent", "VisitModeName", "Attraction"]:
        df[c] = df[c].fillna("Unknown").replace({"-": "Unknown", "nan": "Unknown"})

    if save:
        out = DATA / "master.parquet"
        df.to_parquet(out, index=False)
        print(f"Wrote {out}  shape={df.shape}")
    return df


if __name__ == "__main__":
    m = build_master()
    print(m.head())
    print("\nVisitMode distribution:\n", m["VisitModeName"].value_counts())
