"""Tourism Experience Analytics - Streamlit app.

Run:  streamlit run app.py
Needs: data/master.parquet + artifacts/{reg_model,clf_model,recsys}.joblib
       (build them with: python src/data_prep.py && python src/models.py && python src/recommend.py)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import altair as alt
import streamlit as st

from src.recommend import (recommend_for_user, recommend_collaborative,
                           recommend_content, _popular)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")

# Animated travel background: real world map (equirectangular) + tourist cities + flight arcs.
_WORLD_PATH = (ROOT / "assets" / "world_path.txt").read_text()
# (name, lon, lat) of major world cities, projected onto the equirectangular map below.
_CITIES = [("Bali", 115.2, -8.7), ("Tokyo", 139.7, 35.7), ("Sydney", 151.2, -33.9),
           ("Paris", 2.4, 48.9), ("London", -0.1, 51.5), ("New York", -74.0, 40.7),
           ("Rio", -43.2, -22.9), ("Cairo", 31.2, 30.0), ("Dubai", 55.3, 25.2),
           ("Cape Town", 18.4, -33.9), ("Bangkok", 100.5, 13.8), ("Los Angeles", -118.2, 34.1),
           ("Singapore", 103.8, 1.4), ("Istanbul", 29.0, 41.0), ("Mumbai", 72.9, 19.1),
           ("Beijing", 116.4, 39.9), ("Moscow", 37.6, 55.8), ("Mexico City", -99.1, 19.4),
           ("Toronto", -79.4, 43.7), ("Johannesburg", 28.0, -26.2), ("Rome", 12.5, 41.9),
           ("Buenos Aires", -58.4, -34.6)]
_ROUTE_PAIRS = [(4, 5), (3, 7), (5, 6), (1, 0), (8, 4), (10, 0), (2, 10), (8, 14),
                (12, 15), (16, 13), (5, 17), (18, 4), (19, 8), (11, 1), (20, 7),
                (21, 6), (15, 1), (14, 12), (13, 3), (11, 17), (16, 15), (2, 12)]


def _travel_background():
    W, H = 1440, 720
    proj = lambda lon, lat: ((lon + 180) / 360 * W, (90 - lat) / 180 * H)
    nodes = [proj(lon, lat) for _, lon, lat in _CITIES]
    routes = _ROUTE_PAIRS

    # graticule aligned to the projection (every 30 deg)
    grid = "".join(f'<line x1="{(lon+180)/360*W:.0f}" y1="0" x2="{(lon+180)/360*W:.0f}" y2="{H}"/>'
                   for lon in range(-150, 181, 30))
    grid += "".join(f'<line x1="0" y1="{(90-lat)/180*H:.0f}" x2="{W}" y2="{(90-lat)/180*H:.0f}"/>'
                    for lat in range(-60, 91, 30))

    arcs = ""
    for i, (p, q) in enumerate(routes):
        (x1, y1), (x2, y2) = nodes[p], nodes[q]
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        cx, cy = (x1 + x2) / 2, min(y1, y2) - max(50, dist * 0.22)  # arc bows upward
        d = f"M{x1:.1f} {y1:.1f} Q{cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}"
        # static dashed route ("airline map" look) — fully static, zero animation
        arcs += f'<path d="{d}" fill="none" stroke="url(#route)" stroke-width="1.3" opacity=".22" stroke-dasharray="5 6"/>'

    dots = ""
    for i, (x, y) in enumerate(nodes):
        name = _CITIES[i][0]
        end = x > W * 0.82  # flip label left for far-right cities so it doesn't clip
        tx, anchor = (x - 6, "end") if end else (x + 6, "start")
        dots += (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="#8fd0ff"/>'
                 f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.4" fill="#eaf6ff"/>'
                 f'<text x="{tx:.1f}" y="{y - 6:.1f}" fill="#bfe0ff" font-size="13" '
                 f'font-family="system-ui,sans-serif" text-anchor="{anchor}" opacity=".85" '
                 f'style="paint-order:stroke" stroke="#070a12" stroke-width="2.5">{name}</text>')

    st.markdown(f"""
<style>
.stApp {{ background:#070a12; }}
.travel-bg {{ position:fixed; inset:0; z-index:0; pointer-events:none; opacity:.62; }}
/* scrim keeps text crisp over the busy network */
.travel-scrim {{ position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(70% 55% at 28% 24%, rgba(7,10,18,.55), transparent 70%),
    linear-gradient(180deg, rgba(7,10,18,.15), rgba(7,10,18,.45)); }}
.block-container {{ position:relative; z-index:1; }}
@media (prefers-reduced-motion: reduce) {{ .travel-bg * {{ animation:none!important; }} }}
</style>
<div class="travel-bg"><svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice" width="100%" height="100%">
<defs>
<radialGradient id="bgsky" cx="50%" cy="0%" r="120%"><stop offset="0%" stop-color="#16273f"/><stop offset="55%" stop-color="#0b1120"/><stop offset="100%" stop-color="#070a12"/></radialGradient>
<linearGradient id="route" x1="0" x2="1"><stop offset="0" stop-color="#3f9df0"/><stop offset="1" stop-color="#14b8a6"/></linearGradient>
</defs>
<rect width="{W}" height="{H}" fill="url(#bgsky)"/>
<g stroke="rgba(120,170,235,.09)" stroke-width="1">{grid}</g>
<path d="{_WORLD_PATH}" fill="rgba(74,140,210,.10)" stroke="rgba(130,185,255,.28)" stroke-width="0.6"/>
{arcs}{dots}
</svg></div>
<div class="travel-scrim"></div>
""", unsafe_allow_html=True)


_travel_background()

# Skeuomorphism: raised, beveled, glossy "control-panel" surfaces (gradients + shadows, no blur).
st.markdown("""
<style>
/* embossed panels: cards, tables, metrics, alerts, expanders */
div[data-testid="stDataFrame"], [data-testid="stMetric"], .stAlert,
[data-testid="stExpander"] details {
  background: linear-gradient(180deg,#243350,#18243a) !important;
  border: 1px solid #0b1220 !important; border-top-color: rgba(255,255,255,.14) !important;
  border-radius: 12px !important;
  box-shadow: 0 3px 6px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.10),
              inset 0 -3px 8px rgba(0,0,0,.35) !important;
}
/* recessed / carved inputs */
[data-baseweb="select"] > div {
  background: linear-gradient(180deg,#121b2b,#1a2740) !important;
  border: 1px solid #0a0f1a !important; border-radius: 11px !important;
  box-shadow: inset 0 2px 6px rgba(0,0,0,.6), inset 0 -1px 0 rgba(255,255,255,.06),
              0 1px 0 rgba(255,255,255,.05) !important;
}
/* raised secondary (tab) buttons with a real press */
.stButton > button[kind="secondary"] {
  background: linear-gradient(180deg,#35496557,#1d2b40) , linear-gradient(180deg,#354965,#1d2b40) !important;
  border: 1px solid #0b1220 !important; border-top-color: rgba(255,255,255,.22) !important;
  border-radius: 12px !important; color: #dbe7f5 !important;
  text-shadow: 0 -1px 0 rgba(0,0,0,.5);
  box-shadow: 0 4px 0 #0e1826, 0 6px 10px rgba(0,0,0,.5),
              inset 0 1px 0 rgba(255,255,255,.20) !important;
  transition: transform .04s ease, box-shadow .04s ease;
}
.stButton > button[kind="secondary"]:hover { filter: brightness(1.12); }
.stButton > button[kind="secondary"]:active {
  transform: translateY(3px);
  box-shadow: 0 1px 0 #0e1826, 0 2px 4px rgba(0,0,0,.5),
              inset 0 2px 6px rgba(0,0,0,.5) !important;
}
/* glossy accent primary button */
.stButton > button[kind="primary"] {
  background: linear-gradient(180deg,#ff8286,#e23a3f) !important;
  border: 1px solid #7d1417 !important; border-top-color: rgba(255,255,255,.5) !important;
  border-radius: 12px !important; color:#fff !important; text-shadow:0 -1px 0 rgba(120,0,0,.5);
  box-shadow: 0 4px 0 #8d1c20, 0 7px 12px rgba(0,0,0,.5),
              inset 0 1px 0 rgba(255,255,255,.55), inset 0 -3px 8px rgba(120,0,0,.4) !important;
  transition: transform .04s ease, box-shadow .04s ease;
}
.stButton > button[kind="primary"]:active {
  transform: translateY(3px);
  box-shadow: 0 1px 0 #8d1c20, 0 2px 5px rgba(0,0,0,.5),
              inset 0 2px 7px rgba(120,0,0,.6) !important;
}
/* glossy raised slider knob + grooved track */
.stSlider [role="slider"] {
  background: radial-gradient(circle at 35% 28%, #ffffff, #cdd8e6 42%, #93a3b7) !important;
  border: 1px solid #55606e !important;
  box-shadow: 0 2px 4px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.9) !important; }
</style>
""", unsafe_allow_html=True)

# Typography, spacing and finishing touches.
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1 { font-weight: 800 !important; letter-spacing: -.03em; font-size: 2.35rem !important;
     background: linear-gradient(90deg,#eaf6ff,#9cc9ff 55%,#7ee0d0);
     -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
/* consistent section headers */
.sec { margin: .2rem 0 1rem; }
.sec-t { display:flex; align-items:center; gap:.1rem; font-size:1.45rem; font-weight:700;
         letter-spacing:-.02em; color:#eaf2ff; }
.sec-s { color:#9fb3c8; font-size:.9rem; margin:.15rem 0 0 2.1rem; }
/* widget labels */
.stSelectbox label, .stSlider label, .stRadio label, .stNumberInput label {
  color:#c7d6e6 !important; font-weight:500 !important; }
[data-testid="stMetric"] { padding: .7rem 1rem .8rem !important; }
[data-testid="stMetricValue"] { font-weight:700; letter-spacing:-.02em;
  font-size: clamp(1.15rem, 2.2vw, 1.6rem) !important; line-height:1.2; white-space:nowrap; }
[data-testid="stMetricLabel"] { color:#9fb3c8 !important; }
/* result card + star rating */
.result-card { padding:1.1rem 1.3rem; border-radius:14px;
  background: linear-gradient(180deg,#243350,#18243a);
  border:1px solid #0b1220; border-top-color:rgba(255,255,255,.16);
  box-shadow:0 4px 8px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.12),
             inset 0 -4px 10px rgba(0,0,0,.35); }
.stars { font-size:1.4rem; letter-spacing:2px; color:#ffce3a; }
.stars .dim { color:rgba(255,255,255,.22); }
/* slimmer, themed scrollbar */
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-thumb { background:rgba(143,208,255,.25); border-radius:8px; }
::-webkit-scrollbar-thumb:hover { background:rgba(143,208,255,.4); }
hr { border-color: rgba(255,255,255,.08) !important; }
.app-footer { margin:3rem 0 1rem; padding-top:1.4rem; text-align:center;
  border-top:1px solid rgba(255,255,255,.08);
  background:linear-gradient(180deg, rgba(143,208,255,.05), transparent 60%); }
.app-footer .ft-brand { display:inline-flex; align-items:center; gap:.5rem; font-weight:700;
  color:#eaf2ff; letter-spacing:-.01em; font-size:1rem; }
.app-footer .ft-cap { color:#9fb3c8; font-size:.8rem; margin:.5rem 0 .35rem; }
.app-footer .ft-sub { color:#7d92a8; font-size:.73rem; margin-top:.15rem; }
/* accuracy stat cards */
.stat-grid { display:flex; gap:1rem; flex-wrap:wrap; margin:.3rem 0 1.1rem; }
.stat-card { flex:1 1 210px; padding:1.1rem 1.25rem; border-radius:14px;
  background:linear-gradient(180deg,#243350,#18243a); border:1px solid #0b1220;
  border-top-color:rgba(255,255,255,.14);
  box-shadow:0 4px 8px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.10); }
.stat-card { border-top:2px solid #3f9df0; }
.stat-card .lab { display:flex; align-items:center; gap:.4rem; color:#8ea3b8;
  font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.07em; }
.stat-card .val { font-size:2rem; font-weight:800; letter-spacing:-.03em; color:#eaf2ff;
  margin:.35rem 0 .1rem; line-height:1; }
.stat-card .val small { font-size:.82rem; font-weight:600; color:#9fb3c8; margin-left:.3rem;
  letter-spacing:0; }
.stat-card .bench { color:#7d92a8; font-size:.72rem; margin:.15rem 0 .7rem; }
.stat-card .meter { height:6px; border-radius:99px; background:rgba(255,255,255,.08);
  box-shadow:inset 0 1px 2px rgba(0,0,0,.5); overflow:hidden; }
.stat-card .meter i { display:block; height:100%; border-radius:99px;
  background:linear-gradient(90deg,#3f9df0,#4fd6c4); }
.stat-card .note { color:#aebfd0; font-size:.79rem; line-height:1.45; margin-top:.65rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_master():
    return pd.read_parquet(ROOT / "data" / "master.parquet")


# Lazy per-artifact loaders — each cached, loaded only when a tab actually needs it,
# so the page (and the Trends tab) paints instantly instead of blocking on every model.
@st.cache_resource
def load_reg():
    return joblib.load(ART / "reg_model.joblib")


@st.cache_resource
def load_clf():
    return joblib.load(ART / "clf_model.joblib")


@st.cache_resource
def load_rec():
    return joblib.load(ART / "recsys.joblib")


@st.cache_data
def load_metrics():
    return json.loads((ART / "metrics.json").read_text())


df = load_master()

# name -> id lookup tables built once from the master
def name2id(name_col, id_col):
    return df.drop_duplicates(name_col).set_index(name_col)[id_col].to_dict()

CONT = name2id("Continent", "ContinentId")
REG = name2id("Region", "RegionId")
CTRY = name2id("Country", "CountryId")
ATTR = df.drop_duplicates("Attraction").set_index("Attraction")["AttractionId"].to_dict()
ATTR_META = df.drop_duplicates("AttractionId").set_index("AttractionId")

# --- animated inline SVG icons (SMIL) replacing the emojis ---
_ICON_BODIES = {
    "globe": (lambda LAND:
              '<defs>'
              '<radialGradient id="gOcean" cx="37%" cy="31%" r="78%">'
              '<stop offset="0%" stop-color="#7ec8ff"/><stop offset="52%" stop-color="#2b7fd4"/>'
              '<stop offset="100%" stop-color="#124f8e"/></radialGradient>'
              '<radialGradient id="gLimb" cx="38%" cy="34%" r="72%">'
              '<stop offset="55%" stop-color="rgba(0,20,45,0)"/><stop offset="100%" stop-color="rgba(0,14,34,.55)"/></radialGradient>'
              '<radialGradient id="gAtm" cx="50%" cy="50%" r="52%">'
              '<stop offset="84%" stop-color="rgba(130,205,255,0)"/><stop offset="100%" stop-color="rgba(130,205,255,.55)"/></radialGradient>'
              '<clipPath id="cGlobe"><circle cx="12" cy="12" r="8.4"/></clipPath></defs>'
              '<circle cx="12" cy="12" r="9" fill="url(#gAtm)"/>'
              '<circle cx="12" cy="12" r="8.4" fill="url(#gOcean)"/>'
              '<g clip-path="url(#cGlobe)"><g fill="#3fae6a">'
              f'<g>{LAND}</g><g transform="translate(17,0)">{LAND}</g>'
              '<animateTransform attributeName="transform" type="translate" from="0 0" to="-17 0" dur="7s" repeatCount="indefinite"/>'
              '</g></g>'
              '<circle cx="12" cy="12" r="8.4" fill="url(#gLimb)"/>'
              '<ellipse cx="8.6" cy="8" rx="3" ry="1.7" fill="rgba(255,255,255,.4)"/>'
              '<circle cx="12" cy="12" r="8.4" fill="none" stroke="rgba(160,220,255,.55)" stroke-width=".5"/>'
              )('<ellipse cx="3.5" cy="9" rx="2.3" ry="1.5"/><ellipse cx="7.5" cy="13.5" rx="1.7" ry="2.5"/>'
                '<ellipse cx="11.6" cy="7.8" rx="1.6" ry="1.1"/><ellipse cx="14.6" cy="13" rx="2" ry="1.5"/>'
                '<ellipse cx="9.6" cy="16.6" rx="1.4" ry="1"/>'),
    "target": '<defs><radialGradient id="gTgt"><stop offset="0%" stop-color="#ff8a8d"/>'
              '<stop offset="100%" stop-color="#e63b40"/></radialGradient></defs>'
              '<circle cx="12" cy="12" r="9" fill="none" stroke="rgba(255,90,95,.35)" stroke-width="1.4"/>'
              '<circle cx="12" cy="12" r="6" fill="none" stroke="rgba(255,90,95,.6)" stroke-width="1.4"/>'
              '<circle cx="12" cy="12" r="2.6" fill="url(#gTgt)"/>'
              '<circle cx="12" cy="12" r="6" fill="none" stroke="#ff5a5f" stroke-width="1.6">'
              '<animate attributeName="r" values="2.6;10;2.6" dur="2.4s" repeatCount="indefinite"/>'
              '<animate attributeName="opacity" values=".9;0;.9" dur="2.4s" repeatCount="indefinite"/></circle>'
              '<path d="M20 4 L13 11" stroke="#ffce3a" stroke-width="1.6" stroke-linecap="round"/>'
              '<path d="M19.2 4 L20.4 3.6 L20 4.8 Z" fill="#ffce3a"/>',
    "gauge": '<defs><linearGradient id="gArc" x1="0" x2="1">'
             '<stop offset="0%" stop-color="#2ec76b"/><stop offset="55%" stop-color="#ffce3a"/>'
             '<stop offset="100%" stop-color="#ff5a5f"/></linearGradient></defs>'
             '<path d="M3.5 17 A8.5 8.5 0 0 1 20.5 17" fill="none" stroke="url(#gArc)" '
             'stroke-width="2.6" stroke-linecap="round"/>'
             '<g stroke="rgba(255,255,255,.4)" stroke-width="1" stroke-linecap="round">'
             '<line x1="12" y1="7.5" x2="12" y2="9"/><line x1="5.6" y1="11.6" x2="6.8" y2="12.4"/>'
             '<line x1="18.4" y1="11.6" x2="17.2" y2="12.4"/></g>'
             '<g><line x1="12" y1="17" x2="12" y2="9.2" stroke="#eaf2ff" stroke-width="1.8" stroke-linecap="round"/>'
             '<animateTransform attributeName="transform" type="rotate" values="-74 12 17;74 12 17;-74 12 17" '
             'keyTimes="0;0.5;1" dur="2.8s" repeatCount="indefinite" calcMode="spline" '
             'keySplines="0.45 0 0.55 1;0.45 0 0.55 1"/></g>'
             '<circle cx="12" cy="17" r="1.9" fill="#eaf2ff"/>',
    "thumb": '<defs><linearGradient id="gThumb" x1="0" y1="0" x2="0" y2="1">'
             '<stop offset="0%" stop-color="#ffe08a"/><stop offset="100%" stop-color="#f2a51f"/></linearGradient></defs>'
             '<g><rect x="2.4" y="10.6" width="4.2" height="9.4" rx="1.4" fill="#d98f2a"/>'
             '<path d="M7 11.2 L11 3.4 a1.8 1.8 0 0 1 2.6 1.6 V8.7 h5.1 a1.9 1.9 0 0 1 1.87 2.3 '
             'l-1.24 6.1 a2 2 0 0 1 -2 1.6 H7 z" fill="url(#gThumb)"/>'
             '<line x1="9.4" y1="12.4" x2="9.4" y2="18.4" stroke="rgba(120,70,10,.35)" stroke-width="1"/>'
             '<animateTransform attributeName="transform" type="rotate" '
             'values="0 7 20;-13 7 20;0 7 20;-6 7 20;0 7 20" keyTimes="0;0.12;0.28;0.4;1" '
             'dur="2.4s" repeatCount="indefinite"/></g>',
    "chart": '<defs><linearGradient id="gBar" x1="0" y1="0" x2="0" y2="1">'
             '<stop offset="0%" stop-color="#8fd0ff"/><stop offset="100%" stop-color="#2b7fd4"/></linearGradient></defs>'
             '<g fill="url(#gBar)">'
             '<rect x="3.5" width="4" rx="1.2"><animate attributeName="height" values="4;13;4" dur="1.8s" calcMode="spline" keySplines="0.4 0 0.3 1;0.4 0 0.3 1" keyTimes="0;0.5;1" repeatCount="indefinite"/><animate attributeName="y" values="16;7;16" dur="1.8s" calcMode="spline" keySplines="0.4 0 0.3 1;0.4 0 0.3 1" keyTimes="0;0.5;1" repeatCount="indefinite"/></rect>'
             '<rect x="10" width="4" rx="1.2"><animate attributeName="height" values="4;16;4" dur="1.8s" begin="0.3s" calcMode="spline" keySplines="0.4 0 0.3 1;0.4 0 0.3 1" keyTimes="0;0.5;1" repeatCount="indefinite"/><animate attributeName="y" values="16;4;16" dur="1.8s" begin="0.3s" calcMode="spline" keySplines="0.4 0 0.3 1;0.4 0 0.3 1" keyTimes="0;0.5;1" repeatCount="indefinite"/></rect>'
             '<rect x="16.5" width="4" rx="1.2"><animate attributeName="height" values="4;10;4" dur="1.8s" begin="0.6s" calcMode="spline" keySplines="0.4 0 0.3 1;0.4 0 0.3 1" keyTimes="0;0.5;1" repeatCount="indefinite"/><animate attributeName="y" values="16;10;16" dur="1.8s" begin="0.6s" calcMode="spline" keySplines="0.4 0 0.3 1;0.4 0 0.3 1" keyTimes="0;0.5;1" repeatCount="indefinite"/></rect></g>'
             '<polyline points="5.5,11 12,7 18.5,9" fill="none" stroke="#ffce3a" stroke-width="1.4" '
             'stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="22" stroke-dashoffset="22">'
             '<animate attributeName="stroke-dashoffset" values="22;0;0;22" keyTimes="0;0.4;0.8;1" dur="2.6s" repeatCount="indefinite"/></polyline>',
}


def icon(name, size=26):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'style="vertical-align:middle;margin-right:.5rem">{_ICON_BODIES[name]}</svg>')


def section(name, title, subtitle=""):
    sub = f'<div class="sec-s">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="sec"><div class="sec-t">{icon(name)}<span>{title}</span></div>{sub}</div>',
                unsafe_allow_html=True)


# ---- Altair charts: themed to match the dark glass UI ----
def _grad(c0, c1, vertical=True):
    xy = dict(x1=0, x2=0, y1=1, y2=0) if vertical else dict(x1=0, x2=1, y1=0, y2=0)
    return alt.Gradient(gradient="linear",
                        stops=[alt.GradientStop(color=c0, offset=0),
                               alt.GradientStop(color=c1, offset=1)], **xy)


_BLUE, _TEAL, _GOLD = ("#2b7fd4", "#8fd0ff"), ("#0e7c73", "#4fd6c4"), ("#c98a10", "#ffce3a")


def _theme(chart, h=280):
    return (chart.properties(height=h, background="rgba(0,0,0,0)")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#c7d6e6", titleColor="#9fb3c8", labelFont="Inter",
                            titleFont="Inter", gridColor="rgba(255,255,255,.06)",
                            domainColor="rgba(255,255,255,.15)", tickColor="rgba(255,255,255,.15)")
            .configure_axisX(labelAngle=0))


def bar_chart(series, val_title, pair=_BLUE, horizontal=True, fmt=".0f", height=280):
    """Gradient bar chart with value labels + tooltip from a name->value Series."""
    d = series.rename_axis("cat").reset_index(name="val")
    grad = _grad(*pair, vertical=not horizontal)
    base = alt.Chart(d)
    if horizontal:
        y = alt.Y("cat:N", sort="-x", title=None)
        x = alt.X("val:Q", title=val_title, axis=alt.Axis(grid=True))
        bars = base.mark_bar(cornerRadius=6, color=grad).encode(
            x=x, y=y, tooltip=[alt.Tooltip("cat:N", title=""), alt.Tooltip("val:Q", format=fmt)])
        labels = base.mark_text(align="left", dx=5, color="#dbe7f5", font="Inter", fontSize=12).encode(
            x=x, y=y, text=alt.Text("val:Q", format=fmt))
    else:
        x = alt.X("cat:N", sort="-y", title=None)
        y = alt.Y("val:Q", title=val_title, axis=alt.Axis(grid=True))
        bars = base.mark_bar(cornerRadius=6, color=grad).encode(
            x=x, y=y, tooltip=[alt.Tooltip("cat:N", title=""), alt.Tooltip("val:Q", format=fmt)])
        labels = base.mark_text(baseline="bottom", dy=-4, color="#dbe7f5", font="Inter", fontSize=12).encode(
            x=x, y=y, text=alt.Text("val:Q", format=fmt))
    return _theme(bars + labels, height)


def area_chart(series, val_title, height=280):
    d = series.rename_axis("x").reset_index(name="val")
    d["x"] = d["x"].astype(str)
    base = alt.Chart(d).encode(
        x=alt.X("x:N", title=None),
        y=alt.Y("val:Q", title=val_title),
        tooltip=[alt.Tooltip("x:N", title=""), alt.Tooltip("val:Q", format=".0f")])
    area = base.mark_area(interpolate="monotone", opacity=0.28,
                          color=_grad("rgba(63,157,240,0)", "rgba(63,157,240,.7)"))
    line = base.mark_line(interpolate="monotone", color="#8fd0ff", strokeWidth=2.5)
    pts = base.mark_point(filled=True, size=55, color="#eaf6ff")
    return _theme(area + line + pts, height)


st.markdown('<h1 style="display:flex;align-items:center;margin:0">'
            f'{icon("globe", 40)}Tourism Experience Analytics</h1>', unsafe_allow_html=True)
st.caption("See how much travellers will enjoy a place, who they'll travel with, and where to send them next.")

# Custom nav so each tab carries an animated SVG icon (st.tabs labels can't hold SVG).
_TABS = [("Predict a trip", "gauge"), ("Suggest places", "thumb"),
         ("Trends", "chart"), ("How accurate", "target")]
st.session_state.setdefault("active_tab", "Predict a trip")
for (_label, _ic), _col in zip(_TABS, st.columns(len(_TABS))):
    with _col:
        st.markdown(f'<div style="text-align:center;line-height:1">{icon(_ic, 34)}</div>',
                    unsafe_allow_html=True)
        if st.button(_label, key=f"nav_{_label}", use_container_width=True,
                     type="primary" if st.session_state.active_tab == _label else "secondary"):
            st.session_state.active_tab = _label
            st.rerun()
active = st.session_state.active_tab
st.divider()

# ------------------------------------------------------------------ Predict
if active == "Predict a trip":
    section("gauge", "Predict a trip",
            "Tell us about the traveller and where they're going. We'll estimate how much "
            "they'll enjoy it and who they're likely travelling with.")
    c1, c2, c3 = st.columns(3)
    with c1:
        continent = st.selectbox("Traveller's continent", sorted(CONT))
        regions = sorted(df.loc[df.Continent == continent, "Region"].unique()) or sorted(REG)
        region = st.selectbox("Traveller's region", regions)  # only regions in that continent
    with c2:
        countries = sorted(df.loc[df.Region == region, "Country"].unique()) or sorted(CTRY)
        country = st.selectbox("Traveller's country", countries)  # only countries in that region
        attraction = st.selectbox("Place they're visiting", sorted(ATTR))
    with c3:
        year = st.slider("Year of visit", int(df.VisitYear.min()), int(df.VisitYear.max()), 2022)
        month = st.slider("Month of visit", 1, 12, 6)

    aid = ATTR[attraction]
    meta = ATTR_META.loc[aid]
    row = {
        "ContinentId": CONT[continent], "RegionId": REG[region], "CountryId": CTRY[country],
        "CityId": -1, "AttractionId": aid,
        "AttractionTypeId": float(meta["AttractionTypeId"]),
        "AttractionCityId": int(meta["AttractionCityId"]),
        "VisitYear": year, "VisitMonth": month,
    }

    if st.button("Show prediction", type="primary"):
        reg, clf = load_reg(), load_clf()  # loaded only when actually predicting
        # --- rating (regression) ---
        rrow = dict(row)
        rrow["attr_avg_rating"] = reg["attr_avg"].get(aid, reg["global_avg"])
        rrow["user_avg_rating"] = reg["global_avg"]  # unknown user -> global mean
        rpred = float(reg["model"].predict(pd.DataFrame([rrow])[reg["features"]])[0])

        # --- visit mode (classification) ---
        crow = dict(row)
        crow["Rating"] = round(rpred)
        crow["attr_avg_rating"] = clf["attr_avg"].get(aid, clf["global_avg"])
        X = pd.DataFrame([crow])[clf["features"]]
        if clf["is_xgb"]:
            code = int(clf["model"].predict(X)[0]); mode_id = clf["inv"][code]
        else:
            mode_id = int(clf["model"].predict(X)[0])
        mode_name = clf["mode_map"].get(mode_id, str(mode_id))

        full = int(round(rpred))
        stars = ("★" * full) + f'<span class="dim">{"★" * (5 - full)}</span>'
        st.markdown(
            f'<div class="result-card">'
            f'<div style="color:#9fb3c8;font-size:.85rem">{attraction} · {meta["AttractionType"]}</div>'
            f'<div style="color:#c7d6e6;font-size:.9rem;margin-top:.3rem">Expected enjoyment</div>'
            f'<div style="display:flex;align-items:baseline;gap:.6rem;margin-top:.1rem">'
            f'<span style="font-size:2rem;font-weight:800;letter-spacing:-.03em">{rpred:.1f}</span>'
            f'<span style="color:#9fb3c8">out of 5 stars</span>'
            f'<span class="stars" style="margin-left:.4rem">{stars}</span></div>'
            f'<div style="margin-top:.6rem;color:#c7d6e6">Most likely travelling as: '
            f'<b style="color:#eaf2ff">{mode_name}</b></div></div>',
            unsafe_allow_html=True)

        if hasattr(clf["model"], "predict_proba"):
            proba = clf["model"].predict_proba(X)[0]
            order = clf["model"].classes_
            # XGBoost classes_ are 0-based remapped codes -> convert back to real mode ids
            ids = [clf["inv"][int(c)] for c in order] if clf["is_xgb"] else [int(c) for c in order]
            labels = [clf["mode_map"].get(i, i) for i in ids]
            pdf = pd.DataFrame({"cat": labels, "val": proba})
            pdf["hl"] = pdf["cat"] == mode_name
            st.markdown("**How likely each travel style is**")
            ch = alt.Chart(pdf).mark_bar(cornerRadius=6).encode(
                x=alt.X("val:Q", title="probability", axis=alt.Axis(format="%", grid=True)),
                y=alt.Y("cat:N", sort="-x", title=None),
                color=alt.condition(alt.datum.hl, alt.value("#ff5a5f"), alt.value("#3f7fbf")),
                tooltip=[alt.Tooltip("cat:N", title=""), alt.Tooltip("val:Q", format=".1%")])
            lab = alt.Chart(pdf).mark_text(align="left", dx=5, color="#dbe7f5", font="Inter",
                                           fontSize=12).encode(
                x="val:Q", y=alt.Y("cat:N", sort="-x"), text=alt.Text("val:Q", format=".0%"))
            st.altair_chart(_theme(ch + lab, 220), use_container_width=True)

# ------------------------------------------------------------------ Recommend
if active == "Suggest places":
    section("thumb", "Suggest places to visit",
            "Get a list of attractions a traveller is likely to enjoy next.")
    mode = st.radio("How should we pick?",
                    ["Places like one they loved", "For a specific traveller", "Crowd favourites"],
                    horizontal=True)
    k = st.slider("How many suggestions?", 5, 20, 10)
    rec = load_rec()  # recommendation data, loaded only on this tab

    if mode == "Places like one they loved":
        base = st.selectbox("Pick a place they enjoyed", sorted(ATTR), key="rec_attr")
        approach = st.radio("Match by", ["What similar travellers liked", "Similar type of place"],
                            horizontal=True,
                            help="The first looks at other travellers' choices; the second matches the "
                                 "kind of place (beach, temple, museum…).")
        fn = recommend_collaborative if approach == "What similar travellers liked" else recommend_content
        out = fn(rec, ATTR[base], k)
    elif mode == "For a specific traveller":
        uid = st.number_input("Traveller ID number", min_value=int(df.UserId.min()),
                              max_value=int(df.UserId.max()), value=int(df.UserId.iloc[0]),
                              help="Uses that traveller's past visits to suggest new places.")
        out = recommend_for_user(rec, int(uid), k)
    else:
        out = _popular(rec, k)

    out = out.rename(columns={"Attraction": "Place", "AttractionType": "Type",
                              "AttractionCity": "City", "avg_rating": "Avg rating",
                              "n_visits": "Times visited", "score": "Match"})
    st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ------------------------------------------------------------------ Insights
if active == "Trends":
    section("chart", "Travel trends",
            "A quick picture of who travels, what they visit, and how it's changing over time.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Who people travel with**")
        st.altair_chart(bar_chart(df["VisitModeName"].value_counts(), "number of visits", _BLUE),
                        use_container_width=True)
        st.markdown("**Where travellers come from**")
        st.altair_chart(bar_chart(df["Continent"].value_counts(), "number of visits", _TEAL),
                        use_container_width=True)
    with c2:
        st.markdown("**Highest-rated kinds of places**")
        g = df.groupby("AttractionType")["Rating"].agg(["mean", "size"])
        top = g[g["size"] >= 100]["mean"].sort_values(ascending=False)
        st.altair_chart(bar_chart(top, "average stars", _GOLD, fmt=".2f"), use_container_width=True)
        st.markdown("**Number of visits each year**")
        st.altair_chart(area_chart(df.groupby("VisitYear").size(), "visits"),
                        use_container_width=True)

    fig_dir = ROOT / "reports" / "figures"
    if fig_dir.exists():
        with st.expander("More charts"):
            for p in sorted(fig_dir.glob("*.png")):
                st.image(str(p), caption=p.stem.replace("_", " "))

# ------------------------------------------------------------------ Models
if active == "How accurate":
    section("target", "Prediction accuracy",
            "Performance of the underlying models, measured on trips they were never trained on.")

    metrics = load_metrics()  # tiny JSON, loaded only on this tab
    reg_best = metrics["regression"]["best"]
    clf_best = metrics["classification"]["best"]
    rmae = metrics["regression"]["models"][reg_best]["MAE"]
    acc = metrics["classification"]["models"][clf_best]["accuracy"]
    hit = metrics.get("recommendation", {}).get("HitRate@10")

    def stat(label, value, unit, pct, bench, note):
        return (f'<div class="stat-card"><div class="lab">{label}</div>'
                f'<div class="val">{value}<small>{unit}</small></div>'
                f'<div class="bench">{bench}</div>'
                f'<div class="meter"><i style="width:{pct:.0f}%"></i></div>'
                f'<div class="note">{note}</div></div>')

    cards = [
        stat("Rating accuracy", f"±{rmae:.2f}", "stars", (1 - rmae / 4) * 100,
             "Mean absolute error, 1–5 scale",
             "On average, the predicted rating is within about "
             f"{rmae:.2f} of a star of the actual rating."),
        stat("Travel-style match", f"{acc*100:.0f}", "%", acc * 100,
             "Accuracy vs. 20% random baseline",
             f"Correctly identifies the travel group in {acc*100:.0f}% of cases across five "
             "classes — around 2.5&times; the random baseline."),
    ]
    if hit is not None:
        cards.append(stat("Recommendation hit rate", f"{hit*100:.0f}", "%", hit * 100,
                          "Hit rate @ top-10",
                          f"For {hit*100:.0f}% of travellers, a genuinely visited attraction appears "
                          "in their ten recommended places."))
    st.markdown(f'<div class="stat-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.caption(f"Evaluated on a held-out 20% test split. Best models: {reg_best} (rating), "
               f"{clf_best} (travel style), item-item collaborative filtering (recommendations).")

    with st.expander("Technical details (for data teams)"):
        st.write("**Regression (predict rating)** — best:", metrics["regression"]["best"])
        st.dataframe(pd.DataFrame(metrics["regression"]["models"]).T)
        st.write("**Classification (predict visit mode)** — best:", metrics["classification"]["best"])
        st.dataframe(pd.DataFrame(metrics["classification"]["models"]).T)
        cm_df = pd.DataFrame(metrics["classification"]["models"]).T[["accuracy", "f1_weighted"]]
        long = cm_df.reset_index().melt("index", var_name="metric", value_name="score")
        cmp = alt.Chart(long).mark_bar(cornerRadius=5).encode(
            x=alt.X("score:Q", title=None, axis=alt.Axis(grid=True)),
            y=alt.Y("index:N", title=None, sort="-x"),
            yOffset="metric:N",
            color=alt.Color("metric:N", scale=alt.Scale(range=["#8fd0ff", "#4fd6c4"]),
                            legend=alt.Legend(orient="top", title=None, labelColor="#c7d6e6")),
            tooltip=["index:N", "metric:N", alt.Tooltip("score:Q", format=".3f")])
        st.altair_chart(_theme(cmp, 200), use_container_width=True)
        if "recommendation" in metrics:
            st.write("**Recommendation** — item-item collaborative filtering")
            st.dataframe(pd.DataFrame([metrics["recommendation"]]))
        cm = ART / "confusion_matrix.png"
        if cm.exists():
            st.image(str(cm), caption="Confusion matrix (best classifier)")

st.markdown(
    f'<div class="app-footer">'
    f'<div class="ft-brand">{icon("globe", 20)}Tourism Experience Analytics</div>'
    f'<div class="ft-cap">Rating prediction &middot; Visit-mode classification &middot; '
    f'Attraction recommendation</div>'
    f'<div class="ft-sub">&copy; 2026 Aditya Ganjoo &middot; MIT License &middot; '
    f'Built with Python, scikit-learn &amp; Streamlit</div></div>',
    unsafe_allow_html=True)
