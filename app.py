import base64
import concurrent.futures
import io
import re
import time
import warnings
from pathlib import Path

import altair as alt
import astropy.units as u
import numpy as np
import pandas as pd
import pyvo
import streamlit as st
import streamlit.components.v1 as components
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils.data import conf
from astropy.visualization import (AsinhStretch, ImageNormalize, LinearStretch,
                                   ZScaleInterval)
from astropy.wcs import WCS
from PIL import Image, ImageDraw, ImageFilter, ImageFont

warnings.filterwarnings("ignore")

# ======================================================================
#  Configuration. Everything that changes with new data releases or
#  new services lives here.
# ======================================================================
APP_NAME = "SPHEREx Mover Search"
APP_TAGLINE = ("Find asteroids and comets in SPHEREx infrared data by comparing "
               "images of the same patch of sky taken at different times.")

IRSA_SIA = "https://irsa.ipac.caltech.edu/SIA"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"

# Used only if the SIA service cannot list its collections. Normally the
# list is read from the service, so new releases appear automatically.
FALLBACK_COLLECTIONS = [
    "spherex_dr1", "spherex_dr1_deep",
    "spherex_qr3", "spherex_qr3_deep",
    "spherex_qr2", "spherex_qr2_deep",
]

PIXEL_SCALE = 6.15          # arcsec per SPHEREx pixel
# Flags masked by default when their names appear in the FITS header (names, not bit numbers).
BAD_FLAGS = {"TRANSIENT", "OVERFLOW", "SUR_ERROR", "NONFUNC", "MISSING_DATA", "HOT", "COLD",
             "NONLINEAR", "PERSIST", "OUTLIER"}
DETECTORS = [1, 2, 3, 4, 5, 6]
BACKGROUND_FILES = ["background.jpg", "background.jpeg", "background.png", "background.webp"]

conf.remote_timeout = 25
SIA_LIMIT, TAP_LIMIT, TAP_ATTEMPTS, DOWNLOAD_LIMIT = 40, 50, 1, 100
DOWNLOAD_WORKERS = 4   # parallel cutout downloads; keep low to be gentle on IRSA
SIA_WORKERS = 3
MAX_FOV = 0.25         # degrees; larger cutouts load the archive heavily

PRESETS = {
    "Custom": None,
    "Default target": ("08h55m10.831s", "-07d14m42.53s"),
    "WISE 0855-0714 (fast brown dwarf)": ("08h55m10.831s", "-07d14m42.53s"),
    "Barnard's Star (position for 2026)": ("17h57m47.1s", "+04d46m05s"),
    "Epsilon Indi B (position for 2026)": ("22h04m23.0s", "-56d48m04s"),
    "RW Aurigae": ("05h07m49.57s", "+30d24m05.2s"),
    "R Draconis": ("16h32m40.07s", "+66d45m17.8s"),
    "Pinwheel Galaxy (M101)": ("210.80227", "54.34895"),
}

st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="collapsed")


# ======================================================================
#  Appearance: dark background (own image, uploaded image, or generated)
# ======================================================================
@st.cache_resource(show_spinner=False)
def default_background_b64():
    rng = np.random.default_rng(7)
    w, h = 1600, 900
    yy = np.linspace(0, 1, h)[:, None]
    img = np.zeros((h, w, 3), dtype=float)
    img[..., 0] = 1 + 12 * yy
    img[..., 1] = 7 + 21 * yy
    img[..., 2] = 54 + 12 * yy
    n = 1500
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    br = (rng.uniform(0.15, 1.0, n) ** 3) * 150 + 25
    tint = np.array([[1.0, 0.95, 0.82], [0.88, 0.92, 1.0], [0.99, 0.945, 0.816]])[rng.integers(0, 3, n)]
    np.maximum.at(img, (ys, xs), br[:, None] * tint)
    im = Image.fromarray(np.clip(img, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(0.6))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def image_to_b64(raw: bytes):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((1920, 1200))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def pick_background():
    with st.sidebar:
        st.markdown("**Appearance**")
        up = st.file_uploader("Background image", type=["jpg", "jpeg", "png", "webp"])
        dark = st.slider("Background darkness", 0.40, 0.95, 0.75, 0.01)
        st.caption("To make a background permanent, save the image next to app.py as background.jpg.")
    if up is not None:
        return image_to_b64(up.getvalue()), dark
    here = Path(__file__).resolve().parent
    for name in BACKGROUND_FILES:
        p = here / name
        if p.exists():
            return image_to_b64(p.read_bytes()), dark
    return default_background_b64(), dark


CSS = """
<style>
:root { --n0:#010736; --n1:#0D1C42; --n2:#22396F; --cr:#FCF1D0; --cr-body:rgba(252,241,208,.84); --cr-mute:rgba(252,241,208,.58); }
.stApp {
  background-image: linear-gradient(rgba(1,7,54,__DARK__), rgba(1,7,54,__DARK__)), url("data:image/jpeg;base64,__BG__");
  background-size: cover; background-position: center; background-attachment: fixed;
  color: #FCF1D0;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0D1C42; border-right: 1px solid #22396F; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
.stApp p, .stApp li, .stApp label, .stApp span { color: inherit; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #FCF1D0; font-weight: 600; }
.stApp h3 { font-family: Georgia, 'Times New Roman', serif; font-weight: 400; letter-spacing: .01em; }
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primary"] p { color: #010736 !important; }
div[data-baseweb="tab-highlight"], div[role="tablist"] > div[aria-hidden="true"] { background-color: #FCF1D0 !important; }
button[role="tab"][aria-selected="true"], button[role="tab"][aria-selected="true"] p { color: #FCF1D0 !important; }
button[role="tab"]:hover, button[role="tab"]:hover p { color: #FCF1D0 !important; }
div[role="slider"] { background-color: #FCF1D0 !important; box-shadow: none !important; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: #FCF1D0 !important; }
div[data-baseweb="slider"] div[style*="255, 75, 75"], div[data-baseweb="slider"] div[style*="ff4b4b"] { filter: sepia(1) saturate(.35) brightness(1.5) hue-rotate(5deg); }
span[data-baseweb="tag"] { background-color: #22396F !important; color: #FCF1D0 !important; border: 1px solid #FCF1D0; }
span[data-baseweb="tag"] span, span[data-baseweb="tag"] svg { color: #FCF1D0 !important; fill: #FCF1D0 !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="textarea"] {
  background-color: #0D1C42 !important; border-color: #22396F !important; }
div[data-baseweb="input"] input, div[data-baseweb="base-input"] input, div[data-baseweb="select"] input { color: #FCF1D0 !important; }
button[data-testid="stBaseButton-secondary"] { background-color: #0D1C42 !important; border: 1px solid #22396F !important; color: #FCF1D0 !important; }
button[data-testid="stBaseButton-secondary"]:hover { border-color: #FCF1D0 !important; }
button[data-testid="stBaseButton-primary"] { background-color: #FCF1D0 !important; border-color: #FCF1D0 !important; }
div[data-baseweb="checkbox"] > div:first-child, div[data-baseweb="radio"] > div:first-child { border-color: #FCF1D0 !important; }
.titlerow { display: flex; align-items: baseline; gap: 1.1rem; flex-wrap: wrap; }
.byline { font-size: .74rem; letter-spacing: .16em; text-transform: uppercase; color: #FCF1D0; background: rgba(34,57,111,.55);
  border: 1px solid #22396F; border-radius: 999px; padding: .22rem .8rem; white-space: nowrap; }
div[data-baseweb="tab-border"] { background-color: #22396F !important; }
div[data-testid="stTabs"] button[role="tab"] { font-size: 1rem; }

.kicker { font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; color: var(--cr-mute); margin: 0 0 .35rem 0; }
.hero { padding: .6rem 0 1.5rem 0; border-bottom: 1px solid #22396F; margin-bottom: 1rem; position: relative; }
.hero:after { content: ""; position: absolute; left: 0; bottom: -1px; width: 120px; height: 2px; background: #FCF1D0; }
.hero h1 { font-family: Georgia, 'Times New Roman', serif; font-size: 2.9rem; font-weight: 400;
  letter-spacing: .01em; margin: 0 0 .45rem 0; color: #FCF1D0; }
.hero .tag { font-size: 1.08rem; color: var(--cr-body); max-width: 780px; margin: 0; line-height: 1.55; }

.lead { font-family: Georgia, 'Times New Roman', serif; font-size: 1.55rem; line-height: 1.35; color: #FCF1D0;
  margin: 1.1rem 0 1.1rem 0; max-width: 820px; }
.lead.sm { font-size: 1.2rem; color: var(--cr-body); }

.panel { background: linear-gradient(160deg, rgba(13,28,66,.92), rgba(1,7,54,.88)); border: 1px solid #22396F;
  border-radius: 10px; padding: 1.05rem 1.25rem; margin-bottom: 1rem; }
.panel h4 { margin: 0 0 .45rem 0; font-size: 1.08rem; font-family: Georgia, 'Times New Roman', serif; font-weight: 400; }
.panel p, .panel li { color: var(--cr-body); font-size: .96rem; line-height: 1.65; margin: .25rem 0; }
.panel a { color: #FCF1D0; text-decoration: none; border-bottom: 1px solid #22396F; }
.panel a:hover { border-bottom-color: #FCF1D0; }
.card { position: relative; overflow: hidden; }
.card:before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: #22396F; }
.card:hover:before { background: #FCF1D0; }
.card .big { position: absolute; right: 1rem; top: .35rem; font-family: Georgia, serif; font-size: 2.6rem;
  color: rgba(252,241,208,.14); line-height: 1; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
.grid .panel { margin-bottom: 0; }
.split { display: grid; grid-template-columns: 1.05fr 1fr; gap: 1rem; margin-bottom: 1rem; align-items: stretch; }
.split .panel { margin-bottom: 0; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
@media (max-width: 820px) { .split, .pair { grid-template-columns: 1fr; } }
.art { width: 100%; display: block; border-radius: 8px; border: 1px solid #22396F; }
.cap { font-size: .84rem; color: var(--cr-mute); margin-top: .5rem; line-height: 1.5; }
.fig { font-family: Georgia, serif; font-size: 1.9rem; color: #FCF1D0; }
.figlbl { font-size: .85rem; color: var(--cr-mute); }
.small { font-size: .84rem; color: var(--cr-mute); }
.spec { position: relative; height: 40px; border-radius: 4px; background: rgba(252,241,208,.06);
  border: 1px solid #22396F; margin: 3.4rem 0 2.8rem 0; }
.spec .vis { position: absolute; left: 1.5%; width: 7.1%; top: 0; bottom: 0;
  background: linear-gradient(90deg,#5a2fc0,#2f6fd0,#2fb890,#d8c24a,#d8603a); }
.spec .ir { position: absolute; left: 8.65%; width: 81.7%; top: 0; bottom: 0;
  background: linear-gradient(90deg,rgba(252,241,208,.85),rgba(190,110,70,.65),rgba(90,30,40,.55)); }
.spec .tick { position: absolute; top: -1.35rem; font-size: .72rem; color: #FCF1D0;
  transform: translateX(-50%); white-space: nowrap; }
.spec .tick:after { content: ""; position: absolute; left: 50%; top: 1.1rem; width: 1px; height: 2.2rem;
  background: rgba(252,241,208,.6); }
.spec .lab { position: absolute; bottom: -1.7rem; font-size: .75rem; color: var(--cr-mute); }
.chain { display: flex; flex-wrap: wrap; gap: .5rem; justify-content: center; margin: .3rem 0 .6rem 0; align-items: center; }
.chain .stage { border: 1px solid #22396F; background: rgba(34,57,111,.35); padding: .45rem 1rem; font-size: .92rem;
  color: #FCF1D0; border-radius: 999px; }
.chain .stage b { font-family: Georgia, serif; font-weight: 400; color: var(--cr-mute); margin-right: .5rem; }
.chain .sep { color: #22396F; font-size: 1.1rem; }
</style>
"""
EMBED = st.query_params.get("embed") == "1"
_bg, _dark = pick_background()
st.markdown(CSS.replace("__BG__", _bg).replace("__DARK__", f"{_dark:.2f}"), unsafe_allow_html=True)
if EMBED:
    st.markdown("<style>.hero, div[data-baseweb='tab-list'], [data-testid='stSidebar'], "
                "[data-testid='stSidebarCollapsedControl'] { display: none !important; } "
                ".block-container { padding-top: .5rem; }</style>", unsafe_allow_html=True)


def panel(title, body):
    return f'<div class="panel"><h4>{title}</h4>{body}</div>'


def grid(items):
    st.markdown('<div class="grid">' + "".join(items) + "</div>", unsafe_allow_html=True)


# ======================================================================
#  Illustrations for the information tabs. Everything is generated here,
#  so the app has no extra files and no extra network requests.
# ======================================================================
NAVY0, NAVY1, NAVY2, CREAM = "#010736", "#0D1C42", "#22396F", "#FCF1D0"


def svg_img(svg, alt=""):
    b = base64.b64encode(svg.strip().encode("utf-8")).decode()
    return f'<img class="art" alt="{alt}" src="data:image/svg+xml;base64,{b}">'


def jpg_img(b64, alt=""):
    return f'<img class="art" alt="{alt}" src="data:image/jpeg;base64,{b64}">'


def _stars(seed, w, h, n):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(2, w - 2), rng.uniform(2, h - 2), rng.uniform(.5, 1.6), rng.uniform(.35, 1.0))
            for _ in range(n)]


def _star_svg(stars):
    return "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="{CREAM}" opacity="{o:.2f}"/>'
                   for x, y, r, o in stars)


@st.cache_resource(show_spinner=False)
def blink_strip_svg():
    fw, fh, gap, x0, y0 = 170, 150, 20, 10, 12
    stars = _star_svg(_stars(11, fw, fh, 38))
    track = [(28, 112), (66, 92), (104, 70), (142, 46)]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 240" font-family="Georgia, serif">',
           f'<rect width="760" height="240" rx="10" fill="{NAVY1}"/>']
    for i, (mx, my) in enumerate(track):
        out.append(f'<g transform="translate({x0 + 10 + i * (fw + gap)},{y0 + 10})">')
        out.append(f'<rect width="{fw}" height="{fh}" fill="{NAVY0}" stroke="{NAVY2}"/>{stars}')
        if i == len(track) - 1:
            pts = " ".join(f"{a},{b}" for a, b in track)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{CREAM}" stroke-opacity=".55" '
                       f'stroke-dasharray="4 4"/>')
        out.append(f'<circle cx="{mx}" cy="{my}" r="3.6" fill="{CREAM}"/>'
                   f'<circle cx="{mx}" cy="{my}" r="10" fill="none" stroke="{CREAM}" stroke-width="1.2"/>')
        out.append(f'<text x="{fw / 2}" y="{fh + 24}" fill="{CREAM}" font-size="14" text-anchor="middle">'
                   f'Visit {i + 1}</text></g>')
    out.append(f'<line x1="30" y1="214" x2="728" y2="214" stroke="{CREAM}" stroke-opacity=".6"/>'
               f'<polygon points="738,214 726,209 726,219" fill="{CREAM}" fill-opacity=".8"/>'
               f'<text x="30" y="233" fill="{CREAM}" fill-opacity=".6" font-size="12">time</text></svg>')
    return svg_img("".join(out), "Four frames of the same sky patch. Stars stay fixed and one source moves.")


@st.cache_resource(show_spinner=False)
def blink_anim_svg():
    w = h = 300
    stars = _star_svg(_stars(5, w, h, 70))
    track = [(60, 220), (110, 182), (160, 146), (214, 104)]
    dx = [(x - track[0][0], y - track[0][1]) for x, y in track]
    keys = "".join(f"{k * 25}%{{transform:translate({dx[k][0]}px,{dx[k][1]}px)}}" for k in range(4))
    keys += f"100%{{transform:translate({dx[0][0]}px,{dx[0][1]}px)}}"
    css = (f".m{{animation:hop 4s steps(1,end) infinite}}@keyframes hop{{{keys}}}")
    for i in range(4):
        vals = "".join(f"{k * 25}%{{opacity:{1 if k == i else 0}}}" for k in range(4)) + \
               f"100%{{opacity:{1 if i == 0 else 0}}}"
        css += f".t{i}{{opacity:0;animation:s{i} 4s steps(1,end) infinite}}@keyframes s{i}{{{vals}}}"
    labels = "".join(f'<text class="t{i}" x="150" y="288" fill="{CREAM}" font-size="14" text-anchor="middle">'
                     f'Visit {i + 1} of 4</text>' for i in range(4))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="Georgia, serif">'
           f'<style>{css}</style><rect width="{w}" height="{h}" rx="10" fill="{NAVY0}"/>{stars}'
           f'<g class="m"><circle cx="{track[0][0]}" cy="{track[0][1]}" r="4.2" fill="{CREAM}"/>'
           f'<circle cx="{track[0][0]}" cy="{track[0][1]}" r="11" fill="none" stroke="{CREAM}" stroke-width="1.2"/></g>'
           f'{labels}</svg>')
    return svg_img(svg, "Animated blink comparison: one source hops across a fixed star field.")


BANDS = [(1, .75, 1.09), (2, 1.09, 1.62), (3, 1.62, 2.42), (4, 2.42, 3.82), (5, 3.82, 4.42), (6, 4.42, 5.00)]
BAND_COLORS = ["#4a67b3", "#6f8bcf", "#9fb3df", "#d3d4c8", "#f0e3bb", "#FCF1D0"]


@st.cache_resource(show_spinner=False)
def bands_svg():
    def X(l):
        return 30 + (l - 0.3) / 5.0 * 700
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 250" font-family="Georgia, serif">',
           '<defs><linearGradient id="v" x1="0" x2="1"><stop offset="0" stop-color="#5a2fc0"/>'
           '<stop offset=".3" stop-color="#2f6fd0"/><stop offset=".55" stop-color="#2fb890"/>'
           '<stop offset=".8" stop-color="#d8c24a"/><stop offset="1" stop-color="#d8603a"/></linearGradient></defs>',
           f'<rect width="760" height="250" rx="10" fill="{NAVY1}"/>',
           f'<rect x="{X(.38):.1f}" y="36" width="{X(.75) - X(.38):.1f}" height="16" fill="url(#v)"/>',
           f'<text x="{X(.38):.1f}" y="28" fill="{CREAM}" fill-opacity=".7" font-size="12">visible light</text>']
    for (n, lo, hi), col in zip(BANDS, BAND_COLORS):
        x1, x2 = X(lo), X(hi)
        txt = NAVY0 if n >= 3 else CREAM
        out.append(f'<rect x="{x1:.1f}" y="74" width="{x2 - x1 - 2:.1f}" height="46" fill="{col}"/>'
                   f'<text x="{(x1 + x2) / 2 - 1:.1f}" y="103" fill="{txt}" font-size="15" text-anchor="middle">D{n}</text>'
                   f'<text x="{(x1 + x2) / 2 - 1:.1f}" y="138" fill="{CREAM}" fill-opacity=".7" font-size="10.5" '
                   f'text-anchor="middle">{lo:.2f}-{hi:.2f}</text>')
    out.append(f'<line x1="30" y1="150" x2="730" y2="150" stroke="{NAVY2}"/>')
    for l in range(1, 6):
        out.append(f'<line x1="{X(l):.1f}" y1="146" x2="{X(l):.1f}" y2="154" stroke="{CREAM}" stroke-opacity=".7"/>'
                   f'<text x="{X(l):.1f}" y="168" fill="{CREAM}" fill-opacity=".8" font-size="12" '
                   f'text-anchor="middle">{l} \u00b5m</text>')
    for lam, name, row in [(3.0, "water ice", 200), (4.27, "CO\u2082 ice", 200), (4.67, "CO ice", 228)]:
        out.append(f'<line x1="{X(lam):.1f}" y1="176" x2="{X(lam):.1f}" y2="{row - 14}" stroke="{CREAM}" '
                   f'stroke-opacity=".5" stroke-dasharray="3 3"/>'
                   f'<text x="{X(lam):.1f}" y="{row}" fill="{CREAM}" font-size="12.5" text-anchor="middle">'
                   f'{name} {lam} \u00b5m</text>')
    out.append("</svg>")
    return svg_img("".join(out), "The six SPHEREx detector bands across 0.75 to 5 micrometres.")


@st.cache_resource(show_spinner=False)
def solar_system_svg():
    rng = np.random.default_rng(3)
    cx, cy = 380, 220
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 440" font-family="Georgia, serif">',
           '<defs><radialGradient id="sun"><stop offset="0" stop-color="#FCF1D0"/>'
           '<stop offset="1" stop-color="#FCF1D0" stop-opacity="0"/></radialGradient></defs>',
           f'<rect width="760" height="440" rx="10" fill="{NAVY0}"/>',
           _star_svg(_stars(9, 760, 440, 60))]
    for r, dash in [(38, ""), (58, ""), (80, ""), (108, ""), (205, "4 5")]:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{NAVY2}"{d}/>')
    for _ in range(300):
        a, r = rng.uniform(0, 2 * np.pi), rng.uniform(140, 172)
        out.append(f'<circle cx="{cx + r * np.cos(a):.1f}" cy="{cy + r * np.sin(a):.1f}" r="{rng.uniform(.6, 1.3):.2f}" '
                   f'fill="{CREAM}" opacity="{rng.uniform(.35, .85):.2f}"/>')
    out.append(f'<circle cx="{cx}" cy="{cy}" r="26" fill="url(#sun)"/><circle cx="{cx}" cy="{cy}" r="7" fill="{CREAM}"/>')
    for r, ang, col, rad in [(38, 1.1, "#9fb3df", 2.5), (58, 4.0, "#d3d4c8", 3.2), (80, 2.3, "#6f8bcf", 3.4),
                             (108, 5.3, "#d8603a", 3.0), (205, 2.55, "#f0e3bb", 6)]:
        out.append(f'<circle cx="{cx + r * np.cos(ang):.1f}" cy="{cy + r * np.sin(ang):.1f}" r="{rad}" fill="{col}"/>')
    a, e = 190, .7
    b, c = a * np.sqrt(1 - e * e), a * e
    ecx = cx + c
    out.append(f'<ellipse cx="{ecx:.1f}" cy="{cy}" rx="{a}" ry="{b:.1f}" fill="none" stroke="{CREAM}" '
               f'stroke-opacity=".6" stroke-dasharray="6 4"/>')
    for k, E in enumerate([-1.25, -1.05, -.85, -.65]):
        px, py = ecx + a * np.cos(E), cy + b * np.sin(E)
        if k < 3:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{CREAM}" opacity="{.3 + .2 * k:.2f}"/>')
        else:
            vx, vy = px - cx, py - cy
            n = np.hypot(vx, vy)
            tx, ty = px + vx / n * 46, py + vy / n * 46
            nx, ny = -vy / n * 5, vx / n * 5
            out.append(f'<polygon points="{px + nx:.1f},{py + ny:.1f} {tx:.1f},{ty:.1f} {px - nx:.1f},{py - ny:.1f}" '
                       f'fill="{CREAM}" opacity=".45"/><circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{CREAM}"/>'
                       f'<text x="{px - 12:.1f}" y="{py - 12:.1f}" fill="{CREAM}" font-size="13" text-anchor="end">comet</text>')
    out.append(f'<line x1="130" y1="402" x2="{cx - 100}" y2="{cy + 100}" stroke="{CREAM}" stroke-opacity=".4"/>'
               f'<text x="126" y="406" fill="{CREAM}" font-size="13" text-anchor="end">main asteroid belt</text>'
               f'<text x="{cx - 14}" y="{cy + 52}" fill="{CREAM}" fill-opacity=".7" font-size="11">Sun</text>'
               f'<text x="746" y="428" fill="{CREAM}" fill-opacity=".5" font-size="11" text-anchor="end">'
               f'schematic, not to scale</text></svg>')
    return svg_img("".join(out), "Schematic of the inner Solar System with the asteroid belt and a comet orbit.")


@st.cache_resource(show_spinner=False)
def dust_pair():
    """Simulated, not real data: one star field seen through a dust lane in visible and infrared light."""
    rng = np.random.default_rng(21)
    w, h = 560, 400
    f = rng.normal(size=(h // 8, w // 8))
    f = ((f - f.min()) / (f.max() - f.min()) * 255).astype("uint8")
    d = np.asarray(Image.fromarray(f).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(20)), dtype=float)
    d = (d - d.min()) / (d.max() - d.min())
    yy, xx = np.mgrid[0:h, 0:w]
    band = np.exp(-((yy - h * .5 - 45 * np.sin(xx / w * 6)) ** 2) / (2 * (h * .12) ** 2))
    tau = 3.6 * band * (.35 + .65 * d)
    n = 2800
    S = np.zeros((h, w))
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    np.maximum.at(S, (ys, xs), (rng.uniform(.05, 1, n) ** 4) * 255 + 25)
    S = np.asarray(Image.fromarray(np.clip(S, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(.9)), dtype=float) * 3.4
    glow = np.asarray(Image.fromarray(np.clip(S, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(9)), dtype=float) * 1.6
    bg = np.array([1, 7, 54], dtype=float)

    def render(field, tint):
        im = bg + field[..., None] * np.array(tint)
        buf = io.BytesIO()
        Image.fromarray(np.clip(im, 0, 255).astype("uint8")).save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()

    vis = (S + glow) * np.exp(-tau)
    ir = (S + glow) * np.exp(-.10 * tau) + 26 * d * band
    return render(vis, (.80, .88, 1.0)), render(ir, (1.0, .93, .78))


def _wave(x0, x1, y, amp, period):
    xs = np.linspace(x0, x1, 500)
    ys = y + amp * np.sin((xs - x0) / period * 2 * np.pi)
    return "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))


@st.cache_resource(show_spinner=False)
def redshift_svg():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 230" font-family="Georgia, serif">'
           f'<rect width="760" height="230" rx="10" fill="{NAVY1}"/>'
           f'<text x="30" y="34" fill="{CREAM}" fill-opacity=".7" font-size="13">light leaves a distant galaxy</text>'
           f'<path d="{_wave(30, 730, 62, 14, 28)}" fill="none" stroke="#9fb3df" stroke-width="2"/>'
           f'<text x="380" y="122" fill="{CREAM}" font-size="13" text-anchor="middle">'
           f'\u2193 space expands while the light travels \u2193</text>'
           f'<text x="30" y="152" fill="{CREAM}" fill-opacity=".7" font-size="13">the same light reaches SPHEREx</text>'
           f'<path d="{_wave(30, 730, 186, 14, 84)}" fill="none" stroke="{CREAM}" stroke-width="2"/></svg>')
    return svg_img(svg, "A light wave stretched to a longer wavelength by the expansion of the universe.")


def card(tag, title, body, big=""):
    num = f'<div class="big">{big}</div>' if big else ""
    return f'<div class="panel card">{num}<div class="kicker">{tag}</div><h4>{title}</h4>{body}</div>'


# ======================================================================
#  Archive access
# ======================================================================
def parse_coord(ra_text, dec_text):
    try:
        return SkyCoord(ra_text, dec_text)
    except Exception:
        return SkyCoord(float(ra_text) * u.deg, float(dec_text) * u.deg)


def detector_from_url(url):
    m = re.search(r"/(\d)/level2_", url) or re.search(r"[_/]D(\d)[_/.]", url)
    return int(m.group(1)) if m else None


def get_sia_collections():
    import requests
    import xml.etree.ElementTree as ET

    def _list():
        r = requests.get(IRSA_SIA, params={"maxrec": 0, "responseformat": "votable"}, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        names = []
        for el in root.iter():
            if el.tag.endswith(("PARAM", "FIELD")) and el.attrib.get("name", "").lower() == "collection":
                for opt in el.iter():
                    if opt.tag.endswith("OPTION") and "value" in opt.attrib:
                        names.append(opt.attrib["value"])
        return names

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(_list)
    try:
        names = [n for n in fut.result(timeout=20) if "spherex" in n.lower() and "cal" not in n.lower()]
        if names:
            return names
    except Exception as e:
        print(f"[SIA] collection list unavailable ({e}), using fallback", flush=True)
    finally:
        pool.shutdown(wait=False)
    return FALLBACK_COLLECTIONS


def _sia_one(collection, ra_deg, dec_deg, size_deg, dets):
    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    service = pyvo.dal.SIA2Service(IRSA_SIA)
    tbl = service.search((coord, 1 * u.arcsec), collection=collection).to_table()
    items = []
    for row in tbl:
        url = str(row["access_url"])
        det = detector_from_url(url)
        if det not in dets:
            continue
        try:
            mjd = float(row["t_min"])
        except Exception:
            mjd = float("nan")
        items.append({"uri": f"{url}?center={ra_deg},{dec_deg}d&size={size_deg}", "mjd": mjd, "det": det})
    return items


def search_sia(ra_deg, dec_deg, size_deg, dets):
    collections = get_sia_collections()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(SIA_WORKERS, len(collections)))
    futures = {pool.submit(_sia_one, c, ra_deg, dec_deg, size_deg, dets): c for c in collections}
    items, errors = [], []
    try:
        for fut in concurrent.futures.as_completed(futures, timeout=SIA_LIMIT):
            try:
                items.extend(fut.result())
            except Exception as e:
                errors.append(f"{futures[fut]}: {e}")
    except concurrent.futures.TimeoutError:
        errors.append(f"timeout {SIA_LIMIT} s")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if not items and errors:
        raise RuntimeError("SIA: " + "; ".join(errors[-3:]))
    items = list({it["uri"]: it for it in items}.values())
    items.sort(key=lambda it: (np.isnan(it["mjd"]), it["mjd"]))
    return items


def run_tap(url, query, limit=TAP_LIMIT):
    """Synchronous TAP query with a hard timeout. Returns an astropy Table."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(lambda: pyvo.dal.TAPService(url).search(query).to_table())
    t0 = time.time()
    try:
        tbl = fut.result(timeout=limit)
        print(f"[TAP] {url}: {len(tbl)} rows in {time.time() - t0:.1f} s", flush=True)
        return tbl
    except concurrent.futures.TimeoutError:
        print(f"[TAP] {url}: no answer within {limit} s", flush=True)
        raise TimeoutError(f"TAP did not answer within {limit} s")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def search_tap(ra_deg, dec_deg, size_deg, dets):
    names = ", ".join(f"'SPHEREx-D{d}'" for d in dets)
    query = f"""
    SELECT
        'https://irsa.ipac.caltech.edu/' || a.uri || '?center={ra_deg},{dec_deg}d&size={size_deg}' AS uri,
        p.time_bounds_lower AS mjd,
        p.energy_bandpassname AS bp
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra_deg}, {dec_deg}), p.poly)
        AND p.energy_bandpassname IN ({names})
    """
    last_error = None
    for attempt in range(TAP_ATTEMPTS):
        try:
            table = run_tap(IRSA_TAP, query)
            out = []
            for r in table:
                m = re.search(r"D(\d)", str(r["bp"]))
                out.append({"uri": str(r["uri"]), "mjd": float(r["mjd"]), "det": int(m.group(1)) if m else None})
            out.sort(key=lambda it: (np.isnan(it["mjd"]), it["mjd"]))
            return out
        except Exception as e:
            last_error = e
            print(f"[TAP] error: {e}", flush=True)
    raise last_error


@st.cache_data(show_spinner=False, ttl=3600)
def find_items(ra_deg, dec_deg, size_deg, dets, method):
    if method == "sia":
        return search_sia(ra_deg, dec_deg, size_deg, dets), "SIA"
    if method == "tap":
        return search_tap(ra_deg, dec_deg, size_deg, dets), "TAP"
    tap_error = None
    try:
        items = search_tap(ra_deg, dec_deg, size_deg, dets)
        if items:
            return items, "TAP"
    except Exception as e:
        tap_error = e
    try:
        return search_sia(ra_deg, dec_deg, size_deg, dets), "SIA"
    except Exception as sia_error:
        if tap_error is not None:
            raise RuntimeError(f"TAP: {tap_error} | SIA: {sia_error}")
        raise


def _wave_at(header, hdul, cx, cy):
    """Central wavelength (um) and bandwidth at a pixel, from the spectral WCS (key W)."""
    try:
        w = WCS(header, hdul, key="W")
        vals = w.pixel_to_world_values(cx, cy)
        lam, bw = float(np.ravel(vals[0])[0]), float(np.ravel(vals[1])[0])
        if lam < 1e-3:  # metres -> micrometres
            lam, bw = lam * 1e6, bw * 1e6
        return lam, bw
    except Exception:
        return None, None


def flag_bits(header):
    """Flag names and bit numbers from the FLAGS header.
    Current IRSA format: MSKN<bit> = NAME. Older format: MP_<NAME> = bit or mask."""
    named = {}
    for k, v in header.items():
        ks = str(k)
        if ks.startswith("MSKN") and ks[4:].isdigit() and isinstance(v, str) and v.strip():
            named[v.strip()] = int(ks[4:])
    if named:
        return named
    raw = {k[3:]: int(v) for k, v in header.items()
           if str(k).startswith("MP_") and isinstance(v, (int, np.integer)) and not isinstance(v, bool)}
    vals = set(raw.values())
    as_mask = (len(vals) >= 4 and 0 not in vals and 3 not in vals and 5 not in vals
               and all(v > 0 and (v & (v - 1)) == 0 for v in vals))
    return {k: (int(np.log2(v)) if as_mask else v) for k, v in raw.items()}


def download_cutout(uri, store=None):
    if store is not None and uri in store:
        return store[uri]
    for attempt in range(2):
        try:
            with fits.open(uri, cache=False) as hdul:
                try:
                    hdu = hdul["IMAGE"]
                except KeyError:
                    hdu = hdul[1]
                data = np.array(hdu.data, dtype=float)
                try:
                    wcs = WCS(hdu.header, hdul).celestial
                except Exception:
                    wcs = None
                flags, bits, flag_frac = None, {}, None
                try:
                    fh = hdul["FLAGS"]
                    flags = np.array(fh.data).astype(np.int64)
                    flag_frac = float(np.mean(flags != 0))
                    bits = flag_bits(fh.header)
                except Exception:
                    pass
                try:
                    var = np.array(hdul["VARIANCE"].data, dtype=float)
                except Exception:
                    var = None
                try:
                    struct = "HDUs: " + ", ".join(f"{h.name}({h.header.get('NAXIS1')}x{h.header.get('NAXIS2')})"
                                                  for h in hdul)
                    if "FLAGS" in [h.name for h in hdul]:
                        struct += "\n\nFLAGS header:\n" + hdul["FLAGS"].header.tostring(sep="\n")[:6000]
                except Exception:
                    struct = "unavailable"
                lam, bw = _wave_at(hdu.header, hdul, data.shape[1] / 2, data.shape[0] / 2)
                res = {"data": data, "wcs": wcs, "flag_frac": flag_frac, "wave": lam, "bw": bw,
                       "var": var, "flags": flags, "bits": bits, "struct": struct}
                if store is not None:
                    if len(store) > 150:
                        store.clear()
                    store[uri] = res
                return res
        except Exception as e:
            print(f"[Download] error: {e}", flush=True)
            time.sleep(2)
    return None


@st.cache_resource(show_spinner=False)
def cutout_store():
    return {}


def download_all(uris, limit=DOWNLOAD_LIMIT, progress=None, store=None):
    results = [None] * len(uris)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    futures = {pool.submit(download_cutout, uri, store): k for k, uri in enumerate(uris)}
    done = 0
    try:
        for fut in concurrent.futures.as_completed(futures, timeout=limit):
            try:
                results[futures[fut]] = fut.result()
            except Exception:
                pass
            done += 1
            if progress:
                progress(done, len(uris))
    except concurrent.futures.TimeoutError:
        print("[Download] overall timeout, using what finished", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return results


# ======================================================================
#  Image handling: common sky grid, co-adding, subtraction
# ======================================================================
def grid_size(size_deg):
    return int(np.clip(round(size_deg * 3600 / PIXEL_SCALE), 16, 200))


def make_grid_wcs(ra, dec, n):
    w = WCS(naxis=2)
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.cdelt = [-PIXEL_SCALE / 3600, PIXEL_SCALE / 3600]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def bilinear(img, sx, sy):
    h, w = img.shape
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = sx - x0, sy - y0
    out = np.full(sx.shape, np.nan)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0, y0, fx, fy = x0[ok], y0[ok], fx[ok], fy[ok]
    out[ok] = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
               + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return out


def regrid(data, wcs_src, wcs_dst, n):
    """Resample a cutout onto the common north-up grid so every frame is registered."""
    yy, xx = np.mgrid[0:n, 0:n]
    ra, dec = wcs_dst.wcs_pix2world(xx, yy, 0)
    try:
        sx, sy = wcs_src.all_world2pix(ra, dec, 0)
    except Exception:
        sx, sy = wcs_src.wcs_world2pix(ra, dec, 0)
    return bilinear(data, np.asarray(sx, float), np.asarray(sy, float))


def resample_plain(data, n):
    h, w = data.shape
    sy, sx = np.meshgrid(np.linspace(0, h - 1.001, n), np.linspace(0, w - 1.001, n), indexing="ij")
    return bilinear(data, sx, sy)


def _src_xy(wcs_src, wcs_dst, n):
    yy, xx = np.mgrid[0:n, 0:n]
    ra, dec = wcs_dst.wcs_pix2world(xx, yy, 0)
    try:
        sx, sy = wcs_src.all_world2pix(ra, dec, 0)
    except Exception:
        sx, sy = wcs_src.wcs_world2pix(ra, dec, 0)
    return np.asarray(sx, float), np.asarray(sy, float)


def nearest(arr, sx, sy):
    h, w = arr.shape
    xi, yi = np.rint(sx).astype(int), np.rint(sy).astype(int)
    ok = (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)
    out = np.zeros(sx.shape, dtype=arr.dtype)
    out[ok] = arr[yi[ok], xi[ok]]
    return out


def build_frame(it, res, wcs_t, n):
    data, var, flags = res["data"], res.get("var"), res.get("flags")
    if var is not None and var.shape != data.shape:
        var = None
    if flags is not None and flags.shape != data.shape:
        flags = None
    g = v = fl = None
    if res["wcs"] is not None:
        try:
            sx, sy = _src_xy(res["wcs"], wcs_t, n)
            g = bilinear(data, sx, sy)
            v = bilinear(var, sx, sy) if var is not None else None
            fl = nearest(flags, sx, sy) if flags is not None else None
        except Exception:
            g = None
    if g is None or not np.isfinite(g).any():
        g = resample_plain(data, n)
        v = resample_plain(var, n) if var is not None else None
        fl = None
    return {"data": g, "var": v, "flags": fl, "bits": res.get("bits", {}), "mjd": it["mjd"], "det": it["det"],
            "wave": res["wave"], "bw": res["bw"], "flagged": res["flag_frac"], "uri": it["uri"],
            "struct": res.get("struct")}


def pretty_date(mjd):
    try:
        return Time(mjd, format="mjd").iso[:10]
    except Exception:
        return "date unknown"


def frame_label(f):
    parts = [pretty_date(f["mjd"])]
    if f.get("det"):
        parts.append(f"D{f['det']}")
    if f.get("wave"):
        parts.append(f"{f['wave']:.2f} um")
    return "  ".join(parts)


def bgsub(d):
    m = np.nanmedian(d) if np.isfinite(d).any() else 0.0
    return d - m


def _prep(f, maskbits):
    d = f["data"]
    fl = f.get("flags")
    if maskbits and fl is not None and fl.shape == d.shape:
        d = np.where((fl & maskbits) != 0, np.nan, d)
    return d


def coadd(group, maskbits, weighted):
    D = [bgsub(_prep(g, maskbits)) for g in group]
    vs = [g.get("var") for g in group]
    if weighted and all(v is not None and v.shape == D[0].shape for v in vs):
        W = [np.where(np.isfinite(d) & np.isfinite(v) & (v > 0), 1.0 / np.where(v > 0, v, 1.0), 0.0)
             for d, v in zip(D, vs)]
        num = sum(w * np.nan_to_num(d) for w, d in zip(W, D))
        den = sum(W)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)
    return np.nanmean(np.stack(D), axis=0)


def process(cube, window, sub, per_det=False, order="Time", maskbits=0, weighted=True):
    """Optional masking and co-add in time windows, then optional subtraction of a reference."""
    frames = sorted(cube, key=lambda f: (np.isnan(f["mjd"]), f["mjd"]))
    if window <= 0 and order == "Wavelength":
        frames = sorted(cube, key=lambda f: (f["wave"] is None, f["wave"] or 0.0))
    out = []
    if window > 0:
        subsets = [[f for f in frames if f["det"] == d] for d in sorted({f["det"] for f in frames})] \
            if per_det else [frames]
        for subset in subsets:
            group = []
            def flush():
                if group:
                    tag = f"D{group[0]['det']}  " if per_det else ""
                    lab = (f"{tag}{pretty_date(group[0]['mjd'])} to {pretty_date(group[-1]['mjd'])}"
                           f"  {len(group)} images")
                    out.append({"data": coadd(group, maskbits, weighted), "label": lab, "mjd": group[0]["mjd"]})
            for f in subset:
                if group and f["mjd"] - group[0]["mjd"] > window:
                    flush()
                    group.clear()
                group.append(f)
            flush()
        out.sort(key=lambda o: (np.isnan(o["mjd"]), o["mjd"]))
    else:
        out = [{"data": _prep(f, maskbits), "label": frame_label(f), "mjd": f["mjd"]} for f in frames]

    stretch = "asinh"
    refimg = None
    if sub != "None" and len(out) > 1:
        stretch = "linear"
        bg = [bgsub(o["data"]) for o in out]
        ref = np.nanmedian(np.stack(bg), axis=0) if sub == "Median of all frames" else bg[0]
        refimg = ref if sub == "Median of all frames" else None
        for o, d in zip(out, bg):
            ok = np.isfinite(d) & np.isfinite(ref)
            den = float(np.sum(ref[ok] ** 2))
            sc = max(float(np.sum(d[ok] * ref[ok])) / den, 0.0) if den > 0 else 1.0
            o["data"] = d - sc * ref
    return out, stretch, refimg


def find_tracks(cands, frames, tol=1.5, min_det=3, max_tracks=5):
    """Link candidates from different frames into straight-line tracks (uniform motion)."""
    def tm(c):
        return frames[c["k"]].get("mjd", np.nan)
    remaining = {i for i, c in enumerate(cands) if np.isfinite(tm(c))}
    tracks = []
    for _ in range(max_tracks):
        idx = sorted(remaining)
        best = None
        for a in idx:
            ca = cands[a]
            for b in idx:
                cb = cands[b]
                dt = tm(cb) - tm(ca)
                if dt <= 0 or ca["k"] == cb["k"]:
                    continue
                if np.hypot(cb["x"] - ca["x"], cb["y"] - ca["y"]) < 2:
                    continue
                vx, vy = (cb["x"] - ca["x"]) / dt, (cb["y"] - ca["y"]) / dt
                members = {}
                for m in idx:
                    cm = cands[m]
                    d = np.hypot(cm["x"] - (ca["x"] + vx * (tm(cm) - tm(ca))),
                                 cm["y"] - (ca["y"] + vy * (tm(cm) - tm(ca))))
                    if d <= tol and (cm["k"] not in members or d < members[cm["k"]][0]):
                        members[cm["k"]] = (d, m)
                if len(members) >= min_det:
                    score = (len(members), -sum(v[0] for v in members.values()))
                    if best is None or score > best[0]:
                        best = (score, [v[1] for v in members.values()], vx, vy)
        if best is None:
            break
        _, mem, vx, vy = best
        first = min(mem, key=lambda i: tm(cands[i]))
        east, north = -vx, vy
        tracks.append({"members": mem, "n": len(mem),
                       "rate_arcsec_hr": float(np.hypot(vx, vy) * PIXEL_SCALE / 24.0),
                       "pa_deg": float(np.degrees(np.arctan2(east, north)) % 360.0),
                       "ra": cands[first]["ra"], "dec": cands[first]["dec"]})
        remaining -= set(mem)
    return tracks


def find_movers(frames, ref, wcs_t, nsig=5.0):
    """Peaks in the reference-subtracted frames that are not part of the static sky."""
    if ref is None or wcs_t is None:
        return []
    fin = np.isfinite(ref)
    if fin.sum() < 20:
        return []
    mref = np.nanmedian(ref)
    sref = 1.4826 * np.nanmedian(np.abs(ref[fin] - mref))
    refz = (ref - mref) / (sref if sref > 0 else 1.0)
    cands = []
    for k, fr in enumerate(frames):
        d = fr["data"]
        ok = np.isfinite(d)
        if ok.sum() < 20:
            continue
        med = np.nanmedian(d)
        sig = 1.4826 * np.nanmedian(np.abs(d[ok] - med))
        if not sig > 0:
            continue
        z = np.where(ok, (d - med) / sig, -np.inf)
        h, w = z.shape
        pad = np.pad(z, 1, constant_values=-np.inf)
        nb = np.max([pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)], axis=0)
        ys, xs = np.nonzero((z >= nsig) & (z > nb))
        for y, x in zip(ys, xs):
            if 1 <= x < w - 1 and 1 <= y < h - 1 and refz[y, x] < 3:
                ra_, dec_ = wcs_t.wcs_pix2world([x], [y], 0)
                cands.append({"k": k, "label": fr["label"], "x": int(x), "y": int(y), "snr": float(z[y, x]),
                              "ra": float(ra_[0]), "dec": float(dec_[0])})
    for c in cands:
        c["repeats"] = sum(1 for o in cands if o is not c and abs(o["x"] - c["x"]) <= 1.5 and abs(o["y"] - c["y"]) <= 1.5)
    cands.sort(key=lambda c: -c["snr"])
    return cands[:60]


def _to_deg(v, is_ra, ref):
    if isinstance(v, str):
        from astropy.coordinates import Angle
        return float(Angle(v.strip(), unit=u.hourangle if is_ra else u.deg).deg)
    val = float(v)
    if is_ra and abs(((val - ref + 180) % 360) - 180) > 2.0:
        val *= 15.0
    return val


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_skybot(epoch_jd, ra, dec, radius_arcsec):
    """Known solar-system objects in a cone at a given epoch (IMCCE SkyBoT, text output)."""
    import requests
    r = requests.get("https://ssp.imcce.fr/webservices/skybot/api/conesearch.php",
                     params={"-ep": f"{epoch_jd:.3f}", "-ra": ra, "-dec": dec, "-rs": radius_arcsec,
                             "-mime": "text", "-output": "basic", "-loc": "500"}, timeout=25)
    r.raise_for_status()
    text = r.text.strip()
    if not text:
        return []
    if text.startswith("{") or text.startswith("["):
        payload = r.json()
        data = payload.get("data", []) if isinstance(payload, dict) else payload
        cols = payload.get("columns") if isinstance(payload, dict) else None
        recs = [row if isinstance(row, dict) else dict(zip(cols or [], row)) for row in data]
    else:
        header, recs = None, []
        for line in text.splitlines():
            if "|" not in line:
                continue
            parts = [x.strip() for x in line.lstrip("#").split("|")]
            if line.startswith("#"):
                header = parts
            elif header and len(parts) >= len(header):
                recs.append(dict(zip(header, parts)))
        if header is None:
            raise RuntimeError("unexpected SkyBoT reply: " + text[:120].replace("\n", " "))
    out = []
    for rec in recs:
        low = {str(k).lower(): v for k, v in rec.items()}
        rk = next((k for k in low if k.startswith("ra") and "rate" not in k and not k.startswith("rad")), None)
        dk = next((k for k in low if (k.startswith("dec") or k.startswith("de(") or k == "de") and "rate" not in k), None)
        if rk is None or dk is None:
            continue
        name = next((low[k] for k in ("name", "object", "number", "num") if k in low), "object")
        try:
            out.append((str(name), _to_deg(low[rk], True, ra), _to_deg(low[dk], False, dec)))
        except Exception:
            continue
    return out


def known_objects(frames, ra, dec, radius_deg):
    """One SkyBoT query per distinct date (at most 12). Returns {frame index: [(name, ra, dec)]}."""
    keys = {}
    for k, f in enumerate(frames):
        if np.isfinite(f.get("mjd", np.nan)):
            keys.setdefault(round(f["mjd"], 1), []).append(k)
    ks = sorted(keys)
    if len(ks) > 12:
        ks = [ks[i] for i in np.linspace(0, len(ks) - 1, 12).astype(int)]
    rad = min(600.0, radius_deg * 3600.0)
    found = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(fetch_skybot, key + 2400000.5, ra, dec, rad): key for key in ks}
        for fut in concurrent.futures.as_completed(futs):
            found[futs[fut]] = fut.result()
    by_frame = {}
    for k, f in enumerate(frames):
        if not np.isfinite(f.get("mjd", np.nan)) or not found:
            continue
        key = min(found, key=lambda x: abs(x - round(f["mjd"], 1)))
        by_frame[k] = found[key]
    return by_frame


def composite(cube, out_px=560):
    """False-colour image: detectors D1-2 as blue, D3-4 as green, D5-6 as red."""
    chans = []
    for dets in ((1, 2), (3, 4), (5, 6)):
        fs = [bgsub(f["data"]) for f in cube if f["det"] in dets]
        chans.append(np.nanmean(np.stack(fs), axis=0) if fs else None)
    have = [c for c in chans if c is not None]
    if len(have) < 2:
        return None
    shape = have[0].shape
    planes = []
    for c in (chans[2], chans[1], chans[0]):
        if c is None:
            planes.append(np.zeros(shape))
            continue
        norm = ImageNormalize(c, interval=ZScaleInterval(), stretch=AsinhStretch())
        planes.append(np.nan_to_num(np.ma.filled(norm(c), 0.0)))
    arr = np.flipud(np.clip(np.stack(planes, axis=-1), 0, 1))
    scale = max(1, out_px // max(shape))
    img = Image.fromarray((arr * 255).astype("uint8"))
    return img.resize((img.size[0] * scale, img.size[1] * scale), Image.NEAREST)


# ======================================================================
#  Overlays (Gaia, WISE), drawing and animation
# ======================================================================
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_catalog(kind, ra, dec, radius, glim=16.0):
    if kind == "gaia":
        q = (f"SELECT TOP 400 ra, dec, pmra, pmdec FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
             f"CIRCLE('ICRS',{ra},{dec},{radius})) AND phot_g_mean_mag < {glim} ORDER BY phot_g_mean_mag")
        t = run_tap(GAIA_TAP, q, 25)
        cols = [np.ma.filled(np.ma.array(t[c], dtype=float), np.nan) for c in ("ra", "dec", "pmra", "pmdec")]
        return tuple(cols)
    else:
        q = (f"SELECT TOP 400 ra, dec FROM allwise_p3as_psd WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
             f"CIRCLE('ICRS',{ra},{dec},{radius}))")
        t = run_tap(IRSA_TAP, q, 50)
    return np.array(t["ra"], dtype=float), np.array(t["dec"], dtype=float)


GAIA_EPOCH_MJD = 57388.0   # Gaia DR3 reference epoch J2016.0


def at_epoch(coords, mjd):
    """Positions to draw. Gaia entries carry proper motions and are moved to the date of the frame."""
    if len(coords) == 2:
        return coords
    ra, dec, pmra, pmdec = coords
    if not np.isfinite(mjd):
        return ra, dec
    dt = (mjd - GAIA_EPOCH_MJD) / 365.25
    pmra, pmdec = np.nan_to_num(pmra), np.nan_to_num(pmdec)
    cosd = np.maximum(np.cos(np.radians(dec)), 1e-6)
    return ra + pmra * dt / 3.6e6 / cosd, dec + pmdec * dt / 3.6e6


def resolve_overlays(overlays, mjd):
    return [(at_epoch(c, mjd), col, sh) for c, col, sh in overlays]


def _font(size=15):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def to_picture(data, stretch="asinh", label=None, overlays=None, wcs_t=None, marker=False, out_px=560):
    a = np.array(data, dtype=float)
    if not np.isfinite(a).any():
        a = np.zeros_like(a)
    norm = ImageNormalize(a, interval=ZScaleInterval(),
                          stretch=AsinhStretch() if stretch == "asinh" else LinearStretch())
    arr = np.nan_to_num(np.ma.filled(norm(a), 0.0))
    arr = np.flipud(arr)
    n = arr.shape[0]
    scale = max(1, out_px // max(arr.shape))
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"))
    img = img.resize((img.size[0] * scale, img.size[1] * scale), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(img)
    if overlays and wcs_t is not None:
        for (ras, decs), color, shape in overlays:
            if len(ras) == 0:
                continue
            xs, ys = wcs_t.wcs_world2pix(ras, decs, 0)
            for x, y in zip(xs, ys):
                if -0.5 <= x < n - 0.5 and -0.5 <= y < n - 0.5:
                    px, py = (x + 0.5) * scale, (n - 1 - y + 0.5) * scale
                    r = max(6, scale * 0.45)
                    box = [px - r, py - r, px + r, py + r]
                    if shape == "circle":
                        d.ellipse(box, outline=color, width=2)
                    elif shape == "diamond":
                        d.polygon([(px, py - r), (px + r, py), (px, py + r), (px - r, py)], outline=color, width=2)
                    elif shape == "cross":
                        d.line([(px - r, py), (px + r, py)], fill=color, width=2)
                        d.line([(px, py - r), (px, py + r)], fill=color, width=2)
                    else:
                        d.rectangle(box, outline=color, width=2)
    if marker:
        c = img.size[0] / 2
        for a0, b0 in ((c - 22, c - 10), (c + 10, c + 22)):
            d.line([(a0, c), (b0, c)], fill=(120, 170, 220), width=2)
            d.line([(c, a0), (c, b0)], fill=(120, 170, 220), width=2)
    if label:
        d.rectangle([0, 0, img.size[0], 24], fill=(0, 0, 0))
        d.text((6, 3), label, fill=(225, 230, 238), font=_font(15))
    return img


def make_gif(images, ms):
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:], duration=ms, loop=0)
    return buf.getvalue()


def make_demo_cube(n_frames):
    rng = np.random.default_rng(42)
    side = 40
    yy, xx = np.mgrid[0:side, 0:side]
    stars = [(rng.uniform(3, side - 3), rng.uniform(3, side - 3), rng.uniform(40, 200)) for _ in range(12)]
    base = np.zeros((side, side))
    for sx, sy, amp in stars:
        base += amp * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 1.2 ** 2))
    cube = []
    for k in range(n_frames):
        frac = k / max(n_frames - 1, 1)
        mx, my = 6 + 28 * frac, 30 - 18 * frac
        moving = 120 * np.exp(-((xx - mx) ** 2 + (yy - my) ** 2) / (2 * 1.2 ** 2))
        frame = base + moving + rng.normal(0, 6, (side, side))
        cube.append({"data": frame, "mjd": 60800.0 + 15 * k, "det": 2, "wave": 1.0 + 0.02 * k,
                     "bw": None, "flagged": None, "uri": "demo"})
    return cube


def timed(label, fn, limit):
    t = time.time()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn)
    try:
        res = fut.result(timeout=limit)
        st.success(f"{label}: OK, {time.time() - t:.1f} s")
        return res
    except concurrent.futures.TimeoutError:
        st.error(f"{label}: no answer within {limit} s")
    except Exception as e:
        st.error(f"{label}: {e}")
    finally:
        pool.shutdown(wait=False)
    return None


@st.cache_resource(show_spinner=False)
def archive_slots():
    import threading
    return threading.BoundedSemaphore(4)   # at most 4 visitors query the archive at once


def run_query(*args):
    slots = archive_slots()
    if not slots.acquire(timeout=90):
        st.warning("Many people are using the tool right now. Please try again in a minute.")
        return None
    try:
        return _run_query(*args)
    finally:
        slots.release()


def _run_query(ra_txt, dec_txt, size, det_names, method, max_frames):
    try:
        coord = parse_coord(ra_txt, dec_txt)
    except Exception:
        st.error("Could not read the coordinates. Example: 08h55m10.831s and -07d14m42.53s, or degrees.")
        return None
    ra_deg, dec_deg = float(coord.ra.deg), float(coord.dec.deg)
    dets = tuple(sorted(int(d[1]) for d in det_names))
    status = st.status("Searching the IRSA archive (up to about 3 minutes)...", expanded=True)
    t0 = time.time()
    try:
        items, used = find_items(ra_deg, dec_deg, size, dets, method)
    except Exception as e:
        status.update(label="Search failed", state="error")
        if "502" in str(e) or "Proxy Error" in str(e):
            st.error("The IRSA search servers returned a 502 error. This is a problem on IRSA's side. "
                     "Try again later or switch on demo mode.")
        else:
            st.error("The IRSA archive did not respond. Try again, choose another search method, or use demo mode.")
        st.caption(f"Technical details: {str(e)[:300]}")
        return None
    status.write(f"Search via {used} took {int(time.time() - t0)} s. Images found: {len(items)}.")
    if not items:
        status.update(label="Nothing found", state="error")
        st.warning("No images for this position and these detectors. Try other detectors.")
        return None

    per = max(1, -(-max_frames // len(dets)))
    picked = []
    for d in dets:
        sub = [it for it in items if it["det"] == d]
        if len(sub) > per:
            sub = [sub[i] for i in np.linspace(0, len(sub) - 1, per).astype(int)]
        picked += sub

    status.update(label=f"Downloading {len(picked)} cutouts...")
    bar = st.progress(0.0)
    results = download_all([it["uri"] for it in picked], progress=lambda d, t: bar.progress(d / t), store=cutout_store())
    bar.empty()

    n = grid_size(size)
    wcs_t = make_grid_wcs(ra_deg, dec_deg, n)
    cube = [build_frame(it, r, wcs_t, n) for it, r in zip(picked, results) if r is not None]
    if not cube:
        status.update(label="Download failed", state="error")
        st.error("Images were found but could not be downloaded. Try again later.")
        return None
    status.update(label=f"Loaded {len(cube)} of {len(picked)} cutouts", state="complete", expanded=False)
    return cube, {"ra": ra_deg, "dec": dec_deg, "size": size, "n": n, "demo": False}


# ======================================================================
#  Page
# ======================================================================
st.markdown(
    f'<div class="hero"><div class="kicker">NASA Space Apps Challenge</div><div class="titlerow"><h1>{APP_NAME}</h1><span class="byline">Made by team Lester</span></div>'
    f'<p class="tag">{APP_TAGLINE}</p></div>',
    unsafe_allow_html=True,
)

# The Explorer is the first tab for everyone; ?embed=1 additionally hides the header and the tab bar.
tab_tool, tab_home, tab_see, tab_small, tab_ir = st.tabs(
    ["Explorer", "Overview", "Instrument and science", "Small bodies", "Infrared astronomy"])

# ---------------------------------------------------------------- Overview
with tab_home:
    st.markdown('<p class="lead">Stars stay where they are. Anything that moves between visits is worth a closer look.</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="split">'
        + panel("What this tool does",
                "<p>It retrieves every SPHEREx image of a chosen position from the NASA/IPAC Infrared Science "
                "Archive (IRSA), places the images on a common sky grid and sorts them by time. Stars and galaxies "
                "remain fixed. Asteroids and comets change position from one visit to the next.</p>"
                "<p>The Explorer combines the six detector arrays, masks flagged pixels, co-adds frames with "
                "inverse-variance weights and subtracts the static sky. Sources that move are listed as candidates "
                "and linked into tracks. Known asteroids and comets are marked, Gaia and WISE can be overlaid, "
                "the detectors can be combined into a colour composite, and spectra come with error bars.</p>")
        + "<div>" + blink_anim_svg() + "<div class='cap'>The principle in one picture: the stars are fixed, "
          "the circled source hops between visits.</div></div></div>",
        unsafe_allow_html=True,
    )
    grid([
        card("Step one", "Choose a position", "<p>Type an RA and Dec, pick a ready-made target, or pass the "
             "coordinates in the page URL (see the Explorer tab).</p>", "1"),
        card("Step two", "Retrieve and register", "<p>Cutouts from the selected detectors are downloaded from IRSA "
             "and resampled onto one north-up grid.</p>", "2"),
        card("Step three", "Compare epochs", "<p>Blink the frames, co-add them, or subtract a reference to make "
             "moving sources stand out.</p>", "3"),
    ])
    grid([
        '<div class="panel"><div class="fig">0.75 to 5 \u00b5m</div><div class="figlbl">near-infrared wavelength range</div></div>',
        '<div class="panel"><div class="fig">102</div><div class="figlbl">spectral channels, 17 per detector array</div></div>',
        '<div class="panel"><div class="fig">6.15 arcsec</div><div class="figlbl">pixel size</div></div>',
        '<div class="panel"><div class="fig">4 passes</div><div class="figlbl">of the full sky during the two-year mission</div></div>',
    ])
    st.markdown(
        panel("How a mover reveals itself", blink_strip_svg() +
              "<p class='cap'>Four visits to the same patch of sky. Only the circled source changes position, and "
              "joining its positions gives the track the Explorer looks for.</p>"),
        unsafe_allow_html=True,
    )
    st.markdown(
        panel("The blink comparator",
              "<p>In 1930 Clyde Tombaugh found Pluto by alternating photographic plates of one patch of sky and "
              "watching for a single point that jumped. The method is called a "
              "<a href='https://en.wikipedia.org/wiki/Blink_comparator' target='_blank'>blink comparator</a>. "
              "This tool applies the same idea to infrared survey data.</p>"
              "<p class='small'>Each SPHEREx image is taken at a different wavelength, so the brightness of a star "
              "changes from frame to frame. Judge position, not brightness.</p>"),
        unsafe_allow_html=True,
    )

# ------------------------------------------------------ Instrument/science
with tab_see:
    st.markdown('<p class="lead">One telescope, about a hundred infrared colours, and the whole sky mapped again '
                'and again.</p>', unsafe_allow_html=True)
    st.markdown(
        panel("Six detectors, one continuous spectrum",
              bands_svg() +
              "<p class='cap'>Each detector array covers its own slice of the 0.75 to 5 \u00b5m range through a "
              "filter whose wavelength changes across the array. Six arrays with 17 channels each give the 102 "
              "bands. Wavelength limits in micrometres are printed under each bar.</p>"),
        unsafe_allow_html=True,
    )
    st.markdown("<p class='lead sm'>The same survey serves very different fields of research.</p>",
                unsafe_allow_html=True)
    grid([
        card("Cosmology", "Cosmic inflation", "<p>Maps the positions of hundreds of millions of galaxies to look for "
             "the imprint of <a href='https://en.wikipedia.org/wiki/Cosmic_inflation' target='_blank'>cosmic "
             "inflation</a>, the burst of expansion immediately after the Big Bang.</p>", "01"),
        card("Astrochemistry", "Cosmic ices", "<p>Measures water, carbon dioxide and carbon monoxide ices in dark "
             "clouds where stars and planets form.</p>", "02"),
        card("Galaxies", "Galaxy history", "<p>Adds up the faint light of all galaxies over cosmic time, the "
             "<a href='https://en.wikipedia.org/wiki/Extragalactic_background_light' target='_blank'>"
             "extragalactic background light</a>.</p>", "03"),
        card("Our Galaxy", "Milky Way", "<p>Looks through the dust that hides stars from visible-light cameras.</p>",
             "04"),
        card("Comets", "Interstellar visitors", "<p>SPHEREx observed the interstellar comet "
             "<a href='https://en.wikipedia.org/wiki/3I/ATLAS' target='_blank'>3I/ATLAS</a>.</p>", "05"),
        card("Solar System", "Moving objects", "<p>Bodies of the Solar System appear as sources that shift between "
             "visits. IRSA's <a href='https://irsa.ipac.caltech.edu/applications/MOST/' target='_blank'>Moving "
             "Object Search Tool</a> lists which SPHEREx frames contain a given object.</p>", "06"),
        card("Interplanetary dust", "Zodiacal light", "<p>Dust scatters sunlight across the whole sky. SPHEREx "
             "products include a zodiacal light model so it can be subtracted.</p>", "07"),
    ])
    st.markdown(
        panel("References",
              "<p><a href='https://spherex.caltech.edu/' target='_blank'>SPHEREx mission site</a> | "
              "<a href='https://science.nasa.gov/mission/spherex/' target='_blank'>NASA mission page</a> | "
              "<a href='https://caltech-ipac.github.io/spherex-archive-documentation/' target='_blank'>Archive "
              "documentation</a> | "
              "<a href='https://caltech-ipac.github.io/irsa-tutorials/spherex' target='_blank'>IRSA tutorials</a></p>"),
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------ Small bodies
with tab_small:
    st.markdown('<p class="lead">Asteroids and comets are the sky\'s travellers, and motion is how SPHEREx finds '
                'them.</p>', unsafe_allow_html=True)
    st.markdown(
        panel("Where they live", solar_system_svg() +
              "<p class='cap'>Most asteroids circle the Sun between Mars and Jupiter. Comets follow long, stretched "
              "orbits and grow a tail as they approach the Sun. Faded dots mark where the comet was on earlier visits.</p>"),
        unsafe_allow_html=True,
    )
    st.markdown("<p class='lead sm'>The vocabulary of small bodies is easy to mix up. From parent body to ground:</p>",
                unsafe_allow_html=True)
    st.markdown(
        '<div class="panel"><div class="chain">'
        '<div class="stage"><b>1</b>Asteroid or comet</div><div class="sep">&rsaquo;</div>'
        '<div class="stage"><b>2</b>Meteoroid</div><div class="sep">&rsaquo;</div>'
        '<div class="stage"><b>3</b>Meteor</div><div class="sep">&rsaquo;</div>'
        '<div class="stage"><b>4</b>Fireball</div><div class="sep">&rsaquo;</div>'
        '<div class="stage"><b>5</b>Meteorite</div></div>'
        '<p class="small" style="text-align:center">First in space, then in the atmosphere, then on the ground. Not '
        'every meteoroid becomes a fireball and not every fireball leaves a meteorite.</p></div>',
        unsafe_allow_html=True,
    )
    grid([
        card("In space", "Asteroid", "<p>A rocky or metallic remnant of Solar System formation, mostly between Mars "
             "and Jupiter. It has no tail and shows up as a point that drifts against the stars. "
             "<a href='https://science.nasa.gov/solar-system/asteroids/' target='_blank'>NASA</a></p>", "1"),
        card("In space", "Comet", "<p>A body of ice, dust and rock. Near the Sun it releases gas and dust and grows "
             "a coma and tails. Its dust trail feeds meteor showers. "
             "<a href='https://science.nasa.gov/solar-system/comets/' target='_blank'>NASA</a></p>", "2"),
        card("In space", "Meteoroid", "<p>A small particle, from a grain of sand to about a metre across. "
             "<a href='https://en.wikipedia.org/wiki/Meteoroid' target='_blank'>Wikipedia</a></p>", "3"),
        card("In the air", "Meteor", "<p>The streak of light made when a meteoroid enters the atmosphere at tens of "
             "kilometres per second. "
             "<a href='https://en.wikipedia.org/wiki/Meteor_shower' target='_blank'>Meteor showers</a> happen "
             "when Earth crosses a comet's dust trail.</p>", "4"),
        card("In the air", "Fireball", "<p>A meteor brighter than Venus. One that explodes in the air is a bolide. "
             "<a href='https://cneos.jpl.nasa.gov/fireballs/' target='_blank'>NASA/JPL fireball reports</a></p>", "5"),
        card("On the ground", "Meteorite", "<p>The part of a meteoroid that survives the fall and lands. "
             "<a href='https://en.wikipedia.org/wiki/Meteorite' target='_blank'>Wikipedia</a></p>", "6"),
    ])
    st.markdown(
        panel("Where SPHEREx fits",
              "<p>SPHEREx cannot see meteors, fireballs or meteorites, which happen in or below the atmosphere. "
              "It observes their sources, the asteroids and comets, and the dust that fills the Solar System. "
              "Because it returns to each region of sky, these objects give themselves away by moving between "
              "visits.</p>"),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------- Infrared
with tab_ir:
    st.markdown('<p class="lead">The eye sees a narrow strip of light. SPHEREx looks just past its red end.</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="panel"><div class="spec"><div class="vis"></div><div class="ir"></div>'
        '<div class="tick" style="left:51.9%">3.0 \u00b5m water ice</div>'
        '<div class="tick" style="left:76.3%">4.27 \u00b5m CO2 ice</div>'
        '<div class="tick" style="left:84%;top:-2.7rem">4.67 \u00b5m CO ice</div>'
        '<div class="lab" style="left:1.5%">visible</div>'
        '<div class="lab" style="left:40%">SPHEREx: 0.75 to 5 \u00b5m</div></div>'
        "<p class='small'>Wavelength axis from 0.3 to 5.5 micrometres. Ice feature positions are approximate.</p></div>",
        unsafe_allow_html=True,
    )
    _vis, _ir = dust_pair()
    st.markdown(
        panel("Seeing through the dust",
              '<div class="pair"><div>' + jpg_img(_vis, "Star field hidden by a dust lane in visible light") +
              '<div class="cap">Visible light: a dust lane swallows most of the stars behind it.</div></div>'
              '<div>' + jpg_img(_ir, "The same star field in near-infrared light") +
              '<div class="cap">Near-infrared: the same stars shine through the same dust.</div></div></div>'
              "<p class='small'>Simulated illustration of one field, not SPHEREx data.</p>"),
        unsafe_allow_html=True,
    )
    grid([
        card("Dust", "Penetrating dust", "<p>Dust blocks visible light but lets infrared through far better, "
             "which exposes star-forming clouds and the Galactic centre.</p>", "01"),
        card("Distance", "Redshift", "<p>Expansion of the universe stretches light from distant galaxies to longer "
             "wavelengths (<a href='https://en.wikipedia.org/wiki/Redshift' target='_blank'>redshift</a>), moving "
             "visible and ultraviolet emission into the infrared.</p>", "02"),
        card("Chemistry", "Molecular fingerprints", "<p>Molecules and ices absorb at specific infrared wavelengths. "
             "Water, carbon dioxide and carbon monoxide ices all have features inside the SPHEREx range.</p>", "03"),
        card("Temperature", "Cool bodies", "<p>Dust, planets and small bodies are easier to study in the infrared. "
             "Asteroids and comets are seen in reflected sunlight and in their own thermal emission.</p>", "04"),
        card("Location", "Observing from space", "<p>Water vapour and carbon dioxide absorb much of the infrared, "
             "and the atmosphere itself glows in it. Above the atmosphere, with a cold instrument, the sky is far "
             "darker.</p>", "05"),
    ])
    st.markdown(
        panel("Stretched by the universe", redshift_svg() +
              "<p class='cap'>Light that left a distant galaxy as visible or ultraviolet arrives as infrared. This "
              "is why an infrared survey reaches so far back in cosmic history.</p>"),
        unsafe_allow_html=True,
    )
    st.markdown(
        panel("Further reading",
              "<p><a href='https://science.nasa.gov/ems/07_infraredwaves/' target='_blank'>NASA: Infrared waves</a> | "
              "<a href='https://en.wikipedia.org/wiki/Infrared_astronomy' target='_blank'>Wikipedia: Infrared "
              "astronomy</a></p>"),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------- Explorer
with tab_tool:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    q = st.query_params

    def _f(key, default, lo, hi):
        try:
            return float(np.clip(float(q.get(key, default)), lo, hi))
        except Exception:
            return default

    if "ra" not in st.session_state:
        st.session_state["ra"] = q.get("ra", "08h55m10.831s")
    if "dec" not in st.session_state:
        st.session_state["dec"] = q.get("dec", "-07d14m42.53s")
    if "fov" not in st.session_state:
        st.session_state["fov"] = _f("fov", 0.05, 0.01, MAX_FOV)
    if "dets" not in st.session_state:
        try:
            dd = [f"D{int(x)}" for x in str(q.get("det", "2")).split(",") if int(x) in DETECTORS]
        except Exception:
            dd = []
        st.session_state["dets"] = dd or ["D2"]

    def apply_preset():
        p = PRESETS.get(st.session_state.get("preset"))
        if p:
            st.session_state["ra"], st.session_state["dec"] = p

    def resolve_name():
        nm = st.session_state.get("objname", "").strip()
        if not nm:
            return
        try:
            c = SkyCoord.from_name(nm)
            st.session_state["ra"], st.session_state["dec"] = f"{c.ra.deg:.6f}", f"{c.dec.deg:.6f}"
            st.session_state["resolve_msg"] = f"Resolved {nm}: RA {c.ra.deg:.5f}, Dec {c.dec.deg:.5f} (J2000)."
        except Exception as e:
            st.session_state["resolve_msg"] = f"Could not resolve '{nm}': {str(e)[:100]}"

    st.selectbox("Quick targets", list(PRESETS.keys()), key="preset", on_change=apply_preset)
    n1, n2 = st.columns([3, 1])
    with n1:
        st.text_input("Or look up an object by name", key="objname", placeholder="e.g. Barnard's Star")
    with n2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.button("Look up", on_click=resolve_name)
    if st.session_state.get("resolve_msg"):
        st.caption(st.session_state["resolve_msg"] + " Names give J2000 positions. Stars with large proper "
                   "motion, such as Barnard's Star, have since moved by several arcminutes; use the quick targets.")
    c1, c2 = st.columns(2)
    with c1:
        ra = st.text_input("RA", key="ra", help="Sexagesimal (08h55m10.831s) or degrees (133.79)")
    with c2:
        dec = st.text_input("Dec", key="dec", help="Sexagesimal (-07d14m42.53s) or degrees (-7.25)")

    c3, c4, c5 = st.columns(3)
    with c3:
        size = st.slider("Field of view (degrees)", 0.01, MAX_FOV, step=0.01, key="fov",
                         help="Larger fields follow faster movers but take longer to download.")
    with c4:
        det_sel = st.multiselect("Detectors", [f"D{d}" for d in DETECTORS], key="dets",
                                 help="Each of the six arrays covers a different wavelength range. "
                                      "Select several to combine them.")
    with c5:
        max_frames = st.slider("Maximum cutouts", 6, 60, 12)

    method_label = st.selectbox("Archive search method", ["Auto (TAP first, then SIA)", "SIA only", "TAP only"])
    demo = st.checkbox("Demo mode (artificial frames, no archive access needed)")
    clicked = st.button("Load images", type="primary")

    autorun = q.get("run") == "1" and not st.session_state.get("_autorun_done")
    if autorun:
        st.session_state["_autorun_done"] = True

    if clicked or autorun:
        if demo:
            st.session_state["cube"] = make_demo_cube(max_frames)
            st.session_state["meta"] = {"ra": 0.0, "dec": 0.0, "size": 0.05, "n": 40, "demo": True}
            st.info("Demo mode: the frames are artificial. Find the source that moves among the fixed stars.")
        elif not det_sel:
            st.warning("Select at least one detector.")
        else:
            method = {"Auto (TAP first, then SIA)": "auto", "SIA only": "sia", "TAP only": "tap"}[method_label]
            out = run_query(ra, dec, size, det_sel, method, max_frames)
            if out:
                st.session_state["cube"], st.session_state["meta"] = out
                try:
                    st.query_params.update({"ra": ra, "dec": dec, "fov": f"{size:.2f}",
                                            "det": ",".join(d[1] for d in det_sel)})
                except Exception:
                    pass

    cube = st.session_state.get("cube")
    meta = st.session_state.get("meta")
    if cube and meta:
        st.markdown("### Processing")
        p1, p2, p3 = st.columns(3)
        with p1:
            window = st.slider("Co-add window (days, 0 = off)", 0.0, 400.0, 0.0, 0.5,
                               help="Images inside one window are averaged after background removal. "
                                    "This raises signal-to-noise and merges detectors.")
            order = st.radio("Frame order (no co-add)", ["Time", "Wavelength"], horizontal=True)
        with p2:
            per_det = st.checkbox("Co-add each detector separately", value=True,
                                  help="Gives one stacked image per detector and window, instead of mixing "
                                       "detectors that cover different wavelengths.")
            weighted = st.checkbox("Weight co-add by inverse variance", value=True,
                                   help="Uses the VARIANCE plane of each cutout when it is available.")
            sub = st.radio("Subtract reference", ["None", "Median of all frames", "First frame"],
                           help="Static sources cancel, moving sources remain as positive and negative residuals.")
            nsig = st.slider("Candidate threshold (sigma)", 4.0, 10.0, 5.0, 0.5,
                             help="Used with 'Median of all frames' to list sources that are not part of "
                                  "the static sky.")
        with p3:
            marker = st.checkbox("Mark target position")
            gaia = st.checkbox("Overlay Gaia stars", disabled=meta["demo"])
            glim = st.slider("Gaia magnitude limit (G)", 10.0, 19.0, 16.0, 0.5,
                             help="Only stars brighter than this are drawn. Fainter limits give many more circles.")
            wise = st.checkbox("Overlay WISE sources (AllWISE)", disabled=meta["demo"])
            known = st.checkbox("Mark known asteroids and comets (SkyBoT)", disabled=meta["demo"])
            if st.button("Retry unavailable services"):
                st.session_state["cat_failed"] = set()
            speed = st.slider("Animation speed (frames per second)", 1, 10, 3)

        flag_map = {}
        for f in cube:
            flag_map.update(f.get("bits") or {})
        flag_sel = []
        if flag_map:
            names = sorted(flag_map)
            default = [nm for nm in names if nm.upper() in BAD_FLAGS]
            flag_sel = st.multiselect("Mask pixels with these flags", names, default=default,
                                      help="Flag names are read from the FITS header of each cutout. Masked "
                                           "pixels are excluded from images, co-adds and spectra.")
        else:
            st.caption("Flag names were not found in the cutout headers. You can mask by bit number instead; "
                       "the header text is shown under Diagnostics.")
            manual = st.text_input("Bit numbers to mask (comma-separated, optional)", value="")
        maskbits = 0
        for nm in flag_sel:
            maskbits |= 1 << int(flag_map[nm])
        if not flag_map:
            for tok in manual.replace(";", ",").split(","):
                if tok.strip().isdigit() and int(tok) < 62:
                    maskbits |= 1 << int(tok)

        frames, stretch, refimg = process(cube, window, sub, per_det, order, maskbits, weighted)
        wcs_t = None if meta["demo"] else make_grid_wcs(meta["ra"], meta["dec"], meta["n"])
        overlays = []
        if wcs_t is not None and (gaia or wise):
            radius = meta["size"] * 0.75
            for flag, kind, color, shape, nm in ((gaia, "gaia", (90, 200, 255), "circle", "Gaia"),
                                                 (wise, "wise", (255, 170, 60), "square", "WISE")):
                if flag and nm in st.session_state.setdefault("cat_failed", set()):
                    st.warning(f"{nm} catalogue did not respond earlier in this session, skipped.")
                elif flag:
                    try:
                        with st.spinner(f"Querying {nm}..."):
                            overlays.append((fetch_catalog(kind, meta["ra"], meta["dec"], radius, glim), color, shape))
                    except Exception as e:
                        st.session_state["cat_failed"].add(nm)
                        st.warning(f"{nm} catalogue unavailable: {str(e)[:120]}")

        cands = find_movers(frames, refimg, wcs_t, nsig) if sub == "Median of all frames" else []
        tracks = find_tracks(cands, frames) if cands else []
        linked = {}
        for ti, tr in enumerate(tracks, 1):
            for mi in tr["members"]:
                linked[mi] = ti
        known_by_frame = {}
        if known and wcs_t is not None:
            if "SkyBoT" in st.session_state.setdefault("cat_failed", set()):
                st.warning("SkyBoT did not respond earlier in this session, skipped.")
            else:
                try:
                    with st.spinner("Querying SkyBoT for known solar-system objects..."):
                        known_by_frame = known_objects(frames, meta["ra"], meta["dec"], meta["size"] * 0.75)
                except Exception as e:
                    st.session_state["cat_failed"].add("SkyBoT")
                    st.warning(f"SkyBoT unavailable: {str(e)[:120]}")

        pics = []
        for k, f in enumerate(frames):
            ov = resolve_overlays(overlays, f.get("mjd", np.nan))
            for is_linked, color in ((False, (255, 80, 200)), (True, (255, 225, 70))):
                mine = [c for ci, c in enumerate(cands) if c["k"] == k and (ci in linked) == is_linked]
                if mine:
                    ov.append(((np.array([c["ra"] for c in mine]), np.array([c["dec"] for c in mine])),
                               color, "diamond"))
            kn = known_by_frame.get(k, [])
            if kn:
                ov.append(((np.array([x[1] for x in kn]), np.array([x[2] for x in kn])), (110, 255, 130), "cross"))
            pics.append(to_picture(f["data"], stretch, f["label"], ov, wcs_t, marker))

        st.markdown(f"### Sequence ({len(pics)} frames)")
        t_anim, t_step, t_rgb = st.tabs(["Animation", "Frame by frame", "Colour composite"])
        with t_anim:
            if len(pics) > 1:
                gif = make_gif(pics, int(1000 / speed))
                st.image(gif, width=560)
                st.download_button("Download animation (GIF)", gif, file_name="sequence.gif", mime="image/gif")
            else:
                st.image(pics[0], width=560)
        with t_step:
            i = st.slider("Frame", 1, len(pics), 1) - 1 if len(pics) > 1 else 0
            st.image(pics[i], width=560)
        with t_rgb:
            rgb = composite(cube)
            if rgb is None:
                st.info("Select detectors from at least two of the groups D1-D2, D3-D4 and D5-D6 to build "
                        "a colour composite.")
            else:
                st.image(rgb, width=560)
                st.caption("Blue: D1 and D2 (0.75 to 1.6 um). Green: D3 and D4 (1.6 to 3.8 um). Red: D5 and D6 "
                           "(3.8 to 5.0 um). Very red sources, such as brown dwarfs, stand out. A source that moves "
                           "between the exposures appears as displaced coloured dots.")

        if sub == "Median of all frames":
            st.markdown("### Moving-source candidates")
            if tracks:
                st.markdown("**Linked tracks** (yellow diamonds): candidates that fit uniform straight-line motion "
                            "in at least three frames.")
                tdf = pd.DataFrame([{"track": i, "detections": t["n"], "rate_arcsec_per_hour": round(t["rate_arcsec_hr"], 2),
                                     "position_angle_deg": round(t["pa_deg"], 1), "RA_deg": round(t["ra"], 5),
                                     "Dec_deg": round(t["dec"], 5)} for i, t in enumerate(tracks, 1)])
                st.dataframe(tdf, hide_index=True)
                st.download_button("Download tracks (CSV)", tdf.to_csv(index=False), file_name="tracks.csv")
            elif cands:
                st.caption("No candidates link into a straight-line track yet. Add frames or lower the threshold.")
            if cands:
                st.caption("Peaks above the threshold in the subtracted frames, at positions where the median "
                           "image is empty. Magenta diamonds mark them on the frames. 'repeats' counts other "
                           "detections at the same pixel: a high value suggests a variable star or a detector "
                           "artefact rather than a mover. Always inspect candidates by eye.")
                cdf = pd.DataFrame([{"frame": c["k"] + 1, "image": c["label"], "RA_deg": round(c["ra"], 5),
                                     "Dec_deg": round(c["dec"], 5), "snr": round(c["snr"], 1),
                                     "repeats": c["repeats"], "track": linked.get(ci, "")}
                                    for ci, c in enumerate(cands)])
                st.dataframe(cdf, hide_index=True)
                st.download_button("Download candidates (CSV)", cdf.to_csv(index=False), file_name="candidates.csv")
            else:
                st.caption("No candidates above the threshold. Lower the threshold or add more frames.")
        if known_by_frame:
            names = sorted({x[0] for v in known_by_frame.values() for x in v})
            st.caption(f"Known objects in the field (green crosses): {', '.join(names[:12])}"
                       f"{' and more' if len(names) > 12 else ''}." if names else "No known objects in the field.")

        if len(frames) >= 2:
            st.markdown("### Two-epoch comparison")
            m = len(frames)
            e1, e2 = st.columns(2)
            with e1:
                ra_rng = st.slider("Epoch A (frame range)", 1, m, (1, max(1, m // 2)))
            with e2:
                rb_rng = st.slider("Epoch B (frame range)", 1, m, (min(m, m // 2 + 1), m))
            A = np.nanmean(np.stack([bgsub(f["data"]) for f in frames[ra_rng[0] - 1:ra_rng[1]]]), axis=0)
            B = np.nanmean(np.stack([bgsub(f["data"]) for f in frames[rb_rng[0] - 1:rb_rng[1]]]), axis=0)
            k1, k2, k3 = st.columns(3)
            mjd_a = float(np.nanmean([f["mjd"] for f in frames[ra_rng[0] - 1:ra_rng[1]]]))
            mjd_b = float(np.nanmean([f["mjd"] for f in frames[rb_rng[0] - 1:rb_rng[1]]]))
            k1.image(to_picture(A, "asinh", "Epoch A", resolve_overlays(overlays, mjd_a), wcs_t, marker), width=360)
            k2.image(to_picture(B, "asinh", "Epoch B", resolve_overlays(overlays, mjd_b), wcs_t, marker), width=360)
            k3.image(to_picture(A - B, "linear", "A minus B",
                                resolve_overlays(overlays, float(np.nanmean([mjd_a, mjd_b]))), wcs_t, marker), width=360)
            st.caption("A moving source appears as a bright and a dark spot in the difference image.")

        st.markdown("### Spectrum at the target position")
        n = meta["n"]
        s1, s2, s3 = st.columns(3)
        with s1:
            ap = st.slider("Aperture radius (pixels)", 1.0, 5.0, 1.5, 0.5)
        with s2:
            ox = st.slider("Horizontal offset from target (pixels)", -(n // 2), n // 2, 0)
        with s3:
            oy = st.slider("Vertical offset from target (pixels, up is positive)", -(n // 2), n // 2, 0)
        yy, xx = np.mgrid[0:n, 0:n]
        c0 = (n - 1) / 2
        mask = (xx - (c0 + ox)) ** 2 + (yy - (c0 + oy)) ** 2 <= ap ** 2
        rows = []
        for f in cube:
            if f.get("wave"):
                d = _prep(f, maskbits)
                if np.isfinite(d[mask]).any():
                    flux = float(np.nanmean(d[mask]) - np.nanmedian(d))
                    err, vv = 0.0, f.get("var")
                    if vv is not None and vv.shape == d.shape:
                        mm = mask & np.isfinite(d) & np.isfinite(vv)
                        if mm.any():
                            err = float(np.sqrt(np.sum(vv[mm])) / mm.sum())
                    rows.append({"wavelength_um": f["wave"], "flux_MJy_per_sr": flux, "err": err,
                                 "lo": flux - err, "hi": flux + err,
                                 "detector": f"D{f['det']}", "date": pretty_date(f["mjd"])})
        if rows:
            df = pd.DataFrame(rows).sort_values("wavelength_um")
            base = alt.Chart(df)
            xax = alt.X("wavelength_um:Q", title="Wavelength (um)")
            pts = base.mark_circle(size=70).encode(
                x=xax, y=alt.Y("flux_MJy_per_sr:Q", title="Flux above local background (MJy/sr)"),
                color=alt.Color("detector:N", title="Detector"),
                tooltip=["wavelength_um", "flux_MJy_per_sr", "err", "detector", "date"])
            bars = base.mark_rule().encode(x=xax, y="lo:Q", y2="hi:Q", color=alt.Color("detector:N", legend=None))
            chart = ((bars + pts).properties(height=320)
                .configure(background="transparent")
                .configure_axis(labelColor="#b9c3d1", titleColor="#b9c3d1", gridColor="#232b36", domainColor="#3a4554")
                .configure_legend(labelColor="#b9c3d1", titleColor="#b9c3d1"))
            st.altair_chart(chart, theme=None)
            st.download_button("Download spectrum (CSV)", df.to_csv(index=False), file_name="spectrum.csv")
            st.caption("Values come from the aperture at the chosen position, one point per exposure. "
                       "Each exposure samples a different wavelength. Error bars come from the VARIANCE plane.")
        else:
            st.caption("Wavelength information is not available for these frames.")

        st.markdown("### Image metadata")
        meta_rows = [{"date": pretty_date(f["mjd"]), "mjd": f["mjd"], "detector": f"D{f['det']}",
                      "wavelength_um": None if f["wave"] is None else round(f["wave"], 3),
                      "bandwidth_um": None if f.get("bw") is None else round(f["bw"], 3),
                      "flagged_pixel_fraction": None if f["flagged"] is None else round(f["flagged"], 3),
                      "file": f["uri"].split("?")[0].split("/")[-1]}
                     for f in sorted(cube, key=lambda f: (np.isnan(f["mjd"]), f["mjd"]))]
        st.dataframe(pd.DataFrame(meta_rows), hide_index=True)
        st.caption("Flagged fraction is the share of cutout pixels with any bit set in the FLAGS plane. The meaning of "
                   "individual bits is described in the SPHEREx archive documentation.")

        if not meta["demo"]:
            st.markdown("### Sky context")
            with st.expander("Open Aladin Lite"):
                fov = max(meta["size"] * 4, 0.2)
                html = f"""
                <div id="al" style="width:100%;height:420px"></div>
                <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js" charset="utf-8"></script>
                <script>
                  A.init.then(() => {{
                    let a = A.aladin('#al', {{survey: 'P/allWISE/color', fov: {fov}, target: '{meta["ra"]} {meta["dec"]}'}});
                  }});
                </script>"""
                components.html(html, height=440)

    with st.expander("Share or embed this view"):
        st.markdown("The page reads these URL parameters and loads the images automatically when `run=1` is present.")
        st.code("?ra=133.79&dec=-7.25&fov=0.05&det=2,3&run=1")
        st.markdown("Add `&embed=1` to show only the Explorer, without the header and tabs, for use inside an iframe.")

    with st.expander("How to read the frames"):
        st.markdown(
            "- Every frame shows the same patch of sky, north up and east left, sorted from oldest to newest.\n"
            "- Stars and galaxies stay in place. A source that changes position is a moving object.\n"
            "- Each SPHEREx image samples a different wavelength, so brightness changes. Watch position.\n"
            "- One pixel is about 6 arcseconds. A 0.05 degree field is about 29 by 29 pixels. Widen the field if a "
            "fast mover leaves it.\n"
            "- After subtracting a reference, a mover shows as a bright source with a dark counterpart at its "
            "other positions."
        )

    with st.expander("Diagnostics"):
        if st.session_state.get("cube"):
            st.caption("FITS structure of the first loaded cutout:")
            st.code(st.session_state["cube"][0].get("struct") or "not available")
        st.caption("Checks each step of the archive connection and reports the time taken.")
        if st.button("Run archive diagnostics"):
            import requests
            timed("IRSA website", lambda: requests.get("https://irsa.ipac.caltech.edu/", timeout=25).status_code, 30)
            timed("Simple TAP query", lambda: run_tap(IRSA_TAP, "SELECT TOP 1 planeid FROM spherex.plane", 40), 45)
            try:
                c = parse_coord(ra, dec)
                dn = tuple(sorted(int(d[1]) for d in det_sel)) or (2,)
                r1 = timed("Image search (SIA)", lambda: search_sia(c.ra.deg, c.dec.deg, size, dn), 90)
                if r1:
                    st.write(f"SIA found {len(r1)} images")
                    st.code(r1[0]["uri"])
                    timed("Download one cutout (SIA)", lambda: download_cutout(r1[0]["uri"]), 60)
                r2 = timed("Image search (TAP)", lambda: search_tap(c.ra.deg, c.dec.deg, size, dn), 90)
                if r2:
                    st.write(f"TAP found {len(r2)} images")
                    st.code(r2[0]["uri"])
                    timed("Download one cutout (TAP)", lambda: download_cutout(r2[0]["uri"]), 60)
            except Exception as e:
                st.error(f"Could not read the coordinates: {e}")

st.markdown(
    "<p class='small' style='text-align:center;margin-top:2rem'>Data: SPHEREx and the NASA/IPAC Infrared Science "
    "Archive (IRSA); Gaia (ESA); AllWISE. Unofficial educational project, not affiliated with NASA or Caltech.</p>",
    unsafe_allow_html=True,
)
