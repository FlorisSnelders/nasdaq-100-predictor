"""
Daybreak — Nasdaq-100 next-day directional intelligence console
Group 7 · Stock Predictor

Run with:
    streamlit run dashboard.py

Reads from the SQLite predictions/cleaned_prices tables described in the
design document (Storage Layer, section 5). If the predictions table is
empty or missing, the dashboard falls back to clearly-labelled DEMO data so
the design is visible before the pipeline has been run end-to-end.

Decision-support only. Not investment advice (EU AI Act, Article 50).
"""

from __future__ import annotations

import os
import math
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.storage import load_predictions, load_from_db

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DB_PATH = os.environ.get("STOCKPREDICTOR_DB", "data/nasdaq100.db")
SPARK_WINDOW = 30  # trading days shown in each sparkline

# Warm "candlelit at dawn" palette — intentionally not the usual blue-on-white
C_BG       = "#0D0B0A"
C_SURFACE  = "#16120E"
C_SURFACE2 = "#1E1813"
C_BORDER   = "rgba(232,177,76,0.10)"
C_TEXT     = "#F3EDE2"
C_MUTED    = "#9C9387"
C_AMBER    = "#E8B14C"   # brand / daybreak accent
C_ROSE     = "#E8825C"   # secondary sunrise tone
C_BULL     = "#5FD6A4"   # up
C_BEAR     = "#F0746E"   # down

st.set_page_config(
    page_title="NASDAQ-100 Next-Day Direction Prediction",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Nasdaq-100 ticker -> company name (best-effort; falls back to the ticker)
# --------------------------------------------------------------------------- #
NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet (A)", "GOOG": "Alphabet (C)", "META": "Meta Platforms",
    "AVGO": "Broadcom", "TSLA": "Tesla", "COST": "Costco", "NFLX": "Netflix",
    "AMD": "Advanced Micro Devices", "PEP": "PepsiCo", "ADBE": "Adobe",
    "CSCO": "Cisco", "TMUS": "T-Mobile US", "INTC": "Intel", "CMCSA": "Comcast",
    "QCOM": "Qualcomm", "INTU": "Intuit", "TXN": "Texas Instruments",
    "AMGN": "Amgen", "ISRG": "Intuitive Surgical", "AMAT": "Applied Materials",
    "HON": "Honeywell", "BKNG": "Booking Holdings", "VRTX": "Vertex Pharma",
    "ADP": "Automatic Data Processing", "MU": "Micron", "ADI": "Analog Devices",
    "REGN": "Regeneron", "GILD": "Gilead Sciences", "LRCX": "Lam Research",
    "PANW": "Palo Alto Networks", "MELI": "MercadoLibre", "SBUX": "Starbucks",
    "KLAC": "KLA Corp", "SNPS": "Synopsys", "CDNS": "Cadence Design",
    "MDLZ": "Mondelez", "PYPL": "PayPal", "MAR": "Marriott", "CRWD": "CrowdStrike",
    "CTAS": "Cintas", "ORLY": "O'Reilly Automotive", "ASML": "ASML Holding",
    "ABNB": "Airbnb", "NXPI": "NXP Semiconductors", "PCAR": "PACCAR",
    "CEG": "Constellation Energy", "ROP": "Roper Technologies", "MNST": "Monster",
    "FTNT": "Fortinet", "DASH": "DoorDash", "WDAY": "Workday", "ADSK": "Autodesk",
    "AEP": "American Electric Power", "PAYX": "Paychex", "CHTR": "Charter Comm.",
    "CPRT": "Copart", "ROST": "Ross Stores", "KDP": "Keurig Dr Pepper",
    "ODFL": "Old Dominion", "FAST": "Fastenal", "MRVL": "Marvell Technology",
    "GEHC": "GE HealthCare", "DDOG": "Datadog", "VRSK": "Verisk Analytics",
    "EA": "Electronic Arts", "CTSH": "Cognizant", "EXC": "Exelon",
    "XEL": "Xcel Energy", "KHC": "Kraft Heinz", "TTD": "The Trade Desk",
    "CCEP": "Coca-Cola Europacific", "FANG": "Diamondback Energy", "BKR": "Baker Hughes",
    "IDXX": "IDEXX Laboratories", "CSGP": "CoStar Group", "ON": "ON Semiconductor",
    "MCHP": "Microchip Technology", "DXCM": "DexCom", "CDW": "CDW Corp",
    "TEAM": "Atlassian", "GFS": "GlobalFoundries", "BIIB": "Biogen",
    "ZS": "Zscaler", "ANSS": "Ansys", "WBD": "Warner Bros. Discovery",
    "ILMN": "Illumina", "MDB": "MongoDB", "DLTR": "Dollar Tree",
    "WBA": "Walgreens Boots", "LULU": "Lululemon", "ARM": "Arm Holdings",
    "SMCI": "Super Micro Computer", "PLTR": "Palantir", "APP": "AppLovin",
    "MSTR": "MicroStrategy", "AXON": "Axon Enterprise",
}


def company(ticker: str) -> str:
    return NAMES.get(ticker.upper(), ticker.upper())


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner=False)
def load_data(db_path: str):
    """Return (predictions_df, prices_long_df, as_of, is_demo)."""
    preds = load_predictions()
    if preds.empty:
        preds, prices, as_of = _demo_data()
        return preds, prices, as_of, True

    preds = preds.copy()
    preds["date"] = preds["date"].dt.strftime("%Y-%m-%d")
    as_of = preds["date"].max()
    preds = preds[preds["date"] == as_of]

    prices = load_from_db().copy()
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    prices = prices[prices["date"] <= as_of].sort_values("date")
    return preds, prices, as_of, False


def _demo_data():
    """Synthesise plausible predictions + price walks for the design preview."""
    rng = np.random.default_rng(7)
    tickers = list(NAMES.keys())
    as_of = datetime.now().strftime("%Y-%m-%d")

    # probabilities clustered realistically around the coin-flip line
    probs = np.clip(rng.normal(0.55, 0.11, len(tickers)), 0.34, 0.93)
    preds = pd.DataFrame({
        "ticker": tickers,
        "date": as_of,
        "prediction": (probs >= 0.5).astype(int),
        "probability": probs.round(4),
    })

    rows = []
    # Roll a weekend "today" back to Friday: with an end date on a weekend,
    # bdate_range returns one date fewer than `periods`.
    last_bday = pd.offsets.BDay().rollback(pd.Timestamp(as_of))
    dates = pd.bdate_range(end=last_bday, periods=SPARK_WINDOW)
    for t in tickers:
        price = rng.uniform(40, 600)
        for d in dates:
            open_ = price
            close = max(5.0, open_ * (1 + rng.normal(0, 0.018)))
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.006)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.006)))
            vol = int(rng.uniform(2e6, 6e7))
            rows.append((t, d.strftime("%Y-%m-%d"), round(open_, 2),
                         round(high, 2), round(low, 2), round(close, 2), vol))
            price = close
    prices = pd.DataFrame(
        rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"])
    return preds, prices, as_of


def build_frame(preds: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Merge predictions with price context. confidence = prob of predicted class."""
    spark, last, chg = {}, {}, {}
    for t, grp in prices.groupby("ticker"):
        s = grp.sort_values("date")["close"].tail(SPARK_WINDOW).tolist()
        spark[t] = s
        if s:
            last[t] = s[-1]
            chg[t] = (s[-1] / s[0] - 1.0) if s[0] else 0.0

    df = preds.copy()
    df["ticker"] = df["ticker"].str.upper()
    df["name"] = df["ticker"].map(company)
    df["up"] = df["prediction"].astype(int) == 1
    p = df["probability"].astype(float)
    df["confidence"] = p.where(df["up"], 1 - p)
    df["spark"] = df["ticker"].map(spark)
    df["last"] = df["ticker"].map(last)
    df["window_chg"] = df["ticker"].map(chg)
    df["spark"] = df["spark"].apply(lambda v: v if isinstance(v, list) else [])
    return df.sort_values("confidence", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tiny SVG drawing helpers (no chart library -> distinctive, lightweight)
# --------------------------------------------------------------------------- #
def sparkline(values, color, w=120, h=34):
    if not values or len(values) < 2:
        return f'<svg width="{w}" height="{h}"></svg>'
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    n = len(values)
    pts = [
        (x / (n - 1) * (w - 2) + 1, h - 2 - (v - lo) / rng * (h - 4))
        for x, v in enumerate(values)
    ]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"M{pts[0][0]:.1f},{h} " + \
        " ".join(f"L{x:.1f},{y:.1f}" for x, y in pts) + \
        f" L{pts[-1][0]:.1f},{h} Z"
    gid = f"g{abs(hash((tuple(values), color))) % 99999}"
    return f"""<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" fill="none">
      <defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="{color}" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="{color}" stop-opacity="0"/>
      </linearGradient></defs>
      <path d="{area}" fill="url(#{gid})"/>
      <polyline points="{line}" stroke="{color}" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="2.2" fill="{color}"/>
    </svg>"""


def confidence_arc(conf, color, size=190):
    """270-degree open gauge with a soft glow."""
    r = size / 2 - 16
    cx = cy = size / 2
    start = 135  # degrees; gap at the bottom
    sweep = 270
    circ = 2 * math.pi * r
    arc_len = circ * sweep / 360.0
    filled = arc_len * conf

    def pt(deg):
        a = math.radians(deg)
        return cx + r * math.cos(a), cy + r * math.sin(a)

    x0, y0 = pt(start)
    x1, y1 = pt(start + sweep)
    big = 1 if sweep > 180 else 0
    track = f"M{x0:.1f},{y0:.1f} A{r},{r} 0 {big} 1 {x1:.1f},{y1:.1f}"

    return f"""<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <defs>
        <filter id="gl" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="4" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <path d="{track}" stroke="rgba(255,255,255,0.06)" stroke-width="10"
        fill="none" stroke-linecap="round"/>
      <path d="{track}" stroke="{color}" stroke-width="10" fill="none"
        stroke-linecap="round" filter="url(#gl)"
        stroke-dasharray="{filled:.1f} {arc_len:.1f}"/>
      <text x="{cx}" y="{cy-2}" text-anchor="middle" fill="{C_TEXT}"
        font-family="'IBM Plex Mono',monospace" font-size="40"
        font-weight="600">{conf*100:.0f}<tspan font-size="20">%</tspan></text>
      <text x="{cx}" y="{cy+24}" text-anchor="middle" fill="{C_MUTED}"
        font-family="'Spline Sans',sans-serif" font-size="11"
        letter-spacing="2">CONFIDENCE</text>
    </svg>"""


def conf_bar(conf, color, w=92):
    pct = conf * 100
    return f"""<div class="cbar" style="width:{w}px">
      <div class="cbar-fill" style="width:{pct:.0f}%;background:{color}"></div>
    </div><span class="cbar-num">{pct:.0f}%</span>"""


def arrow(up: bool, size: int = 22) -> str:
    color = C_BULL if up else C_BEAR
    tint = "rgba(95,214,164,0.12)" if up else "rgba(240,116,110,0.12)"
    path = ("M7 2 L11.6 8.4 L8.7 8.4 L8.7 12.6 L5.3 12.6 L5.3 8.4 L2.4 8.4 Z"
            if up else
            "M7 12.6 L2.4 6.2 L5.3 6.2 L5.3 2 L8.7 2 L8.7 6.2 L11.6 6.2 Z")
    return (f'<span class="arrowchip" style="width:{size}px;height:{size}px;'
            f'background:{tint};border:1px solid {color}40">'
            f'<svg width="14" height="14" viewBox="0 0 14 14">'
            f'<path d="{path}" fill="{color}"/></svg></span>')


def next_session_label(as_of) -> str:
    """Next NYSE trading session after `as_of` (falls back to next business day)."""
    try:
        d = pd.Timestamp(str(as_of)[:10])
    except Exception:  # noqa: BLE001
        return "—"
    nxt = None
    try:
        import pandas_market_calendars as mcal
        sched = mcal.get_calendar("XNYS").schedule(
            start_date=(d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end_date=(d + pd.Timedelta(days=12)).strftime("%Y-%m-%d"),
        )
        if len(sched):
            nxt = sched.index[0]
    except Exception:  # noqa: BLE001 — library optional
        nxt = None
    if nxt is None:
        nxt = d + pd.tseries.offsets.BDay(1)
    return pd.Timestamp(nxt).strftime("%a %d %b %Y").upper()


def fmt_vol(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v >= 1e9:
        return f"{v/1e9:.2f}B"
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    return f"{v:.0f}"


def candles(ohlc, w=820, h=180):
    """Candlestick chart from a list of (open, high, low, close) tuples."""
    if not ohlc:
        return f'<svg width="{w}" height="{h}"></svg>'
    hi = max(r[1] for r in ohlc)
    lo = min(r[2] for r in ohlc)
    rng = (hi - lo) or 1.0
    n = len(ohlc)
    slot = w / n
    bw = min(slot * 0.62, 16)

    def y(p):
        return h - 6 - (p - lo) / rng * (h - 12)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" '
             f'preserveAspectRatio="none" style="height:{h}px">']
    for i, (o, hh, ll, c) in enumerate(ohlc):
        cx = i * slot + slot / 2
        col = C_BULL if c >= o else C_BEAR
        parts.append(f'<line x1="{cx:.1f}" y1="{y(hh):.1f}" x2="{cx:.1f}" '
                     f'y2="{y(ll):.1f}" stroke="{col}" stroke-width="1" '
                     f'opacity="0.7"/>')
        top = y(max(o, c))
        bot = y(min(o, c))
        bh = max(bot - top, 1.5)
        parts.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                     f'height="{bh:.1f}" rx="1.5" fill="{col}"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_detail(prices: pd.DataFrame, row) -> str:
    """Detail panel for a single ticker: OHLC numbers + candlesticks + recent sessions."""
    tk = row["ticker"]
    sub = prices[prices["ticker"] == tk].sort_values("date").tail(SPARK_WINDOW)
    col = C_BULL if row["up"] else C_BEAR
    word = "HIGHER" if row["up"] else "LOWER"

    if sub.empty or "open" not in sub.columns:
        return (f'<div class="detail"><div class="detail-tk">{tk}</div>'
                f'<div class="detail-nm">{row["name"]}</div>'
                f'<div style="color:{C_MUTED};margin-top:10px">No price history '
                f'available for this instrument.</div></div>')

    last = sub.iloc[-1]
    prev_close = sub.iloc[-2]["close"] if len(sub) > 1 else last["open"]
    day_ret = (last["close"] / prev_close - 1) if prev_close else 0.0
    day_range = ((last["high"] - last["low"]) / last["close"]) if last["close"] else 0.0
    drift = (last["close"] / sub.iloc[0]["close"] - 1) if sub.iloc[0]["close"] else 0.0
    ret_col = C_BULL if day_ret >= 0 else C_BEAR

    ohlc = list(zip(sub["open"], sub["high"], sub["low"], sub["close"]))

    def stat(k, v, color=C_TEXT):
        return f'<div class="stat"><div class="k">{k}</div>' \
               f'<div class="v" style="color:{color}">{v}</div></div>'

    stats = (
        stat("OPEN", f"${last['open']:,.2f}") +
        stat("HIGH", f"${last['high']:,.2f}", C_BULL) +
        stat("LOW", f"${last['low']:,.2f}", C_BEAR) +
        stat("CLOSE", f"${last['close']:,.2f}") +
        stat("VOLUME", fmt_vol(last.get("volume"))) +
        stat("DAY RANGE", f"{day_range*100:.1f}%") +
        stat("DAY RETURN", f"{day_ret*100:+.2f}%", ret_col) +
        stat("30-SESSION DRIFT", f"{drift*100:+.1f}%",
             C_BULL if drift >= 0 else C_BEAR)
    )

    # recent sessions table (most recent first)
    mini = ('<div class="mini"><div class="mini-row h">'
            '<span>DATE</span><span class="tr">OPEN</span><span class="tr">HIGH</span>'
            '<span class="tr">LOW</span><span class="tr">CLOSE</span>'
            '<span class="tr">VOL</span></div>')
    recent = sub.tail(8).iloc[::-1]
    pcs = recent["close"].shift(-1)  # previous session (older) for colour
    for (_, rr), pc in zip(recent.iterrows(), pcs):
        up = pd.notna(pc) and rr["close"] >= pc
        cc = C_BULL if up else C_BEAR
        mini += (f'<div class="mini-row"><span>{str(rr["date"])[:10]}</span>'
                 f'<span class="tr">{rr["open"]:,.2f}</span>'
                 f'<span class="tr">{rr["high"]:,.2f}</span>'
                 f'<span class="tr">{rr["low"]:,.2f}</span>'
                 f'<span class="tr" style="color:{cc}">{rr["close"]:,.2f}</span>'
                 f'<span class="tr">{fmt_vol(rr["volume"])}</span></div>')
    mini += "</div>"

    return f"""<div class="detail">
      <div class="detail-head">
        <div>
          <div class="detail-tk">{tk}</div>
          <div class="detail-nm">{row['name']}</div>
        </div>
        <div style="text-align:right">
          <div class="detail-call">{arrow(bool(row['up']))}
            <span style="color:{col}">CALLED {word}</span></div>
          <div style="margin-top:8px">{conf_bar(float(row['confidence']), col)}</div>
        </div>
      </div>
      <div class="candle-wrap">{candles(ohlc)}</div>
      <div class="candle-cap">{len(ohlc)} sessions · green body = close ≥ open ·
        red = close &lt; open</div>
      <div class="stats">{stats}</div>
      {mini}
    </div>"""


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Spline+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

#MainMenu, header[data-testid="stHeader"], footer {{visibility:hidden;}}
.stApp {{
  background:
    radial-gradient(1200px 380px at 50% -120px, rgba(232,177,76,0.16), transparent 70%),
    radial-gradient(900px 300px at 80% -60px, rgba(232,130,92,0.10), transparent 70%),
    {C_BG};
  color:{C_TEXT};
}}
.block-container {{padding-top:1.4rem; padding-bottom:3rem; max-width:1180px;}}
* {{font-family:'Spline Sans',sans-serif;}}

/* masthead */
.mast {{display:flex; align-items:baseline; justify-content:space-between;
  border-bottom:1px solid {C_BORDER}; padding-bottom:14px; margin-bottom:8px;}}
.brand {{font-family:'Fraunces',serif; font-size:27px; font-weight:600;
  letter-spacing:-0.4px; color:{C_TEXT}; line-height:1.1;}}
.brand .dot {{color:{C_AMBER};}}
.brand small {{font-family:'IBM Plex Mono',monospace; font-size:11px;
  letter-spacing:3px; color:{C_MUTED}; display:block; margin-top:-4px;}}
.mast-right {{text-align:right; font-family:'IBM Plex Mono',monospace;
  font-size:12px; color:{C_MUTED}; line-height:1.7;}}
.mast-right b {{color:{C_AMBER}; font-weight:600;}}
.demo-tag {{display:inline-block; background:rgba(232,130,92,0.16);
  color:{C_ROSE}; border:1px solid rgba(232,130,92,0.4); border-radius:99px;
  padding:2px 10px; font-size:10px; letter-spacing:2px; margin-left:8px;}}

/* pulse strip */
.pulse {{display:flex; gap:0; border:1px solid {C_BORDER};
  border-radius:14px; overflow:hidden; margin:18px 0 6px;
  background:linear-gradient(180deg,{C_SURFACE},{C_BG});}}
.pulse .cell {{flex:1; padding:14px 18px; border-right:1px solid {C_BORDER};}}
.pulse .cell:last-child {{border-right:none;}}
.pulse .k {{font-family:'IBM Plex Mono',monospace; font-size:10px;
  letter-spacing:2px; color:{C_MUTED};}}
.pulse .v {{font-family:'Fraunces',serif; font-size:28px; font-weight:600;
  margin-top:2px;}}

/* hero */
.hero {{display:grid; grid-template-columns:200px 1fr; gap:24px;
  border:1px solid {C_BORDER}; border-radius:18px; padding:24px 28px;
  margin:14px 0 26px;
  background:
    radial-gradient(420px 200px at 0% 0%, rgba(232,177,76,0.10), transparent 70%),
    {C_SURFACE};}}
.hero-eyebrow {{font-family:'IBM Plex Mono',monospace; font-size:11px;
  letter-spacing:3px; color:{C_AMBER};}}
.hero-ticker {{font-family:'Fraunces',serif; font-size:58px; font-weight:600;
  line-height:1; letter-spacing:-1px; margin:6px 0 2px;}}
.hero-name {{color:{C_MUTED}; font-size:15px; margin-bottom:18px;}}
.hero-call {{display:inline-flex; align-items:center; gap:10px;
  font-family:'IBM Plex Mono',monospace; font-size:15px; letter-spacing:1px;}}
.hero-meta {{display:flex; gap:34px; margin-top:18px;}}
.hero-meta .k {{font-family:'IBM Plex Mono',monospace; font-size:10px;
  letter-spacing:2px; color:{C_MUTED};}}
.hero-meta .v {{font-family:'IBM Plex Mono',monospace; font-size:18px;
  color:{C_TEXT}; margin-top:3px;}}

/* runner-ups */
.runners {{display:grid; grid-template-columns:repeat(3,1fr); gap:14px;
  margin-bottom:24px;}}
.runner {{border:1px solid {C_BORDER}; border-radius:14px; padding:14px 16px;
  background:{C_SURFACE}; display:flex; flex-direction:column; gap:8px;}}
.runner-top {{display:flex; justify-content:space-between; align-items:center;}}
.runner-tk {{font-family:'Fraunces',serif; font-size:22px; font-weight:600;}}
.runner-nm {{color:{C_MUTED}; font-size:11px;}}

/* section label */
.seclabel {{font-family:'IBM Plex Mono',monospace; font-size:11px;
  letter-spacing:3px; color:{C_MUTED}; margin:10px 0 2px;
  display:flex; align-items:center; gap:10px;}}
.seclabel:after {{content:""; flex:1; height:1px; background:{C_BORDER};}}

/* watchlist */
.list {{border:1px solid {C_BORDER}; border-radius:16px; overflow:hidden;
  background:{C_SURFACE};}}
.list-scroll {{max-height:560px; overflow-y:auto;}}
.list-scroll::-webkit-scrollbar {{width:9px;}}
.list-scroll::-webkit-scrollbar-thumb {{background:rgba(232,177,76,0.25);
  border-radius:9px;}}
.row {{display:grid;
  grid-template-columns:28px 150px 92px 150px 120px 96px;
  align-items:center; gap:12px; padding:12px 18px;
  border-bottom:1px solid {C_BORDER}; transition:background .15s;}}
.row:hover {{background:{C_SURFACE2};}}
.row:last-child {{border-bottom:none;}}
.rk {{font-family:'IBM Plex Mono',monospace; font-size:11px; color:{C_MUTED};}}
.tk {{font-family:'Fraunces',serif; font-size:19px; font-weight:600;}}
.nm {{font-size:11px; color:{C_MUTED}; margin-top:-2px;}}
.dir {{font-size:15px;}}
.callword {{font-family:'IBM Plex Mono',monospace; font-size:12px;
  letter-spacing:1px;}}
.cbar {{display:inline-block; height:6px; border-radius:6px;
  background:rgba(255,255,255,0.07); overflow:hidden; vertical-align:middle;}}
.cbar-fill {{height:100%; border-radius:6px;}}
.cbar-num {{font-family:'IBM Plex Mono',monospace; font-size:12px;
  margin-left:8px; color:{C_TEXT};}}
.px {{font-family:'IBM Plex Mono',monospace; font-size:13px; text-align:right;
  color:{C_TEXT};}}
.pxchg {{font-family:'IBM Plex Mono',monospace; font-size:10px; text-align:right;}}
.head-row {{font-family:'IBM Plex Mono',monospace; font-size:10px;
  letter-spacing:1.5px; color:{C_MUTED}; background:{C_BG};}}
.head-row .row {{border-bottom:1px solid {C_BORDER};}}

/* disclaimer */
.disc {{margin-top:26px; border:1px dashed rgba(232,177,76,0.35);
  border-radius:12px; padding:14px 18px; background:rgba(232,177,76,0.04);
  font-size:12px; color:{C_MUTED}; line-height:1.6;}}
.disc b {{color:{C_AMBER}; font-family:'IBM Plex Mono',monospace;
  letter-spacing:1px;}}

.stSelectbox label, .stSlider label, .stTextInput label,
.stRadio label {{color:{C_MUTED} !important; font-size:12px;}}

/* arrow chip */
.arrowchip {{display:inline-flex; align-items:center; justify-content:center;
  border-radius:7px; vertical-align:middle;}}

/* masthead refinements */
.brand-wrap {{display:flex; align-items:center; gap:14px;}}
.sunmark {{flex:none;}}
.next-pill {{display:inline-block; font-family:'IBM Plex Mono',monospace;
  font-size:12px; letter-spacing:1.5px; color:{C_MUTED};
  border:1px solid {C_BORDER}; border-radius:99px; padding:7px 16px;
  background:rgba(232,177,76,0.05);}}
.next-pill b {{color:{C_AMBER}; font-weight:600;}}

/* dateline (broadsheet rule) */
.dateline {{display:flex; align-items:center; gap:16px; justify-content:center;
  font-family:'IBM Plex Mono',monospace; font-size:10.5px; letter-spacing:3px;
  color:{C_MUTED}; margin:16px 0 4px;}}
.dateline:before, .dateline:after {{content:""; flex:1; height:1px;
  background:{C_BORDER};}}

/* breadth strip (replaces metric tiles) */
.breadth {{border:1px solid {C_BORDER}; border-radius:16px; padding:18px 22px;
  margin:18px 0 8px;
  background:linear-gradient(180deg,{C_SURFACE},{C_BG});}}
.breadth-head {{display:flex; justify-content:space-between; align-items:center;
  font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:2px;
  margin-bottom:12px;}}
.b-up {{color:{C_BULL}; display:flex; align-items:center; gap:9px;}}
.b-down {{color:{C_BEAR}; display:flex; align-items:center; gap:9px;}}
.b-mid {{color:{C_MUTED}; letter-spacing:1px; font-size:10.5px;}}
.breadth-bar {{display:flex; height:11px; border-radius:99px; overflow:hidden;
  background:rgba(255,255,255,0.05);}}
.seg {{height:100%;}}
.seg.bull {{background:linear-gradient(90deg,#3FB890,{C_BULL});}}
.seg.bear {{background:linear-gradient(90deg,{C_BEAR},{C_ROSE});}}
.breadth-meta {{font-family:'IBM Plex Mono',monospace; font-size:11px;
  color:{C_MUTED}; margin-top:13px; letter-spacing:0.4px;}}
.breadth-meta b {{color:{C_TEXT}; font-weight:600;}}
.breadth-meta .am {{color:{C_AMBER};}}

/* clickable rows */
.rowlink {{text-decoration:none; color:inherit; cursor:pointer;}}
.rowlink.sel {{background:rgba(232,177,76,0.07);
  box-shadow:inset 3px 0 0 {C_AMBER};}}

/* instrument detail panel */
.detail {{border:1px solid {C_BORDER}; border-radius:18px; padding:22px 26px;
  margin:14px 0 26px;
  background:
    radial-gradient(440px 200px at 100% 0%, rgba(232,177,76,0.08), transparent 70%),
    {C_SURFACE};}}
.detail-head {{display:flex; justify-content:space-between; align-items:flex-start;}}
.detail-tk {{font-family:'Fraunces',serif; font-size:44px; font-weight:600;
  line-height:1; letter-spacing:-0.5px;}}
.detail-nm {{color:{C_MUTED}; font-size:13px; margin-top:3px;}}
.detail-call {{display:flex; align-items:center; justify-content:flex-end; gap:10px;
  font-family:'IBM Plex Mono',monospace; font-size:13px; letter-spacing:1px;}}
.candle-wrap {{margin:18px 0 4px; border:1px solid {C_BORDER}; border-radius:12px;
  padding:10px 8px; background:rgba(0,0,0,0.18);}}
.candle-cap {{font-family:'IBM Plex Mono',monospace; font-size:10px;
  color:{C_MUTED}; letter-spacing:1px; margin-bottom:4px;}}
.stats {{display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
  background:{C_BORDER}; border:1px solid {C_BORDER}; border-radius:12px;
  overflow:hidden; margin:16px 0;}}
.stat {{background:{C_SURFACE}; padding:13px 15px;}}
.stat .k {{font-family:'IBM Plex Mono',monospace; font-size:9.5px;
  letter-spacing:2px; color:{C_MUTED};}}
.stat .v {{font-family:'IBM Plex Mono',monospace; font-size:18px; margin-top:5px;}}
.mini {{margin-top:6px;}}
.mini-row {{display:grid; grid-template-columns:1.4fr 1fr 1fr 1fr 1fr 1fr; gap:10px;
  padding:8px 6px; border-bottom:1px solid {C_BORDER};
  font-family:'IBM Plex Mono',monospace; font-size:12.5px;}}
.mini-row:last-child {{border-bottom:none;}}
.mini-row.h {{color:{C_MUTED}; font-size:9.5px; letter-spacing:1.5px;}}
.mini-row span:first-child {{color:{C_MUTED};}}
.mini-row .tr {{text-align:right;}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Load + assemble
# --------------------------------------------------------------------------- #
preds, prices, as_of, is_demo = load_data(DB_PATH)
df = build_frame(preds, prices)

n_up = int(df["up"].sum())
n_down = len(df) - n_up
avg_conf = df["confidence"].mean()
high_conf = int((df["confidence"] >= 0.65).sum())
top = df.iloc[0]

try:
    _d = datetime.strptime(str(as_of)[:10], "%Y-%m-%d")
    nice_date = _d.strftime("%d %b %Y")
    dateline_date = _d.strftime("%A · %d %B %Y").upper()
except ValueError:
    nice_date = str(as_of)
    dateline_date = str(as_of).upper()
next_sess = next_session_label(as_of)

# which ticker is being inspected (set by clicking a row -> ?ticker=XXX)
_qp = st.query_params.get("ticker")
if isinstance(_qp, (list, tuple)):
    _qp = _qp[0] if _qp else None
selected = (_qp or top["ticker"]).upper()
if selected not in set(df["ticker"]):
    selected = top["ticker"]
sel_row = df[df["ticker"] == selected].iloc[0]

# --------------------------------------------------------------------------- #
# Masthead
# --------------------------------------------------------------------------- #
demo = '<span class="demo-tag">DEMO DATA</span>' if is_demo else ""
sunmark = f"""<svg class="sunmark" width="44" height="44" viewBox="0 0 44 44" fill="none">
  <defs><linearGradient id="sun" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="{C_AMBER}"/>
    <stop offset="100%" stop-color="{C_ROSE}"/></linearGradient></defs>
  <path d="M13 28 a9 9 0 0 1 18 0 Z" fill="url(#sun)"/>
  <line x1="6" y1="28" x2="38" y2="28" stroke="{C_AMBER}" stroke-width="1.8"
    stroke-linecap="round"/>
  <line x1="22" y1="11" x2="22" y2="6" stroke="{C_AMBER}" stroke-width="1.8"
    stroke-linecap="round" opacity="0.9"/>
  <line x1="11" y1="15" x2="8" y2="12" stroke="{C_AMBER}" stroke-width="1.8"
    stroke-linecap="round" opacity="0.65"/>
  <line x1="33" y1="15" x2="36" y2="12" stroke="{C_AMBER}" stroke-width="1.8"
    stroke-linecap="round" opacity="0.65"/>
</svg>"""

st.markdown(f"""
<div class="mast">
  <div class="brand-wrap">
    {sunmark}
    <div>
      <div class="brand">NASDAQ-100 Next-Day Direction Prediction{demo}</div>
      <small>PRE-MARKET DECISION SUPPORT · GROUP 7</small>
    </div>
  </div>
  <div class="mast-right">
    <div class="next-pill">NEXT SESSION&nbsp; →&nbsp; <b>{next_sess}</b></div>
  </div>
</div>
<div class="dateline">{dateline_date} &nbsp;·&nbsp; PRE-MARKET BRIEFING
  &nbsp;·&nbsp; {len(df)} CONSTITUENTS</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disc" style="margin:10px 0 0">
  <b>AI-GENERATED · DECISION SUPPORT ONLY</b> &nbsp; Not investment advice.
  Disclosure provided in line with EU AI Act, Article 50.
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Market pulse
# --------------------------------------------------------------------------- #
breadth = n_up / max(len(df), 1) * 100
st.markdown(f"""
<div class="breadth">
  <div class="breadth-head">
    <span class="b-up">{arrow(True, 18)} {n_up} BULLISH</span>
    <span class="b-mid">{breadth:.0f}% of the index called higher next session</span>
    <span class="b-down">{n_down} BEARISH {arrow(False, 18)}</span>
  </div>
  <div class="breadth-bar">
    <div class="seg bull" style="width:{breadth:.1f}%"></div>
    <div class="seg bear" style="width:{100-breadth:.1f}%"></div>
  </div>
  <div class="breadth-meta">
    avg confidence <b class="am">{avg_conf*100:.0f}%</b> &nbsp;·&nbsp;
    <b>{high_conf}</b> high-conviction calls (≥65%) &nbsp;·&nbsp;
    strongest read <b>{top['ticker']}</b> at
    <b>{top['confidence']*100:.0f}%</b>
  </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Hero — single highest-conviction pick
# --------------------------------------------------------------------------- #
top = df.iloc[0]
top_color = C_BULL if top["up"] else C_BEAR
call_word = "HIGHER" if top["up"] else "LOWER"
last_px = f"${top['last']:,.2f}" if pd.notna(top["last"]) else "—"
wchg = top["window_chg"]
wchg_txt = f"{wchg*100:+.1f}%" if pd.notna(wchg) else "—"
wchg_col = C_BULL if (pd.notna(wchg) and wchg >= 0) else C_BEAR

st.markdown(f"""
<div class="hero">
  <div style="display:flex;align-items:center;justify-content:center">
    {confidence_arc(float(top['confidence']), top_color)}
  </div>
  <div>
    <div class="hero-eyebrow">▌ HIGHEST CONVICTION · NEXT SESSION</div>
    <div class="hero-ticker">{top['ticker']}</div>
    <div class="hero-name">{top['name']}</div>
    <div class="hero-call">{arrow(bool(top['up']))}
      <span style="color:{top_color}">Model expects close to finish {call_word}</span>
    </div>
    <div class="hero-meta">
      <div><div class="k">LAST CLOSE</div><div class="v">{last_px}</div></div>
      <div><div class="k">30-SESSION DRIFT</div>
        <div class="v" style="color:{wchg_col}">{wchg_txt}</div></div>
      <div><div class="k">TRAJECTORY</div>
        <div class="v">{sparkline(top['spark'], top_color, w=160, h=40)}</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Runner-up conviction picks (next 3)
# --------------------------------------------------------------------------- #
st.markdown('<div class="seclabel">NEXT IN LINE</div>', unsafe_allow_html=True)
cards = ""
for _, r in df.iloc[1:4].iterrows():
    col = C_BULL if r["up"] else C_BEAR
    word = "HIGHER" if r["up"] else "LOWER"
    cards += f"""<div class="runner">
      <div class="runner-top">
        <div><div class="runner-tk">{r['ticker']}</div>
          <div class="runner-nm">{r['name']}</div></div>
        {arrow(bool(r['up']))}
      </div>
      {sparkline(r['spark'], col, w=240, h=36)}
      <div style="display:flex;align-items:center;justify-content:space-between">
        <span class="callword" style="color:{col}">{word}</span>
        <span>{conf_bar(float(r['confidence']), col)}</span>
      </div>
    </div>"""
st.markdown(f'<div class="runners">{cards}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Instrument detail (driven by the clicked row)
# --------------------------------------------------------------------------- #
st.markdown('<div class="seclabel">INSTRUMENT DETAIL · CLICK ANY ROW BELOW</div>',
            unsafe_allow_html=True)
st.markdown(render_detail(prices, sel_row), unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Controls + full watchlist
# --------------------------------------------------------------------------- #
st.markdown('<div class="seclabel">THE WATCHLIST · SCROLL THE FULL UNIVERSE</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([2.2, 1.4, 1.4, 1.6])
with c1:
    query = st.text_input("Search", placeholder="Ticker or company…",
                          label_visibility="collapsed")
with c2:
    direction = st.selectbox("Direction", ["All", "Bullish ▲", "Bearish ▼"],
                             label_visibility="collapsed")
with c3:
    order = st.selectbox("Sort", ["Confidence", "Ticker", "Last price"],
                         label_visibility="collapsed")
with c4:
    min_conf = st.slider("Min confidence", 30, 95, 30, 5,
                         label_visibility="collapsed", format="≥ %d%%")

view = df.copy()
if query:
    q = query.lower()
    view = view[view["ticker"].str.lower().str.contains(q) |
                view["name"].str.lower().str.contains(q)]
if direction.startswith("Bullish"):
    view = view[view["up"]]
elif direction.startswith("Bearish"):
    view = view[~view["up"]]
view = view[view["confidence"] * 100 >= min_conf]

if order == "Ticker":
    view = view.sort_values("ticker")
elif order == "Last price":
    view = view.sort_values("last", ascending=False, na_position="last")
else:
    view = view.sort_values("confidence", ascending=False)

# header + rows
rows_html = (
    '<div class="head-row"><div class="row">'
    '<div>#</div><div>INSTRUMENT</div><div>CALL</div>'
    '<div>CONFIDENCE</div><div>30-SESSION</div><div>LAST</div>'
    '</div></div><div class="list-scroll">'
)
for i, (_, r) in enumerate(view.iterrows(), start=1):
    col = C_BULL if r["up"] else C_BEAR
    word = "UP" if r["up"] else "DOWN"
    px = f"${r['last']:,.2f}" if pd.notna(r["last"]) else "—"
    wc = r["window_chg"]
    wc_txt = f"{wc*100:+.1f}%" if pd.notna(wc) else ""
    wc_col = C_BULL if (pd.notna(wc) and wc >= 0) else C_BEAR
    sel_cls = " sel" if r["ticker"] == selected else ""
    rows_html += f"""<a class="row rowlink{sel_cls}" href="?ticker={r['ticker']}"
      target="_self">
      <div class="rk">{i:02d}</div>
      <div><div class="tk">{r['ticker']}</div><div class="nm">{r['name']}</div></div>
      <div>{arrow(bool(r['up']))} <span class="callword"
        style="color:{col}">{word}</span></div>
      <div>{conf_bar(float(r['confidence']), col)}</div>
      <div style="display:flex;align-items:center;gap:8px">
        {sparkline(r['spark'], col, w=90, h=26)}
        <span class="pxchg" style="color:{wc_col}">{wc_txt}</span></div>
      <div class="px">{px}</div>
    </a>"""
rows_html += "</div>"

if view.empty:
    rows_html = ('<div class="list-scroll"><div style="padding:40px;'
                 f'text-align:center;color:{C_MUTED}">No instruments match '
                 'these filters.</div></div>')

st.markdown(f'<div class="list">{rows_html}</div>', unsafe_allow_html=True)
st.markdown(f'<div style="color:{C_MUTED};font-size:11px;'
            f'font-family:IBM Plex Mono,monospace;margin-top:8px">'
            f'Showing {len(view)} of {len(df)} constituents</div>',
            unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Mandatory disclaimer (EU AI Act Article 50)
# --------------------------------------------------------------------------- #
st.markdown(f"""
<div class="disc">
  <b>AI-GENERATED · DECISION SUPPORT ONLY</b><br>
  These directional signals are produced by a machine-learning model and are
  intended to assist, not replace, your own analysis. They are not investment
  advice and carry no guarantee of accuracy. The analyst retains full
  decision-making authority. Disclosure provided in line with EU AI Act,
  Article 50.
</div>
""", unsafe_allow_html=True)