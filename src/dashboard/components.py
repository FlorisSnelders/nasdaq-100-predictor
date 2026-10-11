import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src.storage import load_predictions, load_from_db

BAKEOFF_DIR = ROOT / "results" / "metrics"
MODEL_CARD = ROOT / "docs" / "model_card.md"
LOW_CONFIDENCE = 55.0

DISCLAIMER = (
    "AI generated decision support, not investment advice. These next day "
    "direction signals come from a machine learning model and carry no "
    "guarantee of accuracy. They exist to assist your own analysis, and you "
    "keep full decision making authority. Shown under EU AI Act Article 50."
)

NASDAQ100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "COST",
    "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", "INTC", "CMCSA", "QCOM", "INTU",
    "TXN", "AMGN", "ISRG", "AMAT", "HON", "BKNG", "VRTX", "ADP", "MU", "ADI",
    "REGN", "GILD", "LRCX", "PANW", "MELI", "SBUX", "KLAC", "SNPS", "CDNS", "MDLZ",
    "PYPL", "MAR", "CRWD", "CTAS", "ORLY", "ASML", "ABNB", "NXPI", "PCAR", "CEG",
    "ROP", "MNST", "FTNT", "DASH", "WDAY", "ADSK", "AEP", "PAYX", "CHTR", "CPRT",
    "ROST", "KDP", "ODFL", "FAST", "MRVL", "GEHC", "DDOG", "VRSK", "EA", "CTSH",
    "EXC", "XEL", "KHC", "TTD", "CCEP", "FANG", "BKR", "IDXX", "CSGP", "ON",
    "MCHP", "DXCM", "CDW", "TEAM", "GFS", "BIIB", "ZS", "ANSS", "WBD", "ILMN",
    "MDB", "DLTR", "WBA", "LULU", "ARM", "SMCI", "PLTR", "APP", "MSTR", "AXON",
]


def render_disclaimer():
    st.info(DISCLAIMER)


def latest_predictions():
    preds = load_predictions()
    if preds.empty:
        return preds
    preds = preds.sort_values("date").groupby("ticker", as_index=False).tail(1)
    prices = load_from_db()
    last_close = (
        prices.sort_values("date")
        .groupby("ticker", as_index=False)
        .tail(1)[["ticker", "close"]]
    )
    df = preds.merge(last_close, on="ticker", how="left")
    df["direction"] = df["prediction"].map({1: "Up", 0: "Down"})
    df["confidence"] = (
        df["probability"].where(df["prediction"] == 1, 1 - df["probability"]) * 100
    ).round(1)
    df["updated"] = df["date"].dt.strftime("%Y-%m-%d")
    df["close"] = df["close"].round(2)
    return df[["ticker", "close", "direction", "confidence", "updated"]]


def _demo_frame():
    rng = np.random.default_rng(7)
    probs = np.clip(rng.normal(0.55, 0.12, len(NASDAQ100)), 0.35, 0.95)
    pred = (probs >= 0.5).astype(int)
    conf = np.where(pred == 1, probs, 1 - probs) * 100
    return pd.DataFrame(
        {
            "ticker": NASDAQ100,
            "close": rng.uniform(40, 600, len(NASDAQ100)).round(2),
            "direction": np.where(pred == 1, "Up", "Down"),
            "confidence": conf.round(1),
            "updated": date.today().isoformat(),
        }
    )


def predictions_view():
    df = latest_predictions()
    if df.empty:
        return _demo_frame(), True
    return df, False


def style_predictions(df):
    display = df.rename(
        columns={
            "ticker": "Ticker",
            "close": "Last close",
            "direction": "Direction",
            "confidence": "Confidence",
            "updated": "Updated",
        }
    )

    def color_rows(row):
        if row["Confidence"] < LOW_CONFIDENCE:
            return ["color: #8a8a8a"] * len(row)
        styles = [""] * len(row)
        c = "#1a7f37" if row["Direction"] == "Up" else "#b3261e"
        styles[list(row.index).index("Direction")] = f"color: {c}; font-weight: 600"
        return styles

    return (
        display.style.apply(color_rows, axis=1)
        .format({"Last close": "${:,.2f}", "Confidence": "{:.1f}%"})
    )


def bakeoff_models():
    if not BAKEOFF_DIR.exists():
        return []
    return sorted(p.stem for p in BAKEOFF_DIR.glob("*.json"))


def load_bakeoff(name):
    return json.loads((BAKEOFF_DIR / f"{name}.json").read_text())


def backtesting_tables(payload):
    per_fold = pd.DataFrame(payload["per_fold"])
    keep = [c for c in ["fold", "test_year", "accuracy", "precision", "f1"] if c in per_fold.columns]
    per_fold = per_fold[keep].rename(
        columns={
            "fold": "Fold",
            "test_year": "Test year",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "f1": "F1",
        }
    )
    s = payload["summary"]
    summary = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "F1"],
            "Mean": [s["mean_accuracy"], s["mean_precision"], s["mean_f1"]],
            "Std": [s["std_accuracy"], s["std_precision"], s["std_f1"]],
        }
    )
    return per_fold.round(4), summary.round(4)


def model_card_text():
    if not MODEL_CARD.exists():
        return None
    return MODEL_CARD.read_text(encoding="utf-8")