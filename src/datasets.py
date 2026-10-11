"""Locations and loaders for the raw input files.

The raw inputs live under ``data/raw/`` (gitignored) and are produced by
``scripts/download_data.py``:

    data/raw/prices/<TICKER>.csv   daily OHLCV per ticker
                                   (date, open, high, low, close, volume, transactions)
    data/raw/macro.csv             daily macro series (date, vix_close, treasury_10y)

``scripts/build_database.py`` turns these into the SQLite store
(``data/nasdaq100.db``) that the notebooks, jobs and dashboard read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PRICES_DIR = RAW_DIR / "prices"
MACRO_CSV = RAW_DIR / "macro.csv"

MACRO_COLUMNS = ["date", "vix_close", "treasury_10y"]


def load_prices(prices_dir: str | Path = PRICES_DIR) -> pd.DataFrame:
    """Read every ``<TICKER>.csv`` in ``prices_dir`` into one long frame.

    Column names are lower-cased and a ``ticker`` column is added from the
    file name. Dates are returned as stored; run the frame through
    ``src.data_utils.standardize`` to get tz-aware America/New_York dates
    and the canonical dtypes.
    """
    prices_dir = Path(prices_dir)
    files = sorted(prices_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No price CSVs found in {prices_dir}. "
            "Run `python scripts/download_data.py` first (see data/README.md)."
        )
    frames = []
    for csv in files:
        df = pd.read_csv(csv)
        df.columns = [c.lower() for c in df.columns]
        if "transactions" not in df.columns:
            df["transactions"] = pd.NA
        df["ticker"] = csv.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_macro(path: str | Path = MACRO_CSV) -> pd.DataFrame:
    """Read the daily macro file: ``date`` (tz-naive), ``vix_close``,
    ``treasury_10y``. Non-trading days may be present with NaN values;
    callers clean each series separately."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Macro file not found at {path}. "
            "Run `python scripts/download_data.py` first (see data/README.md)."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(MACRO_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[MACRO_COLUMNS]
