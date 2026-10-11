"""Download the public input data into data/raw/.

Prices: daily OHLCV per ticker from Yahoo Finance via yfinance
(``auto_adjust=False``: split-adjusted closes, no dividend adjustment),
written to ``data/raw/prices/<TICKER>.csv``.

Macro: daily VIX close (FRED ``VIXCLS``) and 10-year Treasury yield
(FRED ``DGS10``), written to ``data/raw/macro.csv``. FRED's public CSV
endpoint needs no API key.

The defaults rebuild the research window used in the notebooks: the 100
tickers in ``config/universe_2015_2025.txt`` from 2015-02-17 to 2025-02-13.
Tickers that Yahoo no longer serves (e.g. delisted after an acquisition)
are reported and skipped.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --tickers-file config/tickers.txt --start 2020-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import sys
import time
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import MACRO_CSV, PRICES_DIR  # noqa: E402
from src.ingest import load_yfinance  # noqa: E402

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"
FRED_SERIES = {"VIXCLS": "vix_close", "DGS10": "treasury_10y"}
PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume", "transactions"]


def read_tickers(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def download_ticker(ticker: str, start: str, end_exclusive: str, retries: int = 3) -> pd.DataFrame:
    """Daily OHLCV for one ticker; empty frame if Yahoo has no data for it."""
    for attempt in range(1, retries + 1):
        try:
            return load_yfinance([ticker], start, end_exclusive)
        except ValueError:
            # load_yfinance concatenates zero frames when Yahoo returns nothing.
            return pd.DataFrame()
        except Exception as exc:  # network hiccup / rate limit: back off and retry
            if attempt == retries:
                print(f"  {ticker}: failed after {retries} attempts ({exc})")
                return pd.DataFrame()
            time.sleep(5 * attempt)
    return pd.DataFrame()


def download_prices(tickers: list[str], start: str, end: str, out_dir: Path, force: bool) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    missing = []
    for i, ticker in enumerate(tickers, start=1):
        path = out_dir / f"{ticker}.csv"
        if path.exists() and not force:
            continue
        df = download_ticker(ticker, start, end_exclusive)
        if df.empty:
            missing.append(ticker)
            print(f"  [{i:3d}/{len(tickers)}] {ticker:6s} no data")
            continue
        df[PRICE_COLUMNS].to_csv(path, index=False)
        print(f"  [{i:3d}/{len(tickers)}] {ticker:6s} {len(df):5d} rows")
    return missing


def download_macro(start: str, end: str, out_path: Path) -> pd.DataFrame:
    frames = []
    for series, column in FRED_SERIES.items():
        url = FRED_CSV_URL.format(series=series, start=start, end=end)
        with urlopen(url, timeout=60) as resp:
            raw = pd.read_csv(StringIO(resp.read().decode("utf-8")))
        raw.columns = ["date", column]
        raw["date"] = pd.to_datetime(raw["date"])
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
        frames.append(raw.set_index("date"))
    macro = pd.concat(frames, axis=1).sort_index().reset_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    macro.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    return macro


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers-file", type=Path, default=ROOT / "config" / "universe_2015_2025.txt")
    parser.add_argument("--start", default="2015-02-17", help="first trading day (inclusive)")
    parser.add_argument("--end", default="2025-02-13", help="last trading day (inclusive)")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-macro", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-download price files that already exist")
    args = parser.parse_args()

    if not args.skip_prices:
        tickers = read_tickers(args.tickers_file)
        print(f"Downloading daily prices for {len(tickers)} tickers ({args.start} -> {args.end}) ...")
        missing = download_prices(tickers, args.start, args.end, PRICES_DIR, args.force)
        n_files = len(list(PRICES_DIR.glob("*.csv")))
        print(f"Price files in {PRICES_DIR.relative_to(ROOT)}: {n_files}")
        if missing:
            print(f"No Yahoo data for {len(missing)} ticker(s): {', '.join(missing)}")

    if not args.skip_macro:
        # Start well before the first price date so the first day-over-day
        # VIX change has a real previous observation.
        macro_start = (pd.Timestamp(args.start) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        macro_end = (pd.Timestamp(args.end) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        print(f"Downloading FRED VIXCLS + DGS10 ({macro_start} -> {macro_end}) ...")
        macro = download_macro(macro_start, macro_end, MACRO_CSV)
        print(f"Wrote {len(macro)} rows to {MACRO_CSV.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
