"""Build the SQLite store (data/nasdaq100.db) from the raw files in data/raw/.

Steps:
  1. cleaned_prices   - raw price CSVs -> standardize -> calendar / dedupe /
                        OHLC-consistency cleaning -> save_to_db, which also
                        computes the gap-aware next-day label
  2. features         - return_1d per ticker (strictly per-ticker pct_change)
  3. macro_indicators - vix_close / vix_change_1d via clean_vix_frame
  4. predictions      - scores the newest day with the trained model in
                        models/ so the dashboard has something to show
                        (skip with --no-predictions)

No network calls: run scripts/download_data.py first.

Usage:
    python scripts/build_database.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaning import (  # noqa: E402
    remove_duplicate_rows,
    remove_weekends_and_holidays,
    validate_price_consistency,
)
from src.data_utils import standardize  # noqa: E402
from src.datasets import load_macro, load_prices  # noqa: E402
from src.macro import clean_vix_frame  # noqa: E402
from src.storage import (  # noqa: E402
    DB_PATH,
    EXPECTED_TICKERS,
    load_from_db,
    save_features,
    save_macro_indicators,
    save_to_db,
)


def build_cleaned_prices() -> int:
    df = standardize(load_prices())
    df = remove_weekends_and_holidays(df)
    df = remove_duplicate_rows(df)
    df = validate_price_consistency(df)
    n_tickers = df["ticker"].nunique()
    if n_tickers != EXPECTED_TICKERS:
        print(f"  note: {n_tickers} tickers (expected {EXPECTED_TICKERS}); continuing")
    # `transactions` is not persisted in cleaned_prices.
    df = df.drop(columns=["transactions"])
    return save_to_db(df, check_ticker_count=False)


def build_features() -> int:
    prices = load_from_db().sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["return_1d"] = prices.groupby("ticker")["close"].pct_change()
    return save_features(prices[["ticker", "date", "return_1d"]])


def build_macro_indicators() -> int:
    vix = clean_vix_frame(load_macro(), date_col="date", vix_col="vix_close")
    return save_macro_indicators(vix)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build data/nasdaq100.db from data/raw/.")
    parser.add_argument("--no-predictions", action="store_true",
                        help="skip scoring the newest day with the trained model")
    args = parser.parse_args()

    print(f"Building {DB_PATH.relative_to(ROOT)} ...")
    print(f"[1/4] cleaned_prices   : {build_cleaned_prices():,} rows")
    print(f"[2/4] features         : {build_features():,} rows")
    print(f"[3/4] macro_indicators : {build_macro_indicators():,} rows")
    if args.no_predictions:
        print("[4/4] predictions      : skipped")
    else:
        from src.inference import write_predictions

        print(f"[4/4] predictions      : {write_predictions():,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
