"""Daily live VIX macro ingest for the NASDAQ-100 direction-prediction project.

Run:
    python -m src.jobs.daily_macro

Fetches the full VIXCLS series from FRED (every run, from HISTORY_START
to today), pipes it through clean_vix_frame (the shared cleaner that
also powers the batch build in scripts/build_database.py), and upserts the
result into macro_indicators via save_macro_indicators.

Why full history every run, not incremental:
  - clean_vix_frame's pct_change needs the real previous observation;
    a full-history fetch is the simplest way to guarantee that.
  - FRED revisions to historical values come along for free.
  - save_macro_indicators is delete-then-insert per (date), so
    re-writing the entire series is idempotent.
  - FRED is fast; the bandwidth cost is negligible.

Requires the FRED_API_KEY environment variable.

Out of scope (separate jobs):
  - OHLCV ingest -> src/jobs/daily_ingest.py
  - inference   -> separate scoring step
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from fredapi import Fred

from src.macro import clean_vix_frame
from src.storage import save_macro_indicators

FRED_SERIES_ID = "VIXCLS"
API_KEY_ENV = "FRED_API_KEY"

# Project-wide history start. Matches the start of the CSV-batch macro
# series (first cleaned row in macro_indicators is 2015-02-13, the first
# row after clean_vix_frame's leading-NaN drop) and aligns with the
# cleaned_prices window. Infrastructure constant -- not a runtime knob.
HISTORY_START = "2015-01-01"


def _get_fred_client() -> Fred:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} environment variable not set -- "
            f"set it before running this job."
        )
    return Fred(api_key=key)


def fetch_vix_series(client: Fred, start: str, end: str) -> pd.Series:
    """Fetch the raw VIXCLS series from FRED. Returns a pd.Series indexed
    by a tz-naive DatetimeIndex; values are float with NaN on non-trading
    days. No cleaning here -- clean_vix_frame is the single source of
    truth for that."""
    return client.get_series(
        FRED_SERIES_ID,
        observation_start=start,
        observation_end=end,
    )


def series_to_raw_frame(series: pd.Series) -> pd.DataFrame:
    """Reshape the FRED Series into the DataFrame layout that
    clean_vix_frame expects: a date column and a VIX column, with the
    date column tz-naive."""
    return pd.DataFrame({
        "date": series.index,
        FRED_SERIES_ID: series.values,
    })


def main() -> int:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    print(
        f"[fetch] FRED series={FRED_SERIES_ID}  "
        f"start={HISTORY_START}  end={today}"
    )

    client = _get_fred_client()
    series = fetch_vix_series(client, HISTORY_START, today)

    if series.empty:
        print("[fred] 0 observations returned (empty response). Exiting cleanly.")
        return 0
    print(
        f"[fred] {len(series)} observations fetched "
        f"(raw range: {series.index.min().date()} -> {series.index.max().date()})"
    )

    raw = series_to_raw_frame(series)

    # Cheap insurance against a future fredapi version that would return
    # a tz-aware index. clean_vix_frame does tz_localize("America/
    # New_York") on the date column and would crash on an already-aware
    # series. Failing here with a clear message beats producing wrong
    # data silently.
    if raw["date"].dt.tz is not None:
        raise ValueError(
            f"FRED date column unexpectedly tz-aware ({raw['date'].dt.tz}); "
            f"clean_vix_frame requires tz-naive input."
        )

    vix = clean_vix_frame(raw, date_col="date", vix_col=FRED_SERIES_ID)
    if vix.empty:
        print("[clean] 0 rows after cleaning. Exiting cleanly.")
        return 0
    print(
        f"[clean] {len(vix)} rows after clean_vix_frame "
        f"(date range: {vix['date'].min().date()} -> {vix['date'].max().date()})"
    )

    stored = save_macro_indicators(vix)
    print(
        f"[save] {len(vix)} rows upserted to macro_indicators; "
        f"table now contains {stored} rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
