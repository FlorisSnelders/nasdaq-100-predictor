"""Daily live OHLCV ingest for the NASDAQ-100 direction-prediction project.

Run:
    python -m src.jobs.daily_ingest

Pulls the last ~10 calendar days of OHLCV via yfinance, applies the
same cleaning chain that originally produced cleaned_prices for
training (no macro, no build_feature_matrix), upserts the result into
cleaned_prices, derives return_1d from add_daily_returns, and upserts
that into features. Two tables touched: cleaned_prices and features.

Out of scope (separate jobs):
  - macro/VIX -> dedicated FRED-driven job
  - inference -> dedicated scoring step

Why a 10-day window when only today's row is "new":
  1. add_daily_returns needs at least one prior trading day per ticker,
     otherwise today's pct_change is NaN and the row gets dropped in
     load_model_frame.
  2. save_incremental re-derives labels via storage._compute_label over
     the full batch against the NASDAQ trading calendar. Including the
     preceding handful of trading days backfills the label for D-1
     (and any earlier row still NULL from a prior day's run) correctly,
     including across weekends/holidays where the next trading day is
     several calendar days away.

Label of the newest row per ticker stays NULL by design (see comment at
save_incremental call below). That row is exactly what the model is
about to score; its forward-looking label is not determined yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.cleaning import (
    remove_duplicate_rows,
    remove_weekends_and_holidays,
    validate_price_consistency,
)
from src.data_cleaner import (
    add_daily_returns,
    find_missing_trading_days,
    flag_pregap_rows,
)
from src.data_utils import check_schema, standardize
from src.ingest import load_yfinance
from src.storage import save_features, save_incremental

TICKERS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "tickers.txt"
)
NY = "America/New_York"

# Calendar-day lookback for the yfinance pull. Wide enough to cover both
# the pct_change anchor and weekend/holiday-spanning label backfill (a
# Friday row whose label needs Monday's close survives a 3-day gap).
WINDOW_DAYS = 10

# Columns that must never be NULL after the cleaning chain. `label` is
# excluded deliberately (newest row per ticker is supposed to be NULL).
# `transactions` is excluded because yfinance never provides it.
# `label_valid` is excluded because it's a transient flag from
# flag_pregap_rows and gets dropped before save.
NON_NULLABLE_COLS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def read_tickers(path: Path = TICKERS_PATH) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def compute_window(
    today_ny: pd.Timestamp, lookback_days: int = WINDOW_DAYS
) -> tuple[str, str]:
    """Return (start, end) date strings for yfinance. End is exclusive."""
    start = (today_ny - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (today_ny + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return start, end


def assert_no_nulls(df: pd.DataFrame, cols: list[str]) -> None:
    """Fail loudly if any listed column contains NULL after cleaning.
    The cleaning chain does not enforce this explicitly; a silent NULL
    here would propagate straight into cleaned_prices."""
    offenders = {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()}
    if offenders:
        raise ValueError(
            f"NULLs found in non-nullable columns after cleaning: {offenders}"
        )


def assert_no_future_dates(df: pd.DataFrame) -> None:
    """Fail loudly if any row is dated after the current NY clock. Catches
    yfinance/tz misalignment that would otherwise let a fabricated future
    row sneak into the DB."""
    now_ny = pd.Timestamp.now(tz=NY)
    future_mask = df["date"] > now_ny
    if future_mask.any():
        n = int(future_mask.sum())
        examples = (
            df.loc[future_mask, ["ticker", "date"]]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{n} rows have dates after {now_ny}. Examples: {examples}"
        )


def main() -> int:
    tickers = read_tickers()
    today_ny = pd.Timestamp.now(tz=NY).normalize()
    start, end = compute_window(today_ny)
    print(
        f"[window] tickers={len(tickers)}  start={start}  end={end}  "
        f"(today_NY={today_ny.date()})"
    )

    raw = load_yfinance(tickers, start, end)
    if raw is None or raw.empty:
        print(
            "[load] yfinance returned 0 rows "
            "(non-trading window or upstream issue). Exiting cleanly."
        )
        return 0
    print(f"[load] {len(raw)} rows across {raw['ticker'].nunique()} tickers")

    df = standardize(raw)
    check_schema(df)

    # Cleaning chain -- same sequence as the original run_pipeline, minus
    # the (now-removed) macro merge. Order matters: dedupe and OHLC
    # consistency assume the calendar filter has already run.
    df = remove_weekends_and_holidays(df)
    df = remove_duplicate_rows(df)
    df = validate_price_consistency(df)
    print(f"[clean] {len(df)} rows survived calendar + dedupe + OHLC filters")

    if df.empty:
        print("[clean] 0 rows survived cleaning. Exiting cleanly.")
        return 0

    missing = find_missing_trading_days(df)
    df = flag_pregap_rows(df, missing)
    df = add_daily_returns(df)
    n_non_null = int(df["daily_return"].notna().sum())
    print(
        f"[returns] {len(df)} rows total, "
        f"{n_non_null} with non-null daily_return"
    )

    # Invariants the cleaning chain does NOT enforce. validate_stage from
    # src.validation is intentionally NOT used here: STAGE_PROFILES is
    # dimensioned for the training batch (100k-200k rows) and would fail
    # on a daily slice. Project decision is to not touch shared training
    # infrastructure for the live path; we duplicate just these two
    # invariants locally (no_nulls on the OHLCV+key columns, no future
    # dates). OHLC consistency and duplicates are already enforced by
    # validate_price_consistency and remove_duplicate_rows above.
    assert_no_nulls(df, NON_NULLABLE_COLS)
    assert_no_future_dates(df)

    # cleaned_prices: save_incremental does the scoped (ticker, date)
    # upsert AND re-derives the label per row via storage._compute_label
    # over the full batch. The newest row per ticker emerges with
    # label=NULL: _compute_label only assigns a label when the row's
    # next NASDAQ trading day is present in the batch. For the most
    # recent row there is no next-day close, so is_consecutive is False
    # and the initialised pd.NA stays put. Exactly what we want -- that
    # row is what the model will score next, its label is not determined
    # until tomorrow.
    #
    # flag_pregap_rows added `label_valid`, which doesn't belong in
    # cleaned_prices (storage.LEGACY_COLUMNS lists it). Drop it
    # explicitly so save_incremental doesn't warn on every run.
    df_for_save = df.drop(columns=["label_valid"], errors="ignore")
    n_prices = save_incremental(df_for_save)
    print(f"[save_incremental] {n_prices} rows upserted to cleaned_prices")

    # features: add_daily_returns produced `daily_return`, which is
    # bit-identical to fill_features.py's `return_1d` (same sort, same
    # groupby+pct_change). Rename and persist.
    #
    # Drop NaN return rows so we never overwrite an existing non-NULL
    # return_1d in features with NULL. save_features is delete-then-
    # insert on the (ticker, date) keys we send, so writing a NULL for
    # the oldest row in the window would clobber a real value that came
    # from a prior run's larger window. Leaving NaN rows out keeps the
    # operation strictly additive for return_1d.
    feats = df.rename(columns={"daily_return": "return_1d"})
    feats = feats.dropna(subset=["return_1d"])
    feats = feats[["ticker", "date", "return_1d"]]
    if feats.empty:
        print("[save_features] no non-null return_1d rows to write")
    else:
        n_feats_total = save_features(feats)
        print(
            f"[save_features] {len(feats)} rows upserted; "
            f"features table now contains {n_feats_total} rows"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
