import sqlite3
import warnings
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nasdaq100.db"

EXPECTED_TICKERS = 100

CLEANED_PRICES_COLUMNS = [
    "ticker", "date", "open", "high", "low", "close", "volume", "label",
]

MACRO_INDICATORS_COLUMNS = ["date", "vix_close", "vix_change_1d"]

FEATURES_COLUMNS = ["ticker", "date", "return_1d"]

PREDICTIONS_COLUMNS = ["ticker", "date", "prediction", "probability"]

# The exact feature set + order consumed by the production model bundle.
# Must stay sorted alphabetically — the joblib bundle's prod_features was
# derived via sorted(), so any other order silently misaligns the input
# matrix at serve time.
MODEL_FEATURE_CONTRACT = ["return_1d", "vix_change_1d", "vix_close"]

LEGACY_COLUMNS = ("label_valid", "daily_return", "transactions")


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_tables() -> None:
    """Create all four tables (cleaned_prices, macro_indicators, features,
    predictions) if they do not already exist. Idempotent."""
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cleaned_prices (
                ticker  TEXT    NOT NULL,
                date    TEXT    NOT NULL,
                open    REAL    NOT NULL,
                high    REAL    NOT NULL,
                low     REAL    NOT NULL,
                close   REAL    NOT NULL,
                volume  INTEGER,
                label   INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cleaned_prices_date "
            "ON cleaned_prices(date)"
        )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_indicators (
                date          TEXT NOT NULL PRIMARY KEY,
                vix_close     REAL,
                vix_change_1d REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                ticker    TEXT NOT NULL,
                date      TEXT NOT NULL,
                return_1d REAL,
                PRIMARY KEY (ticker, date),
                FOREIGN KEY (ticker, date) REFERENCES cleaned_prices(ticker, date)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_date "
            "ON features(date)"
        )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                ticker      TEXT NOT NULL,
                date        TEXT NOT NULL,
                prediction  INTEGER,
                probability REAL,
                PRIMARY KEY (ticker, date),
                FOREIGN KEY (ticker, date) REFERENCES cleaned_prices(ticker, date)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_date "
            "ON predictions(date)"
        )

        conn.commit()
    finally:
        conn.close()


def _compute_label(df: pd.DataFrame) -> pd.Series:
    """Compute the next-day-direction label per (ticker, date).

    1 if next-day close > today close, 0 if next-day close <= today close,
    NA if the next row in `df` is NOT the immediately following NASDAQ
    trading day (gap in data, or last row per ticker). The gap check uses
    the official NASDAQ calendar so weekends/holidays don't count as gaps,
    but real missing trading days do invalidate the label.
    """
    import pandas_market_calendars as mcal

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    dates_raw = pd.to_datetime(df["date"])
    if getattr(dates_raw.dt, "tz", None) is not None:
        dates_n = dates_raw.dt.tz_localize(None).dt.normalize()
    else:
        dates_n = dates_raw.dt.normalize()

    sched = mcal.get_calendar("NASDAQ").schedule(
        start_date=dates_n.min(),
        end_date=dates_n.max() + pd.Timedelta(days=14),
    )
    trading_days = pd.DatetimeIndex(sched.index)
    if trading_days.tz is not None:
        trading_days = trading_days.tz_localize(None)
    trading_days = trading_days.normalize()

    # The trading day immediately after each row's date.
    pos = trading_days.searchsorted(dates_n.values, side="right")
    # Clip so out-of-range positions map to the last calendar entry; we mask
    # those rows out below so the bogus lookup is never used.
    clipped = pos.clip(max=len(trading_days) - 1)
    expected_next_arr = trading_days.values[clipped]
    expected_next = pd.Series(expected_next_arr, index=df.index)
    expected_next[pos >= len(trading_days)] = pd.NaT

    # The next row's actual date within the same ticker.
    next_close = df.groupby("ticker")["close"].shift(-1)
    next_date_raw = df.groupby("ticker")["date"].shift(-1)
    next_date = pd.to_datetime(next_date_raw)
    if getattr(next_date.dt, "tz", None) is not None:
        next_date = next_date.dt.tz_localize(None)
    next_date = next_date.dt.normalize()

    is_consecutive = (next_date == expected_next) & next_close.notna()

    label = pd.Series(pd.NA, index=df.index, dtype="Int64")
    direction = (next_close > df["close"]).astype("Int64")
    label[is_consecutive] = direction[is_consecutive]
    return label


def _warn_if_legacy(df: pd.DataFrame) -> None:
    legacy = [c for c in LEGACY_COLUMNS if c in df.columns]
    if legacy:
        warnings.warn(
            f"Columns {legacy} are no longer persisted in cleaned_prices. "
            "Drop them from your pipeline output or recompute on read. "
            "This warning will become an error in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )


def save_to_db(df: pd.DataFrame, check_ticker_count: bool = True) -> int:
    """Persist cleaned OHLCV + computed label to SQLite.

    Expects at minimum: ticker, date, open, high, low, close, volume.
    Computes `label` (1 if next-day close > today close per ticker, 0 otherwise,
    NULL for the last row per ticker). Replaces all rows in cleaned_prices on
    each run (delete-then-insert) so reruns are idempotent.
    Set `check_ticker_count=False` to skip the 100-ticker guard (e.g. when a
    rebuilt universe is missing a delisted ticker).
    Returns the number of rows written.
    """
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    n_tickers = df["ticker"].nunique()
    if check_ticker_count and n_tickers != EXPECTED_TICKERS:
        raise ValueError(
            f"Expected {EXPECTED_TICKERS} tickers, got {n_tickers}"
        )

    _warn_if_legacy(df)

    rows = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    rows["label"] = _compute_label(rows)
    rows = rows[CLEANED_PRICES_COLUMNS].copy()
    rows["date"] = rows["date"].astype(str)

    conn = _get_connection()
    try:
        create_tables()
        conn.execute("DELETE FROM cleaned_prices")
        rows.to_sql(
            "cleaned_prices",
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )

        stored = conn.execute("SELECT COUNT(*) FROM cleaned_prices").fetchone()[0]
        if stored != len(rows):
            raise RuntimeError(
                f"Row count mismatch: expected {len(rows)}, stored {stored}"
            )
        conn.commit()
    finally:
        conn.close()
    return stored

def save_incremental(df: pd.DataFrame) -> int:
    """Upsert cleaned OHLCV + computed label into cleaned_prices without
    wiping history. Replaces only the (ticker, date) rows present in df.
    Use this for the daily ingest job; use save_to_db for a full reload.
    Returns the number of rows upserted.
    """
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    rows = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    rows["label"] = _compute_label(rows)
    rows = rows[CLEANED_PRICES_COLUMNS].copy()
    rows["date"] = rows["date"].astype(str)

    cols = list(rows.columns)
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("ticker", "date"))
    sql = (
        f"INSERT INTO cleaned_prices ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(ticker, date) DO UPDATE SET {updates}"
    )
    records = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in rows.itertuples(index=False, name=None)
    ]

    conn = _get_connection()
    try:
        create_tables()
        conn.executemany(sql, records)
        conn.commit()
    finally:
        conn.close()
    return len(rows)

def load_from_db(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load cleaned OHLCV + label from SQLite with optional filters.

    Args:
        tickers:    List of ticker symbols, or None for all.
        start_date: Inclusive start date (YYYY-MM-DD), or None.
        end_date:   Inclusive end date (YYYY-MM-DD), or None.

    Returns:
        DataFrame with columns: ticker, date, open, high, low, close, volume, label.
        `date` is parsed back to tz-aware America/New_York.

    Note: `label_valid`, `daily_return`, and `transactions` are no longer
    persisted. Recompute them downstream if needed.
    """
    conn = _get_connection()
    try:
        create_tables()
        df = _read_filtered(
            conn,
            table="cleaned_prices",
            columns=CLEANED_PRICES_COLUMNS,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        conn.close()

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York")
    return df


def save_features(df: pd.DataFrame) -> int:
    """Persist computed technical features to the `features` table.

    Required columns: ticker, date. Any of the FEATURES_COLUMNS that are
    present will be written; missing ones are stored as NULL.
    Idempotent: replaces all (ticker, date) rows that appear in df.
    """
    if not {"ticker", "date"}.issubset(df.columns):
        raise ValueError("features DataFrame must include 'ticker' and 'date'")

    present = [c for c in FEATURES_COLUMNS if c in df.columns]
    rows = df[present].copy()
    rows["date"] = rows["date"].astype(str)

    conn = _get_connection()
    try:
        create_tables()
        _delete_then_insert(conn, "features", rows, key_cols=("ticker", "date"))
        conn.commit()
        stored = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    finally:
        conn.close()
    return stored


def load_features(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load technical features from SQLite with optional filters."""
    conn = _get_connection()
    try:
        create_tables()
        df = _read_filtered(
            conn,
            table="features",
            columns=FEATURES_COLUMNS,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        conn.close()

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York")
    return df


def save_macro_indicators(df: pd.DataFrame) -> int:
    """Persist macroeconomic indicators (VIX, 10y treasury, …) to the
    `macro_indicators` table. One row per date.

    Required column: date. Any of MACRO_INDICATORS_COLUMNS that are present
    will be written; missing ones are stored as NULL.
    Idempotent: replaces all dates that appear in df.
    """
    if "date" not in df.columns:
        raise ValueError("macro_indicators DataFrame must include 'date'")

    present = [c for c in MACRO_INDICATORS_COLUMNS if c in df.columns]
    rows = df[present].copy()
    rows["date"] = rows["date"].astype(str)

    conn = _get_connection()
    try:
        create_tables()
        _delete_then_insert(conn, "macro_indicators", rows, key_cols=("date",))
        conn.commit()
        stored = conn.execute("SELECT COUNT(*) FROM macro_indicators").fetchone()[0]
    finally:
        conn.close()
    return stored


def load_macro_indicators(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load macroeconomic indicators from SQLite with optional date filter."""
    conn = _get_connection()
    try:
        create_tables()
        df = _read_filtered(
            conn,
            table="macro_indicators",
            columns=MACRO_INDICATORS_COLUMNS,
            tickers=None,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        conn.close()

    df["date"] = pd.to_datetime(df["date"])
    return df


def save_predictions(df: pd.DataFrame) -> int:
    """Persist model predictions to the `predictions` table.

    Required columns: ticker, date, prediction, probability.
    Idempotent: replaces all (ticker, date) rows that appear in df.
    """
    required = set(PREDICTIONS_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions DataFrame is missing columns: {missing}")

    rows = df[PREDICTIONS_COLUMNS].copy()
    rows["date"] = rows["date"].astype(str)

    conn = _get_connection()
    try:
        create_tables()
        _delete_then_insert(conn, "predictions", rows, key_cols=("ticker", "date"))
        conn.commit()
        stored = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    finally:
        conn.close()
    return stored


def load_predictions(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load model predictions from SQLite with optional filters."""
    conn = _get_connection()
    try:
        create_tables()
        df = _read_filtered(
            conn,
            table="predictions",
            columns=PREDICTIONS_COLUMNS,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        conn.close()

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York")
    return df


def load_model_frame(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Build the production model input frame from the database.

    Joins `features` (return_1d, per ticker/date) with `macro_indicators`
    (vix_close, vix_change_1d, per date) and `cleaned_prices` (label) on a
    normalized tz-naive date key — joining on the raw timestamp string would
    break because cleaned_prices/features store tz-aware America/New_York
    timestamps while macro_indicators stores them differently per writer.

    Returns columns in this exact order:
        ticker, date, label, *MODEL_FEATURE_CONTRACT
    so the caller can slice df[MODEL_FEATURE_CONTRACT] straight into the
    joblib bundle without re-sorting.

    Rows with any NULL in MODEL_FEATURE_CONTRACT are dropped (the model
    cannot score without all three features). Rows with a NULL label are
    kept — by definition the most recent row per ticker has no label yet
    (its next-day close has not happened), and that row is exactly the
    one we want to score.
    """
    prices = load_from_db(
        tickers=tickers, start_date=start_date, end_date=end_date
    )
    feats = load_features(
        tickers=tickers, start_date=start_date, end_date=end_date
    )
    # Read macro_indicators raw: load_macro_indicators() parses dates via
    # pd.to_datetime without utc=True, which raises on mixed EST/EDT
    # offsets (modern pandas) once the table contains tz-localized
    # America/New_York strings. We want the raw strings here so we can
    # parse them ourselves with utc=True and feed a unified date-key
    # through the join.
    conn = _get_connection()
    try:
        create_tables()
        macro = _read_filtered(
            conn,
            table="macro_indicators",
            columns=MACRO_INDICATORS_COLUMNS,
            tickers=None,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        conn.close()

    def _date_key(s: pd.Series) -> pd.Series:
        # utc=True handles mixed offsets in raw strings AND already
        # tz-aware Series. Convert back to NY then strip tz so the key
        # represents the local NASDAQ calendar day, not a UTC instant.
        ts = pd.to_datetime(s, utc=True)
        return ts.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()

    prices = prices.assign(_dkey=_date_key(prices["date"]))
    feats = feats.assign(_dkey=_date_key(feats["date"]))
    macro = macro.assign(_dkey=_date_key(macro["date"]))

    df = (
        prices[["ticker", "date", "_dkey", "label"]]
        .merge(feats[["ticker", "_dkey", "return_1d"]], on=["ticker", "_dkey"], how="left")
        .merge(macro[["_dkey", "vix_close", "vix_change_1d"]], on="_dkey", how="left")
    )

    df = df.dropna(subset=MODEL_FEATURE_CONTRACT)
    df = df[["ticker", "date", "label", *MODEL_FEATURE_CONTRACT]]
    return df.reset_index(drop=True)


def _read_filtered(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
    tickers: list[str] | None,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    query = f"SELECT {', '.join(columns)} FROM {table}"
    conditions = []
    params: list[str] = []

    if tickers and "ticker" in columns:
        placeholders = ", ".join("?" for _ in tickers)
        conditions.append(f"ticker IN ({placeholders})")
        params.extend(tickers)

    # Date filters compare on the YYYY-MM-DD prefix, not on the raw
    # stored string. The raw values are tz-aware timestamps like
    # "2026-06-03 00:00:00-04:00", which lexicographically sort AFTER a
    # date-only filter argument like "2026-06-03" -- so a plain
    # `date <= '2026-06-03'` silently excludes that day's rows.
    # substr(date, 1, 10) reduces both sides to the calendar day, robust
    # across EDT/EST offset boundaries and any wallclock time the writer
    # happened to use. Matches the calendar-day semantics that
    # load_model_frame already applies via _date_key.
    # The [:10] slicing on the Python side is a defensive safety net: if a
    # caller ever passes a longer string like "2026-06-03 12:00:00", we
    # still compare on the YYYY-MM-DD prefix instead of producing a
    # subtly wrong window.
    if start_date:
        conditions.append("substr(date, 1, 10) >= ?")
        params.append(str(start_date)[:10])

    if end_date:
        conditions.append("substr(date, 1, 10) <= ?")
        params.append(str(end_date)[:10])

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    order = "ticker, date" if "ticker" in columns else "date"
    query += f" ORDER BY {order}"

    return pd.read_sql(query, conn, params=params)


def _delete_then_insert(
    conn: sqlite3.Connection,
    table: str,
    rows: pd.DataFrame,
    key_cols: tuple[str, ...],
) -> None:
    """Replace any existing rows in `table` that share a key with `rows`,
    then bulk-insert `rows`. Keeps writes idempotent without nuking unrelated
    keys (unlike save_to_db, which intentionally clears the full table)."""
    if rows.empty:
        return

    if len(key_cols) == 1:
        key = key_cols[0]
        values = rows[key].unique().tolist()
        # SQLite has a default 999 parameter limit; chunk to stay safe.
        for i in range(0, len(values), 500):
            chunk = values[i : i + 500]
            placeholders = ", ".join("?" for _ in chunk)
            conn.execute(
                f"DELETE FROM {table} WHERE {key} IN ({placeholders})",
                chunk,
            )
    else:
        # Composite key delete: tuple-by-tuple.
        keys = list(rows[list(key_cols)].itertuples(index=False, name=None))
        where = " AND ".join(f"{k} = ?" for k in key_cols)
        conn.executemany(f"DELETE FROM {table} WHERE {where}", keys)

    rows.to_sql(
        table,
        conn,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
