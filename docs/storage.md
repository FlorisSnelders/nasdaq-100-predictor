# Storage layer: schema, indexing and benchmark

All persisted state for the NASDAQ-100 pipeline lives in a single SQLite
database at `data/nasdaq100.db`. The public API lives in
[`src/storage.py`](../src/storage.py) and groups four tables:

| Table              | Purpose                                                | Reader / Writer                          |
| ------------------ | ------------------------------------------------------ | ---------------------------------------- |
| `cleaned_prices`   | Post-pipeline OHLCV + next-day direction label         | `save_to_db` / `load_from_db`            |
| `macro_indicators` | VIX close + 1d pct change, one row per trading date    | `save_macro_indicators` / `load_macro_indicators` |
| `features`         | Per-ticker per-date model inputs (`return_1d` only)    | `save_features` / `load_features`        |
| `predictions`      | Model output (direction + probability)                 | `save_predictions` / `load_predictions`  |

`create_tables()` creates all four (with `IF NOT EXISTS`) and is called
automatically by every save/load function. `PRAGMA foreign_keys = ON`
is enabled on every connection.

## Schema

### `cleaned_prices` (source-of-truth OHLCV + label)

```sql
CREATE TABLE cleaned_prices (
    ticker  TEXT    NOT NULL,
    date    TEXT    NOT NULL,
    open    REAL    NOT NULL,
    high    REAL    NOT NULL,
    low     REAL    NOT NULL,
    close   REAL    NOT NULL,
    volume  INTEGER,
    label   INTEGER,   -- 1 if next-day close > today close, 0 otherwise,
                       -- NULL for the last row per ticker
    PRIMARY KEY (ticker, date)
);
CREATE INDEX idx_cleaned_prices_date ON cleaned_prices(date);
```

`label` is computed inside `save_to_db` from `close` (per ticker,
sorted by date), so callers do not need to materialise it themselves.

`date` is stored as TEXT because SQLite has no native datetime type and
the pipeline produces tz-aware timestamps in `America/New_York` that need
to survive the round-trip across EST/EDT. `load_from_db` parses the
stored strings back to tz-aware datetimes via
`pd.to_datetime(..., utc=True).dt.tz_convert("America/New_York")`.

### `macro_indicators`

```sql
CREATE TABLE macro_indicators (
    date          TEXT NOT NULL PRIMARY KEY,
    vix_close     REAL,
    vix_change_1d REAL
);
```

Slimmed to just the VIX columns because the production model contract
only consumes `vix_close` and `vix_change_1d`; the previous
`treasury_10y` / `treasury_10y_change_1d` fields were removed from
both schema and pipeline.

Two writers feed this table through the **same** cleaning helper
(`clean_vix_frame`), so they cannot drift apart:

- **Live:** [`src/jobs/daily_macro.py`](../src/jobs/daily_macro.py)
  fetches the full `VIXCLS` series from FRED (requires `FRED_API_KEY`)
  on every run and upserts it.
- **Batch / offline:** [`scripts/build_database.py`](../scripts/build_database.py)
  reads `data/raw/macro.csv` (FRED `VIXCLS`, downloaded by
  `scripts/download_data.py`). `clean_vix_frame` lives in
  [`src/macro.py`](../src/macro.py).

Both coerce to numeric, drop NaN VIX rows, sort, compute `vix_change_1d`
via `pct_change` on the clean sorted series **before** any join (so the
change references the real previous observation, not a join artefact),
drop the first row whose change is undefined, then tz-localise the date
to `America/New_York` midnight and write via `save_macro_indicators`.

Note: `treasury_10y` is intentionally **not** loaded here, the
production model contract is only `return_1d`, `vix_change_1d`,
`vix_close`. The Yahoo-Finance path in `src/macro.py` (`^VIX`/`^TNX`)
still produces a 10-year-Treasury column, but only for the in-memory
bakeoff comparison; it never reaches this table.

`save_macro_indicators` is idempotent for the dates supplied: it
deletes the matching dates and re-inserts them, leaving other dates
untouched.

### `features` (per-ticker model inputs)

```sql
CREATE TABLE features (
    ticker    TEXT NOT NULL,
    date      TEXT NOT NULL,
    return_1d REAL,
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker, date) REFERENCES cleaned_prices(ticker, date)
);
CREATE INDEX idx_features_date ON features(date);
```

Slimmed to a single feature column because the chosen production
model bundle only consumes `return_1d`. The previous wide table
(16 indicators including SMA/EMA/MACD/RSI/Bollinger/OBV/index series)
is no longer materialised here; teammates' parallel candidate models
(random forest, LSTM, ...) compute their own features outside this DB
and are unaffected by the slim schema.

Source: step 2 of [`scripts/build_database.py`](../scripts/build_database.py), which reads
`cleaned_prices`, sorts by `(ticker, date)`, and computes `return_1d`
strictly per-ticker via `groupby("ticker")["close"].pct_change()` ,
never a frame-level `pct_change`, so ticker A's last close cannot
leak into ticker B's first row. The first observation per ticker has
no prior close and lands as NULL (one NULL per ticker).

`save_features` still accepts a DataFrame with any subset of the
schema columns (plus the required `ticker`/`date`); missing ones are
stored as NULL. With the slimmed schema that effectively means
`return_1d` is the only optional column.

### `predictions`

```sql
CREATE TABLE predictions (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    prediction  INTEGER,
    probability REAL,
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker, date) REFERENCES cleaned_prices(ticker, date)
);
CREATE INDEX idx_predictions_date ON predictions(date);
```

## Production model feature contract

The production joblib bundle was trained on exactly three features, in
this exact (alphabetically sorted) order:

```python
MODEL_FEATURE_CONTRACT = ["return_1d", "vix_change_1d", "vix_close"]
```

The order is **not** cosmetic: the bundle's `prod_features` was derived
via `sorted()`, so any other column order silently misaligns the input
matrix at serve time. The constant lives in [`src/storage.py`](../src/storage.py)
and is the single source of truth, slice `df[MODEL_FEATURE_CONTRACT]`
to feed the bundle, never hard-code the list elsewhere.

### `load_model_frame(tickers=None, start_date=None, end_date=None)`

The shared serving function that builds the model-input frame in one
place, so training-vs-serving skew can't sneak in:

- Joins `features` (`return_1d` per ticker/date) with `macro_indicators`
  (`vix_close`, `vix_change_1d` per date) and `cleaned_prices`
  (`label`).
- Joins on a **normalised tz-naive calendar-day key**, not on the raw
  TEXT timestamp. `cleaned_prices` / `features` store tz-aware
  `America/New_York`; `macro_indicators` stores tz-localised
  `America/New_York` midnight with mixed EST/EDT offsets across the
  year. Both sides are parsed via `pd.to_datetime(s, utc=True)` →
  `tz_convert("America/New_York")` → `tz_localize(None)` →
  `normalize()` before the merge, so DST transitions and offset jitter
  cannot create off-by-one mismatches.
- Returns columns in the exact order
  `ticker, date, label, return_1d, vix_change_1d, vix_close`, the
  last three are `MODEL_FEATURE_CONTRACT` verbatim, ready to slice
  straight into the bundle.
- Drops rows where **any** feature is NULL (the model cannot score
  without all three).
- **Keeps** rows where `label` is NULL. By definition the most-recent
  row per ticker has no label yet (its next-day close hasn't happened)
  and that is exactly the row the live pipeline needs to score.
- `date` in the output is tz-aware `America/New_York`, consistent with
  `load_from_db`.

Respects `tickers` / `start_date` / `end_date` filters on all three
underlying tables.

Implementation note: `load_model_frame` currently bypasses
`load_macro_indicators` and reads `macro_indicators` directly via
`_read_filtered` as a workaround. `load_macro_indicators` parses
dates without `utc=True` and raises on the mixed EST/EDT offsets
written by `clean_vix_frame` (pandas 3.0.2 strict-fails on
mixed offsets; older versions tolerated it). Once
`load_macro_indicators` is updated to handle the mixed-offset case
(e.g. `pd.to_datetime(..., utc=True)`), `load_model_frame` can revert
to the standard call.

## Building the database

[`scripts/build_database.py`](../scripts/build_database.py) builds the
whole store from the raw files in `data/raw/` (see
[`data/README.md`](../data/README.md)), offline:

1. **`cleaned_prices`** - raw price CSVs -> `standardize` -> calendar
   filter, de-duplication and OHLC-consistency checks -> `save_to_db`,
   which recomputes the gap-aware `label` per ticker using the NASDAQ
   calendar.
2. **`features`** - `return_1d` per `(ticker, date)`.
3. **`macro_indicators`** - `vix_close` + `vix_change_1d` via
   `clean_vix_frame`.
4. **`predictions`** - scores the newest complete day with the bundle in
   `models/` (skip with `--no-predictions`).

```bash
python scripts/build_database.py
```

Every step is idempotent (delete-then-insert on the key columns), so the
script is safe to re-run. If a DataFrame passed to `save_to_db` still
contains `label_valid`, `daily_return` or `transactions`, it emits a
`DeprecationWarning` and drops them rather than writing them.

## Indexing

| Index                                | Columns         | Created by   | Purpose                              |
| ------------------------------------ | --------------- | ------------ | ------------------------------------ |
| `sqlite_autoindex_cleaned_prices_1`  | (ticker, date)  | `PRIMARY KEY`| Ticker-scoped lookups and ranges     |
| `idx_cleaned_prices_date`            | (date)          | explicit     | Date-only filters across all tickers |
| `sqlite_autoindex_features_1`        | (ticker, date)  | `PRIMARY KEY`| Per-ticker feature lookups           |
| `idx_features_date`                  | (date)          | explicit     | Date-only feature queries            |
| `sqlite_autoindex_predictions_1`     | (ticker, date)  | `PRIMARY KEY`| Per-ticker prediction lookups        |
| `idx_predictions_date`               | (date)          | explicit     | Date-only prediction queries         |
| `sqlite_autoindex_macro_indicators_1`| (date)          | `PRIMARY KEY`| Date lookups                         |

### Why these

The composite PK on `(ticker, date)` is a B-tree with `ticker` as the
leading column. It is used for any of these access patterns:

- `WHERE ticker = ?`
- `WHERE ticker = ? AND date = ?`
- `WHERE ticker = ? AND date BETWEEN ? AND ?`
- `WHERE ticker IN (?, ?, …) AND date BETWEEN ? AND ?` *(the reference
  benchmark query, each ticker becomes an independent index seek + range
  scan)*

A composite index does **not** help when the leading column is omitted,
so a date-only query like *"give me everything from the last 2 years"*
would otherwise do a full table scan. The extra `idx_*_date` indexes
close that gap.

### Why no more indexes

The pipeline currently only filters on `ticker` and `date`. Adding
indexes on price/feature/prediction columns would slow down writes
(every `INSERT` updates every index) without speeding up any query
we actually issue.

## Benchmark (cleaned_prices)

The reference workload is **"load all rows for 10 tickers over the last
2 years"**, with a sub-second target.

### Setup

- DB: `data/nasdaq100.db`, 224,030 rows, 100 tickers, dates 2015-02-17
  → 2025-02-13. File size 36.4 MB.
- Tickers used: `AAPL, MSFT, NVDA, AMZN, GOOG, META, TSLA, AVGO, COST, NFLX`
- Window: 2023-02-13 → 2025-02-13 (the most recent 2 years)
- Hardware/runtime: Windows 11, Python 3.14.4, SQLite 3.50.4
- Method: one warm-up call (to populate OS/SQLite caches), then 5 timed
  runs of `load_from_db(tickers=…, start_date=…, end_date=…)`.

### Query plan

```
SEARCH cleaned_prices USING INDEX sqlite_autoindex_cleaned_prices_1
       (ticker=? AND date>? AND date<?)
```

This confirms the composite PK index is being used, no full scan.

### Results

| Query                                     | Mean      | Min      | Max      | Notes                                       |
| ----------------------------------------- | --------- | -------- | -------- | ------------------------------------------- |
| **10 tickers × 2 years (reference)**      | **37 ms** | 36 ms    | 38 ms    | Uses PK composite index. 5,020 rows.        |
| 1 ticker, full history                    | 22 ms     | 20 ms    | 26 ms    | Uses PK composite index. ~2,500 rows.       |
| All 100 tickers, last 2 years (date-only) | 361 ms    | 356 ms   | 365 ms   | Uses `idx_cleaned_prices_date`. ~50k rows.  |
| All rows (no filter)                      | 1,422 ms  | 1,410 ms | 1,443 ms | Full table scan, bounded by data volume.   |

The reference query lands at ~37 ms, **roughly 27× under the 1-second
target**.

> Benchmark numbers were measured against the previous schema (which had
> three extra columns). The new schema is narrower, so timings should be
> the same or slightly better, re-run after the next refresh if you
> need fresh figures.
