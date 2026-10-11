# Data

No data is committed to this repository. The original project used a daily price snapshot and a macro-indicator file supplied through the course; those files are not redistributable, so this folder only contains this README. Everything the code needs can be rebuilt from free public sources with two commands:

```bash
python scripts/download_data.py    # ~1 minute, needs internet
python scripts/build_database.py   # ~20 seconds, offline
```

## What gets created

| Path | Contents | Source |
|---|---|---|
| `data/raw/prices/<TICKER>.csv` | Daily OHLCV per stock, 2015-02-17 to 2025-02-13 | Yahoo Finance via `yfinance` (split-adjusted, not dividend-adjusted) |
| `data/raw/macro.csv` | Daily VIX close and 10-year Treasury yield | FRED series `VIXCLS` and `DGS10` (public CSV endpoint, no API key) |
| `data/nasdaq100.db` | SQLite store with four tables: `cleaned_prices`, `features`, `macro_indicators`, `predictions` (schema in `docs/storage.md`) | built by `scripts/build_database.py` |
| `data/processed/` | Derived tables written by the notebooks (e.g. EDA summary statistics) | notebooks |

The stock universe is the 100 NASDAQ-100 members of the original snapshot (`config/universe_2015_2025.txt`). `config/tickers.txt` holds the current index members, used by the live daily-ingest job.

## How close is the public rebuild to the original data?

- **Macro:** FRED `VIXCLS` and `DGS10` values are identical to the original macro file.
- **Prices:** Yahoo closing prices match the original snapshot; daily volumes differ slightly.
- **Universe:** As of October 2026, Yahoo no longer serves 3 of the 100 tickers (ANSS, EA and WBD, delisted after acquisitions), so the rebuild has 97 stocks. The original data also contained ticker-symbol reuse (e.g. `APP` belonged to a different company before AppLovin listed in 2021), which Yahoo's history does not have.

Because of these differences, numbers from a rebuilt dataset are close to, but not identical to, the notebook outputs (which were produced on the original snapshot). The main README lists both.

## Keeping the data current (optional)

The jobs in `src/jobs/` append new trading days and score them with the trained model:

```bash
python -m src.jobs.daily_ingest      # latest OHLCV from Yahoo -> cleaned_prices + features
python -m src.jobs.daily_macro       # VIX from the FRED API -> macro_indicators
python -m src.jobs.daily_inference   # score the newest complete day -> predictions
```

`daily_macro` uses the FRED API and needs a free key in the `FRED_API_KEY` environment variable (for example in a `.env` file, which is gitignored). Get one at https://fredaccount.stlouisfed.org/apikeys.
