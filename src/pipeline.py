from src.ingest import load_snapshot, load_yfinance
from src.data_utils import standardize, check_schema
from src.cleaning import remove_weekends_and_holidays, remove_duplicate_rows, validate_price_consistency
from src.data_cleaner import add_daily_returns, find_missing_trading_days, flag_pregap_rows
from src.macro import load_and_clean_macro
from src.validation import validate_stage
from src.storage import save_to_db
from src.quality_report import StageStats, ValidationResult, generate_quality_report


def run_pipeline(source=None, tickers=None, start=None, end=None, save_db=False, write_report=False):
    stage_stats = []
    validation_results = []

    if tickers is not None:
        df = load_yfinance(tickers, start, end)
    else:
        df = load_snapshot(source)
    stage_stats.append(StageStats("ingest", rows_in=0, rows_out=len(df)))

    rows_in = len(df)
    df = standardize(df)
    check_schema(df)
    validate_stage(df, stage="after_load", verbose=False)
    validation_results.append(ValidationResult("after_load", "all_checks", True, "passed"))
    stage_stats.append(StageStats("standardize", rows_in=rows_in, rows_out=len(df)))

    rows_in = len(df)
    df = remove_weekends_and_holidays(df)
    df = remove_duplicate_rows(df)
    df = validate_price_consistency(df)
    stage_stats.append(StageStats("clean", rows_in=rows_in, rows_out=len(df)))

    missing = find_missing_trading_days(df)
    df = flag_pregap_rows(df, missing)
    df = add_daily_returns(df)

    rows_in = len(df)
    macro_df = load_and_clean_macro(
        start=df["date"].min().strftime("%Y-%m-%d"),
        end=df["date"].max().strftime("%Y-%m-%d"),
    )
    df["date_key"] = df["date"].dt.tz_localize(None).dt.normalize()
    df = df.merge(macro_df, left_on="date_key", right_on="date", how="left", suffixes=("", "_macro"))
    df = df.drop(columns=["date_key", "date_macro"])
    stage_stats.append(StageStats("macro_merge", rows_in=rows_in, rows_out=len(df)))

    validate_stage(df, stage="after_clean", verbose=False)
    validation_results.append(ValidationResult("after_clean", "all_checks", True, "passed"))

    if save_db:
        save_to_db(df)

    if write_report:
        generate_quality_report(
            stage_stats=stage_stats,
            validation_results=validation_results,
            final_df=df,
        )

    return df


def build_bakeoff_frame(prices_dir=None, macro_csv=None):
    """Offline version of the bake-off input frame, built from data/raw/.

    Same cleaning chain as run_pipeline (calendar filter, dedupe, OHLC
    consistency, pre-gap label flags), but the macro columns come from the
    local macro file (FRED VIX + 10-year yield) instead of a live yfinance
    pull, so the result is reproducible offline. Dates are returned as
    tz-naive calendar days. Feed the result to
    src.features.build_feature_matrix.
    """
    from src.datasets import MACRO_CSV, PRICES_DIR, load_macro, load_prices

    df = load_prices(prices_dir or PRICES_DIR)
    df = standardize(df)
    check_schema(df)
    df = remove_weekends_and_holidays(df)
    df = remove_duplicate_rows(df)
    df = validate_price_consistency(df)
    df = flag_pregap_rows(df, find_missing_trading_days(df))
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()

    macro = load_macro(macro_csv or MACRO_CSV).rename(
        columns={"vix_close": "vix", "treasury_10y": "treasury_yield_10y"}
    )
    return df.merge(macro, on="date", how="left")
