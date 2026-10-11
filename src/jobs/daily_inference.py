"""Daily inference: score next-day direction for the newest complete day.

Run:
    python -m src.jobs.daily_inference

Reads the production model bundle from models/prod_model_training_xgboost
.joblib, calls load_model_frame to build the joined serving frame
(features + macro + cleaned_prices), picks the newest date where all
three features are complete and the label is still NULL (the day whose
next-day close has not happened yet), scores every ticker on that day,
and upserts predictions via save_predictions.

The prediction date is not always today: OHLCV (yfinance) and macro
(FRED) can update on different days, and load_model_frame inner-joins
on date, so the newest row in the frame may be a day or two behind the
calendar. That is expected; we log which date is actually scored.

Inference is decoupled from ingest. This job reads only the DB; the
OHLCV + macro fills (daily_ingest.py, daily_macro.py) must have run
first. No yfinance/FRED imports here.

Out of scope: back-filling predictions for older NULL-label rows
(separate back-test / back-fill use case).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

from src.storage import MODEL_FEATURE_CONTRACT, load_model_frame, save_predictions

BUNDLE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "prod_model_training_xgboost.joblib"
)
NY = "America/New_York"

# Calendar-day lookback for load_model_frame. Has to cover whatever lag
# might exist between the OHLCV side and the macro side of the inner
# join. 10 days is comfortably more than needed and keeps the read
# volume small.
WINDOW_DAYS = 10


def _load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Bundle not found at {BUNDLE_PATH}. Train the model first."
        )
    bundle = joblib.load(BUNDLE_PATH)
    required = {"model", "features"}
    missing = required - set(bundle.keys())
    if missing:
        raise KeyError(
            f"Bundle is missing required keys {sorted(missing)}; "
            f"got keys: {sorted(bundle.keys())}"
        )
    return bundle


def _select_prediction_day(
    df: pd.DataFrame,
) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    """Pick the newest date with complete features, labeled or not.
    load_model_frame already dropped rows missing any feature, so the
    newest date in the frame is the newest fully-featured day. We score
    it regardless of label: FRED VIX lags the price feed by a day, so the
    newest VIX-complete day is usually already labeled, and refusing
    labeled days (the old rule) left nothing to score on a normal day."""
    if df.empty:
        return None, df
    pred_date = df["date"].max()
    to_predict = df[df["date"] == pred_date]
    return pred_date, to_predict


def main() -> int:
    bundle = _load_bundle()
    model = bundle["model"]
    features = list(bundle["features"])
    print(f"[model] loaded from {BUNDLE_PATH}; features={features}")

    # Warn-only: bundle and storage.MODEL_FEATURE_CONTRACT are two
    # independent declarations of the same list. If they diverge it is
    # equally likely that the constant is stale as that the bundle is
    # wrong, so don't block the run -- just surface it. Bundle stays
    # the single source of truth for column order.
    if features != MODEL_FEATURE_CONTRACT:
        print(
            f"[warn] bundle features {features} differ from "
            f"storage.MODEL_FEATURE_CONTRACT {MODEL_FEATURE_CONTRACT}. "
            f"Using bundle order; verify whether the constant in "
            f"src/storage.py needs updating."
        )

    end = pd.Timestamp.now(tz=NY).normalize()
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    df = load_model_frame(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        print(
            f"[frame] 0 rows loaded for window "
            f"{start.date()} -> {end.date()}. Exiting cleanly."
        )
        return 0
    print(
        f"[frame] {len(df)} rows loaded for window "
        f"{start.date()} -> {end.date()}"
    )

    # Hard-raise: a feature the bundle expects is genuinely absent from
    # the frame. This is a real data gap (load_model_frame can't supply
    # it), not a naming or constant mismatch. Order is enforced by the
    # df[features] selection below, not by this check.
    missing_features = set(features) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Bundle expects features {sorted(missing_features)} that "
            f"load_model_frame did not return (got columns: "
            f"{list(df.columns)}). Cannot score."
        )

    pred_date, to_predict = _select_prediction_day(df)
    if pred_date is None:
        print(
            "[predict] no complete prediction day available "
            "(all complete-feature rows already have a label). "
            "Exiting cleanly."
        )
        return 0

    # bundle["features"] is the sole place column order is decided.
    # Do not substitute MODEL_FEATURE_CONTRACT here -- that would
    # silently re-introduce the dual-source-of-truth we just warned about.
    X = to_predict[features]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    n_pred = len(to_predict)
    print(
        f"[predict] date={pred_date.date()} tickers={n_pred} "
        f"mean_proba={proba.mean():.4f}"
    )

    # Build the predictions frame off to_predict directly so the date
    # column keeps its tz-aware dtype. .values on a tz-aware datetime
    # Series silently strips the tz and converts the wallclock to UTC,
    # producing strings like "2026-06-03 04:00:00" that no longer match
    # cleaned_prices.date ("2026-06-03 00:00:00-04:00") -- which trips
    # the predictions->cleaned_prices FK on save.
    #
    # Positional alignment: pred and proba come from
    # model.predict_proba(to_predict[features]), so they share
    # to_predict's row order. reset_index(drop=True) lines everything up
    # on a fresh 0..n-1 index so the two assignments below align by
    # position. Do not reorder these three lines.
    out = to_predict[["ticker", "date"]].reset_index(drop=True).copy()
    out["prediction"] = pred
    out["probability"] = proba
    stored = save_predictions(out)
    print(
        f"[save] {n_pred} predictions upserted; "
        f"predictions table now contains {stored} rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
