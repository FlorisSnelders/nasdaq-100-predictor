# tests/test_inference.py
#
# Unit tests for src/inference.py (write_predictions).
# write_predictions is a 4-call orchestration over joblib.load,
# load_model_frame, model.predict_proba, and save_predictions. All four
# edges are stubbed; a hand-rolled FakeModel plays the predict_proba role.

import numpy as np
import pandas as pd
import pytest

from src import inference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ny(date_str):
    return pd.Timestamp(date_str, tz="America/New_York")


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["ticker", "date", "return_1d", "vix_close", "vix_change_1d"],
    )


FEATURES = ["return_1d", "vix_change_1d", "vix_close"]


class FakeModel:
    """Stand-in for the xgboost classifier. Records every X passed to
    predict_proba and returns a fixed up-probability vector. Column 1 of
    the returned array is the 'up' probability (matches the [:, 1] slice
    in write_predictions)."""

    def __init__(self, p_up):
        self._p_up = list(p_up)
        self.calls = []

    def predict_proba(self, X):
        self.calls.append(X.copy())
        if len(X) != len(self._p_up):
            raise AssertionError(
                f"FakeModel expected {len(self._p_up)} rows, got {len(X)}"
            )
        col0 = [1.0 - p for p in self._p_up]
        return np.array(list(zip(col0, self._p_up)))


def _install(monkeypatch, *, bundle, frame, save_return=42):
    captured = {"loaded_path": None, "saved": None}

    def fake_joblib_load(path):
        captured["loaded_path"] = path
        return bundle

    def fake_load_frame():
        return frame.copy()

    def fake_save(df):
        captured["saved"] = df.copy()
        return save_return

    monkeypatch.setattr(inference.joblib, "load", fake_joblib_load)
    monkeypatch.setattr(inference, "load_model_frame", fake_load_frame)
    monkeypatch.setattr(inference, "save_predictions", fake_save)
    return captured


# ---------------------------------------------------------------------------
# write_predictions
# ---------------------------------------------------------------------------

def test_write_predictions_selects_latest_per_ticker(monkeypatch):
    # Two tickers x three dates -> save_predictions must receive exactly
    # the newest row per ticker. Regression for the groupby.tail(1) step.
    frame = _frame([
        ("T00", _ny("2024-01-02"), 0.01, 15.0, 0.01),
        ("T00", _ny("2024-01-03"), 0.02, 15.5, 0.03),
        ("T00", _ny("2024-01-04"), 0.03, 16.0, 0.03),
        ("T01", _ny("2024-01-02"), -0.01, 15.0, 0.01),
        ("T01", _ny("2024-01-03"), -0.02, 15.5, 0.03),
        ("T01", _ny("2024-01-04"), -0.03, 16.0, 0.03),
    ])
    model = FakeModel(p_up=[0.7, 0.3])
    captured = _install(
        monkeypatch,
        bundle={"model": model, "features": FEATURES},
        frame=frame,
    )

    n = inference.write_predictions()
    assert n == 42

    saved = captured["saved"]
    assert list(saved.columns) == ["ticker", "date", "prediction", "probability"]
    assert len(saved) == 2

    saved = saved.sort_values("ticker").reset_index(drop=True)
    assert list(saved["ticker"]) == ["T00", "T01"]
    assert saved.loc[0, "date"] == _ny("2024-01-04")
    assert saved.loc[1, "date"] == _ny("2024-01-04")


def test_write_predictions_threshold_is_geq_half(monkeypatch):
    # proba >= 0.5 -> prediction 1, strict below -> 0. The 0.50 boundary
    # itself maps to 1 because the comparison is >=, not >.
    frame = _frame([
        ("A", _ny("2024-01-02"), 0.01, 15.0, 0.01),
        ("B", _ny("2024-01-02"), 0.01, 15.0, 0.01),
        ("C", _ny("2024-01-02"), 0.01, 15.0, 0.01),
    ])
    model = FakeModel(p_up=[0.30, 0.50, 0.70])
    captured = _install(
        monkeypatch,
        bundle={"model": model, "features": FEATURES},
        frame=frame,
    )

    inference.write_predictions()

    saved = captured["saved"].sort_values("ticker").reset_index(drop=True)
    assert list(saved["prediction"]) == [0, 1, 1]
    # probability column must carry the raw proba positionally aligned to
    # the ticker order (sort by ticker reproduces input order here).
    assert list(saved["probability"]) == pytest.approx([0.30, 0.50, 0.70])


def test_write_predictions_preserves_tz_aware_date(monkeypatch):
    # Regression: a .values-based shortcut on a tz-aware datetime Series
    # silently strips the tz and converts wallclock to UTC. The same skew
    # bug is documented in daily_inference.py. Test that the date column
    # in the saved frame is still tz-aware America/New_York.
    frame = _frame([("A", _ny("2024-01-02"), 0.01, 15.0, 0.01)])
    model = FakeModel(p_up=[0.6])
    captured = _install(
        monkeypatch,
        bundle={"model": model, "features": FEATURES},
        frame=frame,
    )

    inference.write_predictions()

    saved = captured["saved"]
    assert pd.api.types.is_datetime64_any_dtype(saved["date"])
    assert str(saved["date"].dt.tz) == "America/New_York"


def test_write_predictions_forwards_model_path(monkeypatch):
    # Custom model_path must reach joblib.load; MODEL_PATH is only the
    # default. Catches accidental hard-coding of the constant.
    frame = _frame([("A", _ny("2024-01-02"), 0.01, 15.0, 0.01)])
    model = FakeModel(p_up=[0.6])
    captured = _install(
        monkeypatch,
        bundle={"model": model, "features": FEATURES},
        frame=frame,
    )

    inference.write_predictions(model_path="custom/path.joblib")
    assert captured["loaded_path"] == "custom/path.joblib"


def test_write_predictions_uses_bundle_feature_order(monkeypatch):
    # X passed to predict_proba must respect bundle["features"] order, not
    # the alphabetically-sorted MODEL_FEATURE_CONTRACT. If the call ever
    # falls back to the contract, a non-sorted bundle silently misaligns.
    custom_order = ["vix_close", "return_1d", "vix_change_1d"]
    frame = _frame([("A", _ny("2024-01-02"), 0.01, 15.0, 0.01)])
    model = FakeModel(p_up=[0.6])
    _install(
        monkeypatch,
        bundle={"model": model, "features": custom_order},
        frame=frame,
    )

    inference.write_predictions()
    X = model.calls[0]
    assert list(X.columns) == custom_order
