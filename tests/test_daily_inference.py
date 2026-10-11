import joblib
import pandas as pd
import pytest

from src.jobs import daily_inference
from src.jobs.daily_inference import _select_prediction_day


def test_select_prediction_day_scores_newest_even_if_labeled():
    df = pd.DataFrame({"date": [pd.Timestamp("2026-06-09")], "label": [1]})
    pred_date, to_predict = _select_prediction_day(df)
    assert pred_date == pd.Timestamp("2026-06-09")
    assert len(to_predict) == 1


def test_select_prediction_day_none_when_empty():
    df = pd.DataFrame({"date": [], "label": []})
    pred_date, to_predict = _select_prediction_day(df)
    assert pred_date is None
    assert to_predict.empty

def test_select_prediction_day_picks_newest_null():
    df = pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "date": [
            pd.Timestamp("2026-06-08"),
            pd.Timestamp("2026-06-09"),
            pd.Timestamp("2026-06-09"),
        ],
        "label": [1, pd.NA, pd.NA],
    })
    pred_date, to_predict = _select_prediction_day(df)
    assert pred_date == pd.Timestamp("2026-06-09")
    assert len(to_predict) == 2


def test_load_bundle_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_inference, "BUNDLE_PATH", tmp_path / "nope.joblib")
    with pytest.raises(FileNotFoundError):
        daily_inference._load_bundle()


def test_load_bundle_missing_keys(monkeypatch, tmp_path):
    p = tmp_path / "b.joblib"
    joblib.dump({"model": object()}, p)
    monkeypatch.setattr(daily_inference, "BUNDLE_PATH", p)
    with pytest.raises(KeyError):
        daily_inference._load_bundle()


def test_load_bundle_ok(monkeypatch, tmp_path):
    p = tmp_path / "b.joblib"
    joblib.dump({"model": object(), "features": ["a", "b"]}, p)
    monkeypatch.setattr(daily_inference, "BUNDLE_PATH", p)
    bundle = daily_inference._load_bundle()
    assert bundle["features"] == ["a", "b"]