import math

import pandas as pd
import pytest


def test_module_imports_and_builds_demo(app):
    assert app.is_demo is True
    assert len(app.df) == len(app.NAMES)


# ---- company ----
def test_company_known(app):
    assert app.company("aapl") == "Apple"
    assert app.company("MSFT") == "Microsoft"


def test_company_unknown_returns_upper_ticker(app):
    assert app.company("zzzz") == "ZZZZ"


# ---- _demo_data ----
def test_demo_data_shapes_and_columns(app):
    preds, prices, as_of = app._demo_data()
    assert list(preds.columns) == ["ticker", "date", "prediction", "probability"]
    assert len(preds) == len(app.NAMES)
    assert set(prices.columns) == {"ticker", "date", "open", "high", "low", "close", "volume"}
    assert len(prices) == len(app.NAMES) * app.SPARK_WINDOW


def test_demo_data_prediction_is_binary(app):
    preds, _, _ = app._demo_data()
    assert set(preds["prediction"].unique()).issubset({0, 1})
    # prediction tracks probability except at the rounding boundary (prob rounds
    # to 4dp after the >=0.5 cut, so a raw 0.49997 -> stored 0.5 with prediction 0)
    mism = preds[preds["prediction"] != (preds["probability"] >= 0.5).astype(int)]
    assert (abs(mism["probability"] - 0.5) < 1e-3).all()


def test_demo_data_probabilities_in_bounds(app):
    preds, _, _ = app._demo_data()
    assert preds["probability"].between(0.34, 0.93).all()


def test_demo_data_is_deterministic(app):
    p1, _, _ = app._demo_data()
    p2, _, _ = app._demo_data()
    pd.testing.assert_frame_equal(p1, p2)


# ---- build_frame ----
def _preds():
    return pd.DataFrame({
        "ticker": ["aapl", "msft"],
        "date": ["2024-01-05", "2024-01-05"],
        "prediction": [1, 0],
        "probability": [0.8, 0.7],
    })


def _prices():
    rows = []
    for t in ["AAPL", "MSFT"]:
        for i, c in enumerate([100.0, 110.0, 120.0]):
            rows.append((t, f"2024-01-0{i+2}", c))
    return pd.DataFrame(rows, columns=["ticker", "date", "close"])


def test_build_frame_confidence_is_prob_of_predicted_class(app):
    df = app.build_frame(_preds(), _prices())
    aapl = df[df["ticker"] == "AAPL"].iloc[0]
    msft = df[df["ticker"] == "MSFT"].iloc[0]
    assert bool(aapl["up"]) is True
    assert abs(aapl["confidence"] - 0.8) < 1e-9          # predicted up -> p
    assert abs(msft["confidence"] - (1 - 0.7)) < 1e-9    # predicted down -> 1-p


def test_build_frame_uppercases_and_names(app):
    df = app.build_frame(_preds(), _prices())
    assert set(df["ticker"]) == {"AAPL", "MSFT"}
    assert df[df["ticker"] == "AAPL"].iloc[0]["name"] == "Apple"


def test_build_frame_sorted_by_confidence_desc(app):
    df = app.build_frame(_preds(), _prices())
    assert list(df["confidence"]) == sorted(df["confidence"], reverse=True)


def test_build_frame_window_change_and_last(app):
    df = app.build_frame(_preds(), _prices())
    aapl = df[df["ticker"] == "AAPL"].iloc[0]
    assert aapl["last"] == 120.0
    assert abs(aapl["window_chg"] - (120.0 / 100.0 - 1.0)) < 1e-9


def test_build_frame_missing_prices_gives_empty_spark(app):
    preds = pd.DataFrame({
        "ticker": ["nvda"], "date": ["2024-01-05"],
        "prediction": [1], "probability": [0.6],
    })
    df = app.build_frame(preds, _prices())  # NVDA absent from prices
    row = df.iloc[0]
    assert row["spark"] == []
    assert pd.isna(row["last"])


# ---- sparkline ----
def test_sparkline_empty_returns_blank_svg(app):
    assert app.sparkline([], "#fff") == '<svg width="120" height="34"></svg>'


def test_sparkline_single_point_blank(app):
    assert app.sparkline([1.0], "#fff").endswith("></svg>")


def test_sparkline_draws_polyline(app):
    svg = app.sparkline([1.0, 2.0, 3.0], "#E8B14C")
    assert "<polyline" in svg and "#E8B14C" in svg


def test_sparkline_flat_series_no_crash(app):
    svg = app.sparkline([5.0, 5.0, 5.0], "#fff")  # rng == 0 branch
    assert "<polyline" in svg


# ---- confidence_arc ----
def test_confidence_arc_contains_percentage(app):
    svg = app.confidence_arc(0.73, "#E8B14C")
    assert "73" in svg and "CONFIDENCE" in svg and svg.startswith("<svg")


# ---- conf_bar ----
def test_conf_bar_renders_percent(app):
    html = app.conf_bar(0.42, "#fff")
    assert "42%" in html and "cbar" in html


# ---- arrow ----
def test_arrow_up_uses_bull_color(app):
    assert app.C_BULL in app.arrow(True)


def test_arrow_down_uses_bear_color(app):
    assert app.C_BEAR in app.arrow(False)


# ---- fmt_vol ----
@pytest.mark.parametrize("value,expected", [
    (2_500_000_000, "2.50B"),
    (3_400_000, "3.4M"),
    (12_000, "12K"),
    (500, "500"),
])
def test_fmt_vol_scales(app, value, expected):
    assert app.fmt_vol(value) == expected


def test_fmt_vol_invalid_returns_dash(app):
    assert app.fmt_vol(None) == "—"
    assert app.fmt_vol("abc") == "—"


# ---- next_session_label ----
def test_next_session_label_returns_future_weekday(app):
    out = app.next_session_label("2024-01-05")  # Fri -> next session Mon 08 Jan
    assert "08 JAN 2024" in out


def test_next_session_label_bad_input(app):
    assert app.next_session_label("not-a-date") == "—"


# ---- candles ----
def test_candles_empty_blank_svg(app):
    assert app.candles([]) == '<svg width="820" height="180"></svg>'


def test_candles_renders_rects(app):
    svg = app.candles([(10, 12, 9, 11), (11, 13, 10, 10)])
    assert svg.count("<rect") == 2
    assert app.C_BULL in svg and app.C_BEAR in svg  # one up, one down


# ---- render_detail ----
def _detail_prices():
    rows = []
    for i in range(5):
        c = 100.0 + i
        rows.append(("AAPL", f"2024-01-0{i+2}", c, c + 2, c - 2, c + 1, 1_000_000))
    return pd.DataFrame(
        rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"])


def _row(up=True):
    return pd.Series({"ticker": "AAPL", "name": "Apple", "up": up, "confidence": 0.8})


def test_render_detail_with_data(app):
    html = app.render_detail(_detail_prices(), _row(up=True))
    assert "AAPL" in html and "<svg" in html and "DAY RETURN" in html


def test_render_detail_no_data(app):
    empty = pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
    html = app.render_detail(empty, _row())
    assert "No price history" in html