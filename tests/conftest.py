import sys
import types
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# point this at your dashboard file (app.py / dashboard.py)
APP_PATH = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"

class _Stub:
    """Callable + dict-like + context-manager no-op.
    Covers st.markdown(...), st.query_params.get(...), with st.container(), etc."""
    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        return _Stub()

    def get(self, *a, **k):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeSt(_Stub):
    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Stub() for _ in range(n)]

    # widgets used in the module body must return concrete values
    def text_input(self, *a, **k):
        return ""

    def selectbox(self, label, options, **k):
        return options[0]

    def slider(self, label, mn, mx, value, *a, **k):
        return value


@pytest.fixture
def app(monkeypatch):
    """Import the dashboard with streamlit + src.storage stubbed. Empty preds ->
    the app's built-in demo-data path runs, so loading needs no real database."""
    monkeypatch.setitem(sys.modules, "streamlit", _FakeSt())

    storage = types.ModuleType("src.storage")
    storage.load_predictions = lambda *a, **k: pd.DataFrame()
    storage.load_from_db = lambda *a, **k: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "src.storage", storage)

    spec = importlib.util.spec_from_file_location("dashboard_app", APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_app"] = mod
    spec.loader.exec_module(mod)
    return mod