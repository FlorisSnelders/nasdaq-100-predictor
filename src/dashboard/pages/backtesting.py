import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.dashboard.components import bakeoff_models, load_bakeoff, backtesting_tables, render_disclaimer

st.set_page_config(page_title="Backtesting", layout="wide")

st.title("Walk forward backtesting")
render_disclaimer()

models = bakeoff_models()
if not models:
    st.info("No walk-forward results found in results/metrics.")
else:
    default = models.index("xgboost") if "xgboost" in models else 0
    choice = st.selectbox("Model", models, index=default)
    per_fold, summary = backtesting_tables(load_bakeoff(choice))
    st.subheader("Per fold")
    st.dataframe(per_fold, hide_index=True, use_container_width=True)
    st.subheader("Mean and standard deviation across folds")
    st.dataframe(summary, hide_index=True, use_container_width=True)

render_disclaimer()
