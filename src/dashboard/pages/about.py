import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.dashboard.components import model_card_text, render_disclaimer

st.set_page_config(page_title="About", layout="wide")

render_disclaimer()

text = model_card_text()
if text is None:
    st.info("Model card not found at docs/model_card.md.")
else:
    st.markdown(text)

render_disclaimer()
