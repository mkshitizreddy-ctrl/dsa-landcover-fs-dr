import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ("scripts", "src"):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
import streamlit as st

RESULTS = ROOT / "results"


def setup_page(title, caption=None):
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    if caption:
        st.caption(caption)


def make_show(limit=15):
    box = st.empty()
    lines = []

    def show(message):
        lines.append(str(message))
        box.code("\n".join(lines[-limit:]), language=None)
    return show


@st.cache_data
def read_result(relative_path):
    path = RESULTS / relative_path
    return pd.read_csv(path) if path.exists() else None
