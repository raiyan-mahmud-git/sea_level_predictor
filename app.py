"""
app.py – Streamlit entrypoint for the Sea Level Predictor app.

Navigation
----------
Uses st.navigation (Streamlit ≥ 1.36) to define four pages:
  - Overview       – dataset facts and attribution
  - Trend Explorer – interactive chart and predictions
  - Model Lab      – model comparison and residuals
  - Methodology    – formulas and limitations

Page config is set here once (must be the first Streamlit call in the script).
"""

import streamlit as st

# ── Page config must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="Sea Level Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/raiyan-mahmud-git/sea_level_predictor",
        "Report a bug": "https://github.com/raiyan-mahmud-git/sea_level_predictor/issues",
        "About": (
            "**Sea Level Predictor** – freeCodeCamp Data Analysis project.  \n"
            "Built with Streamlit, Plotly, pandas, scipy, numpy.  \n"
            "Educational use only. Not for policy decisions."
        ),
    },
)

# ── Pages ───────────────────────────────────────────────────────────────────
from views import methodology, model_lab, overview, trends  # noqa: E402

pages = {
    "📊 Overview": overview.show,
    "📈 Trend Explorer": trends.show,
    "🔬 Model Lab": model_lab.show,
    "📐 Methodology": methodology.show,
}

# ── Sidebar navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/Sea_level_rise_observations.svg/640px-Sea_level_rise_observations.svg.png",
        caption="Global mean sea level rise (NASA)",
        use_container_width=True,
    )
    st.markdown("---")
    st.caption("🌊 Sea Level Predictor  \n*freeCodeCamp Data Analysis Project*")
    st.markdown("---")
    selected = st.radio(
        "Navigate",
        list(pages.keys()),
        key="nav_page",
        label_visibility="collapsed",
    )

# ── Render selected page ─────────────────────────────────────────────────────
pages[selected]()
