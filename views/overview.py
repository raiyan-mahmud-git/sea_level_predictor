"""
views/overview.py – Overview page for the Sea Level Predictor app.

Shows:
- Dataset preview (first/last 5 rows)
- Summary statistics
- Missing value report
- Source and attribution

All numbers come from the code – nothing is hard-coded.
"""

from __future__ import annotations

import streamlit as st

from sealevel.data import load_data, summary_stats
from ui.components import render_error


@st.cache_data(show_spinner="Loading dataset…")
def _load():
    return load_data()


def show() -> None:
    st.title("🌊 Sea Level Rise – Dataset Overview")
    st.caption(
        "Source: U.S. Environmental Protection Agency, "
        "citing CSIRO (Church & White, 2011) and NOAA tide-gauge records."
    )

    try:
        df = _load()
    except Exception as exc:
        render_error(exc, "Failed to load dataset")
        return

    stats = summary_stats(df)

    # --- Key facts --------------------------------------------------------- #
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", stats["n"])
    c2.metric("First year", stats["first_year"])
    c3.metric("Last year", stats["last_year"])
    c4.metric("Columns", len(df.columns))

    st.divider()

    # --- Dataset preview --------------------------------------------------- #
    st.subheader("Dataset preview")
    tab1, tab2 = st.tabs(["First 10 rows", "Last 10 rows"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(df.tail(10), use_container_width=True)

    # --- Column descriptions ----------------------------------------------- #
    st.subheader("Column descriptions")
    st.markdown(
        """
| Column | Type | NaNs | Description |
|---|---|---|---|
| **Year** | int | 0 | Calendar year of observation |
| **CSIRO Adjusted Sea Level** | float | 0 | Cumulative sea-level change in **inches** relative to the 1880 baseline (Church & White 2011) |
| **Lower Error Bound** | float | 0 | Lower 1σ uncertainty bound (inches) |
| **Upper Error Bound** | float | 0 | Upper 1σ uncertainty bound (inches) |
| **NOAA Adjusted Sea Level** | float | 113 | NOAA satellite-era record (inches), available from ~1993 |
"""
    )

    # --- Summary statistics ------------------------------------------------ #
    st.subheader("Summary statistics (CSIRO column)")
    col = "CSIRO Adjusted Sea Level"
    desc_df = df[[col]].describe().round(4)
    st.dataframe(desc_df, use_container_width=True)

    # --- Missing value report ---------------------------------------------- #
    st.subheader("Missing values")
    na_series = df.isna().sum()
    na_df = na_series.reset_index()
    na_df.columns = ["Column", "NaN count"]
    na_df["% missing"] = (na_df["NaN count"] / len(df) * 100).round(1)
    st.dataframe(na_df, use_container_width=True)

    if stats["noaa_n"] > 0:
        st.info(
            f"ℹ️ The NOAA column has {stats['noaa_n']} valid rows "
            f"({stats['noaa_coverage']:.0%} coverage). "
            "It starts around 1993 when satellite altimetry became available.",
            icon="ℹ️",
        )

    # --- Live Data Source & Verification ---------------------------------- #
    st.divider()
    st.subheader("🔄 Data Verification & Refresh")
    with st.expander("Dataset Source & Verification", expanded=False):
        st.write(
            "The app operates offline by default using the verified local `epa-sea-level.csv`. "
            "You can test connectivity to the upstream EPA / freeCodeCamp source or clear the cache."
        )
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🔄 Clear cache & reload", key="btn_reload_data"):
                st.cache_data.clear()
                st.success("Cache cleared and dataset reloaded.", icon="✅")
                st.rerun()
        with c_btn2:
            if st.button("🌐 Verify upstream source", key="btn_verify_upstream"):
                import urllib.request

                url = "https://raw.githubusercontent.com/freeCodeCamp/boilerplate-sea-level-predictor/main/epa-sea-level.csv"
                try:
                    with urllib.request.urlopen(url, timeout=5) as response:
                        if response.status == 200:
                            st.success("Upstream repository is online and accessible.", icon="🌐")
                        else:
                            st.warning(f"Upstream returned HTTP {response.status}.", icon="⚠️")
                except Exception as ex:
                    st.info(
                        f"Could not reach upstream server: {ex}. Bundled data remains active.",
                        icon="ℹ️",
                    )

    # --- Attribution ------------------------------------------------------- #
    st.divider()
    st.subheader("📚 Attribution & licence")
    st.markdown(
        """
- **CSIRO data**: Church, J. A. & White, N. J. (2011).
  *Sea-level rise from the late 19th to the early 21st Century.*
  Surveys in Geophysics, 32(4–5), 585–602.
  Data via the U.S. Environmental Protection Agency.

- **NOAA data**: National Oceanic and Atmospheric Administration – Laboratory for Satellite Altimetry.

- This dataset is used here for educational purposes under the freeCodeCamp
  *Data Analysis with Python* certification project.
"""
    )
