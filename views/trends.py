"""
views/trends.py – Trend Explorer page.

Provides the main interactive analysis:
- Sidebar controls (column, start year, horizon, units, model)
- Interactive Plotly chart
- Metric cards (slope, R², p-value, prediction at horizon)
- Download buttons (PNG, CSV)
- Query-param sync for shareable URLs
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from sealevel.data import filter_by_year, load_data
from sealevel.errors import SeaLevelError
from sealevel.models import (
    fit_linear,
    fit_piecewise,
    fit_polynomial,
    predict_linear,
)
from sealevel.plotting import (
    make_piecewise_figure,
    make_trend_figure,
)
from ui.components import (
    SidebarConfig,
    render_error,
    render_fit_stats,
    render_metric_cards,
    render_sidebar,
)


@st.cache_data(show_spinner="Loading dataset…")
def _load():
    return load_data()


def show() -> None:
    st.title("📈 Trend Explorer")
    st.caption(
        "Fit linear, quadratic, or piecewise regression models and extrapolate to your chosen horizon."
    )

    # --- Load data --------------------------------------------------------- #
    try:
        df = _load()
    except Exception as exc:
        render_error(exc, "Failed to load dataset")
        return

    first_year = int(df["Year"].min())
    last_year = int(df["Year"].max())

    # Available numeric columns for analysis
    numeric_cols = [
        c for c in ["CSIRO Adjusted Sea Level", "NOAA Adjusted Sea Level"] if c in df.columns
    ]

    # --- Sidebar ----------------------------------------------------------- #
    cfg: SidebarConfig = render_sidebar(
        first_year=first_year,
        last_year=last_year,
        available_columns=numeric_cols,
    )

    # --- Sync query params ------------------------------------------------- #
    try:
        st.query_params.update(
            {
                "col": cfg.column,
                "start": str(cfg.start_year),
                "horizon": str(cfg.horizon),
                "unit": cfg.unit,
                "model": cfg.model,
            }
        )
    except Exception:
        pass  # query param sync is best-effort

    # --- Fit models -------------------------------------------------------- #
    try:
        # Full-period linear fit (always computed – used as baseline)
        fit_all = fit_linear(df, column=cfg.column)

        # Filter to recent window
        df_recent = filter_by_year(df, start_year=cfg.start_year, column=cfg.column)
        fit_recent = fit_linear(df_recent, column=cfg.column)

    except SeaLevelError as exc:
        render_error(exc, "Fitting error")
        return
    except Exception as exc:
        render_error(exc, "Unexpected error during model fitting")
        return

    # --- Predictions ------------------------------------------------------- #
    years_all = np.arange(first_year, cfg.horizon + 1)
    years_recent = np.arange(cfg.start_year, cfg.horizon + 1)

    preds_all = predict_linear(fit_all, years_all, interval="prediction")
    preds_recent = predict_linear(fit_recent, years_recent, interval="prediction")

    pred_all_horizon = float(fit_all.intercept + fit_all.slope * cfg.horizon)
    pred_recent_horizon = float(fit_recent.intercept + fit_recent.slope * cfg.horizon)

    # --- Metric cards ------------------------------------------------------ #
    render_metric_cards(
        fit_all=fit_all,
        fit_recent=fit_recent,
        pred_all_at_horizon=pred_all_horizon,
        pred_recent_at_horizon=pred_recent_horizon,
        horizon=cfg.horizon,
        unit=cfg.unit,
    )

    # --- Chart ------------------------------------------------------------- #
    st.subheader("Chart")

    if cfg.model == "Linear":
        fig = make_trend_figure(
            df=df,
            fit_all=fit_all,
            fit_recent=fit_recent,
            horizon=cfg.horizon,
            unit=cfg.unit,
            show_intervals=cfg.show_intervals,
            show_error_bounds=cfg.show_error_bounds,
            predictions_all=preds_all,
            predictions_recent=preds_recent,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif cfg.model == "Quadratic":
        try:
            fit_poly = fit_polynomial(df, degree=2, column=cfg.column)
            fit_poly_recent = fit_polynomial(df_recent, degree=2, column=cfg.column)
        except Exception as exc:
            render_error(exc, "Polynomial fit failed")
            return

        import plotly.graph_objects as go

        scale = 25.4 if cfg.unit == "mm" else 1.0
        unit_label = cfg.unit
        fig = go.Figure()
        # Scatter
        fig.add_scatter(
            x=df["Year"],
            y=df[cfg.column] * scale,
            mode="markers",
            marker=dict(size=5, color="#648FFF", opacity=0.7),
            name="Observed",
        )
        # Full polynomial
        yr_poly = np.arange(first_year, cfg.horizon + 1, dtype=float)
        y_poly = np.polyval(fit_poly.coeffs, yr_poly) * scale
        fig.add_scatter(
            x=yr_poly,
            y=y_poly,
            mode="lines",
            line=dict(color="#FE6100", width=2),
            name=f"Quadratic ({first_year}–{cfg.horizon})",
        )
        # Recent polynomial
        yr_poly_r = np.arange(cfg.start_year, cfg.horizon + 1, dtype=float)
        y_poly_r = np.polyval(fit_poly_recent.coeffs, yr_poly_r) * scale
        fig.add_scatter(
            x=yr_poly_r,
            y=y_poly_r,
            mode="lines",
            line=dict(color="#785EF0", width=2, dash="dash"),
            name=f"Quadratic ({cfg.start_year}–{cfg.horizon})",
        )
        fig.update_layout(
            title="Sea Level Rise – Quadratic Fit",
            xaxis_title="Year",
            yaxis_title=f"Sea Level Change ({unit_label})",
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stats for polynomial
        with st.expander("Quadratic fit statistics"):
            c1, c2 = st.columns(2)
            c1.metric("R² (full period)", f"{fit_poly.r_squared:.4f}")
            c2.metric("R² (recent window)", f"{fit_poly_recent.r_squared:.4f}")
            pred_poly_horizon = float(np.polyval(fit_poly.coeffs, cfg.horizon))
            pred_poly_r_horizon = float(np.polyval(fit_poly_recent.coeffs, cfg.horizon))
            c1.metric(
                f"Prediction at {cfg.horizon} ({unit_label})", f"{pred_poly_horizon * scale:.2f}"
            )
            c2.metric(
                f"Prediction at {cfg.horizon} (recent, {unit_label})",
                f"{pred_poly_r_horizon * scale:.2f}",
            )

    elif cfg.model == "Piecewise":
        try:
            fit_pw = fit_piecewise(df, column=cfg.column)
        except Exception as exc:
            render_error(exc, "Piecewise fit failed")
            return

        fig = make_piecewise_figure(df=df, fit=fit_pw, horizon=cfg.horizon, unit=cfg.unit)
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            f"🔍 Automatically detected breakpoint: **{fit_pw.breakpoint_year}**  \n"
            f"Early slope: {fit_pw.fit_early.slope:.5f} {cfg.unit}/yr  \n"
            f"Late slope:  {fit_pw.fit_late.slope:.5f} {cfg.unit}/yr  \n"
            f"Slope acceleration: ×{fit_pw.fit_late.slope / fit_pw.fit_early.slope:.1f}",
            icon="📐",
        )

    # --- Fit statistics expander ------------------------------------------ #
    render_fit_stats(fit_all, label=f"Full-period linear fit ({first_year}–{last_year})")
    render_fit_stats(fit_recent, label=f"Recent linear fit ({cfg.start_year}–{last_year})")

    # --- Download buttons -------------------------------------------------- #
    st.divider()
    st.subheader("⬇️ Downloads")

    dcols = st.columns(2)

    # Predictions CSV
    pred_csv_df = preds_all.df.copy()
    pred_csv_df.columns = [
        "year",
        f"prediction_{cfg.unit}",
        f"lower_{cfg.unit}",
        f"upper_{cfg.unit}",
    ]
    if cfg.unit == "mm":
        for c in pred_csv_df.columns[1:]:
            pred_csv_df[c] = pred_csv_df[c] * 25.4
    csv_bytes = pred_csv_df.to_csv(index=False).encode()

    with dcols[0]:
        st.download_button(
            label="📄 Download predictions (CSV)",
            data=csv_bytes,
            file_name=f"sea_level_predictions_{cfg.start_year}_{cfg.horizon}_{cfg.unit}.csv",
            mime="text/csv",
            key="dl_csv",
        )

    # Chart PNG
    with dcols[1]:
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
            st.download_button(
                label="🖼️ Download chart (PNG)",
                data=img_bytes,
                file_name=f"sea_level_chart_{cfg.model.lower()}_{cfg.unit}.png",
                mime="image/png",
                key="dl_png",
            )
        except Exception:
            st.info("Install `kaleido` (`pip install kaleido`) to enable PNG export.")
