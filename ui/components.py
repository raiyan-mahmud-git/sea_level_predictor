"""
ui/components.py – Reusable Streamlit UI components for the sealevel app.

Keeps all widget definitions in one place so the view pages stay readable.
Each function renders one logical UI chunk and returns the user's current
selection via its return value.

DESIGN NOTES
------------
- Use keyword argument `key=` for all widgets so state survives page
  navigation and query-param sync works correctly.
- Labels always include units to reduce cognitive load.
- Error display wraps st.error() – never bare tracebacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


# Sidebar control panel



@dataclass
class SidebarConfig:
    """User-selected parameters returned by render_sidebar()."""

    column: str  # sea-level column to analyse
    start_year: int  # first year of the "recent" window
    horizon: int  # forecast horizon year
    unit: str  # 'inches' or 'mm'
    model: str  # 'Linear', 'Quadratic', or 'Piecewise'
    show_intervals: bool  # show prediction intervals on chart
    show_error_bounds: bool  # show CSV error bounds on chart


def render_sidebar(
    first_year: int = 1880,
    last_year: int = 2013,
    available_columns: list[str] | None = None,
) -> SidebarConfig:
    
    if available_columns is None:
        available_columns = ["CSIRO Adjusted Sea Level"]

    with st.sidebar:
        st.header("⚙️ Controls")

        column = st.selectbox(
            "Sea-level series",
            options=available_columns,
            index=0,
            key="sel_column",
            help="Choose which sea-level measurement series to analyse.",
        )

        start_year = st.slider(
            "Recent window start year",
            min_value=first_year + 5,  # need ≥ 5 points per segment
            max_value=last_year - 2,
            value=min(2000, last_year - 2),
            step=1,
            key="sel_start_year",
            help=(
                "The 'recent' trend is fit from this year to the last year "
                f"in the dataset ({last_year})."
            ),
        )

        horizon = st.slider(
            "Forecast horizon (year)",
            min_value=last_year + 1,
            max_value=2100,
            value=2050,
            step=1,
            key="sel_horizon",
            help="Extrapolate the fitted lines to this year.",
        )

        unit = st.radio(
            "Display units",
            options=["inches", "mm"],
            index=0,
            key="sel_unit",
            horizontal=True,
            help="1 inch = 25.4 mm",
        )

        model = st.selectbox(
            "Trend model",
            options=["Linear", "Quadratic", "Piecewise"],
            index=0,
            key="sel_model",
            help=(
                "Linear: OLS straight-line fit.  "
                "Quadratic: degree-2 polynomial (captures acceleration).  "
                "Piecewise: two straight lines with automatic breakpoint detection."
            ),
        )

        st.divider()
        st.subheader("Chart options")

        show_intervals = st.toggle(
            "Show prediction intervals",
            value=True,
            key="sel_intervals",
            help=(
                "95 % prediction intervals (shaded bands). "
                "⚠️ These are likely too narrow because OLS assumes "
                "independent residuals, which is violated for annual time series."
            ),
        )

        show_error_bounds = st.toggle(
            "Show dataset error bounds",
            value=True,
            key="sel_err_bounds",
            help="The Lower/Upper Error Bound columns from the original CSV.",
        )

    return SidebarConfig(
        column=column,
        start_year=start_year,
        horizon=horizon,
        unit=unit,
        model=model,
        show_intervals=show_intervals,
        show_error_bounds=show_error_bounds,
    )



# Metric cards

def render_metric_cards(
    fit_all,
    fit_recent,
    pred_all_at_horizon: float,
    pred_recent_at_horizon: float,
    horizon: int,
    unit: str,
) -> None:
    """Render a row of st.metric cards summarising the two linear fits.

    Parameters
    ----------
    fit_all:               LinearFit for the full period.
    fit_recent:            LinearFit for the recent window.
    pred_all_at_horizon:   Predicted value (full fit) at horizon year.
    pred_recent_at_horizon: Predicted value (recent fit) at horizon year.
    horizon:               Forecast year.
    unit:                  'inches' or 'mm'.
    """
    scale = 25.4 if unit == "mm" else 1.0
    unit_label = unit

    cols = st.columns(4)
    with cols[0]:
        st.metric(
            label=f"Full-period slope ({unit_label}/yr)",
            value=f"{fit_all.slope * scale:.5f}",
            help=f"Linear trend from {fit_all.start_year} to {fit_all.end_year}.",
        )
    with cols[1]:
        st.metric(
            label=f"Recent slope ({unit_label}/yr)",
            value=f"{fit_recent.slope * scale:.5f}",
            delta=f"{(fit_recent.slope - fit_all.slope) * scale:+.5f}",
            delta_color="inverse",
            help=f"Linear trend from {fit_recent.start_year} to {fit_recent.end_year}.",
        )
    with cols[2]:
        st.metric(
            label=f"Full-period {horizon} prediction ({unit_label})",
            value=f"{pred_all_at_horizon * scale:.2f}",
            help=f"Predicted sea-level rise above the {fit_all.start_year} baseline at year {horizon}.",
        )
    with cols[3]:
        st.metric(
            label=f"Recent {horizon} prediction ({unit_label})",
            value=f"{pred_recent_at_horizon * scale:.2f}",
            delta=f"{(pred_recent_at_horizon - pred_all_at_horizon) * scale:+.2f} vs full",
            help=(
                f"Predicted sea-level rise above the {fit_all.start_year} baseline at year {horizon} "
                f"using the slope since {fit_recent.start_year}."
            ),
        )


def render_fit_stats(fit, label: str = "Fit statistics") -> None:
    """Render an expander with detailed regression statistics."""
    with st.expander(label, expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "R²", f"{fit.r_squared:.4f}", help="1.0 = perfect fit, 0.0 = no better than the mean."
        )
        c2.metric(
            "p-value",
            f"{fit.p_value:.2e}",
            help="P(slope = 0 | data). Very small = strong evidence of a trend.",
        )
        c3.metric(
            "Std error of slope", f"{fit.std_err:.6f}", help="Standard error of the slope estimate."
        )
        c4.metric("n (data points)", str(fit.n))

        if fit.autocorrelation_warning:
            st.warning(
                "⚠️ **Autocorrelation caveat**: OLS standard errors assume independent "
                "residuals, but annual sea-level data is autocorrelated. "
                "Confidence and prediction intervals are likely **too narrow**. "
                "See Methodology for details.",
                icon="⚠️",
            )


def render_error(error: Exception, context: str = "") -> None:
    """Display a user-friendly error message instead of a raw traceback."""
    msg = str(error)
    if context:
        msg = f"**{context}**: {msg}"
    st.error(msg, icon="🚨")
