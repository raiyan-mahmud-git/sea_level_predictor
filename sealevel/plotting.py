"""
plotting.py – Plotly chart builders for the sealevel package.

All functions in this module return plotly.graph_objects.Figure objects.
They are free of Streamlit calls so they can be tested independently and
reused from other frontends.

DESIGN DECISIONS
-----------------
- Colorblind-safe palette: IBM Carbon palette (blue #648FFF, orange #FE6100,
  purple #785EF0, green #009E73, red #DC267F).
- Axis labels always include units.
- All chart titles and labels are generated from actual data values, never
  hard-coded strings like "1880-2050".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sealevel.models import LinearFit, PiecewiseFit, Predictions

# Colorblind-safe IBM Carbon palette
COLORS = {
    "scatter": "#648FFF",  # blue – raw data points
    "fit_all": "#FE6100",  # orange – full-period line
    "fit_recent": "#785EF0",  # purple – recent-period line
    "fit_poly": "#009E73",  # green – polynomial
    "fit_piece": "#DC267F",  # red – piecewise
    "interval": "rgba(100,143,255,0.15)",  # light blue fill
    "error_bound": "rgba(150,150,150,0.25)",
}


def make_trend_figure(
    df: pd.DataFrame,
    fit_all: LinearFit,
    fit_recent: LinearFit,
    horizon: int = 2050,
    unit: str = "inches",
    show_intervals: bool = True,
    show_error_bounds: bool = True,
    predictions_all: Predictions | None = None,
    predictions_recent: Predictions | None = None,
) -> go.Figure:
    """Build the main Trend Explorer figure.

    Parameters
    ----------
    df:                  Full dataset from load_data().
    fit_all:             LinearFit on the full dataset.
    fit_recent:          LinearFit on the recent window.
    horizon:             Last forecast year (e.g. 2050).
    unit:                'inches' or 'mm'.
    show_intervals:      Whether to shade prediction intervals.
    show_error_bounds:   Whether to shade the CSV's own error bounds.
    predictions_all:     Pre-computed Predictions from predict_linear().
    predictions_recent:  Pre-computed Predictions from predict_linear().

    Returns
    -------
    plotly.graph_objects.Figure
    """
    scale = 25.4 if unit == "mm" else 1.0
    unit_label = "mm" if unit == "mm" else "inches"
    col = fit_all.column

    fig = go.Figure()

    # Error bounds from CSV (if available and requested)
    if (
        show_error_bounds
        and "Lower Error Bound" in df.columns
        and "Upper Error Bound" in df.columns
    ):
        lo = df["Lower Error Bound"] * scale
        hi = df["Upper Error Bound"] * scale
        fig.add_trace(
            go.Scatter(
                x=pd.concat([df["Year"], df["Year"][::-1]]),
                y=pd.concat([hi, lo[::-1]]),
                fill="toself",
                fillcolor=COLORS["error_bound"],
                line=dict(color="rgba(0,0,0,0)"),
                name="CSIRO error bounds",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # Scatter: raw observations
    fig.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df[col] * scale,
            mode="markers",
            marker=dict(size=5, color=COLORS["scatter"], opacity=0.7),
            name=f"Observed ({col})",
            hovertemplate="Year: %{x}<br>Level: %{y:.3f} " + unit_label + "<extra></extra>",
        )
    )

    # NOAA series (if available)
    noaa_col = "NOAA Adjusted Sea Level"
    if noaa_col in df.columns:
        noaa = df.dropna(subset=[noaa_col])
        if not noaa.empty:
            fig.add_trace(
                go.Scatter(
                    x=noaa["Year"],
                    y=noaa[noaa_col] * scale,
                    mode="markers",
                    marker=dict(size=5, color="#DC267F", opacity=0.5, symbol="diamond"),
                    name="NOAA Adjusted",
                    hovertemplate="Year: %{x}<br>NOAA: %{y:.3f} " + unit_label + "<extra></extra>",
                )
            )

    # Full-period linear trend
    if predictions_all is not None:
        pred_df = predictions_all.df
        y_all = pred_df["prediction"].to_numpy() * scale
        yr_all = pred_df["year"].to_numpy()

        if show_intervals and not pred_df["lower"].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([yr_all, yr_all[::-1]]),
                    y=np.concatenate(
                        [
                            pred_df["upper"].to_numpy() * scale,
                            pred_df["lower"].to_numpy()[::-1] * scale,
                        ]
                    ),
                    fill="toself",
                    fillcolor="rgba(254,97,0,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{fit_all.start_year}–{horizon} PI",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=yr_all,
                y=y_all,
                mode="lines",
                line=dict(color=COLORS["fit_all"], width=2),
                name=(
                    f"{fit_all.start_year}–{horizon} trend  "
                    f"(slope={fit_all.slope * scale:.4f} {unit_label}/yr)"
                ),
                hovertemplate="Year: %{x}<br>Trend: %{y:.3f} " + unit_label + "<extra></extra>",
            )
        )

    # Recent linear trend
    if predictions_recent is not None:
        pred_r = predictions_recent.df
        y_rec = pred_r["prediction"].to_numpy() * scale
        yr_rec = pred_r["year"].to_numpy()

        if show_intervals and not pred_r["lower"].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([yr_rec, yr_rec[::-1]]),
                    y=np.concatenate(
                        [
                            pred_r["upper"].to_numpy() * scale,
                            pred_r["lower"].to_numpy()[::-1] * scale,
                        ]
                    ),
                    fill="toself",
                    fillcolor="rgba(120,94,240,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{fit_recent.start_year}–{horizon} PI",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=yr_rec,
                y=y_rec,
                mode="lines",
                line=dict(color=COLORS["fit_recent"], width=2, dash="dash"),
                name=(
                    f"{fit_recent.start_year}–{horizon} trend  "
                    f"(slope={fit_recent.slope * scale:.4f} {unit_label}/yr)"
                ),
                hovertemplate="Year: %{x}<br>Trend: %{y:.3f} " + unit_label + "<extra></extra>",
            )
        )

    # Layout
    fig.update_layout(
        title=dict(text="Sea Level Rise", font=dict(size=18)),
        xaxis=dict(title="Year", gridcolor="#e0e0e0"),
        yaxis=dict(title=f"Sea Level Change ({unit_label})", gridcolor="#e0e0e0"),
        legend=dict(orientation="v", x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        hovermode="x unified",
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def make_residual_figure(
    df: pd.DataFrame,
    fit: LinearFit,
) -> go.Figure:
    """Plot residuals (observed – predicted) vs year.

    A pattern in residuals indicates the model is missing something
    (e.g. the quadratic acceleration visible in sea-level data).
    """
    col = fit.column
    sub = df.dropna(subset=[col]).sort_values("Year")
    x = sub["Year"].to_numpy(float)
    y = sub[col].to_numpy(float)
    y_hat = fit.intercept + fit.slope * x
    resid = y - y_hat

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["Year"],
            y=resid,
            mode="markers",
            marker=dict(color=COLORS["scatter"], size=6),
            name="Residual",
            hovertemplate="Year: %{x}<br>Residual: %{y:.4f} in<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Residuals (Observed − Predicted)",
        xaxis_title="Year",
        yaxis_title="Residual (inches)",
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
    )
    return fig


def make_piecewise_figure(
    df: pd.DataFrame,
    fit: PiecewiseFit,
    horizon: int = 2050,
    unit: str = "inches",
) -> go.Figure:
    """Plot piecewise linear fit with two distinct segments."""
    scale = 25.4 if unit == "mm" else 1.0
    unit_label = "mm" if unit == "mm" else "inches"
    col = fit.fit_early.column

    fig = go.Figure()

    # Scatter
    fig.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df[col] * scale,
            mode="markers",
            marker=dict(size=5, color=COLORS["scatter"], opacity=0.7),
            name="Observed",
        )
    )

    # Early segment
    yr_e = np.arange(fit.fit_early.start_year, fit.breakpoint_year + 1, dtype=float)
    y_e = fit.fit_early.intercept + fit.fit_early.slope * yr_e
    fig.add_trace(
        go.Scatter(
            x=yr_e,
            y=y_e * scale,
            mode="lines",
            line=dict(color=COLORS["fit_all"], width=2),
            name=f"{fit.fit_early.start_year}–{fit.breakpoint_year}",
        )
    )

    # Late segment extended to horizon
    yr_l = np.arange(fit.breakpoint_year + 1, horizon + 1, dtype=float)
    y_l = fit.fit_late.intercept + fit.fit_late.slope * yr_l
    fig.add_trace(
        go.Scatter(
            x=yr_l,
            y=y_l * scale,
            mode="lines",
            line=dict(color=COLORS["fit_recent"], width=2, dash="dash"),
            name=f"{fit.breakpoint_year + 1}–{horizon}",
        )
    )

    # Breakpoint marker
    bp_y = fit.fit_early.intercept + fit.fit_early.slope * fit.breakpoint_year
    fig.add_trace(
        go.Scatter(
            x=[fit.breakpoint_year],
            y=[bp_y * scale],
            mode="markers",
            marker=dict(color=COLORS["fit_piece"], size=10, symbol="x"),
            name=f"Breakpoint ({fit.breakpoint_year})",
        )
    )

    fig.update_layout(
        title="Piecewise Linear Fit",
        xaxis_title="Year",
        yaxis_title=f"Sea Level Change ({unit_label})",
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
    )
    return fig
