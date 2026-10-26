"""
views/model_lab.py – Model comparison page.

Shows:
- Side-by-side comparison table: Linear (all), Linear (2000+), Quadratic, Piecewise
  – RMSE and MAE from time-based backtest (fit 1880-1993, test 1993-2013)
  – Slope ratio recent vs full
  – R²
- Residual plot for the selected model
- Estimated breakpoint vs the arbitrary 2000 cutoff
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sealevel.data import load_data
from sealevel.metrics import backtest
from sealevel.models import (
    find_breakpoint,
    fit_linear,
    fit_piecewise,
    fit_polynomial,
)
from sealevel.plotting import make_residual_figure
from ui.components import render_error


@st.cache_data(show_spinner="Loading dataset…")
def _load():
    return load_data()


@st.cache_data(show_spinner="Fitting models…")
def _build_comparison_table(cutoff: int = 1993):
    df = _load()
    col = "CSIRO Adjusted Sea Level"
    rows = []

    y_full = df[col].dropna().to_numpy(float)
    ss_tot = float(np.sum((y_full - y_full.mean()) ** 2))

    # --- Linear (all data) ------------------------------------------------ #
    fit_all = fit_linear(df, column=col)
    bt_all = backtest(df, cutoff_year=cutoff, column=col)
    rows.append(
        {
            "Model": f"Linear (all, {fit_all.start_year}–{fit_all.end_year})",
            "R²": f"{fit_all.r_squared:.4f}",
            "Slope (in/yr)": f"{fit_all.slope:.5f}",
            "Backtest RMSE (in)": f"{bt_all.rmse_val:.4f}",
            "Backtest MAE (in)": f"{bt_all.mae_val:.4f}",
            "2050 prediction (in)": f"{fit_all.intercept + fit_all.slope * 2050:.2f}",
        }
    )

    # --- Linear (2000+) --------------------------------------------------- #
    fit_rec = fit_linear(df, start_year=2000, column=col)
    bt_rec = backtest(df, cutoff_year=cutoff, column=col)
    rows.append(
        {
            "Model": f"Linear (2000–{fit_rec.end_year})",
            "R²": f"{fit_rec.r_squared:.4f}",
            "Slope (in/yr)": f"{fit_rec.slope:.5f}",
            "Backtest RMSE (in)": f"{bt_rec.rmse_val:.4f}",
            "Backtest MAE (in)": f"{bt_rec.mae_val:.4f}",
            "2050 prediction (in)": f"{fit_rec.intercept + fit_rec.slope * 2050:.2f}",
        }
    )

    # --- Quadratic --------------------------------------------------------- #
    fit_poly = fit_polynomial(df, degree=2, column=col)
    train = df[df["Year"] < cutoff].dropna(subset=[col])
    test = df[df["Year"] >= cutoff].dropna(subset=[col])
    fp = fit_polynomial(train, degree=2, column=col)
    pred_test = np.polyval(fp.coeffs, test["Year"].to_numpy(float))
    actual_test = test[col].to_numpy(float)
    from sealevel.metrics import mae as _mae
    from sealevel.metrics import rmse as _rmse

    poly_rmse = _rmse(actual_test, pred_test)
    poly_mae = _mae(actual_test, pred_test)
    pred_poly_2050 = float(np.polyval(fit_poly.coeffs, 2050))
    inst_slope = 2 * fit_poly.coeffs[0] * fit_poly.end_year + fit_poly.coeffs[1]
    rows.append(
        {
            "Model": "Quadratic (all data)",
            "R²": f"{fit_poly.r_squared:.4f}",
            "Slope (in/yr)": f"{inst_slope:.5f} (in {fit_poly.end_year})",
            "Backtest RMSE (in)": f"{poly_rmse:.4f}",
            "Backtest MAE (in)": f"{poly_mae:.4f}",
            "2050 prediction (in)": f"{pred_poly_2050:.2f}",
        }
    )

    # --- Piecewise --------------------------------------------------------- #
    try:
        fit_pw = fit_piecewise(df, column=col)
        tr = df[df["Year"] < cutoff].dropna(subset=[col])
        te = df[df["Year"] >= cutoff].dropna(subset=[col])
        fp_pw = fit_piecewise(tr, column=col)
        bp = fp_pw.breakpoint_year
        te_early = te[te["Year"] <= bp]
        te_late = te[te["Year"] > bp]
        pred_pw = np.concatenate(
            [
                fp_pw.fit_early.intercept
                + fp_pw.fit_early.slope * te_early["Year"].to_numpy(float),
                fp_pw.fit_late.intercept + fp_pw.fit_late.slope * te_late["Year"].to_numpy(float),
            ]
        )
        actual_pw = np.concatenate([te_early[col].to_numpy(float), te_late[col].to_numpy(float)])
        pw_rmse = _rmse(actual_pw, pred_pw)
        pw_mae = _mae(actual_pw, pred_pw)
        pred_pw_2050 = fp_pw.fit_late.intercept + fp_pw.fit_late.slope * 2050
        r2_pw = 1.0 - fit_pw.total_rss / ss_tot if ss_tot > 0 else 0.0

        rows.append(
            {
                "Model": f"Piecewise (bp={fit_pw.breakpoint_year})",
                "R²": f"{r2_pw:.4f}",
                "Slope (in/yr)": f"{fit_pw.fit_late.slope:.5f} (late)",
                "Backtest RMSE (in)": f"{pw_rmse:.4f}",
                "Backtest MAE (in)": f"{pw_mae:.4f}",
                "2050 prediction (in)": f"{pred_pw_2050:.2f}",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Model": "Piecewise",
                "R²": f"Error: {exc}",
                "Slope (in/yr)": "—",
                "Backtest RMSE (in)": "—",
                "Backtest MAE (in)": "—",
                "2050 prediction (in)": "—",
            }
        )

    return pd.DataFrame(rows)


def show() -> None:
    st.title("🔬 Model Lab")
    st.caption("Compare models on a time-based backtest (train 1880–1992, test 1993–2013).")

    try:
        df = _load()
    except Exception as exc:
        render_error(exc, "Failed to load dataset")
        return

    col = "CSIRO Adjusted Sea Level"

    # --- Comparison table -------------------------------------------------- #
    st.subheader("Model comparison")
    try:
        table = _build_comparison_table(cutoff=1993)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Backtest: fitted on data up to 1992 (inclusive), evaluated on 1993–2013.  "
            "RMSE and MAE are in inches.  "
            "For the piecewise model the backpoint is re-estimated within the training window."
        )
    except Exception as exc:
        render_error(exc, "Could not build comparison table")

    st.divider()

    # --- Slope ratio ------------------------------------------------------- #
    st.subheader("Slope acceleration")
    try:
        fit_all = fit_linear(df, column=col)
        fit_rec = fit_linear(df, start_year=2000, column=col)
        ratio = fit_rec.slope / fit_all.slope
        st.metric(
            "2000+ slope ÷ 1880+ slope",
            f"×{ratio:.2f}",
            help=(
                "If this is > 1, sea-level rise has accelerated since 2000 "
                "compared with the long-term average."
            ),
        )
        st.progress(min(ratio / 4, 1.0), text=f"Acceleration ratio: {ratio:.2f}×")
    except Exception as exc:
        render_error(exc, "Slope ratio computation failed")

    st.divider()

    # --- Estimated breakpoint ---------------------------------------------- #
    st.subheader("Automatic breakpoint vs arbitrary 2000 cutoff")
    try:
        bp = find_breakpoint(df, column=col, min_segment=5)
        st.metric("Automatically detected breakpoint", bp)
        if bp != 2000:
            diff = bp - 2000
            direction = "earlier" if diff < 0 else "later"
            st.info(
                f"The optimal breakpoint is **{abs(diff)} years {direction}** than the "
                f"arbitrary 2000 cutoff.  The freeCodeCamp project hard-codes 2000; "
                f"the data suggests {bp} is where the slope change is most pronounced.",
                icon="📐",
            )
        else:
            st.success("The automatic breakpoint agrees with the 2000 cutoff.")
    except Exception as exc:
        render_error(exc, "Breakpoint detection failed")

    st.divider()

    # --- Residual plot ----------------------------------------------------- #
    st.subheader("Residual plot (linear fit, all data)")
    try:
        fit_all = fit_linear(df, column=col)
        resid_fig = make_residual_figure(df, fit_all)
        st.plotly_chart(resid_fig, use_container_width=True)
        st.caption(
            "A curved pattern in the residuals indicates the linear model "
            "underestimates sea-level rise in recent decades – consistent with acceleration."
        )
    except Exception as exc:
        render_error(exc, "Residual plot failed")
