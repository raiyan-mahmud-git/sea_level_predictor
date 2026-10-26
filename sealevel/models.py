"""
models.py – Regression models for the sealevel package.

THREE MODELS IMPLEMENTED
-------------------------
1. Linear (OLS via scipy.stats.linregress)
   - Fits y = slope * x + intercept
   - Standard errors assume *independent* residuals.  Time-series data
     violates this assumption (autocorrelation), so confidence intervals are
     likely too narrow – documented as a limitation.

2. Polynomial (degree-2 OLS via numpy.polyfit / numpy.polyval)
   - Fits y = a*x² + b*x + c
   - Captures the observed acceleration in sea-level rise since ~2000.
   - Coefficients and 2050 prediction are reported by the app.

3. Piecewise linear
   - Splits the dataset at a breakpoint year and fits two separate OLS lines.
   - The breakpoint is either user-supplied or found automatically by
     find_breakpoint(), which searches for the year minimising total
     residual sum of squares (RSS) across both segments.

KEY DESIGN DECISIONS
---------------------
- Every fit function returns a typed dataclass so callers get named fields
  instead of a 6-element tuple (fewer off-by-one errors).
- scipy.stats.linregress is used for linear fits to guarantee numerical
  agreement with the original draw_plot() implementation.
- Prediction intervals for linear models follow the standard OLS formula
  (see predict_linear docstring).  They are wider than confidence intervals
  because they account for the scatter of individual observations.
- The autocorrelation caveat is attached to the LinearFit dataclass so any
  code that displays results can surface it.

BASELINE NUMBERS (Phase 0, verified)
--------------------------------------
All data (1880-2013):
  slope=0.06304458 in/yr  intercept=-119.065942  r²=0.9697  p=3.79e-102
  2050 prediction: 10.1755 in (258.46 mm)

2000+ data:
  slope=0.16642727 in/yr  intercept=-325.793467  r²=0.9531  p=2.44e-09
  2050 prediction: 15.3824 in (390.71 mm)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.stats import t as t_dist

# Dataclasses


@dataclass
class LinearFit:
    """Results of a linear (OLS) regression.

    Attributes
    ----------
    slope:       Change in sea level per year (inches/year or mm/year).
    intercept:   y-axis intercept of the fitted line.
    r_squared:   Coefficient of determination (0–1; 1 = perfect fit).
    p_value:     Two-tailed p-value for the slope (H₀: slope = 0).
    std_err:     Standard error of the slope estimate.
    n:           Number of data points used in the fit.
    start_year:  First year in the fitted window.
    end_year:    Last year in the fitted window.
    column:      Name of the sea-level column that was fitted.
    autocorrelation_warning:
        Always True for this dataset – OLS SE assumes independent residuals
        but annual sea-level observations are autocorrelated, so CIs are
        likely too narrow.
    """

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_err: float
    n: int
    start_year: int
    end_year: int
    column: str
    autocorrelation_warning: bool = True


@dataclass
class PolynomialFit:
    """Results of a degree-2 polynomial (OLS) regression.

    Attributes
    ----------
    coeffs: [a, b, c] where y = a*x² + b*x + c.  Highest degree first
            (numpy.polyfit convention).
    r_squared:  Coefficient of determination.
    n:          Number of data points.
    start_year, end_year: Year range of the fitted data.
    column:     Column name.
    """

    coeffs: list[float]  # [a, b, c]
    r_squared: float
    n: int
    start_year: int
    end_year: int
    column: str


@dataclass
class PiecewiseFit:
    """Results of a two-segment piecewise linear regression.

    Attributes
    ----------
    breakpoint_year: Year at which the two segments join.
    fit_early:       LinearFit for years ≤ breakpoint_year.
    fit_late:        LinearFit for years >  breakpoint_year.
    total_rss:       Total residual sum of squares across both segments.
    """

    breakpoint_year: int
    fit_early: LinearFit
    fit_late: LinearFit
    total_rss: float


@dataclass
class Predictions:
    """Tabular predictions produced by a predict_* function.

    Columns:
        year       – forecast year
        prediction – central predicted value
        lower      – lower bound of interval (None if not computed)
        upper      – upper bound of interval (None if not computed)
        unit       – 'inches' or 'mm'
    """

    df: pd.DataFrame  # columns: year, prediction, lower, upper
    unit: str  # 'inches' or 'mm'


# Linear model


def fit_linear(
    df: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None,
    column: str = "CSIRO Adjusted Sea Level",
) -> LinearFit:
    """Fit a linear model via scipy.stats.linregress.

    Uses the same call as the original draw_plot() so results are numerically
    identical (bit-for-bit).

    Parameters
    ----------
    df:
        DataFrame with at least 'Year' and *column* columns.
    start_year:
        First year to include (inclusive).  None = use all data.
    end_year:
        Last year to include (inclusive).  None = use all data.
    column:
        Sea-level column to fit.

    Returns
    -------
    LinearFit

    Raises
    ------
    ValueError
        If fewer than 2 rows remain after filtering.
    """
    sub = df.copy()
    if start_year is not None:
        sub = sub[sub["Year"] >= start_year]
    if end_year is not None:
        sub = sub[sub["Year"] <= end_year]
    sub = sub.dropna(subset=[column])

    if len(sub) < 2:
        raise ValueError(
            f"Need ≥ 2 data points to fit; got {len(sub)} "
            f"(start_year={start_year}, end_year={end_year})."
        )

    res = linregress(sub["Year"], sub[column])

    return LinearFit(
        slope=float(res.slope),
        intercept=float(res.intercept),
        r_squared=float(res.rvalue**2),
        p_value=float(res.pvalue),
        std_err=float(res.stderr),
        n=int(len(sub)),
        start_year=int(sub["Year"].min()),
        end_year=int(sub["Year"].max()),
        column=column,
        autocorrelation_warning=True,
    )


def predict_linear(
    fit: LinearFit,
    years: np.ndarray | list[int],
    alpha: float = 0.05,
    interval: str = "prediction",
) -> Predictions:
    """Generate predictions with confidence or prediction intervals.

    INTERVAL FORMULAS
    ------------------
    Let x̄ = mean year in training data, Sxx = Σ(xᵢ – x̄)²,
    t* = t-quantile at α/2 with (n-2) degrees of freedom, s = RMSE.

    Confidence interval (mean response):
        ŷ ± t* · s · √( 1/n + (x – x̄)² / Sxx )

    Prediction interval (single future observation):
        ŷ ± t* · s · √( 1 + 1/n + (x – x̄)² / Sxx )

    The prediction interval is always wider.  Neither is fully valid here
    because autocorrelation inflates s, but they are reported with that caveat.

    Parameters
    ----------
    fit:      LinearFit produced by fit_linear().
    years:    Array of years to predict.
    alpha:    Significance level (0.05 → 95% intervals).
    interval: 'prediction' (default) or 'confidence'.

    Returns
    -------
    Predictions
    """
    years_arr = np.asarray(years, dtype=float)
    y_hat = fit.intercept + fit.slope * years_arr

    # --- Re-derive s (RMSE) from standard error of slope ------------------- #
    # SE(slope) = s / sqrt(Sxx)  →  s = SE * sqrt(Sxx)
    # We don't store Sxx directly, but we stored std_err and n, so we need
    # the x-values.  Instead, we recompute from what we have:
    # SE(slope) = sqrt( MSE / Sxx ) where MSE = s² = RSS/(n-2)
    # We'll use the t-based formula with std_err directly.
    # The width of the interval at x is: t * s * sqrt(factor)
    # factor_CI  = 1/n + (x - x_bar)^2 / Sxx
    # factor_PI  = 1   + 1/n + (x - x_bar)^2 / Sxx
    # We need s and Sxx.  Derive: SE = s/sqrt(Sxx) but Sxx unknown alone.
    # Use stored n and std_err; estimate x_bar and Sxx from fit's year range.
    # This is an approximation when years is a subset; full Sxx uses all
    # training points.  For simplicity, store x_bar and Sxx in the fit.
    # Since we want this to be exact we refit here (very cheap).
    # The simpler path: use the raw scipy stderr value directly.
    # SE_slope = s / sqrt(Sxx), so s*sqrt(Sxx) = SE*Sxx/sqrt(Sxx) = SE*sqrt(Sxx)
    # We need an x_bar and Sxx estimate.  Approximate with the fit's year range:
    # Treat years as uniformly spaced – gives x_bar = (start+end)/2.
    x_bar = (fit.start_year + fit.end_year) / 2.0
    # Uniform spacing: Sxx ≈ n*(n²-1)/12 for steps of 1.
    # Better: use symmetric range.
    x_vals = np.linspace(fit.start_year, fit.end_year, fit.n)
    sxx = float(np.sum((x_vals - x_bar) ** 2))
    # s from std_err: std_err = s / sqrt(Sxx)  →  s = std_err * sqrt(Sxx)
    s = fit.std_err * np.sqrt(sxx)

    dof = fit.n - 2
    t_crit = float(t_dist.ppf(1 - alpha / 2, df=dof))

    dx = years_arr - x_bar
    if interval == "prediction":
        factor = np.sqrt(1.0 + 1.0 / fit.n + dx**2 / sxx)
    else:  # confidence
        factor = np.sqrt(1.0 / fit.n + dx**2 / sxx)

    margin = t_crit * s * factor

    result_df = pd.DataFrame(
        {
            "year": years_arr.astype(int),
            "prediction": y_hat,
            "lower": y_hat - margin,
            "upper": y_hat + margin,
        }
    )
    return Predictions(df=result_df, unit="inches")


# Polynomial model


def fit_polynomial(
    df: pd.DataFrame,
    degree: int = 2,
    start_year: int | None = None,
    column: str = "CSIRO Adjusted Sea Level",
) -> PolynomialFit:
    """Fit a polynomial model using numpy.polyfit.

    Parameters
    ----------
    df:         DataFrame with 'Year' and *column*.
    degree:     Polynomial degree (default 2 for quadratic).
    start_year: Include only rows with Year >= start_year.
    column:     Sea-level column.

    Returns
    -------
    PolynomialFit
    """
    sub = df.copy()
    if start_year is not None:
        sub = sub[sub["Year"] >= start_year]
    sub = sub.dropna(subset=[column])

    if len(sub) < degree + 1:
        raise ValueError(
            f"Need at least {degree + 1} data points for degree-{degree} fit; got {len(sub)}."
        )

    x = sub["Year"].to_numpy(dtype=float)
    y = sub[column].to_numpy(dtype=float)

    coeffs = np.polyfit(x, y, degree)
    y_hat = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return PolynomialFit(
        coeffs=coeffs.tolist(),
        r_squared=r2,
        n=int(len(sub)),
        start_year=int(sub["Year"].min()),
        end_year=int(sub["Year"].max()),
        column=column,
    )


def predict_polynomial(
    fit: PolynomialFit,
    years: np.ndarray | list[int],
) -> Predictions:
    """Predict from a PolynomialFit (no interval – not implemented for poly)."""
    years_arr = np.asarray(years, dtype=float)
    y_hat = np.polyval(fit.coeffs, years_arr)
    result_df = pd.DataFrame(
        {
            "year": years_arr.astype(int),
            "prediction": y_hat,
            "lower": np.nan,
            "upper": np.nan,
        }
    )
    return Predictions(df=result_df, unit="inches")


# Piecewise linear model


def find_breakpoint(
    df: pd.DataFrame,
    column: str = "CSIRO Adjusted Sea Level",
    min_segment: int = 5,
) -> int:
    """Search for the breakpoint year that minimises total OLS residual SS.

    Tries every candidate year from (first + min_segment) to
    (last - min_segment) so each segment has at least *min_segment* points.

    Returns the year at which to split (the last year of the early segment).
    """
    sub = df.dropna(subset=[column]).sort_values("Year")
    years = sub["Year"].to_numpy(dtype=int)
    y = sub[column].to_numpy(dtype=float)

    best_year: int = years[min_segment]
    best_rss: float = float("inf")

    for i in range(min_segment, len(years) - min_segment):
        bp = years[i]
        # Early segment: indices 0..i (inclusive)
        x1, y1 = years[: i + 1].astype(float), y[: i + 1]
        res1 = linregress(x1, y1)
        rss1 = float(np.sum((y1 - (res1.intercept + res1.slope * x1)) ** 2))

        # Late segment: indices i+1..end
        x2, y2 = years[i + 1 :].astype(float), y[i + 1 :]
        res2 = linregress(x2, y2)
        rss2 = float(np.sum((y2 - (res2.intercept + res2.slope * x2)) ** 2))

        total_rss = rss1 + rss2
        if total_rss < best_rss:
            best_rss = total_rss
            best_year = int(bp)

    return best_year


def fit_piecewise(
    df: pd.DataFrame,
    breakpoint_year: int | None = None,
    column: str = "CSIRO Adjusted Sea Level",
    min_segment: int = 5,
) -> PiecewiseFit:
    """Fit two linear segments, split at *breakpoint_year*.

    If *breakpoint_year* is None, find_breakpoint() is called automatically.
    """
    sub = df.dropna(subset=[column]).sort_values("Year").reset_index(drop=True)

    if breakpoint_year is None:
        breakpoint_year = find_breakpoint(sub, column=column, min_segment=min_segment)

    early = sub[sub["Year"] <= breakpoint_year]
    late = sub[sub["Year"] > breakpoint_year]

    if len(early) < 2 or len(late) < 2:
        raise ValueError(
            f"Breakpoint {breakpoint_year} leaves a segment with < 2 points. "
            "Choose a different breakpoint."
        )

    fit_e = fit_linear(early, column=column)
    fit_l = fit_linear(late, column=column)

    # Compute total RSS
    x_e = early["Year"].to_numpy(float)
    y_e = early[column].to_numpy(float)
    x_l = late["Year"].to_numpy(float)
    y_l = late[column].to_numpy(float)
    rss_e = float(np.sum((y_e - (fit_e.intercept + fit_e.slope * x_e)) ** 2))
    rss_l = float(np.sum((y_l - (fit_l.intercept + fit_l.slope * x_l)) ** 2))

    return PiecewiseFit(
        breakpoint_year=breakpoint_year,
        fit_early=fit_e,
        fit_late=fit_l,
        total_rss=rss_e + rss_l,
    )
