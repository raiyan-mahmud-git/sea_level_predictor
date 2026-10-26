"""
views/methodology.py – Methodology and limitations page.

Explains:
- OLS linear regression formula with st.latex
- How prediction intervals are computed
- Limitations: extrapolation, autocorrelation, dataset coverage
"""

from __future__ import annotations

import streamlit as st


def show() -> None:
    st.title("📐 Methodology")
    st.caption(
        "Formulas, assumptions, and limitations – so you know exactly what the numbers mean."
    )

    # -----------------------------------------------------------------------
    # 1. Linear regression
    # -----------------------------------------------------------------------
    st.header("1. Linear Regression (OLS)")
    st.markdown(
        """
We fit a straight line to *n* observations (year xᵢ, sea level yᵢ) using
**Ordinary Least Squares (OLS)**: minimise the sum of squared residuals.
"""
    )
    st.latex(
        r"""
        \hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x
        \quad \text{where} \quad
        \hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
                            {\sum_{i=1}^{n}(x_i - \bar{x})^2},
        \quad
        \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}
        """
    )

    st.markdown("**Goodness of fit – R²:**")
    st.latex(
        r"R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = "
        r"1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}"
    )

    st.markdown("**Standard error of the slope:**")
    st.latex(
        r"\text{SE}(\hat{\beta}_1) = \sqrt{\frac{s^2}{S_{xx}}}, "
        r"\quad s^2 = \frac{SS_{\text{res}}}{n-2}, "
        r"\quad S_{xx} = \sum(x_i - \bar{x})^2"
    )

    # -----------------------------------------------------------------------
    # 2. Prediction intervals
    # -----------------------------------------------------------------------
    st.header("2. Prediction Intervals")
    st.markdown(
        """
A **prediction interval** covers where a *new individual observation* will fall,
as opposed to a **confidence interval** which covers the *mean response*.
Prediction intervals are always wider.
"""
    )
    st.latex(
        r"\hat{y}(x^*) \pm t^*_{n-2,\, \alpha/2} \cdot s \cdot "
        r"\sqrt{1 + \frac{1}{n} + \frac{(x^* - \bar{x})^2}{S_{xx}}}"
    )
    st.markdown(r"For a *confidence interval*, drop the leading $1$ inside the square root.")

    # -----------------------------------------------------------------------
    # 3. Quadratic (polynomial) fit
    # -----------------------------------------------------------------------
    st.header("3. Quadratic (Degree-2) Polynomial Fit")
    st.latex(r"\hat{y} = a x^2 + b x + c")
    st.markdown(
        "Coefficients are found by `numpy.polyfit`, which solves the same least-squares "
        "problem but for a quadratic basis.  Sea-level rise has been accelerating, "
        "so the quadratic term *a* should be positive and significant."
    )

    # -----------------------------------------------------------------------
    # 4. Piecewise linear fit
    # -----------------------------------------------------------------------
    st.header("4. Piecewise Linear Fit")
    st.markdown(
        """
Two straight-line segments joined at a **breakpoint year** $\\tau$:

$$
\\hat{y}(x) = \\begin{cases}
    \\hat{\\beta}_{0,1} + \\hat{\\beta}_{1,1}\\, x & x \\leq \\tau \\\\
    \\hat{\\beta}_{0,2} + \\hat{\\beta}_{1,2}\\, x & x > \\tau
\\end{cases}
$$

The **automatic breakpoint** $\\hat{\\tau}$ is found by a grid search:
$$
\\hat{\\tau} = \\arg\\min_{\\tau} \\left[ SS_{\\text{res},1}(\\tau) + SS_{\\text{res},2}(\\tau) \\right]
$$
Each candidate split must leave at least 5 observations in each segment.
"""
    )

    # -----------------------------------------------------------------------
    # 5. Time-based backtest
    # -----------------------------------------------------------------------
    st.header("5. Time-Based Backtest")
    st.markdown(
        """
For time-series data, a **random** train/test split is invalid: you would
be predicting the past from the future.  Instead we use a **temporal split**:

- **Training set**: all years $< 1993$
- **Test set**: all years $\\geq 1993$

We fit the model on the training set and compute RMSE and MAE on the test set.
This simulates making a prediction in 1993 about what sea level would do over
the next 20 years.
"""
    )
    st.latex(
        r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}, "
        r"\quad \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|"
    )

    # -----------------------------------------------------------------------
    # 6. Limitations
    # -----------------------------------------------------------------------
    st.header("⚠️ Limitations")
    with st.expander("Read the full limitations section", expanded=True):
        st.markdown(
            """
### Extrapolation
All trend lines are **extrapolated beyond the range of the data** (the dataset
ends in 2013; forecasts extend to 2050 or 2100).  Extrapolation assumes the
fitted relationship continues unchanged into the future.  Sea-level rise
depends on greenhouse-gas emissions, ice-sheet dynamics, and ocean heat
content – none of which are captured by a curve fit to historical data.
**Do not use these projections for policy decisions** without consulting
authoritative sources such as the IPCC Sixth Assessment Report.

### Autocorrelated Residuals
OLS standard errors assume that the error terms εᵢ are **independently and
identically distributed** (i.i.d.).  Annual sea-level measurements are
strongly autocorrelated: if this year's level is above the trend, next year's
is likely to be too.  Autocorrelation inflates the apparent fit quality,
causing OLS to **underestimate standard errors** and produce **prediction
intervals that are too narrow**.  Robust alternatives include HAC
(Newey-West) standard errors or time-series models (ARIMA).

### Dataset Coverage
The CSIRO series covers 1880–2013 (134 years).  The NOAA satellite altimetry
series starts around 1993 and has 113 missing rows in the earlier period.
Trends estimated from a single observational record embed systematic
measurement errors and the specific conditions of the stations used.

### Two-point Structural Break
The piecewise model assumes exactly **one breakpoint**.  Real sea-level
acceleration may be more gradual or have multiple inflection points.

### Straight-line Extrapolation Can Mislead
The long-period linear fit (slope ≈ 0.063 in/yr) and the 2000+ linear fit
(slope ≈ 0.166 in/yr) differ by a factor of **2.6×**.  A 2050 prediction
ranges from **10.2 in (258 mm)** to **15.4 in (391 mm)** depending purely on
which window is chosen.  This illustrates how sensitive extrapolations are to
model and window selection.

### This Is an Educational Project
This app was built as part of the freeCodeCamp *Data Analysis with Python*
certification.  It is intended to demonstrate data analysis techniques,
not to provide authoritative sea-level projections.
"""
        )
