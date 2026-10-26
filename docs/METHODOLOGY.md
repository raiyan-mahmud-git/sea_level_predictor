# Methodology – Extended Notes

This document supplements the in-app **Methodology** page with deeper explanations
and references.

---

## Ordinary Least Squares (OLS) Linear Regression

OLS is a method for estimating the coefficients of a linear equation so that the
sum of squared differences between observed and predicted values is minimised.

**Why use OLS here?**
- The original freeCodeCamp script used `scipy.stats.linregress`, which is OLS.
- Keeping OLS ensures numerical compatibility with the legacy test suite.
- OLS has a closed-form solution (no iterative solver needed) and is interpretable.

**Key assumption violated:** OLS assumes errors are *independently and identically
distributed* (i.i.d.).  Annual sea-level data is a time series with strong positive
autocorrelation – an above-average year tends to be followed by another above-average
year.  This makes OLS standard errors too small, prediction intervals too narrow, and
p-values too optimistic.

**Remedies (not implemented, noted as stretch goal):**
- Newey-West (HAC) standard errors
- ARIMA or state-space models
- Block bootstrap confidence intervals

---

## Why the 2000 Cutoff Is Arbitrary

The freeCodeCamp project hard-codes year 2000 as the start of the "recent" window.
This choice is plausible (satellite altimetry improved around then; the rate of rise
visibly accelerated post-2000) but arbitrary.  

The piecewise breakpoint detector in `sealevel.models.find_breakpoint()` searches for
the year that minimises total residual sum of squares across both segments.  The
result depends on the data and is reported to the user.

---

## Model Comparison Rationale

| Model | Strengths | Weaknesses |
|---|---|---|
| Linear (all) | Simple, interpretable | Misses post-2000 acceleration |
| Linear (2000+) | Captures recent rate | Short window (14 pts), unstable |
| Quadratic | Captures acceleration | Unbounded growth, hard to interpret |
| Piecewise | Explicit change-point, two interpretable slopes | Assumes sharp break |

No model captures the physics of sea-level rise.  All are curve-fitting exercises
on a single 134-year record.

---

## References

1. Church, J. A. & White, N. J. (2011). Sea-level rise from the late 19th to the
   early 21st Century. *Surveys in Geophysics*, 32(4–5), 585–602.
2. IPCC (2021). *Climate Change 2021: The Physical Science Basis* (AR6 WGI).
3. Nerem, R. S. et al. (2018). Climate-change–driven accelerated sea-level rise
   detected in the altimeter era. *PNAS*, 115(9), 2022–2025.
4. Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite,
   heteroskedasticity and autocorrelation consistent covariance matrix.
   *Econometrica*, 55(3), 703–708.
