"""
tests/test_models.py – Unit tests for sealevel.models

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress

from sealevel.data import load_data
from sealevel.models import (
    find_breakpoint,
    fit_linear,
    fit_piecewise,
    fit_polynomial,
    predict_linear,
    predict_polynomial,
)

# Fixtures


@pytest.fixture(scope="session")
def real_df():
    return load_data()


@pytest.fixture(scope="session")
def fit_all(real_df):
    return fit_linear(real_df)


@pytest.fixture(scope="session")
def fit_2000(real_df):
    return fit_linear(real_df, start_year=2000)


# 1. Agreement with scipy.stats.linregress


class TestLinearFitAgreesWithScipy:
    def test_all_data_slope(self, real_df, fit_all):
        ref = linregress(real_df["Year"], real_df["CSIRO Adjusted Sea Level"])
        assert abs(fit_all.slope - ref.slope) < 1e-10

    def test_all_data_intercept(self, real_df, fit_all):
        ref = linregress(real_df["Year"], real_df["CSIRO Adjusted Sea Level"])
        assert abs(fit_all.intercept - ref.intercept) < 1e-10

    def test_all_data_r2(self, real_df, fit_all):
        ref = linregress(real_df["Year"], real_df["CSIRO Adjusted Sea Level"])
        assert abs(fit_all.r_squared - ref.rvalue**2) < 1e-10

    def test_all_data_pvalue(self, real_df, fit_all):
        ref = linregress(real_df["Year"], real_df["CSIRO Adjusted Sea Level"])
        assert abs(fit_all.p_value - ref.pvalue) < 1e-10

    def test_2000_slope(self, real_df, fit_2000):
        sub = real_df[real_df["Year"] >= 2000]
        ref = linregress(sub["Year"], sub["CSIRO Adjusted Sea Level"])
        assert abs(fit_2000.slope - ref.slope) < 1e-10

    def test_2000_intercept(self, real_df, fit_2000):
        sub = real_df[real_df["Year"] >= 2000]
        ref = linregress(sub["Year"], sub["CSIRO Adjusted Sea Level"])
        assert abs(fit_2000.intercept - ref.intercept) < 1e-10


# 2. Phase-0 baseline regression tests
# Computed independently in Phase 0; tests fail if numbers change.


BASELINE_ALL = {
    "slope": 0.06304458,
    "intercept": -119.065942,
    "r_squared": 0.969747,
    "p_value": 3.79e-102,
    "n": 134,
    "pred_2050": 10.1755,  # inches
    "pred_2050_mm": 258.46,  # mm (rounded to 2 dp)
}

BASELINE_2000 = {
    "slope": 0.16642727,
    "intercept": -325.793467,
    "r_squared": 0.953137,
    "n": 14,  # 2000-2013 inclusive = 14 years
    "pred_2050": 15.3824,  # inches
    "pred_2050_mm": 390.71,  # mm
}


class TestPhase0Baseline:
    def test_all_slope(self, fit_all):
        assert abs(fit_all.slope - BASELINE_ALL["slope"]) < 1e-6

    def test_all_intercept(self, fit_all):
        assert abs(fit_all.intercept - BASELINE_ALL["intercept"]) < 1e-4

    def test_all_r2(self, fit_all):
        assert abs(fit_all.r_squared - BASELINE_ALL["r_squared"]) < 1e-4

    def test_all_n(self, fit_all):
        assert fit_all.n == BASELINE_ALL["n"]

    def test_all_pred_2050_inches(self, fit_all):
        pred = fit_all.intercept + fit_all.slope * 2050
        assert abs(pred - BASELINE_ALL["pred_2050"]) < 1e-3

    def test_all_pred_2050_mm(self, fit_all):
        pred = (fit_all.intercept + fit_all.slope * 2050) * 25.4
        assert abs(pred - BASELINE_ALL["pred_2050_mm"]) < 0.05

    def test_2000_slope(self, fit_2000):
        assert abs(fit_2000.slope - BASELINE_2000["slope"]) < 1e-6

    def test_2000_intercept(self, fit_2000):
        assert abs(fit_2000.intercept - BASELINE_2000["intercept"]) < 1e-4

    def test_2000_r2(self, fit_2000):
        assert abs(fit_2000.r_squared - BASELINE_2000["r_squared"]) < 1e-4

    def test_2000_pred_2050_inches(self, fit_2000):
        pred = fit_2000.intercept + fit_2000.slope * 2050
        assert abs(pred - BASELINE_2000["pred_2050"]) < 1e-3

    def test_2000_pred_2050_mm(self, fit_2000):
        pred = (fit_2000.intercept + fit_2000.slope * 2050) * 25.4
        assert abs(pred - BASELINE_2000["pred_2050_mm"]) < 0.05


# 3. Synthetic data with known slope/intercep


def make_synthetic(slope: float, intercept: float, n: int = 50) -> pd.DataFrame:

    years = np.arange(1950, 1950 + n, dtype=int)
    levels = intercept + slope * years.astype(float)
    return pd.DataFrame({"Year": years, "CSIRO Adjusted Sea Level": levels})


class TestSyntheticData:
    def test_exact_slope_recovery(self):
        known_slope = 0.08
        known_intercept = -150.0
        df = make_synthetic(known_slope, known_intercept)
        fit = fit_linear(df)
        assert abs(fit.slope - known_slope) < 1e-10
        assert abs(fit.intercept - known_intercept) < 1e-6

    def test_r2_is_one_for_perfect_data(self):
        df = make_synthetic(0.05, -90.0)
        fit = fit_linear(df)
        assert abs(fit.r_squared - 1.0) < 1e-10

    def test_n_matches_input(self):
        df = make_synthetic(0.05, -90.0, n=30)
        fit = fit_linear(df)
        assert fit.n == 30


# 4. Polynomial fit


class TestPolynomialFit:
    def test_fit_quadratic_on_real_data(self, real_df):
        fit = fit_polynomial(real_df, degree=2)
        assert len(fit.coeffs) == 3
        assert fit.r_squared > 0.97

    def test_prediction_at_2050_is_reasonable(self, real_df):
        fit = fit_polynomial(real_df, degree=2)
        pred = predict_polynomial(fit, [2050])
        val = pred.df["prediction"].iloc[0]
        # Should be positive and within a plausible range
        assert 5 < val < 50

    def test_quadratic_fits_parabola_exactly(self):
        x = np.arange(1900, 1950, dtype=float)
        y = 0.001 * x**2 - 3 * x + 2000.0
        df = pd.DataFrame({"Year": x.astype(int), "CSIRO Adjusted Sea Level": y})
        fit = fit_polynomial(df, degree=2)
        assert abs(fit.r_squared - 1.0) < 1e-6


# 5 & 6. Piecewise fit with known breakpoint


def make_piecewise_data(bp: int = 1950) -> pd.DataFrame:
    """Synthetic two-slope data with a hard kink at *bp*."""
    years1 = np.arange(1880, bp + 1, dtype=int)
    years2 = np.arange(bp + 1, 2014, dtype=int)
    y1 = 0.03 * years1.astype(float) - 56.0  # slow rise
    # Ensure continuity at breakpoint
    y_bp = 0.03 * bp - 56.0
    y2 = y_bp + 0.15 * (years2.astype(float) - bp)  # fast rise
    years = np.concatenate([years1, years2])
    levels = np.concatenate([y1, y2])
    return pd.DataFrame({"Year": years, "CSIRO Adjusted Sea Level": levels})


class TestPiecewiseFit:
    def test_find_breakpoint_recovers_known_kink(self):
        known_bp = 1950
        df = make_piecewise_data(bp=known_bp)
        bp = find_breakpoint(df, min_segment=5)
        # Allow ±2 years tolerance
        assert abs(bp - known_bp) <= 2

    def test_fit_piecewise_slopes_correct(self):
        known_bp = 1950
        df = make_piecewise_data(bp=known_bp)
        fit = fit_piecewise(df, breakpoint_year=known_bp)
        assert abs(fit.fit_early.slope - 0.03) < 0.001
        assert abs(fit.fit_late.slope - 0.15) < 0.001

    def test_fit_piecewise_on_real_data_returns_valid(self, real_df):
        fit = fit_piecewise(real_df)
        assert isinstance(fit.breakpoint_year, int)
        assert fit.fit_early.n >= 5
        assert fit.fit_late.n >= 5
        assert fit.total_rss > 0


# 7. predict_linear – intervals


class TestPredictLinear:
    def test_prediction_interval_wider_than_confidence(self, fit_all):
        years = np.array([2050])
        pi = predict_linear(fit_all, years, interval="prediction")
        ci = predict_linear(fit_all, years, interval="confidence")
        pi_width = float((pi.df["upper"] - pi.df["lower"]).iloc[0])
        ci_width = float((ci.df["upper"] - ci.df["lower"]).iloc[0])
        assert pi_width > ci_width

    def test_prediction_monotone_with_year(self, fit_all):
        # Future predictions should increase for positive slope
        years = np.arange(2020, 2060)
        pred = predict_linear(fit_all, years)
        preds = pred.df["prediction"].to_numpy()
        assert (np.diff(preds) > 0).all()

    def test_central_prediction_matches_formula(self, fit_all):
        year = 2050
        expected = fit_all.intercept + fit_all.slope * year
        pred = predict_linear(fit_all, [year])
        assert abs(float(pred.df["prediction"].iloc[0]) - expected) < 1e-8
