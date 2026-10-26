"""
tests/test_metrics.py – Unit tests for sealevel.metrics
"""

from __future__ import annotations

import numpy as np
import pytest

from sealevel.data import load_data
from sealevel.metrics import BacktestResult, backtest, mae, rmse


@pytest.fixture(scope="module")
def real_df():
    return load_data()


class TestRmse:
    def test_perfect_prediction(self):
        a = np.array([1.0, 2.0, 3.0])
        assert rmse(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        actual = np.array([0.0, 0.0])
        predicted = np.array([3.0, 4.0])
        # sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        assert rmse(actual, predicted) == pytest.approx(3.5355, rel=1e-3)

    def test_empty_returns_nan(self):
        assert np.isnan(rmse(np.array([]), np.array([])))


class TestMae:
    def test_perfect_prediction(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mae(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        actual = np.array([0.0, 0.0, 0.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert mae(actual, predicted) == pytest.approx(2.0)


class TestBacktest:
    def test_returns_backtest_result(self, real_df):
        result = backtest(real_df, cutoff_year=2000)
        assert isinstance(result, BacktestResult)

    def test_train_test_split_is_correct(self, real_df):
        result = backtest(real_df, cutoff_year=2000)
        assert result.train_end < 2000
        assert result.test_start >= 2000

    def test_rmse_is_positive(self, real_df):
        result = backtest(real_df, cutoff_year=2000)
        assert result.rmse_val > 0

    def test_not_enough_train_data_raises(self, real_df):
        with pytest.raises(ValueError, match="Not enough training"):
            backtest(real_df, cutoff_year=1881)  # only 1 training row (1880)

    def test_no_test_data_raises(self, real_df):
        with pytest.raises(ValueError, match="No test data"):
            backtest(real_df, cutoff_year=2014)  # dataset ends at 2013
