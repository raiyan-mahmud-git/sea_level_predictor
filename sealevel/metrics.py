"""
metrics.py – Model evaluation metrics for the sealevel package.

Provides:
- rmse():     Root Mean Squared Error
- mae():      Mean Absolute Error
- backtest(): Time-based train/test split evaluation

TIME-BASED BACKTESTING
-----------------------
For time-series data a random train/test split is wrong: if you train on 1990
and test on 1985 you're predicting the past.  Instead we use an expanding
window: train on [start, cutoff), test on [cutoff, end].  This simulates
making predictions at the cutoff date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sealevel.models import LinearFit, fit_linear

# --------------------------------------------------------------------------- #
# Basic metrics                                                                #
# --------------------------------------------------------------------------- #


def rmse(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Root Mean Squared Error.

    RMSE penalises large errors more than MAE.  Reported in the same units as
    the sea-level column (inches or mm).
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if len(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error.

    MAE is easier to interpret: "on average, predictions are off by X inches."
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.abs(a - p)))


# --------------------------------------------------------------------------- #
# Backtest                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class BacktestResult:
    """Results of a time-based backtest.

    Attributes
    ----------
    train_start, train_end: Year range of the training window.
    test_start,  test_end:  Year range of the test window.
    rmse:   RMSE on the test set (inches).
    mae:    MAE on the test set (inches).
    n_train, n_test: Number of observations in each split.
    fit:    The LinearFit trained on the training window.
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    rmse_val: float
    mae_val: float
    n_train: int
    n_test: int
    fit: LinearFit


def backtest(
    df: pd.DataFrame,
    cutoff_year: int,
    column: str = "CSIRO Adjusted Sea Level",
    start_year: int | None = None,
) -> BacktestResult:
    """Train on years < cutoff_year, evaluate on years >= cutoff_year.

    Parameters
    ----------
    df:          Full dataset from load_data().
    cutoff_year: First year of the test set.
    column:      Sea-level column to evaluate.
    start_year:  Optionally restrict the training start year.

    Returns
    -------
    BacktestResult
    """
    sub = df.dropna(subset=[column]).sort_values("Year")

    if start_year is not None:
        sub = sub[sub["Year"] >= start_year]

    train = sub[sub["Year"] < cutoff_year]
    test = sub[sub["Year"] >= cutoff_year]

    if len(train) < 2:
        raise ValueError(f"Not enough training data before year {cutoff_year}.")
    if len(test) == 0:
        raise ValueError(f"No test data at or after year {cutoff_year}.")

    fit = fit_linear(train, column=column)
    test_years = test["Year"].to_numpy(float)
    predictions = fit.intercept + fit.slope * test_years
    actual = test[column].to_numpy(float)

    return BacktestResult(
        train_start=int(train["Year"].min()),
        train_end=int(train["Year"].max()),
        test_start=int(test["Year"].min()),
        test_end=int(test["Year"].max()),
        rmse_val=rmse(actual, predictions),
        mae_val=mae(actual, predictions),
        n_train=int(len(train)),
        n_test=int(len(test)),
        fit=fit,
    )
