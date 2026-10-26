"""
tests/test_data.py – Unit tests for sealevel.data

Tests cover:
- load_data() with the real CSV
- load_data() with synthetic CSVs (missing column, NaN handling)
- filter_by_year() validation errors
- inches_to_mm / mm_to_inches round-trip
- summary_stats()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sealevel.data import (
    filter_by_year,
    inches_to_mm,
    load_data,
    mm_to_inches,
    summary_stats,
)
from sealevel.errors import (
    DataLoadError,
    EmptyWindowError,
    InsufficientDataError,
    MissingColumnError,
)

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def real_df():
    """Load the bundled CSV once for the whole test session."""
    return load_data()


@pytest.fixture()
def minimal_csv(tmp_path):
    """Write a tiny valid CSV and return its path."""
    data = pd.DataFrame(
        {
            "Year": [1880, 1881, 1882, 1883, 1884],
            "CSIRO Adjusted Sea Level": [0.0, 0.1, 0.2, 0.3, 0.4],
            "Lower Error Bound": [-0.1] * 5,
            "Upper Error Bound": [0.1] * 5,
            "NOAA Adjusted Sea Level": [np.nan] * 5,
        }
    )
    p = tmp_path / "mini.csv"
    data.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# load_data() – happy path                                                     #
# --------------------------------------------------------------------------- #


class TestLoadDataHappyPath:
    def test_returns_dataframe(self, real_df):
        assert isinstance(real_df, pd.DataFrame)

    def test_has_required_columns(self, real_df):
        assert "Year" in real_df.columns
        assert "CSIRO Adjusted Sea Level" in real_df.columns

    def test_year_range(self, real_df):
        assert real_df["Year"].min() == 1880
        assert real_df["Year"].max() == 2013

    def test_row_count(self, real_df):
        # 134 years from 1880 to 2013 inclusive
        assert len(real_df) == 134

    def test_no_nans_in_required_columns(self, real_df):
        assert real_df["Year"].isna().sum() == 0
        assert real_df["CSIRO Adjusted Sea Level"].isna().sum() == 0

    def test_sorted_by_year(self, real_df):
        assert (real_df["Year"].diff().dropna() >= 0).all()

    def test_noaa_has_nans(self, real_df):
        # Phase-0 audit: 113 NaNs in NOAA column
        assert real_df["NOAA Adjusted Sea Level"].isna().sum() > 0

    def test_minimal_csv(self, minimal_csv):
        df = load_data(minimal_csv)
        assert len(df) == 5


# --------------------------------------------------------------------------- #
# load_data() – error cases                                                    #
# --------------------------------------------------------------------------- #


class TestLoadDataErrors:
    def test_missing_file_raises_data_load_error(self, tmp_path):
        with pytest.raises(DataLoadError, match="not found"):
            load_data(tmp_path / "nonexistent.csv")

    def test_missing_required_column_raises_missing_column_error(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"Year": [1880], "WrongColumn": [0.0]}).to_csv(bad, index=False)
        with pytest.raises(MissingColumnError):
            load_data(bad)

    def test_non_numeric_values_are_coerced_to_nan_and_dropped(self, tmp_path):
        bad = tmp_path / "text.csv"
        df = pd.DataFrame(
            {
                "Year": ["1880", "text", "1882"],
                "CSIRO Adjusted Sea Level": ["0.0", "bad", "0.2"],
            }
        )
        df.to_csv(bad, index=False)
        with pytest.warns(UserWarning):
            result = load_data(bad)
        assert len(result) == 2  # 'text' row dropped


# --------------------------------------------------------------------------- #
# filter_by_year()                                                             #
# --------------------------------------------------------------------------- #


class TestFilterByYear:
    def test_normal_filter(self, real_df):
        filtered = filter_by_year(real_df, start_year=2000)
        assert filtered["Year"].min() == 2000

    def test_empty_window_raises(self, real_df):
        with pytest.raises(EmptyWindowError):
            filter_by_year(real_df, start_year=9999)

    def test_insufficient_data_raises(self, tmp_path):
        tiny = tmp_path / "one.csv"
        pd.DataFrame(
            {
                "Year": [2000],
                "CSIRO Adjusted Sea Level": [5.0],
            }
        ).to_csv(tiny, index=False)
        df = load_data(tiny)
        with pytest.raises(InsufficientDataError):
            filter_by_year(df, start_year=2000)

    def test_missing_column_raises(self, real_df):
        with pytest.raises(MissingColumnError):
            filter_by_year(real_df, start_year=1880, column="Nonexistent")


# --------------------------------------------------------------------------- #
# Unit conversion                                                              #
# --------------------------------------------------------------------------- #


class TestUnitConversion:
    def test_inches_to_mm_scalar(self):
        assert abs(inches_to_mm(1.0) - 25.4) < 1e-10

    def test_mm_to_inches_scalar(self):
        assert abs(mm_to_inches(25.4) - 1.0) < 1e-10

    def test_round_trip(self):
        val = 7.654
        assert abs(mm_to_inches(inches_to_mm(val)) - val) < 1e-10

    def test_array_input(self):
        arr = np.array([0.0, 1.0, 2.0])
        result = inches_to_mm(arr)
        np.testing.assert_allclose(result, [0.0, 25.4, 50.8])


# --------------------------------------------------------------------------- #
# summary_stats()                                                              #
# --------------------------------------------------------------------------- #


class TestSummaryStats:
    def test_returns_expected_keys(self, real_df):
        stats = summary_stats(real_df)
        for key in ("n", "first_year", "last_year", "min_val", "max_val"):
            assert key in stats

    def test_year_bounds(self, real_df):
        stats = summary_stats(real_df)
        assert stats["first_year"] == 1880
        assert stats["last_year"] == 2013

    def test_missing_column_raises(self, real_df):
        with pytest.raises(MissingColumnError):
            summary_stats(real_df, column="Nonexistent")
