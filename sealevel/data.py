"""
data.py – Data loading and validation for the sealevel package.

KEY DESIGN DECISIONS
---------------------
1. Paths are resolved with pathlib so the function works regardless of the
   current working directory (fixes Problem 1 from the spec).
2. load_data() is a pure function that only reads and validates.  Caching is
   handled by the Streamlit layer (st.cache_data) so this module stays free of
   Streamlit imports and is easy to test.
3. Validation raises typed exceptions (MissingColumnError, etc.) instead of
   letting pandas crash with an unhelpful KeyError.

CSV COLUMNS (confirmed by Phase-0 audit)
-----------------------------------------
Year                      int64   1880 – 2013   no NaNs
CSIRO Adjusted Sea Level  float64               no NaNs
Lower Error Bound         float64               no NaNs
Upper Error Bound         float64               no NaNs
NOAA Adjusted Sea Level   float64               113 NaNs (starts ~1993)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sealevel.errors import (
    DataLoadError,
    EmptyWindowError,
    InsufficientDataError,
    MissingColumnError,
)

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

# Path to the bundled CSV, resolved relative to *this file* so it works from
# any working directory.
_HERE = Path(__file__).parent
_CSV_PATH: Path = _HERE.parent / "epa-sea-level.csv"

# The required column that every analysis depends on.
REQUIRED_COLUMNS = {"Year", "CSIRO Adjusted Sea Level"}

# Optional columns that the app can display if present.
OPTIONAL_COLUMNS = {
    "Lower Error Bound",
    "Upper Error Bound",
    "NOAA Adjusted Sea Level",
}

# Inches → millimetres conversion factor.
INCHES_TO_MM: float = 25.4


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def load_data(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the EPA sea-level dataset.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.  Defaults to the bundled ``epa-sea-level.csv``
        at the repository root.  Pass an explicit path in tests to load
        synthetic data.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame sorted by Year.  All required columns are present,
        numeric, and contain no NaNs.  Optional columns may contain NaNs.

    Raises
    ------
    DataLoadError
        If the file cannot be found or cannot be parsed as CSV.
    MissingColumnError
        If a required column is absent.
    """
    path = Path(csv_path) if csv_path is not None else _CSV_PATH

    # --- 1. File existence ------------------------------------------------- #
    if not path.exists():
        raise DataLoadError(
            f"CSV file not found at '{path}'. "
            "Make sure 'epa-sea-level.csv' is at the repository root."
        )

    # --- 2. Parse ---------------------------------------------------------- #
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataLoadError(f"Could not parse '{path}': {exc}") from exc

    # --- 3. Required columns ----------------------------------------------- #
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise MissingColumnError(col, df.columns.tolist())

    # --- 4. Coerce types --------------------------------------------------- #
    #   pandas usually infers these correctly, but we enforce numeric to catch
    #   files with stray text in the data rows.
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 5. Drop rows where required columns are NaN ----------------------- #
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    dropped = before - len(df)
    if dropped > 0:
        import warnings

        warnings.warn(
            f"Dropped {dropped} rows with NaN in required columns.",
            stacklevel=2,
        )

    # --- 6. Sort by year --------------------------------------------------- #
    df = df.sort_values("Year").reset_index(drop=True)

    return df


def filter_by_year(
    df: pd.DataFrame,
    start_year: int,
    column: str = "CSIRO Adjusted Sea Level",
) -> pd.DataFrame:
    """Return rows where Year >= start_year, validating that the window is usable.

    Parameters
    ----------
    df:
        Full validated DataFrame from load_data().
    start_year:
        First year to include (inclusive).
    column:
        The analysis column (checked for NaNs after filtering).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.

    Raises
    ------
    MissingColumnError
        If ``column`` is not in *df*.
    EmptyWindowError
        If no rows remain after filtering.
    InsufficientDataError
        If fewer than 2 rows remain (cannot fit a regression line).
    """
    if column not in df.columns:
        raise MissingColumnError(column, df.columns.tolist())

    last_year = int(df["Year"].max())
    filtered = df[df["Year"] >= start_year].dropna(subset=[column])

    if filtered.empty:
        raise EmptyWindowError(start_year, last_year)

    if len(filtered) < 2:
        raise InsufficientDataError(len(filtered))

    return filtered.reset_index(drop=True)


def inches_to_mm(value: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    """Convert sea-level measurement(s) from inches to millimetres."""
    return value * INCHES_TO_MM


def mm_to_inches(value: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    """Convert sea-level measurement(s) from millimetres to inches."""
    return value / INCHES_TO_MM


def summary_stats(df: pd.DataFrame, column: str = "CSIRO Adjusted Sea Level") -> dict:
    """Return a summary-stats dict for display in the Overview page.

    Keys: n, first_year, last_year, min_val, max_val, mean_val, std_val,
          noaa_coverage (fraction of non-NaN NOAA rows if column exists).
    """
    if column not in df.columns:
        raise MissingColumnError(column, df.columns.tolist())

    stats: dict = {
        "n": int(len(df)),
        "first_year": int(df["Year"].min()),
        "last_year": int(df["Year"].max()),
        "min_val": float(df[column].min()),
        "max_val": float(df[column].max()),
        "mean_val": float(df[column].mean()),
        "std_val": float(df[column].std()),
    }

    noaa_col = "NOAA Adjusted Sea Level"
    if noaa_col in df.columns:
        n_noaa = int(df[noaa_col].notna().sum())
        stats["noaa_n"] = n_noaa
        stats["noaa_coverage"] = n_noaa / len(df)
    else:
        stats["noaa_n"] = 0
        stats["noaa_coverage"] = 0.0

    return stats
