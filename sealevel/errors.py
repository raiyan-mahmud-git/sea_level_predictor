"""
errors.py – Custom exception hierarchy for the sealevel package.

Having named exceptions lets callers catch specific problems instead of
bare Exception or ValueError.  The Streamlit UI catches these and calls
st.error() so the user sees a friendly message, not a raw traceback.
"""


class SeaLevelError(Exception):
    """Base class for all sealevel package errors."""


class DataLoadError(SeaLevelError):
    """Raised when the CSV cannot be read or is fundamentally broken."""


class MissingColumnError(SeaLevelError):
    """Raised when an expected column is absent from the DataFrame."""

    def __init__(self, column: str, available: list[str]) -> None:
        self.column = column
        self.available = available
        super().__init__(f"Column '{column}' not found. Available columns: {available}")


class EmptyWindowError(SeaLevelError):
    """Raised when the start-year filter leaves zero rows to fit."""

    def __init__(self, start_year: int, last_year: int) -> None:
        self.start_year = start_year
        self.last_year = last_year
        super().__init__(
            f"No data after filtering from year {start_year}. "
            f"The dataset only covers up to {last_year}."
        )


class InsufficientDataError(SeaLevelError):
    """Raised when fewer than 2 rows remain after filtering (can't fit a line)."""

    def __init__(self, n: int) -> None:
        self.n = n
        super().__init__(f"Need at least 2 data points to fit a line, but only {n} remain.")
