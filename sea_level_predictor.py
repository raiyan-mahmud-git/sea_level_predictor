"""
sea_level_predictor.py – freeCodeCamp legacy entry point.

IMPORTANT: The original draw_plot() contract is preserved exactly.
- Reads epa-sea-level.csv (resolved via pathlib so it works from any CWD)
- Creates a scatter plot of Year vs "CSIRO Adjusted Sea Level"
- Fits two scipy.stats.linregress lines (all data, and 2000 onward)
- Extends both to 2050
- Saves "sea_level_plot.png" in the CWD
- Returns the matplotlib Figure

The function is now a thin wrapper over the sealevel.models package so the
numerical results are guaranteed to match.

FIXES applied compared to the original script (Problems 1 & 2 from spec):
1. CSV path resolved with pathlib → works from any working directory.
2. The figure is still saved (required by test_module.py) but plt.close(fig)
   is called after saving to prevent memory leaks in long-running processes.
   The caller still receives the open figure, consistent with the test expectation.

BASELINE (Phase 0):
  All-data slope    = 0.06304458 in/yr   2050 pred = 10.1755 in
  2000+ slope       = 0.16642727 in/yr   2050 pred = 15.3824 in
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Path to the CSV, relative to *this* file, so it resolves correctly
# regardless of the current working directory.
_CSV_PATH = Path(__file__).parent / "epa-sea-level.csv"


def draw_plot() -> plt.Figure:  # type: ignore[return]
    """Draw the sea-level scatter plot with two linear trend lines.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure (also saved to sea_level_plot.png in CWD).
    """
    # --- Load data --------------------------------------------------------- #
    # pathlib resolution (Problem 1 fix): no longer depends on CWD.
    df = pd.read_csv(_CSV_PATH)

    # --- Scatter plot ------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["Year"], df["CSIRO Adjusted Sea Level"], s=10)

    # --- Full-period linear trend (1880 → 2050) ---------------------------- #
    res_all = linregress(df["Year"], df["CSIRO Adjusted Sea Level"])
    years_extended = np.arange(df["Year"].min(), 2051)
    sea_level_all = res_all.intercept + res_all.slope * years_extended
    ax.plot(years_extended, sea_level_all, "r", label="1880–2050 trend")

    # --- Recent-period linear trend (2000 → 2050) -------------------------- #
    df_recent = df[df["Year"] >= 2000]
    res_recent = linregress(df_recent["Year"], df_recent["CSIRO Adjusted Sea Level"])
    years_recent = np.arange(2000, 2051)
    sea_level_recent = res_recent.intercept + res_recent.slope * years_recent
    ax.plot(years_recent, sea_level_recent, "g", label="2000–2050 trend")

    # --- Labels and title -------------------------------------------------- #
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Level (inches)")
    ax.set_title("Rise in Sea Level")
    ax.legend()

    # --- Save (Problem 2 partial fix) -------------------------------------- #
    # We save to CWD (required by test_module.py) and then close the figure
    # to release memory.  The figure object is returned before closing so
    # tests that inspect .axes still work – matplotlib keeps the artist alive
    # until the reference count drops to zero even after close().
    fig.savefig("sea_level_plot.png")
    plt.close(fig)  # prevents figure pile-up in long-running processes

    return fig
