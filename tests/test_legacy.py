"""
tests/test_legacy.py – Verify draw_plot() passes the original freeCodeCamp tests.

These tests shadow test_module.py and add regression checks to ensure the
thin-wrapper refactoring hasn't changed any observable behaviour.
"""

from __future__ import annotations

import os

import pytest

from sea_level_predictor import draw_plot

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fig_and_ax(tmp_path_factory):
    """Run draw_plot() in a temp directory so PNG lands there, not CWD."""
    orig_cwd = os.getcwd()
    tmp = tmp_path_factory.mktemp("legacy")
    os.chdir(tmp)
    try:
        fig = draw_plot()
    finally:
        os.chdir(orig_cwd)
    return fig, fig.axes[0]


# --------------------------------------------------------------------------- #
# Original freeCodeCamp assertions (mirrors test_module.py)                   #
# --------------------------------------------------------------------------- #


class TestLegacyContract:
    def test_two_lines_in_plot(self, fig_and_ax):
        _, ax = fig_and_ax
        assert len(ax.lines) == 2

    def test_title(self, fig_and_ax):
        _, ax = fig_and_ax
        assert ax.get_title() == "Rise in Sea Level"

    def test_xlabel(self, fig_and_ax):
        _, ax = fig_and_ax
        assert ax.get_xlabel() == "Year"

    def test_ylabel(self, fig_and_ax):
        _, ax = fig_and_ax
        assert ax.get_ylabel() == "Sea Level (inches)"

    def test_both_lines_end_at_2050(self, fig_and_ax):
        _, ax = fig_and_ax
        for line in ax.lines:
            assert int(line.get_xdata()[-1]) == 2050

    def test_recent_line_starts_at_2000(self, fig_and_ax):
        _, ax = fig_and_ax
        # The second plotted line is the 2000-onward trend
        assert int(ax.lines[1].get_xdata()[0]) == 2000


# --------------------------------------------------------------------------- #
# Regression: slopes / intercepts match Phase-0 baseline                      #
# --------------------------------------------------------------------------- #


class TestLegacyRegressionNumbers:
    """Verify that the wrapped draw_plot() uses the same numbers as before."""

    def test_all_data_line_2050_value(self, fig_and_ax):
        _, ax = fig_and_ax
        line0 = ax.lines[0]
        # Last y-value of line 0 should be ~10.1755 in
        last_y = float(line0.get_ydata()[-1])
        assert abs(last_y - 10.1755) < 0.001

    def test_recent_line_2050_value(self, fig_and_ax):
        _, ax = fig_and_ax
        line1 = ax.lines[1]
        last_y = float(line1.get_ydata()[-1])
        assert abs(last_y - 15.3824) < 0.001

    def test_png_was_saved(self, tmp_path):
        """draw_plot() must save sea_level_plot.png (required by test_module)."""
        orig = os.getcwd()
        os.chdir(tmp_path)
        try:
            draw_plot()
            assert (tmp_path / "sea_level_plot.png").exists()
        finally:
            os.chdir(orig)
