"""
test_module.py – freeCodeCamp original tests. Do not modify.

These tests are part of the certification project and must pass unchanged.
"""

import unittest
import pandas as pd
from sea_level_predictor import draw_plot


class LinePlotTestCase(unittest.TestCase):
    def setUp(self):
        self.fig = draw_plot()
        self.ax = self.fig.axes[0]

    # scatter plot point check
    def test_scatter_plot_contains_correct_number_of_items_in_legend(self):
        actual = len(self.ax.lines)
        expected = 2
        self.assertEqual(
            actual,
            expected,
            "Expected two lines in the plot.",
        )

    def test_plot_title(self):
        actual = self.ax.get_title()
        expected = "Rise in Sea Level"
        self.assertEqual(actual, expected, "Expected title to be 'Rise in Sea Level'")

    def test_plot_xlabel(self):
        actual = self.ax.get_xlabel()
        expected = "Year"
        self.assertEqual(actual, expected, "Expected xlabel to be 'Year'")

    def test_plot_ylabel(self):
        actual = self.ax.get_ylabel()
        expected = "Sea Level (inches)"
        self.assertEqual(
            actual, expected, "Expected ylabel to be 'Sea Level (inches)'"
        )

    def test_plot_lines_correct_end_year(self):
        actual = int(self.ax.lines[0].get_xdata()[-1])
        expected = 2050
        self.assertEqual(
            actual,
            expected,
            "Expected the line to end at year 2050.",
        )

    def test_plot_lines_2000_starts_at_2000(self):
        actual = int(self.ax.lines[1].get_xdata()[0])
        expected = 2000
        self.assertEqual(
            actual,
            expected,
            "Expected the second line to start at year 2000.",
        )


if __name__ == "__main__":
    unittest.main()
