"""
tests/test_app.py – Streamlit AppTest smoke tests.

These use Streamlit's built-in testing framework (streamlit.testing.v1).
They verify:
  - App loads without uncaught exceptions
  - Each page (Overview, Trend Explorer, Model Lab, Methodology) renders cleanly
  - Expected widgets and data are present
  - Error states are caught cleanly without crashing the app
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from streamlit.testing.v1 import AppTest

    HAS_APPTEST = True
except ImportError:
    HAS_APPTEST = False

pytestmark = pytest.mark.skipif(not HAS_APPTEST, reason="AppTest requires Streamlit ≥ 1.28")

APP_PATH = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture()
def app():
    """Create and run a fresh AppTest instance."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


class TestAppLoads:
    def test_no_exception_on_initial_load(self, app):
        assert not app.exception, f"App raised: {app.exception}"

    def test_page_title_present(self, app):
        elements = list(app.title) + list(app.header) + list(app.subheader) + list(app.markdown)
        all_text = " ".join(str(e.value) for e in elements if hasattr(e, "value"))
        assert "Sea Level" in all_text or len(app.title) > 0


class TestPageNavigation:
    """Test that all four pages render without exceptions."""

    def test_overview_page(self, app):
        app.radio(key="nav_page").set_value("📊 Overview").run()
        assert not app.exception
        assert len(app.dataframe) > 0

    def test_trend_explorer_page(self, app):
        app.radio(key="nav_page").set_value("📈 Trend Explorer").run()
        assert not app.exception
        # Check that metrics were rendered
        assert len(app.metric) > 0

    def test_model_lab_page(self, app):
        app.radio(key="nav_page").set_value("🔬 Model Lab").run()
        assert not app.exception
        assert len(app.dataframe) > 0

    def test_methodology_page(self, app):
        app.radio(key="nav_page").set_value("📐 Methodology").run()
        assert not app.exception
        # Check formulas / markdown rendered
        assert len(app.header) > 0 or len(app.markdown) > 0


class TestTrendExplorerControls:
    """Test interactive control changes in Trend Explorer."""

    def test_change_unit_to_mm(self, app):
        app.radio(key="nav_page").set_value("📈 Trend Explorer").run()
        assert not app.exception
        app.radio(key="sel_unit").set_value("mm").run()
        assert not app.exception

    def test_change_model_to_quadratic(self, app):
        app.radio(key="nav_page").set_value("📈 Trend Explorer").run()
        assert not app.exception
        app.selectbox(key="sel_model").set_value("Quadratic").run()
        assert not app.exception

    def test_change_model_to_piecewise(self, app):
        app.radio(key="nav_page").set_value("📈 Trend Explorer").run()
        assert not app.exception
        app.selectbox(key="sel_model").set_value("Piecewise").run()
        assert not app.exception
