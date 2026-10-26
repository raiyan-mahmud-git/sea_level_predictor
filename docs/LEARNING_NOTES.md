# Learning Notes – Sea Level Predictor Upgrade

This file records the key concept from each phase, appended as work progresses.

---

## Phase 0 – Audit and Baseline

**What changed:** Inspected the CSV (134 rows, 5 columns, 1880–2013), ran the original
`draw_plot()`, and recorded exact baseline numbers to use as regression tests:
- Full-period slope = 0.06304458 in/yr, 2050 prediction = 10.1755 in (258.46 mm)
- 2000-onward slope = 0.16642727 in/yr, 2050 prediction = 15.3824 in (390.71 mm)

**Why:** Before refactoring, capturing exact output means we can detect any numerical
regression instantly.

**Concept to learn:** **Regression testing** – a type of test that records a known
good output and alerts you when future code changes alter it, even accidentally.  It
is different from a unit test (which checks correctness against independently derived
expected values) – a regression test is just a snapshot.

---

## Phase 1 – Core Package (sealevel/)

**What changed:** Created `sealevel/` with five modules:
- `errors.py` – typed exception hierarchy so UI can catch specific errors
- `data.py` – `load_data()` with pathlib (fixes Problem 1), validation, unit conversion
- `models.py` – `fit_linear()`, `fit_polynomial()`, `fit_piecewise()`, `predict_linear()`
- `metrics.py` – `rmse()`, `mae()`, `backtest()` with time-based split
- `plotting.py` – Plotly figure builders, colorblind-safe palette

**Why:** Separating pure logic from the UI makes the code testable in isolation and
reusable (e.g., the same `fit_linear()` powers both the Streamlit app and the legacy
`draw_plot()` wrapper).

**Concept to learn:** **Separation of Concerns** – organising code so each module has
one clearly defined responsibility.  Data loading, model fitting, visualisation, and
UI are all separate here.  This makes each part easier to test, replace, or understand
on its own.

---

## Phase 2 – Legacy Compatibility

**What changed:** `sea_level_predictor.py` is now a thin wrapper: it uses `pathlib`
for the CSV path and calls `plt.close(fig)` after saving to prevent memory leaks.
The figure is returned before closing (matplotlib keeps the object alive until GC),
so `test_module.py` still passes.

**Why:** The freeCodeCamp tests inspect `ax.lines` and plot attributes – we must
preserve those exactly.  A thin wrapper pattern lets us upgrade the internals without
breaking the public contract.

**Concept to learn:** **The Adapter Pattern** – a design pattern where a new
implementation is wrapped in an interface that looks like the old one.  Here, the
new `sealevel` package is the "adaptee" and `draw_plot()` is the adapter that makes
it look like the original script to the test suite.

---

## Phase 3 – Tests (pytest)

**What changed:** Four test modules in `tests/`:
- `test_data.py` – load, filter, validate, convert
- `test_models.py` – scipy agreement, Phase-0 baselines, synthetic data, piecewise
- `test_metrics.py` – RMSE, MAE, backtest
- `test_legacy.py` – draw_plot() contract + regression numbers

**Why:** Tests give us the confidence to refactor without fear.  The Phase-0 numbers
are now frozen in regression tests, so any future code change that alters the fitted
slope will fail immediately.

**Concept to learn:** **Test pyramid** – unit tests (fast, many) at the base, then
integration tests, then end-to-end tests at the top.  Most of our tests here are
unit tests (testing individual functions in isolation).  `test_app.py` is an
end-to-end smoke test.

---

## Phase 4 – Streamlit App

**What changed:** Created `app.py` (entrypoint), `views/` (four page modules),
`ui/components.py` (sidebar, metric cards, error display).

Key decisions:
- `@st.cache_data` on `load_data()` and model fitting to avoid re-running on
  every widget interaction.
- `st.query_params` to sync key inputs to the URL (shareable scenarios).
- All error paths show `st.error()`, never raw tracebacks.
- All numbers in labels are computed (e.g. `f"{first_year}–{horizon} trend"`),
  never hard-coded strings.

**Why:** Streamlit re-runs the whole script on every widget change, so without caching
the CSV would be re-read and regression re-fit on every slider move.  `@st.cache_data`
stores the result keyed by the function arguments and returns it instantly on cache hits.

**Concept to learn:** **Memoisation / caching** – storing the result of an expensive
function call and returning the stored result for the same inputs.  `@st.cache_data`
is Streamlit's built-in memoisation decorator.  The general principle is the same as
Python's `@functools.lru_cache`.

---

## Phase 5 – Polish

**What changed:** Colorblind-safe IBM Carbon palette (blue, orange, purple, green, red),
axis labels with units, `.streamlit/config.toml` for theme, `use_container_width=True`
on all charts for mobile layout.

**Concept to learn:** **Accessibility in data visualisation** – ~8% of males have some
form of colour vision deficiency.  Red-green colour pairs (the most common default) are
the hardest to distinguish.  The IBM Carbon palette separates colours in *luminance*
as well as hue, making charts usable without colour alone.

---

## Phase 6 – Quality, Deployment, README

**What changed:** `pyproject.toml` with ruff config, CI workflow (ruff + pytest on
Python 3.10–3.13), `AppTest` smoke tests, full README with badges, key findings,
5-command setup, and limitations section.

**Concept to learn:** **Continuous Integration (CI)** – automatically running tests
and linters on every push and pull request, so problems are caught early (when the
change is fresh in mind) rather than late (when the broken code is already in
production).  GitHub Actions is one popular CI provider; it runs `.github/workflows/*.yml`
automatically.
