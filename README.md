# 🌊 Sea Level Predictor

[![CI](https://github.com/raiyan-mahmud-git/sea_level_predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/raiyan-mahmud-git/sea_level_predictor/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive Streamlit data-analysis app exploring global mean sea-level rise from 1880 to 2013 — upgraded from a single freeCodeCamp certification script into a polished, tested, deployable tool.

> **Educational use only.** Do not use these extrapolations for policy decisions.  
> See [Limitations](#️-limitations) below.

---

## 🔑 Key Findings (all numbers computed from the data)

| Metric | Value |
|---|---|
| Dataset coverage | 1880 – 2013 (134 years) |
| Full-period OLS slope | **0.06304 in/yr** (1.602 mm/yr) |
| 2000-onward OLS slope | **0.16643 in/yr** (4.227 mm/yr) |
| Slope acceleration factor | **×2.64** |
| Full-period 2050 prediction | **10.18 in (258 mm)** above 1880 baseline |
| 2000-onward 2050 prediction | **15.38 in (391 mm)** above 1880 baseline |
| Full-period R² | 0.9697 |
| 2000-onward R² | 0.9531 |
| Automatic breakpoint year | detected by minimising split RSS |

> The 37 % spread between the two 2050 predictions (10.2 vs 15.4 in) illustrates how sensitive extrapolations are to window choice — see [Methodology](#-methodology) for why this matters.

---

## 🚀 Setup (5 commands)

```bash
git clone https://github.com/raiyan-mahmud-git/sea_level_predictor.git
cd sea_level_predictor
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 🗂️ Project Structure

```
sea_level_predictor/
├── app.py                        # Streamlit entrypoint
├── sea_level_predictor.py        # freeCodeCamp legacy entry (draw_plot kept)
├── main.py                       # freeCodeCamp boilerplate (unchanged)
├── test_module.py                # freeCodeCamp tests (unchanged)
├── epa-sea-level.csv             # Bundled dataset (EPA / CSIRO / NOAA)
│
├── sealevel/                     # Core analysis package (no Streamlit imports)
│   ├── data.py                   # load_data(), filter_by_year(), unit conversion
│   ├── models.py                 # fit_linear(), fit_polynomial(), fit_piecewise()
│   ├── metrics.py                # rmse(), mae(), backtest()
│   ├── plotting.py               # Plotly figure builders
│   └── errors.py                 # Custom exception hierarchy
│
├── views/                        # Streamlit page modules
│   ├── overview.py               # Dataset preview + attribution
│   ├── trends.py                 # Interactive trend explorer
│   ├── model_lab.py              # Model comparison + residuals
│   └── methodology.py           # Formulas + limitations
│
├── ui/
│   └── components.py             # Sidebar, metric cards, error display
│
├── tests/
│   ├── test_data.py              # Unit tests for sealevel.data
│   ├── test_models.py            # Unit tests for sealevel.models (+ baseline regression)
│   ├── test_metrics.py           # Unit tests for sealevel.metrics
│   ├── test_legacy.py            # draw_plot() regression tests
│   └── test_app.py               # Streamlit AppTest smoke tests
│
├── docs/
│   ├── METHODOLOGY.md            # Extended methodology notes
│   └── LEARNING_NOTES.md         # Phase-by-phase learning journal
│
├── .streamlit/config.toml        # Theme (colorblind-safe ocean palette)
├── .github/workflows/ci.yml      # CI: ruff + pytest on Python 3.10–3.13
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## 📱 App Pages

| Page | Description |
|---|---|
| **Overview** | Dataset preview, summary stats, missing values, attribution |
| **Trend Explorer** | Interactive Plotly chart; linear / quadratic / piecewise models; prediction intervals; metric cards; CSV + PNG downloads |
| **Model Lab** | Side-by-side RMSE/MAE comparison (time-based backtest), slope acceleration, automatic breakpoint, residual plot |
| **Methodology** | LaTeX formulas, limitations expander |

---

## 🧪 Running Tests

```bash
# All tests (including original freeCodeCamp test_module.py)
pytest tests/ test_module.py -v

# Linting
ruff check . --exclude test_module.py
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (keep `epa-sea-level.csv` at the repo root).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and entrypoint `app.py`.
4. Click **Deploy** – no extra secrets needed.

The app will be live at `https://<your-subdomain>.streamlit.app`.

---

## ⚠️ Limitations

- **Extrapolation**: All forecasts extend well beyond the 2013 data endpoint. Sea-level rise depends on emissions, ice-sheet dynamics, and ocean heat — none captured by curve fitting.
- **Autocorrelated residuals**: OLS standard errors assume i.i.d. errors. Annual sea-level data is autocorrelated, so prediction intervals are likely **too narrow**.
- **Dataset coverage**: CSIRO series ends at 2013; NOAA satellite data starts ~1993 with 113 missing rows in the earlier period.
- **Window sensitivity**: The 2050 prediction ranges from 10.2 to 15.4 in depending solely on which window is used (1880+ vs 2000+).
- **Educational purpose**: This project demonstrates data analysis techniques. For authoritative projections, consult the IPCC AR6 or NOAA Technical Reports.

---

## 📚 Data Attribution

- **CSIRO**: Church, J. A. & White, N. J. (2011). *Sea-level rise from the late 19th to the early 21st Century.* Surveys in Geophysics, 32(4–5), 585–602.
- **NOAA**: Laboratory for Satellite Altimetry.
- Dataset provided via the U.S. Environmental Protection Agency and the freeCodeCamp *Data Analysis with Python* curriculum.

---

## 📄 License

MIT © raiyan-mahmud-git – see [LICENSE](LICENSE).
