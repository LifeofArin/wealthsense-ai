# WealthSense AI

Stock forecasting and goal-based financial planning in one Streamlit app. I built it to show how deep learning, classical baselines, and uncertainty methods can work together in a product-style dashboard, with an optional Gemini chat layer on top.

**Live demo:** [wealthsenseai.streamlit.app](https://wealthsenseai.streamlit.app/)

The pipeline covers five tickers (AAPL, MSFT, NVDA, TSLA, SPY) on daily data from 2015 to 2023. It includes LSTM, GRU, and Transformer models, a rolling inverse-RMSE ensemble, ARIMA and naive baselines, Monte Carlo dropout bands, Diebold-Mariano tests, feature ablation, and a Monte Carlo goal planner with Sharpe, Sortino, drawdown, and transaction-cost-aware backtests.

> For research and portfolio analytics demos only. Not financial advice.

## What this project includes

| Layer | Module | What it does |
|---|---|---|
| Data | `src/data_pipeline.py` | Pulls Yahoo Finance OHLCV and macro series (VIX, yield spread, DXY), builds a log-return target, and fits scalers on training data only. |
| Models | `src/models.py` | LSTM, GRU, and Transformer heads with LayerNorm and GELU; the Transformer uses a causal mask and returns attention weights. |
| Baselines | `src/baselines.py` | ARIMA(5,0,0) walk-forward forecasts and a naive persistence baseline for comparison. |
| Uncertainty | `src/uncertainty.py` | Monte Carlo dropout (100 passes) plus calibration checks at 50%, 80%, and 90%. |
| Ensemble | `src/ensemble.py` | Rolling inverse-RMSE weights across LSTM, GRU, and Transformer. |
| Training | `src/train.py` | End-to-end training with SMAPE on returns and MAPE on reconstructed prices. |
| Planning | `src/monte_carlo.py` | 5,000-path Monte Carlo engine with model outlook tilt, Sharpe, Sortino, max drawdown, and cost-aware backtests. |
| Ablation | `src/ablation.py` | Five feature sets: full, no macro, no RSI, no volume, price only. |
| Chat | `src/chat.py` | Google Gemini with structured context, plus a free rule-based fallback when no API key is set. |
| Attention | `src/attention.py` | Day-importance chart and attention heatmap from the Transformer. |
| Walk-forward | `src/walk_forward.py` | Expanding-window validation in 6-month folds. |
| Statistics | `src/statistical_tests.py` | Diebold-Mariano tests with Harvey-Leybourne-Newbold correction. |
| Sensitivity | `src/hyperparam_sensitivity.py` | Hyperparameter sensitivity runs. |
| Feature audit | `src/feature_audit.py` | Feature quality and importance checks. |
| Critical analysis | `src/critical_analysis.py` | Structured review of model strengths and limits. |
| Results | `run_results_summary.py` | One consolidated results report for reviewers. |
| UI | `app.py` | Multi-tab Streamlit dashboard. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env       # optional: add GOOGLE_API_KEY (free at aistudio.google.com/apikey)
```

## Run

```bash
# 1. Pull data and train all models (LSTM, GRU, Transformer, ensemble, ARIMA, naive)
python src/train.py

# 2. Build the consolidated results report
python run_results_summary.py

# 3. Launch the dashboard
streamlit run app.py
```

## Research methodology

- Walk-forward validation with an expanding window in 6-month folds
- Diebold-Mariano testing (Harvey-Leybourne-Newbold correction) for pairwise significance
- Feature ablation in core and extended configurations
- Uncertainty calibration at 50%, 80%, and 90% interval targets

## Reproducibility

- Random seed fixed in `config.SEED = 42`
- Data window: `START_DATE=2015-01-01`, `END_DATE=2023-12-31`
- Train: 2015-2021, validation: 2022, test: 2023
- 30-day input window; target is the log-return on day 31

## How metrics are reported

Models predict log-returns because they are more stable than raw prices. MAPE on log-returns is misleading when values sit near zero, so I report:

- **On returns:** MAE, RMSE, SMAPE, directional accuracy
- **On reconstructed prices:** MAE (USD), RMSE (USD), MAPE (%), which are easier to read in a business context

ARIMA and naive baselines use the same evaluation setup on the same test split.

## Project structure

```
wealthsenseAI/
├── config.py
├── app.py                       # Streamlit dashboard (4 tabs)
├── requirements.txt
├── .env.example
├── data/                        # cached CSVs (auto-populated)
├── artifacts/
│   ├── models/                  # *.pt + scalers
│   └── results/                 # summary.json + *_preds.npz + ablation_*.json
└── src/
    ├── data_pipeline.py
    ├── models.py
    ├── baselines.py
    ├── uncertainty.py
    ├── ensemble.py
    ├── train.py
    ├── monte_carlo.py
    ├── ablation.py
    ├── attention.py
    └── chat.py
```

## Documentation

- [data.md](data.md): dataset notes
- [overview.md](overview.md): non-technical walkthrough
