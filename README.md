# Cross-Market Correlation Strategy: BTC vs Nasdaq Futures

Research repository for studying whether short-horizon alignment between Bitcoin and Nasdaq futures can be turned into a tradable regime filter, now extended with ML-enhanced position sizing.

## What This Repo Does

- Loads and synchronizes 1-minute BTCUSDT and NQ futures data.
- Normalizes timestamps into a shared timeline to avoid cross-market misalignment.
- Computes rolling Pearson and Spearman correlation plus directional hit-ratio metrics.
- Analyzes intraday seasonality, signal quality, event clustering, and forward returns.
- Runs a sensitivity grid search over signal windows and holding horizons.
- Implements ML filtering (logistic regression) for trade approval/rejection.
- Compares equity curves with flat position sizing across baseline, dynamic exits, volatility filters, and ML-enhanced strategies.

## Repository Structure

- [`data/`](data): local minute-bar datasets used by the notebooks.
- [`datasets/`](datasets): additional data copies for redundancy.
- [`research/correlation_research.ipynb`](research/correlation_research.ipynb): main research notebook for correlation analysis and signal generation.
- [`research/ml_position_sizing.ipynb`](research/ml_position_sizing.ipynb): ML-enhanced position sizing notebook with equity curve comparisons.
- [`results/`](results): output plots, metrics, and backtest results.
- [`src/__init__.py`](src/__init__.py): packages src as a Python module.
- [`src/config.py`](src/config.py): repo-relative paths and default research constants.
- [`src/data_pipeline.py`](src/data_pipeline.py): CSV loading, schema normalization, timezone alignment, and dataset joins.
- [`src/indicators.py`](src/indicators.py): rolling correlation, hit-ratio, ADF helper, and volume-weighted feature logic.
- [`src/simulation.py`](src/simulation.py): signal-state construction, event clustering, and event-study helpers.
- [`src/optimization.py`](src/optimization.py): sensitivity grid search across windows and hold periods.
- [`Archive - Previous Works/`](Archive - Previous Works): historical notebooks and outputs kept for reference.

## Data

The repo expects these local CSVs:

- `data/BTCUSDT_1m_2024-03-07_to_2026-03-07.csv`
- `data/NQ_stitched_1min_2024-03-07_to_2026-03-07.csv`

BTC data is treated as UTC-native. NQ data is parsed as `America/Chicago` exchange time and converted to UTC before alignment.

## Workflow

1. For correlation research: Open [`research/correlation_research.ipynb`](research/correlation_research.ipynb). Run setup cells to import from `src`, then explore stats, plots, and interpretations.
2. For ML position sizing: Open [`research/ml_position_sizing.ipynb`](research/ml_position_sizing.ipynb). It builds on correlation signals, adds ML filtering, and compares equity curves on OOS data.
3. Extend `src` when code becomes reusable or needed outside notebooks.

## Current State

The project has progressed from initial correlation analysis to ML-enhanced position sizing. Key achievements:
- Implemented flat sizing (size=1.0) for simplicity and robustness, replacing vol-adjusted/Kelly approaches.
- Added logistic regression classifier for trade filtering, trained on IS data and evaluated OOS.
- Compares 4 equity curves: baseline, Path B2 exits, SD-filtered, and ML-filtered.
- Updated `src` files for compatibility; added `__init__.py` for packaging.

Note: Logistic regression underperforms on OOS (rejects all trades), so ML tuning is needed (e.g., probability thresholds, hyperparameters, or alternative models like RandomForest). The core strategy is ready—run notebooks to validate, then integrate ML if improved.
