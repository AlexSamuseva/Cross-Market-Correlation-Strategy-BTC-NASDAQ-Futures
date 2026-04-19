# Cross-Market Correlation Strategy: BTC vs Nasdaq Futures

Research repository for studying whether short-horizon alignment between Bitcoin and Nasdaq futures can be turned into a tradable regime filter.

## What This Repo Does

- Loads and synchronizes 1-minute BTCUSDT and NQ futures data.
- Normalizes timestamps into a shared timeline to avoid cross-market misalignment.
- Computes rolling Pearson and Spearman correlation plus directional hit-ratio metrics.
- Analyzes intraday seasonality, signal quality, event clustering, and forward returns.
- Runs a sensitivity grid search over signal windows and holding horizons.

## Repository Structure

- [`data/`](data): local minute-bar datasets used by the notebook.
- [`research/correlation_research.ipynb`](research/correlation_research.ipynb): main research notebook.
- [`src/config.py`](src/config.py): repo-relative paths and default research constants.
- [`src/data_pipeline.py`](src/data_pipeline.py): CSV loading, schema normalization, timezone alignment, and dataset joins.
- [`src/indicators.py`](src/indicators.py): rolling correlation, hit-ratio, ADF helper, and volume-weighted feature logic.
- [`src/simulation.py`](src/simulation.py): signal-state construction, event clustering, and event-study helpers.
- [`src/optimization.py`](src/optimization.py): sensitivity grid search across windows and hold periods.
- [`Archive - Previous Works/`](<Archive - Previous Works>): historical notebooks and outputs kept for reference.

## Data

The repo expects these local CSVs:

- `data/BTCUSDT_1m_2024-03-07_to_2026-03-07.csv`
- `data/NQ_stitched_1min_2024-03-07_to_2026-03-07.csv`

BTC data is treated as UTC-native. NQ data is parsed as `America/Chicago` exchange time and converted to UTC before alignment.

## Workflow

1. Open [`research/correlation_research.ipynb`](research/correlation_research.ipynb).
2. Run the setup cells, which import reusable functions from `src`.
3. Use the notebook for statistical tests, plots, and interpretation.
4. Extend `src` when a block becomes reusable or is needed outside the notebook.

## Current State

The active refactor goal is to keep research-specific narrative and visual exploration in the notebook while moving repeatable data prep, metric calculation, signal logic, and robustness checks into `src`.
