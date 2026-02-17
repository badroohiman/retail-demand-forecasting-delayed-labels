# 📦 Retail Demand Forecasting with Delayed Labels

[![CI](https://github.com/badroohiman/retail-demand-forecasting-delayed-labels/actions/workflows/ci.yml/badge.svg)](https://github.com/badroohiman/retail-demand-forecasting-delayed-labels/actions/workflows/ci.yml)

## Overview

This project explores **retail demand forecasting** using daily item-level sales data, with a focus on **realistic production constraints** such as **delayed and revised labels**.

Rather than treating forecasting as a static supervised learning problem, the project explicitly models how **label availability changes over time**, which is a common but often ignored challenge in real-world ML systems.

---

## Project goals

* Build a **reproducible end-to-end pipeline** for demand forecasting
* Perform **decision-driven exploratory data analysis (EDA)**
* Understand **intermittent / zero-inflated demand**
* Simulate **label delay and revision scenarios**
* Design **time-aware evaluation and backtesting**
* Compare baselines and ML models under realistic constraints

---
## Key results (TL;DR)

- Demand is highly intermittent (~60% zero-sales days).
- **Tuned LightGBM (L1 objective)** achieves ~7% MAE improvement over a strong 28-day rolling baseline (≈1.016 vs ≈1.10) under rolling-origin backtesting.
- **Roll28** still wins on **sMAPE** (≈0.99 vs ≈1.24); metric choice materially affects model ranking.
- Evaluation by demand type shows LGBM improves MAE most on **erratic** and **smooth** series; roll28 remains better on sMAPE across all types.
- Results confirm that **evaluation design and data realism matter more than model complexity**.

## For quick reviewers

If you are short on time, the key sections are:
- Overview
- Key results (TL;DR)
- Modeling (Summary + Results and comparison)

## What makes this project different

Most demand forecasting examples assume immediate and final ground truth.
This project explicitly models **label latency and revision**, a common but under-documented
challenge in real production systems.

Key differentiators:
- Explicit simulation of delayed and revised labels (v0 / v1 / v2)
- Time-aware backtesting with no leakage
- Honest comparison against strong statistical baselines
- Focus on decision-making under realistic constraints, not leaderboard optimization

## Dataset

* Source: **M5 Forecasting dataset (Kaggle)**
* Granularity: **Daily sales per item per store**
* Features:

  * Target: `sales`
  * Calendar features (events, SNAP flags)
  * Weekly sell prices
* Data access is handled via the Kaggle API (credentials are **not** committed).

> ⚠️ For reproducibility and resource constraints, most analysis is performed on a **controlled sample**
> (1 store × 200 items). All logic is designed to scale to larger subsets or the full dataset.

---

## Repository structure

```
.
├── src/
│   ├── data/
│   │   ├── download.py           # Kaggle data download
│   │   └── preprocess.py         # Raw → canonical dataset
│   ├── labels/
│   │   └── make_label_versions.py # Delayed labels (y_v0, y_v1, y_v2)
│   ├── features/
│   │   └── build_features.py     # Lag, rolling, calendar, availability
│   ├── models/
│   │   ├── train_lgbm.py         # Single-split LightGBM
│   │   ├── tune_lgbm.py          # Rolling-origin CV + hyperparameter grid
│   │   ├── train_two_stage.py    # Intermittent-demand (P>0 × E[y|y>0])
│   │   └── tune_two_stage_minimal.py
│   └── eval/
│       ├── backtest_baselines.py # Rolling-origin baseline evaluation
│       └── eval_best_by_demand_type.py  # Model vs baseline by demand type
├── notebooks/
│   └── 01_eda.ipynb             # Decision-driven EDA
├── data/
│   ├── raw/                     # Ignored (Kaggle data)
│   └── processed/               # Ignored (parquet outputs)
├── models/                      # Saved models (e.g. lgbm_tuned.txt)
├── reports/                     # Backtest results, figures, eval_by_demand_type.csv
├── requirements.txt
└── README.md
```


## Exploratory Data Analysis (EDA)

The EDA (`notebooks/01_eda.ipynb`) focuses on **understanding the data before modeling**, rather than premature cleaning.

### Key findings (sample: 1 store × 200 items)

* **Strong zero-inflation**: ~60% of daily observations have zero sales.
* **Right-skewed, bursty demand**: sales occur intermittently rather than smoothly.
* **Event features are sparse by design** and primarily represent “no event” states.
* **Sell prices have non-trivial missingness (~15%)**, which appears structural but requires validation.
* **Temporal structure dominates** cross-sectional variation.

### Modeling implications

* Zero sales are treated as **valid observations**, not missing values.
* Evaluation metrics should be robust to intermittency (e.g. MAE, sMAPE).
* Time-aware validation and lag/rolling features are essential.
* Cleaning and imputation decisions are deferred until after EDA.

---

## Reproducibility

* All paths are resolved **relative to the repository root**.
* Output directories are created programmatically.
* Raw and processed data are excluded from version control.
* Notebooks can be run end-to-end after executing preprocessing.

### Minimal setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data preparation and pipeline

```bash
# 1. Download and preprocess
python src/data/download.py
python src/data/preprocess.py --sample_stores 1 --sample_items 200 \
  --out_path data/processed/m5_daily_sample.parquet

# 2. Create delayed label versions
python src/labels/make_label_versions.py \
  --in_path data/processed/m5_daily_sample.parquet \
  --out_path data/processed/m5_labeled_sample.parquet

# 3. Build features
python src/features/build_features.py \
  --in_path data/processed/m5_labeled_sample.parquet \
  --out_path data/processed/m5_features_sample.parquet

# 4. Tune LightGBM (optional)
python -m src.models.tune_lgbm --results_path reports/tune_lgbm_results.csv

# 5. Evaluate baselines
python -m src.eval.backtest_baselines --in_path data/processed/m5_labeled_sample.parquet

# 6. Evaluate best model by demand type
python -m src.eval.eval_best_by_demand_type
```

---

## Next steps

* ~~Implement delayed-label variants (v0 / v1 / v2)~~ ✓
* ~~Design time-aware backtesting across label versions~~ ✓
* ~~Establish strong baselines and tuned LightGBM~~ ✓
* Further experimentation: two-stage intermittent model, weighted MAE for M5 scale differences.

---

## Why this project

Many forecasting examples assume immediate and perfect labels.
This project focuses on **how forecasting systems behave when labels arrive late or are revised**, which is often the reality in production environments.

---

## Notes

* Kaggle credentials and data are intentionally excluded.
* The repository prioritizes **clarity, reproducibility, and decision transparency** over leaderboard optimization.

### Delayed ground truth simulation (v0 / v1 / v2)

Real retail systems often have delayed and revised sales reporting. To reflect this, we create three label versions:

- **y_v0**: same-day label (most incomplete)
- **y_v1**: 7-day matured label (less incomplete)
- **y_v2**: final label (treated as ground truth)

Labels are simulated by under-reporting a subset of non-zero days with a reproducible random seed.
This allows evaluation of how reporting latency impacts model performance and operational decisions.

### Time-aware baselines (rolling-origin backtest)

We evaluate naive baselines using rolling-origin backtesting (no shuffling, no leakage).
Baselines:
- naive (lag-1)
- seasonal naive (lag-7)
- rolling mean (28-day)

Results are reported for delayed label maturities `y_v0`, `y_v1`, `y_v2` to quantify the impact of reporting latency.

## Time-aware baseline evaluation — results

We evaluated three baseline forecasters using rolling-origin backtesting:
- naive (lag-1)
- seasonal naive (lag-7)
- rolling mean (28 days)

Results were compared across delayed label maturities (`y_v0`, `y_v1`, `y_v2`).

### Key observations
- Error increases as label maturity decreases (`y_v2` → `y_v1` → `y_v0`), validating the delayed-label simulation.
- The 28-day rolling mean achieves the lowest MAE across all label versions.
- The naive baseline achieves the lowest sMAPE due to frequent zero predictions.

### Interpretation
- Rolling averages reduce absolute error by smoothing demand but tend to over-predict on zero-sales days.
- Relative error metrics (sMAPE) strongly penalize non-zero predictions when true demand is zero, favoring sparse predictors.
- Metric choice materially affects model ranking in zero-inflated demand settings.

## 📐 Modeling

**Summary**

A tuned global LightGBM model (L1 objective) outperforms the 28-day rolling-mean baseline on **MAE** under rolling-origin evaluation, while roll28 remains better on **sMAPE**. Evaluation by demand type shows MAE gains are largest on erratic and smooth series. A two-stage intermittent-demand model is also available for comparison.

### Modeling strategy

Modeling was approached **after** establishing strong statistical baselines and a robust time-aware evaluation framework.

Given the observed characteristics of the data:

* strong zero-inflation
* intermittent, bursty demand
* weak short-term autocorrelation
* delayed and revised labels

the goal of modeling was **not to maximize accuracy at all costs**, but to assess whether a **global machine learning model** could meaningfully outperform well-calibrated statistical baselines under realistic constraints.

---

### Feature engineering

Features were constructed to be **time-safe**, meaning that all inputs for a given prediction date were available strictly **before** that date.

The feature set includes:

* **Lag features**: `lag_1`, `lag_7`, `lag_14`, `lag_28`
* **Rolling statistics**: rolling mean (7, 28 days), rolling std (28 days)
* **Calendar features**: day of week, week of year, month
* **Price features**: current sell price, 7-day relative price change
* **Availability features**: `in_catalog` (product in catalog), `days_since_first_price` (handles structural zeros)
* **Categorical identifiers**: item, department, category, store, state, event name/type (handled natively)

No target leakage or forward-looking aggregation is used.

---

### Machine learning model

A **global LightGBM model** is trained with **rolling-origin CV** and hyperparameter tuning. The primary objective is **regression_l1 (MAE)**; **Tweedie** is also explored for zero-inflated data.

Key modeling choices:

* single global model across all items
* native handling of categorical features (no one-hot encoding)
* **rolling-origin CV** (horizon=28, step=28, min_train_days=365) with early stopping
* grid over `num_leaves`, `min_data_in_leaf`, `reg_alpha`, `reg_lambda`, `max_depth`, and objective (L1 vs Tweedie)
* **sMAPE as tie-breaker** when MAE is (nearly) equal
* evaluation against the same horizon and metrics as the baselines

The model is trained on final labels (`y_v2`) and evaluated using MAE and sMAPE.

---

### Results and comparison

On final labels (`y_v2`) with rolling-origin evaluation:

| Model | MAE | sMAPE |
|-------|-----|-------|
| **Tuned LightGBM (L1)** | **1.016** | 1.24 |
| Rolling mean (28-day) | 1.10 | **0.99** |

* **LightGBM** outperforms roll28 on **MAE**.
* **Roll28** outperforms LightGBM on **sMAPE** (metric choice drives ranking).

### Evaluation by demand type

Using ADI/CV² classification (Syntetos–Boylan), we break down performance by demand type (smooth / intermittent / erratic / lumpy):

| Demand type | MAE improvement (LGBM vs roll28) | sMAPE |
|-------------|----------------------------------|-------|
| Erratic | **+0.28** | roll28 better |
| Smooth | **+0.19** | roll28 better |
| Lumpy | +0.13 | roll28 better |
| Intermittent | +0.10 | roll28 better |

**MAE improvement** is largest on erratic and smooth series; **sMAPE** favors roll28 across all types.

---

### Interpretation

This result is **expected and informative** in the context of intermittent retail demand:

* Tuned LightGBM **does** outperform roll28 on MAE, especially on erratic and smooth demand.
* sMAPE strongly favors roll28 (sparse, zero-biased predictions).
* Metric choice drives model ranking; the same models swap rank on MAE vs sMAPE.

The outcome highlights:

* the strength of well-chosen baselines
* the importance of honest evaluation and metric alignment
* the value of demand-type breakdown to understand where models improve



### Modeling takeaways

* Machine learning should be **justified**, not assumed to be superior.
* Strong statistical baselines are essential reference points.
* Evaluation design matters more than model complexity.
* Label maturity and data availability materially affect measured performance.

### Hyperparameter tuning

LightGBM is tuned via rolling-origin CV over a grid of:

* `num_leaves`, `min_data_in_leaf`, `reg_alpha`, `reg_lambda`, `max_depth`
* Objective: `regression_l1` and `tweedie` (variance power 1.1, 1.2, 1.3 for zero-inflated data)

Results are saved to CSV (e.g. `reports/tune_lgbm_results.csv`); the best config (by MAE, sMAPE tie-breaker) is retrained and saved to `models/lgbm_tuned.txt`.

👤 Author
Iman Badrooh Data Scientist / Machine Learning Engineer (UK)
