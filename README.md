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
- Strong statistical baselines (28-day rolling mean) are extremely competitive.
- A global Poisson LightGBM model performed comparably but did not outperform the rolling baseline on MAE.
- A two-stage intermittent-demand model (P(y>0) × E[y|y>0]) achieved marginal improvements after minimal tuning.
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
.
├── src/
│   └── data/
│       ├── download.py        # Kaggle data download (competition-aware)
│       └── preprocess.py      # Raw → canonical dataset (sample-aware)
│
├── notebooks/
│   └── 01_eda.ipynb           # Decision-driven exploratory data analysis
│
├── data/
│   ├── raw/                   # Ignored (Kaggle data)
│   └── processed/             # Ignored (parquet outputs)
│
├── reports/
│   └── figures/               # Generated plots (small artifacts only)
│
├── requirements.txt
├── .gitignore
└── README.md


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

### Data preparation

```bash
python src/data/download.py
python src/data/preprocess.py --sample_stores 1 --sample_items 200 \
  --out_path data/processed/m5_daily_sample.parquet
```

---

## Next steps

* Implement **delayed-label variants** (v0 / v1 / v2) to simulate production label availability.
* Design **time-aware backtesting** across label versions.
* Establish strong baselines before introducing ML models.
* Evaluate how label delay impacts model selection and performance.

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

A global LightGBM model (Poisson objective) and a two-stage intermittent-demand model were evaluated under strict time-based splits. Neither substantially outperformed a well-calibrated rolling-mean baseline on MAE, highlighting the competitiveness of statistical methods in sparse retail demand.

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

* **Lag features**

  * `lag_1`, `lag_7`, `lag_14`, `lag_28`
* **Rolling statistics**

  * rolling mean (7, 28 days)
  * rolling standard deviation (28 days)
* **Calendar features**

  * day of week
  * week of year
  * month
* **Price features**

  * current sell price
  * 7-day relative price change
* **Categorical identifiers**

  * item, department, category, store, state
  * event name and event type (handled natively by the model)

No target leakage or forward-looking aggregation is used.

---

### Machine learning model

A **global LightGBM model** was trained using a **Poisson objective**, which is appropriate for non-negative count data and commonly used in demand forecasting.

Key modeling choices:

* single global model across all items
* native handling of categorical features (no one-hot encoding)
* strict time-based train/test split
* evaluation against the same horizon and metrics as the baselines

The model was trained on final labels (`y_v2`) and evaluated using MAE and sMAPE.

---

### Results and comparison

On final labels (`y_v2`):

* **LightGBM (Poisson)** achieved MAE ≈ **1.13**
* **Rolling mean (28-day)** baseline achieved MAE ≈ **1.10**

The machine learning model performed **comparably but did not outperform** the strongest statistical baseline on MAE.

---

### Interpretation

This result is **expected and informative** in the context of intermittent retail demand:

* Rolling averages are highly competitive in sparse demand regimes.
* MAE favors conservative predictors that avoid over-predicting on zero-sales days.
* A single global model may struggle to outperform item-level statistical heuristics without more specialized structure.

Rather than indicating modeling failure, this outcome highlights:

* the strength of well-chosen baselines
* the importance of honest evaluation
* the limits of generic ML models in zero-inflated demand settings



### Modeling takeaways

* Machine learning should be **justified**, not assumed to be superior.
* Strong statistical baselines are essential reference points.
* Evaluation design matters more than model complexity.
* Label maturity and data availability materially affect measured performance.

### Minimal hyperparameter tuning

A bounded 6-run tuning experiment was conducted on the two-stage model,
varying only high-impact LightGBM parameters (`num_leaves`, `min_child_samples`)
under a fixed time-based split.

The best configuration slightly improved MAE (≈1% relative gain) by increasing
capacity in the regression stage, indicating that performance is primarily
limited by modeling the magnitude of non-zero demand rather than occurrence.

Further tuning was intentionally stopped to avoid overfitting the evaluation split
and to preserve the interpretability and credibility of the results.
