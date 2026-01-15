# 📦 Retail Demand Forecasting with Delayed Labels

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
```

---

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
