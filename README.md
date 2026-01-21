# Retail Demand Forecasting (Delayed Labels) — Two-Stage LightGBM

## What this repo shows
An end-to-end, production-style pipeline for retail demand forecasting with delayed/partial labels and memory-safe chunked processing (Parquet shards).

## Pipeline
- Chunked preprocess → data/processed/m5_daily_full_ds/
- Delayed labels (y_v0, y_v1, y_v2) → data/processed/m5_labeled_full_ds/
- Chunked features → data/processed/m5_features_full_ds/
- Two-stage training (per shard) → models + reports

## Model
Two-stage for sparse demand:
1) Classifier: P(demand > 0)
2) Regressor: E(demand | demand > 0)
Prediction: y_hat = p × mu

## Results (20 shards, split at 2015-01-01)
- Weighted MAE: 0.9260
- Weighted sMAPE: 1.4813
- Test rows: ~1.9M

## Key folders
- src/data/, src/labels/, src/features/, src/models/
- reports/
- artifacts/models/
