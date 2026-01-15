# AI Coding Agent Instructions for Retail Demand Forecasting

## Project Overview
This is a machine learning project for retail demand forecasting using the M5 dataset, with a focus on realistic production constraints like delayed and revised labels. The codebase emphasizes reproducibility, decision-driven EDA, and time-aware evaluation over leaderboard optimization.

## Architecture & Data Flow
- **Data Pipeline**: Raw M5 CSVs → canonical daily parquet via `src/data/download.py` and `src/data/preprocess.py`
- **EDA**: Decision-driven analysis in `notebooks/01_eda.ipynb` on sampled data (1 store × 200 items)
- **Modeling**: Future ML models in `src/modeling/` (currently empty)
- **Deployment/Monitoring**: Infrastructure in `src/deploy/` and `src/monitoring/` (currently empty)

Key data transformations:
- Wide sales format (d_1...d_N) melted to long format with `to_long_sales()`
- Joined with calendar (on 'd') and prices (on store_id, item_id, wm_yr_wk)
- Zero sales treated as valid observations, not missing values

## Critical Workflows
- **Setup**: `make venv && make install && make install-dev`
- **Development**: `make run-notebook` for Jupyter, `make format` for code/notebook formatting
- **Quality**: `make check` runs lint (ruff), format-check (black), notebook validation, and tests
- **Data Prep**: `python src/data/download.py` then `python src/data/preprocess.py --sample_stores 1 --sample_items 200`

## Project-Specific Conventions
- **Paths**: All paths resolved relative to repository root; output dirs created programmatically
- **Data Handling**: Raw/processed data ignored in git; use parquet for processed data
- **Sampling**: Sample stores/items early in preprocessing to avoid OOM in dev environments
- **Time Awareness**: Evaluation must account for label delays; use date-based splits, not random
- **Zero Inflation**: ~60% zero sales in sample; metrics like MAE/sMAPE preferred over MSE
- **Script Structure**: Use argparse for CLI scripts; functions return DataFrames for composability

## Code Patterns & Examples
- **Data Loading**: Check file variants (validation vs evaluation) in `load_raw()`
- **Merging**: Left joins with sanity checks for missing dates/prices
- **Column Selection**: Keep minimal columns initially, expand during feature engineering
- **Notebook Formatting**: Use `nbqa black` and `nbqa ruff` for consistent notebook code
- **Error Handling**: Raise descriptive errors for missing files or failed joins

## Dependencies & Environment
- **Python**: Virtual env in `.venv/` with requirements.txt (includes pandas, scikit-learn, matplotlib, jupyter)
- **Linting**: ruff for fast Python linting
- **Formatting**: black for code, nbqa for notebooks
- **Testing**: pytest (currently no tests implemented)
- **External**: Kaggle API for data download (credentials in ~/.kaggle/kaggle.json)

## Key Files to Reference
- `src/data/preprocess.py`: Exemplifies data pipeline patterns and sampling logic
- `notebooks/01_eda.ipynb`: Shows EDA approach prioritizing understanding over cleaning
- `Makefile`: Defines all development workflows and tool integrations
- `README.md`: Documents project goals, dataset details, and reproducibility setup</content>
<parameter name="filePath">/workspaces/retail-demand-forecasting-delayed-labels/.github/copilot-instructions.md