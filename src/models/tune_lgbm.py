"""
Tune single LightGBM with rolling-origin CV.
Primary metric: MAE (minimize). Secondary: sMAPE (reported).
"""

import argparse
import itertools
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.eval.backtest_baselines import BacktestConfig, mae, smape


def _get_cat_cols_and_features(df: pd.DataFrame, drop_cols: list[str]):
    # Exclude "d" (high-cardinality day index causes "too many bins" warning)
    cat_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]
    features = [c for c in df.columns if c not in drop_cols]
    return cat_cols, features


def rolling_origin_lgbm_cv(
    df: pd.DataFrame,
    label_col: str,
    cfg: BacktestConfig,
    lgbm_params: dict,
    cat_cols: list[str],
    features: list[str],
    random_state: int = 42,
    max_folds: int | None = None,
) -> tuple[float, float]:
    """
    Run rolling-origin CV for one LightGBM config.
    Uses early stopping (eval on last horizon days of train, eval_metric=l1/MAE).
    Returns (mean_mae, mean_smape) over folds.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = np.array(sorted(df["date"].unique()))

    if len(dates) < cfg.min_train_days + cfg.horizon:
        raise ValueError(
            f"Not enough dates for backtest. Have {len(dates)} unique days, "
            f"need at least {cfg.min_train_days + cfg.horizon}."
        )

    first_origin_idx = cfg.min_train_days
    last_origin_idx = len(dates) - cfg.horizon - 1
    origin_indices = list(range(first_origin_idx, last_origin_idx + 1, cfg.step))
    if max_folds is not None and len(origin_indices) > max_folds:
        origin_indices = origin_indices[-max_folds:]

    mae_folds = []
    smape_folds = []

    for origin_idx in origin_indices:
        origin_date = pd.Timestamp(dates[origin_idx])
        val_start = origin_date - pd.Timedelta(days=cfg.horizon - 1)
        test_start = origin_date + pd.Timedelta(days=1)
        test_end = origin_date + pd.Timedelta(days=cfg.horizon)

        train2_mask = df["date"] < val_start
        val_mask = (df["date"] >= val_start) & (df["date"] <= origin_date)
        test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)

        X_train = df.loc[train2_mask, features]
        y_train = df.loc[train2_mask, label_col]
        X_val = df.loc[val_mask, features]
        y_val = df.loc[val_mask, label_col]
        X_test = df.loc[test_mask, features]
        y_test = df.loc[test_mask, label_col].to_numpy()

        if len(y_test) == 0 or len(y_val) == 0:
            continue
        if train2_mask.sum() < 1000:
            continue

        model = lgb.LGBMRegressor(**lgbm_params, random_state=random_state)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        n_iter = model.best_iteration_ or lgbm_params.get("n_estimators", 500)
        preds = model.predict(X_test, num_iteration=n_iter)
        preds = np.clip(preds, 0, None)

        mae_folds.append(mae(y_test, preds))
        smape_folds.append(smape(y_test, preds))

    min_folds = 3
    if len(mae_folds) < min_folds:
        raise ValueError(
            f"Too few valid folds: {len(mae_folds)} (need at least {min_folds}). "
            "Check data coverage or reduce min_train_days / horizon."
        )

    mean_mae = float(np.mean(mae_folds))
    mean_smape = float(np.mean(smape_folds))
    return mean_mae, mean_smape


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune LightGBM with rolling-origin CV (MAE-first, sMAPE reported)."
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default="data/processed/m5_features_sample.parquet",
    )
    parser.add_argument("--label", type=str, default="y_v2")
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--min_train_days", type=int, default=365)
    parser.add_argument("--step", type=int, default=28)
    parser.add_argument(
        "--max_folds",
        type=int,
        default=10,
        help="Max number of CV folds (use last N origins to limit runtime).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="models/lgbm_tuned.txt",
        help="Path to save best model (retrained on full train window).",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="reports/tune_lgbm_results.csv",
        help="Path to save grid search results (CSV) for tracking and comparison.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing input: {in_path}\n"
            "Run: python src/features/build_features.py  (after make_label_versions)"
        )

    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"])

    drop_cols = ["date", "sales", "y_v0", "y_v1", "y_v2"]
    cat_cols, features = _get_cat_cols_and_features(df, drop_cols)
    for c in cat_cols:
        df[c] = df[c].astype("category")
    # Ensure object columns (e.g. event names) are categorical
    obj_cols = [c for c in features if df[c].dtype == "object"]
    for c in obj_cols:
        df[c] = df[c].astype("category")
        if c not in cat_cols:
            cat_cols.append(c)

    cfg = BacktestConfig(
        horizon=args.horizon,
        min_train_days=args.min_train_days,
        step=args.step,
    )

    # Grid: L1 + Tweedie (1.1, 1.2, 1.3 for zero-inflated data ~60% zeros)
    base_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "force_row_wise": True,
        "verbosity": -1,
        "n_jobs": -1,
    }
    objective_configs = [
        {"objective": "regression_l1"},
        {"objective": "tweedie", "tweedie_variance_power": 1.1},
        {"objective": "tweedie", "tweedie_variance_power": 1.2},
        {"objective": "tweedie", "tweedie_variance_power": 1.3},
    ]
    grid = {
        "num_leaves": [15, 31],
        "min_data_in_leaf": [30, 70],
        "reg_alpha": [0.1, 1.0],
        "reg_lambda": [0.1, 1.0],
        "max_depth": [-1, 8],
    }

    results = []
    keys = list(grid.keys())
    for obj_cfg in objective_configs:
        for values in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, values))
            lgbm_params = {**base_params, **obj_cfg, **params}
            mean_mae, mean_smape = rolling_origin_lgbm_cv(
                df,
                args.label,
                cfg,
                lgbm_params,
                cat_cols,
                features,
                random_state=args.seed,
                max_folds=args.max_folds,
            )
            results.append(
                {"mean_mae": mean_mae, "mean_smape": mean_smape, "params": lgbm_params}
            )

    # Sort by MAE (primary), sMAPE as tie-breaker
    results.sort(key=lambda x: (x["mean_mae"], x["mean_smape"]))
    best = results[0]

    # Save grid results for tracking and comparison
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        row = {"mean_mae": r["mean_mae"], "mean_smape": r["mean_smape"]}
        row.update(r["params"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"[OK] Grid results saved to {results_path}")

    print("Tune results (sorted by mean MAE):")
    print("-" * 60)
    for i, r in enumerate(results[:10], 1):
        p = r["params"]
        obj = p.get("objective", "regression_l1")
        tvp = f" tvp={p.get('tweedie_variance_power', '')}" if obj == "tweedie" else ""
        print(
            f"  {i}. MAE={r['mean_mae']:.4f}  sMAPE={r['mean_smape']:.4f}  "
            f"obj={obj}{tvp}  num_leaves={p['num_leaves']} min_d={p['min_data_in_leaf']}"
        )
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more")
    print("-" * 60)
    print("Best (MAE-first):")
    print(f"  mean_mae   = {best['mean_mae']:.4f}")
    print(f"  mean_smape = {best['mean_smape']:.4f}")
    print("  params:", best["params"])

    # Retrain best on full train window aligned with CV: date <= last_origin_date
    dates = np.array(sorted(df["date"].unique()))
    last_origin_idx = len(dates) - cfg.horizon - 1
    last_origin_date = pd.Timestamp(dates[last_origin_idx])
    print(
        f"\nRetrain up to last_origin_date={last_origin_date.date()} "
        "(exclude final horizon window)"
    )
    train_mask = df["date"] <= last_origin_date
    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, args.label]
    final_model = lgb.LGBMRegressor(**best["params"], random_state=args.seed)
    final_model.fit(X_train, y_train, categorical_feature=cat_cols)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(out_path))
    print(f"[OK] Best model saved to {out_path}")


if __name__ == "__main__":
    main()
