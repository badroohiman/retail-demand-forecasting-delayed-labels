"""
Tune two-stage model under rolling-origin backtesting (same protocol as tune_lgbm).
Focus: decision rule (soft vs hard gating), not tree capacity.
Compares (1) soft p*μ, (2) soft_gated p_adj*μ with threshold, (3) hard μ if p≥τ else 0.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.eval.backtest_baselines import BacktestConfig
from src.eval.backtest_two_stage import (
    _get_cat_cols_and_features,
    rolling_origin_two_stage,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage model tuning under rolling-origin backtesting. "
        "Compares soft vs hard gating with validation-based threshold tuning."
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default="data/processed/m5_features_sample.parquet",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="reports/tuning_two_stage_minimal.csv",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="y_v2",
        choices=["y_v0", "y_v1", "y_v2"],
    )
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--min_train_days", type=int, default=365)
    parser.add_argument("--step", type=int, default=28)
    parser.add_argument("--max_folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing input: {in_path}\n"
            "Run: python src/features/build_features.py (after make_label_versions)"
        )

    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"])

    drop_cols = ["date", "sales", "y_v0", "y_v1", "y_v2", "d"]
    cat_cols, features = _get_cat_cols_and_features(df, drop_cols)
    for c in cat_cols:
        df[c] = df[c].astype("category")
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

    # Fixed params (regression_l1, best LGBM-like); focus on decision rule
    clf_params = dict(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
    )
    reg_params = dict(
        objective="regression_l1",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=30,
        reg_alpha=1.0,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    p_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

    # Compare soft, soft_gated, hard
    rows = []
    for mode in ["soft", "soft_gated", "hard"]:
        mean_mae, mean_smape = rolling_origin_two_stage(
            df,
            args.label,
            cfg,
            cat_cols,
            features,
            clf_params,
            reg_params,
            combination_mode=mode,
            p_thresholds=p_thresholds if mode != "soft" else None,
            random_state=args.seed,
            max_folds=args.max_folds,
        )
        rows.append(
            {
                "combination_mode": mode,
                "label": args.label,
                "mean_mae": mean_mae,
                "mean_smape": mean_smape,
                "horizon": args.horizon,
                "max_folds": args.max_folds,
            }
        )
        print(f"{mode}: MAE={mean_mae:.4f} sMAPE={mean_smape:.4f}")

    out = pd.DataFrame(rows).sort_values("mean_mae")
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)

    print(f"\n[OK] Saved: {args.out_path}")
    print("\nResults (sorted by MAE):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
