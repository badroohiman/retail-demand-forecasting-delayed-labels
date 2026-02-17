"""
Rolling-origin backtest for the two-stage model (P(y>0) × E[y|y>0]).
Uses same BacktestConfig as baselines and tuned LGBM for fair comparison.
Tunes probability threshold on validation set per fold.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.eval.backtest_baselines import BacktestConfig, mae, smape


def _get_cat_cols_and_features(df: pd.DataFrame, drop_cols: list[str]):
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


def _combine_stages(
    p: np.ndarray,
    mu: np.ndarray,
    mode: str,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Combine classifier output p and regressor output mu.
    - soft: y_pred = p * mu (conditional mean)
    - hard: y_pred = mu if p >= tau else 0 (hard gating)
    """
    mu = np.clip(mu, 0, None)
    if mode == "soft":
        return p * mu
    if mode == "hard":
        if threshold is None:
            raise ValueError("Threshold required for hard gating.")
        return np.where(p >= threshold, mu, 0.0)
    raise ValueError(f"Unknown mode: {mode}")


def _tune_threshold_soft(
    p_val: np.ndarray,
    mu_val: np.ndarray,
    y_val: np.ndarray,
    thresholds: list[float],
) -> float:
    """Pick threshold minimizing MAE for soft gating: p_adj * mu, p_adj = p if p>=th else 0."""
    best_mae = float("inf")
    best_thresh = 0.0
    for th in thresholds:
        p_adj = np.where(p_val >= th, p_val, 0.0)
        pred = p_adj * mu_val
        m = mae(y_val, pred)
        if m < best_mae:
            best_mae = m
            best_thresh = th
    return best_thresh


def _tune_threshold_hard(
    p_val: np.ndarray,
    mu_val: np.ndarray,
    y_val: np.ndarray,
    thresholds: list[float],
) -> float:
    """Pick threshold minimizing MAE for hard gating: mu if p>=tau else 0."""
    best_mae = float("inf")
    best_thresh = 0.0
    for th in thresholds:
        pred = np.where(p_val >= th, mu_val, 0.0)
        m = mae(y_val, pred)
        if m < best_mae:
            best_mae = m
            best_thresh = th
    return best_thresh


def rolling_origin_two_stage(
    df: pd.DataFrame,
    label_col: str,
    cfg: BacktestConfig,
    cat_cols: list[str],
    features: list[str],
    clf_params: dict,
    reg_params: dict,
    combination_mode: str,
    p_thresholds: list[float] | None = None,
    random_state: int = 42,
    max_folds: int | None = None,
) -> tuple[float, float]:
    """
    Run rolling-origin CV for the two-stage model.
    combination_mode: "soft" (p*mu), "soft_gated" (p_adj*mu, tune th), "hard" (mu if p>=th else 0, tune th)
    For soft_gated and hard, p_thresholds must be provided for validation tuning.
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

        train_mask = df["date"] < val_start
        val_mask = (df["date"] >= val_start) & (df["date"] <= origin_date)
        test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)

        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, label_col].astype(float).to_numpy()
        X_val = df.loc[val_mask, features]
        y_val = df.loc[val_mask, label_col].astype(float).to_numpy()
        X_test = df.loc[test_mask, features]
        y_test = df.loc[test_mask, label_col].astype(float).to_numpy()

        if len(y_test) == 0 or len(y_val) == 0:
            continue
        if train_mask.sum() < 1000:
            continue

        nz_mask = y_train > 0
        if nz_mask.sum() == 0:
            continue

        # Stage 1: classifier
        y_train_bin = nz_mask.astype(int)
        clf = lgb.LGBMClassifier(**clf_params, random_state=random_state)
        clf.fit(X_train, y_train_bin, categorical_feature=cat_cols)

        p_val = clf.predict_proba(X_val)[:, 1]
        p_test = clf.predict_proba(X_test)[:, 1]

        # Stage 2: regressor on non-zero only
        reg = lgb.LGBMRegressor(**reg_params, random_state=random_state)
        X_train_nz = X_train.iloc[nz_mask]
        y_train_nz = y_train[nz_mask]
        reg.fit(X_train_nz, y_train_nz, categorical_feature=cat_cols)
        mu_val = np.clip(reg.predict(X_val), 0, None)
        mu_test = np.clip(reg.predict(X_test), 0, None)

        # Combine stages: soft (no threshold), soft_gated/hard (tune on val)
        if combination_mode == "soft":
            pred_test = _combine_stages(p_test, mu_test, "soft")
        elif combination_mode == "soft_gated":
            if p_thresholds is None:
                raise ValueError("p_thresholds required for soft_gated.")
            best_th = _tune_threshold_soft(p_val, mu_val, y_val, p_thresholds)
            p_adj = np.where(p_test >= best_th, p_test, 0.0)
            pred_test = p_adj * mu_test
        elif combination_mode == "hard":
            if p_thresholds is None:
                raise ValueError("p_thresholds required for hard.")
            best_th = _tune_threshold_hard(p_val, mu_val, y_val, p_thresholds)
            pred_test = _combine_stages(p_test, mu_test, "hard", threshold=best_th)
        else:
            raise ValueError(f"Unknown combination_mode: {combination_mode}")

        mae_folds.append(mae(y_test, pred_test))
        smape_folds.append(smape(y_test, pred_test))

    min_folds = 3
    if len(mae_folds) < min_folds:
        raise ValueError(
            f"Too few valid folds: {len(mae_folds)} (need at least {min_folds})."
        )

    return float(np.mean(mae_folds)), float(np.mean(smape_folds))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rolling-origin backtest for two-stage intermittent-demand model."
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
    parser.add_argument("--max_folds", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        default="soft_gated",
        choices=["soft", "soft_gated", "hard"],
        help="soft=p*μ; soft_gated=thresholded p*μ; hard=μ if p≥τ else 0",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="reports/two_stage_backtest_results.csv",
    )
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

    mean_mae, mean_smape = rolling_origin_two_stage(
        df,
        args.label,
        cfg,
        cat_cols,
        features,
        clf_params,
        reg_params,
        combination_mode=args.mode,
        p_thresholds=p_thresholds if args.mode != "soft" else None,
        random_state=args.seed,
        max_folds=args.max_folds,
    )

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": "two_stage",
                "label": args.label,
                "mean_mae": mean_mae,
                "mean_smape": mean_smape,
                "horizon": args.horizon,
                "max_folds": args.max_folds,
            }
        ]
    ).to_csv(args.out_path, index=False)

    print("Two-stage rolling-origin backtest:")
    print(f"  mean_mae   = {mean_mae:.4f}")
    print(f"  mean_smape = {mean_smape:.4f}")
    print(f"[OK] Saved: {args.out_path}")


if __name__ == "__main__":
    main()
