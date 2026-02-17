"""
Evaluate the best tuned LightGBM by demand type (smooth / intermittent / erratic / lumpy).

- Loads models/lgbm_tuned.txt and runs same rolling-origin folds as tune_lgbm.
- Compares predictions to roll28 baseline.
- Classifies each (store_id, item_id) with ADI/CV² (Syntetos–Boylan).
- Reports MAE and sMAPE by demand_type for both model and baseline, plus improvement.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.eval.backtest_baselines import BacktestConfig, mae, smape


def _get_cat_cols_and_features(df: pd.DataFrame, drop_cols: list[str]) -> tuple[list[str], list[str]]:
    cat_cols = [
        "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]
    features = [c for c in df.columns if c not in drop_cols]
    return cat_cols, features


def _add_roll28(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Add pred_roll28 = mean(y over t-28..t-1) per series."""
    out = df.copy()
    g = out.groupby(["store_id", "item_id"], sort=False, observed=True)[label_col]
    out["pred_roll28"] = g.shift(1).transform(lambda s: s.rolling(28, min_periods=1).mean())
    return out


def _adi_cv2(series: pd.Series) -> dict:
    """ADI = mean gap (periods) between non-zero demand; CV² = (std/mean)² of non-zero sizes."""
    sales = series.values
    nonzero_idx = np.where(sales > 0)[0]
    if len(nonzero_idx) < 2:
        return {"ADI": np.nan, "CV2": np.nan}
    gaps = np.diff(nonzero_idx)
    adi = float(np.mean(gaps))
    sizes = sales[nonzero_idx]
    mean_d = float(np.mean(sizes))
    cv2 = (np.std(sizes) / mean_d) ** 2 if mean_d > 0 else np.nan
    return {"ADI": adi, "CV2": float(cv2) if not np.isnan(cv2) else np.nan}


def _classify_demand_type(adi: float, cv2: float) -> str:
    """Syntetos–Boylan: smooth / intermittent / erratic / lumpy."""
    if pd.isna(adi) or pd.isna(cv2):
        return "unknown"
    if adi < 1.32 and cv2 < 0.49:
        return "smooth"
    if adi >= 1.32 and cv2 < 0.49:
        return "intermittent"
    if adi < 1.32 and cv2 >= 0.49:
        return "erratic"
    return "lumpy"


def compute_demand_type_per_series(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """One row per (store_id, item_id) with columns ADI, CV2, demand_type."""
    def adi_cv2_row(g):
        d = _adi_cv2(g[label_col])
        d["demand_type"] = _classify_demand_type(d["ADI"], d["CV2"])
        return pd.Series(d)

    out = (
        df.sort_values(["store_id", "item_id", "date"])
        .groupby(["store_id", "item_id"], sort=False, observed=True)
        .apply(adi_cv2_row, include_groups=False)
        .reset_index()
    )
    return out


def run_eval(
    df: pd.DataFrame,
    label_col: str,
    cfg: BacktestConfig,
    booster: lgb.Booster,
    features: list[str],
    cat_cols: list[str],
    pandas_categorical: list | None = None,
    max_folds: int = 10,
) -> pd.DataFrame:
    """
    Run rolling-origin folds; for each test row return store_id, item_id, date, y_true, pred_lgbm, pred_roll28.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = np.array(sorted(df["date"].unique()))

    if len(dates) < cfg.min_train_days + cfg.horizon:
        raise ValueError(
            f"Not enough dates. Have {len(dates)}, need {cfg.min_train_days + cfg.horizon}."
        )

    first_origin_idx = cfg.min_train_days
    last_origin_idx = len(dates) - cfg.horizon - 1
    origin_indices = list(range(first_origin_idx, last_origin_idx + 1, cfg.step))
    if len(origin_indices) > max_folds:
        origin_indices = origin_indices[-max_folds:]

    df = _add_roll28(df, label_col)

    rows = []
    for origin_idx in origin_indices:
        origin_date = pd.Timestamp(dates[origin_idx])
        test_start = origin_date + pd.Timedelta(days=1)
        test_end = origin_date + pd.Timedelta(days=cfg.horizon)

        test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)
        fold_df = df.loc[test_mask].copy()
        if len(fold_df) == 0:
            continue

        # Drop test rows where roll28 is NaN (not enough history)
        fold_df = fold_df[fold_df["pred_roll28"].notna()]
        if len(fold_df) == 0:
            continue

        X_test = fold_df[features].copy()
        # Only the first len(pandas_categorical) columns must be category; rest must not be (object→numeric)
        for i, c in enumerate(cat_cols):
            if c not in X_test.columns:
                continue
            levels = pandas_categorical[i] if pandas_categorical and i < len(pandas_categorical) else None
            if levels is not None:
                X_test[c] = pd.Categorical(X_test[c].astype(str), categories=[str(x) for x in levels])
            else:
                X_test[c] = X_test[c].astype("category")
        # Columns not in cat_cols must not be object/category or LightGBM infers extra categoricals
        for c in X_test.columns:
            if c not in cat_cols and X_test[c].dtype.name in ("object", "category"):
                X_test[c] = pd.Categorical(X_test[c]).codes

        pred_lgbm = booster.predict(X_test)
        pred_lgbm = np.clip(pred_lgbm, 0, None)

        fold_df = fold_df[["store_id", "item_id", "date", label_col, "pred_roll28"]].copy()
        fold_df = fold_df.rename(columns={label_col: "y_true"})
        fold_df["pred_lgbm"] = pred_lgbm
        fold_df["pred_roll28"] = fold_df["pred_roll28"].astype(float)
        rows.append(fold_df)

    if not rows:
        raise ValueError("No valid fold data.")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate best LightGBM by demand type (smooth/intermittent/erratic/lumpy)."
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default="data/processed/m5_features_sample.parquet",
    )
    parser.add_argument("--model_path", type=str, default="models/lgbm_tuned.txt")
    parser.add_argument("--label", type=str, default="y_v2")
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--min_train_days", type=int, default=365)
    parser.add_argument("--step", type=int, default=28)
    parser.add_argument("--max_folds", type=int, default=10)
    parser.add_argument(
        "--out_path",
        type=str,
        default="reports/eval_by_demand_type.csv",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    model_path = Path(args.model_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}. Run tune_lgbm first and save best model."
        )

    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"])

    drop_cols = ["date", "sales", "y_v0", "y_v1", "y_v2"]
    cat_cols, features = _get_cat_cols_and_features(df, drop_cols)
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Feature order and categorical set must match the saved model
    booster = lgb.Booster(model_file=str(model_path))
    model_features = booster.feature_name()
    features = [f for f in model_features if f in df.columns]
    if len(features) != len(model_features):
        raise ValueError(
            f"Model has {len(model_features)} features but only {len(features)} found in data. "
            "Missing: " + str(set(model_features) - set(df.columns))
        )
    # Use model's pandas_categorical so predict() sees same categorical columns as training
    dump = booster.dump_model()
    pandas_cat = dump.get("pandas_categorical", [])
    cat_cols_from_model = [model_features[i] for i in range(len(pandas_cat))]
    cat_cols = [c for c in cat_cols_from_model if c in df.columns]

    cfg = BacktestConfig(
        horizon=args.horizon,
        min_train_days=args.min_train_days,
        step=args.step,
    )

    pred_df = run_eval(
        df,
        args.label,
        cfg,
        booster,
        features,
        cat_cols,
        pandas_categorical=dump.get("pandas_categorical"),
        max_folds=args.max_folds,
    )

    # Demand type per series (on full history)
    demand_type_df = compute_demand_type_per_series(df, args.label)
    pred_df = pred_df.merge(
        demand_type_df[["store_id", "item_id", "demand_type"]],
        on=["store_id", "item_id"],
        how="left",
    )
    pred_df["demand_type"] = pred_df["demand_type"].fillna("unknown")

    # Metrics by demand_type
    def metrics_by_type(g):
        y = g["y_true"].to_numpy()
        p_lgb = g["pred_lgbm"].to_numpy()
        p_r28 = g["pred_roll28"].to_numpy()
        return pd.Series({
            "n_obs": len(y),
            "mae_lgbm": mae(y, p_lgb),
            "smape_lgbm": smape(y, p_lgb),
            "mae_roll28": mae(y, p_r28),
            "smape_roll28": smape(y, p_r28),
        })

    by_type = (
        pred_df.groupby("demand_type", sort=False, observed=True)
        .apply(metrics_by_type, include_groups=False)
        .reset_index()
    )
    by_type["mae_improvement"] = by_type["mae_roll28"] - by_type["mae_lgbm"]
    by_type["smape_improvement"] = by_type["smape_roll28"] - by_type["smape_lgbm"]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_type.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")

    # Overall (all types)
    y_all = pred_df["y_true"].to_numpy()
    p_lgb_all = pred_df["pred_lgbm"].to_numpy()
    p_r28_all = pred_df["pred_roll28"].to_numpy()
    print("\nOverall (all demand types):")
    print(f"  n_obs    = {len(y_all)}")
    print(f"  MAE  (LGBM)  = {mae(y_all, p_lgb_all):.4f}")
    print(f"  MAE  (roll28)= {mae(y_all, p_r28_all):.4f}")
    print(f"  sMAPE (LGBM)  = {smape(y_all, p_lgb_all):.4f}")
    print(f"  sMAPE (roll28)= {smape(y_all, p_r28_all):.4f}")
    print("\nBy demand_type:")
    print(by_type.to_string(index=False))


if __name__ == "__main__":
    main()
