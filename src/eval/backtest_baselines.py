import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    horizon: int = 28          # forecast horizon (days)
    min_train_days: int = 365  # minimum history before first test window
    step: int = 28             # how far to move the origin each fold


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    required = {"store_id", "item_id", "date", label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[["store_id", "item_id", "date", label_col]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    out = out.rename(columns={label_col: "y"})
    return out


def _add_baseline_features(ts: pd.DataFrame) -> pd.DataFrame:
    """
    For each series, create simple baseline predictors available at time t:
    - pred_naive: y_{t-1}
    - pred_seasonal: y_{t-7}
    - pred_roll28: mean(y_{t-28:t-1})
    """
    ts = ts.copy()
    g = ts.groupby(["store_id", "item_id"], sort=False)

    ts["pred_naive"] = g["y"].shift(1)
    ts["pred_seasonal"] = g["y"].shift(7)
    ts["pred_roll28"] = g["y"].shift(1).transform(lambda s: s.rolling(28).mean())

    return ts


def rolling_origin_backtest(df: pd.DataFrame, label_col: str, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Rolling-origin evaluation:
    - choose multiple cutoffs ("origins") in time
    - for each origin, evaluate predictions for the next horizon days
    - never uses future labels to predict past (time-aware)
    """
    data = _prepare_df(df, label_col=label_col)
    data = _add_baseline_features(data)

    # Determine fold cutoffs using global dates (simple and transparent)
    dates = np.array(sorted(data["date"].unique()))
    if len(dates) < cfg.min_train_days + cfg.horizon:
        raise ValueError(
            f"Not enough dates for backtest. Have {len(dates)} unique days, "
            f"need at least {cfg.min_train_days + cfg.horizon}."
        )

    # origins are indices into dates[]; each origin predicts the next horizon days
    first_origin_idx = cfg.min_train_days
    last_origin_idx = len(dates) - cfg.horizon - 1
    origin_indices = list(range(first_origin_idx, last_origin_idx + 1, cfg.step))

    results = []

    for fold, origin_idx in enumerate(origin_indices, start=1):
        origin_date = dates[origin_idx]
        test_start = origin_date + np.timedelta64(1, "D")
        test_end = origin_date + np.timedelta64(cfg.horizon, "D")

        fold_mask = (data["date"] >= test_start) & (data["date"] <= test_end)
        fold_df = data.loc[fold_mask].copy()

        # Drop rows where predictors are not available (early history)
        # (This is fair: those baselines cannot predict without history.)
        for model_col in ["pred_naive", "pred_seasonal", "pred_roll28"]:
            valid = fold_df[model_col].notna()
            y_true = fold_df.loc[valid, "y"].to_numpy(dtype=float)
            y_pred = fold_df.loc[valid, model_col].to_numpy(dtype=float)

            if len(y_true) == 0:
                continue

            results.append(
                {
                    "fold": fold,
                    "origin_date": pd.Timestamp(origin_date),
                    "test_start": pd.Timestamp(test_start),
                    "test_end": pd.Timestamp(test_end),
                    "label": label_col,
                    "model": model_col.replace("pred_", ""),
                    "n_obs": int(len(y_true)),
                    "mae": mae(y_true, y_pred),
                    "smape": smape(y_true, y_pred),
                }
            )

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-aware rolling-origin backtest for baseline forecasters.")
    parser.add_argument("--in_path", type=str, default="data/processed/m5_labeled_sample.parquet")
    parser.add_argument("--out_path", type=str, default="reports/baseline_backtest_results.csv")
    parser.add_argument("--label", type=str, choices=["y_v0", "y_v1", "y_v2", "all"], default="all")
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--min_train_days", type=int, default=365)
    parser.add_argument("--step", type=int, default=28)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing input: {in_path}\n"
            "Run label versioning first:\n"
            "  python src/labels/make_label_versions.py "
            "--in_path data/processed/m5_daily_sample.parquet "
            "--out_path data/processed/m5_labeled_sample.parquet"
        )

    df = pd.read_parquet(in_path)

    cfg = BacktestConfig(horizon=args.horizon, min_train_days=args.min_train_days, step=args.step)

    labels = ["y_v0", "y_v1", "y_v2"] if args.label == "all" else [args.label]

    all_results = []
    for label_col in labels:
        res = rolling_origin_backtest(df, label_col=label_col, cfg=cfg)
        all_results.append(res)

    out = pd.concat(all_results, ignore_index=True)
    _ensure_parent(Path(args.out_path))
    out.to_csv(args.out_path, index=False)

    print(f"[OK] Saved: {args.out_path}")
    print("\nSummary (mean over folds):")
    summary = (
        out.groupby(["label", "model"], as_index=False)[["mae", "smape", "n_obs"]]
           .mean()
           .sort_values(["label", "mae"])
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
