import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def time_split(df: pd.DataFrame, split_date: str):
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    return train, test


def prep_df(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Categoricals (same list you used elsewhere)
    cat_cols = [
        "d",
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
    for c in cat_cols:
        df[c] = df[c].astype("category")

    # IMPORTANT: remove non-numeric/string identifiers
    drop_cols = ["id", "date", "sales", "y_v0", "y_v1", "y_v2"]
    features = [c for c in df.columns if c not in drop_cols]

    return df, features, cat_cols


def train_two_stage_on_part(
    df: pd.DataFrame,
    label: str,
    split_date: str,
    clf_params: dict,
    reg_params: dict,
    seed: int,
):
    df, features, cat_cols = prep_df(df)
    train, test = time_split(df, split_date)

    X_train = train[features]
    y_train = train[label].astype(float).to_numpy()

    X_test = test[features]
    y_test = test[label].astype(float).to_numpy()

    # Stage 1 classifier
    y_train_bin = (y_train > 0).astype(int)
    clf = lgb.LGBMClassifier(**clf_params, random_state=seed)
    clf.fit(X_train, y_train_bin, categorical_feature=cat_cols)
    p_test = clf.predict_proba(X_test)[:, 1]

    # Stage 2 regressor (train only on non-zero)
    nz = y_train > 0
    if nz.sum() == 0:
        raise ValueError("No non-zero targets in training set for stage-2 regressor.")

    reg = lgb.LGBMRegressor(**reg_params, random_state=seed)
    reg.fit(X_train.loc[nz], y_train[nz], categorical_feature=cat_cols)
    mu_test = np.clip(reg.predict(X_test), 0, None)

    y_pred = p_test * mu_test

    metrics = {
        "mae": mae(y_test, y_pred),
        "smape": smape(y_test, y_pred),
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "train_nz_rows": int(nz.sum()),
    }
    return clf, reg, metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train two-stage models on first N parquet parts (RAM-safe).")
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", default="artifacts/models")
    p.add_argument("--report_path", default="reports/train_two_stage_20parts.csv")
    p.add_argument("--label", default="y_v1", choices=["y_v0", "y_v1", "y_v2"])
    p.add_argument("--split_date", default="2015-01-01")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_parts", type=int, default=20)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"--in_dir must be a directory: {in_dir}")

    parts = sorted(in_dir.glob("part_*.parquet"))
    if not parts:
        parts = sorted(in_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found in: {in_dir}")

    parts = parts[: args.n_parts]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)

    # Use your best params (from tuning)
    clf_params = dict(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=63,
        min_child_samples=50,
    )
    reg_params = dict(
        objective="poisson",
        n_estimators=900,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=127,
        min_child_samples=50,
    )

    rows = []
    w_mae_num = 0.0
    w_smape_num = 0.0
    w_den = 0

    for i, part in enumerate(parts, start=1):
        df = pd.read_parquet(part)

        clf, reg, m = train_two_stage_on_part(
            df=df,
            label=args.label,
            split_date=args.split_date,
            clf_params=clf_params,
            reg_params=reg_params,
            seed=args.seed,
        )

        # Save models
        stem = part.stem  # part_00007
        clf_path = out_dir / f"two_stage_clf_{stem}.txt"
        reg_path = out_dir / f"two_stage_reg_{stem}.txt"
        clf.booster_.save_model(str(clf_path))
        reg.booster_.save_model(str(reg_path))

        rows.append(
            {
                "part": part.name,
                "label": args.label,
                "split_date": args.split_date,
                "train_rows": m["train_rows"],
                "train_nz_rows": m["train_nz_rows"],
                "test_rows": m["test_rows"],
                "mae": m["mae"],
                "smape": m["smape"],
                "clf_model": str(clf_path),
                "reg_model": str(reg_path),
            }
        )

        if m["test_rows"] > 0:
            w_mae_num += m["mae"] * m["test_rows"]
            w_smape_num += m["smape"] * m["test_rows"]
            w_den += m["test_rows"]

        print(
            f"[OK] {i}/{len(parts)} {part.name} | "
            f"MAE={m['mae']:.6f} sMAPE={m['smape']:.6f} | "
            f"saved: {clf_path.name}, {reg_path.name}"
        )

    out = pd.DataFrame(rows).sort_values("mae")
    out.to_csv(args.report_path, index=False)

    w_mae = w_mae_num / w_den if w_den else float("nan")
    w_smape = w_smape_num / w_den if w_den else float("nan")

    print(f"\n[OK] Saved report: {args.report_path}")
    print(f"[SUMMARY] Parts={len(parts)} | Weighted MAE={w_mae:.6f} | Weighted sMAPE={w_smape:.6f} | Total test rows={w_den}")
    print("\nTop 10 parts (lowest MAE):")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

