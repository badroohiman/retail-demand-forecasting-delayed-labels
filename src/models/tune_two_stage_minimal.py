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


def run_two_stage(
    df: pd.DataFrame,
    label: str,
    split_date: str,
    clf_params: dict,
    reg_params: dict,
    seed: int,
):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Categoricals
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

    train, test = time_split(df, split_date)

    drop_cols = ["date", "sales", "y_v0", "y_v1", "y_v2"]
    features = [c for c in df.columns if c not in drop_cols]

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
    return {
        "mae": mae(y_test, y_pred),
        "smape": smape(y_test, y_pred),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Minimal tuning experiment for two-stage model."
    )
    parser.add_argument(
        "--in_path", type=str, default="data/processed/m5_features_sample.parquet"
    )
    parser.add_argument(
        "--out_path", type=str, default="reports/tuning_two_stage_minimal.csv"
    )
    parser.add_argument(
        "--label", type=str, default="y_v2", choices=["y_v0", "y_v1", "y_v2"]
    )
    parser.add_argument("--split_date", type=str, default="2015-01-01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_parquet(in_path)

    # ---- Baseline-ish parameter defaults (stable, not too complex) ----
    clf_base = dict(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=63,
        min_child_samples=50,
    )

    reg_base = dict(
        objective="poisson",
        n_estimators=900,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=63,
        min_child_samples=50,
    )

    # ---- Minimal experiment: 6 runs ----
    experiments = []

    # Classifier tuning (keep reg fixed)
    experiments += [
        ("clf_A_base", clf_base, reg_base),
        (
            "clf_B_more_reg",
            {**clf_base, "min_child_samples": 150, "num_leaves": 63},
            reg_base,
        ),
        (
            "clf_C_more_cap",
            {**clf_base, "min_child_samples": 50, "num_leaves": 127},
            reg_base,
        ),
    ]

    # Regressor tuning (keep best/standard clf fixed; start from base)
    experiments += [
        ("reg_A_base", clf_base, reg_base),
        (
            "reg_B_more_reg",
            clf_base,
            {**reg_base, "min_child_samples": 150, "num_leaves": 63},
        ),
        (
            "reg_C_more_cap",
            clf_base,
            {**reg_base, "min_child_samples": 50, "num_leaves": 127},
        ),
    ]

    rows = []
    for name, clf_p, reg_p in experiments:
        metrics = run_two_stage(
            df=df,
            label=args.label,
            split_date=args.split_date,
            clf_params=clf_p,
            reg_params=reg_p,
            seed=args.seed,
        )
        row = {
            "exp": name,
            "label": args.label,
            "split_date": args.split_date,
            "clf_num_leaves": clf_p["num_leaves"],
            "clf_min_child_samples": clf_p["min_child_samples"],
            "reg_num_leaves": reg_p["num_leaves"],
            "reg_min_child_samples": reg_p["min_child_samples"],
            "mae": metrics["mae"],
            "smape": metrics["smape"],
        }
        rows.append(row)
        print(f"{name}: MAE={row['mae']:.6f} sMAPE={row['smape']:.6f}")

    out = pd.DataFrame(rows).sort_values("mae")
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)

    print(f"\n[OK] Saved: {args.out_path}")
    print("\nTop results (sorted by MAE):")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
