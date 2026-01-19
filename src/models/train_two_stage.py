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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage model: P(y>0) classifier + E[y|y>0] regressor."
    )
    parser.add_argument(
        "--in_path", type=str, default="data/processed/m5_features_sample.parquet"
    )
    parser.add_argument(
        "--label", type=str, default="y_v2", choices=["y_v0", "y_v1", "y_v2"]
    )
    parser.add_argument("--split_date", type=str, default="2015-01-01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.in_path)
    df["date"] = pd.to_datetime(df["date"])

    # Categorical columns
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

    train, test = time_split(df, args.split_date)

    target = args.label
    drop_cols = ["date", "sales", "y_v0", "y_v1", "y_v2"]
    features = [c for c in df.columns if c not in drop_cols]

    X_train = train[features]
    y_train = train[target].astype(float).to_numpy()

    X_test = test[features]
    y_test = test[target].astype(float).to_numpy()

    # ---- Stage 1: classifier for y > 0 ----
    y_train_bin = (y_train > 0).astype(int)

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        random_state=args.seed,
    )

    clf.fit(X_train, y_train_bin, categorical_feature=cat_cols)

    p_test = clf.predict_proba(X_test)[:, 1]  # probability of non-zero demand

    # ---- Stage 2: regressor on non-zero rows only ----
    nz_mask = y_train > 0
    if nz_mask.sum() == 0:
        raise ValueError(
            "No non-zero sales in training data; cannot train stage-2 regressor."
        )

    X_train_nz = X_train.loc[nz_mask]
    y_train_nz = y_train[nz_mask]

    reg = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        random_state=args.seed,
    )

    reg.fit(X_train_nz, y_train_nz, categorical_feature=cat_cols)

    mu_test = reg.predict(X_test)
    mu_test = np.clip(mu_test, 0, None)

    # Final prediction
    y_pred = p_test * mu_test

    print(f"Two-stage results | label={target} | split_date={args.split_date}")
    print("MAE:", mae(y_test, y_pred))
    print("sMAPE:", smape(y_test, y_pred))

    # Save models
    Path("models").mkdir(exist_ok=True)
    clf.booster_.save_model("models/lgbm_two_stage_clf.txt")
    reg.booster_.save_model("models/lgbm_two_stage_reg.txt")
    print(
        "[OK] Saved models to models/lgbm_two_stage_clf.txt and models/lgbm_two_stage_reg.txt"
    )


if __name__ == "__main__":
    main()
