import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.eval.backtest_baselines import mae, smape


def time_split(df: pd.DataFrame, split_date: str):
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM on time-series features."
    )
    parser.add_argument(
        "--in_path", type=str, default="data/processed/m5_features_sample.parquet"
    )
    parser.add_argument("--label", type=str, default="y_v2")
    parser.add_argument("--split_date", type=str, default="2015-01-01")
    args = parser.parse_args()

    df = pd.read_parquet(args.in_path)
    df["date"] = pd.to_datetime(df["date"])
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

    # keep only columns that exist
    cat_cols = [c for c in cat_cols if c in df.columns]

    for c in cat_cols:
        df[c] = df[c].astype("category")
    train, test = time_split(df, args.split_date)

    target = args.label
    drop_cols = ["date", "sales", "y_v0", "y_v1", "y_v2"]
    features = [c for c in df.columns if c not in drop_cols]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    model = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(
    X_train,
    y_train,
    categorical_feature=cat_cols
)


    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    print("MAE:", mae(y_test.values, preds))
    print("sMAPE:", smape(y_test.values, preds))

    Path("models").mkdir(exist_ok=True)
    model.booster_.save_model("models/lgbm.txt")
    print("[OK] Model saved to models/lgbm.txt")


if __name__ == "__main__":
    main()
