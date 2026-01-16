import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["dow"] = out["date"].dt.weekday
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    return out


def add_lag_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    g = df.groupby(["store_id", "item_id"], sort=False)
    out = df.copy()

    for lag in [1, 7, 14, 28]:
        out[f"lag_{lag}"] = g[target].shift(lag)

    out["roll_mean_7"] = g[target].shift(1).rolling(7).mean()
    out["roll_mean_28"] = g[target].shift(1).rolling(28).mean()
    out["roll_std_28"] = g[target].shift(1).rolling(28).std()

    return out


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(["store_id", "item_id"], sort=False)

    out["price_lag_7"] = g["sell_price"].shift(7)
    out["price_change_7"] = (out["sell_price"] - out["price_lag_7"]) / out["price_lag_7"]

    return out


def build_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    out = add_time_features(out)
    out = add_lag_features(out, target)
    out = add_price_features(out)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build time-safe features for LightGBM.")
    parser.add_argument("--in_path", type=str, default="data/processed/m5_labeled_sample.parquet")
    parser.add_argument("--out_path", type=str, default="data/processed/m5_features_sample.parquet")
    parser.add_argument("--label", type=str, default="y_v2", help="Target label to build lags from")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_parquet(in_path)

    df_feat = build_features(df, target=args.label)

    # Drop rows where features are not available yet
    feature_cols = [c for c in df_feat.columns if c.startswith(("lag_", "roll_", "price_"))]
    df_feat = df_feat.dropna(subset=feature_cols)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(args.out_path, index=False)

    print(f"[OK] Saved features: {args.out_path}")
    print("Rows after feature build:", len(df_feat))


if __name__ == "__main__":
    main()
