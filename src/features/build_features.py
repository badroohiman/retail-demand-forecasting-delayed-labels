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
    out["price_change_7"] = (out["sell_price"] - out["price_lag_7"]) / out[
        "price_lag_7"
    ]

    return out


def add_intermittent_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Add intermittent-demand features: non-zero rate (rolling proportion of
    non-zero sales) and days since last sale. Helps the classifier predict
    demand occurrence.
    """
    out = df.copy()
    g = out.groupby(["store_id", "item_id"], sort=False)

    out["non_zero_rate_7"] = g[target].shift(1).transform(
        lambda s: s.gt(0).rolling(7, min_periods=1).mean()
    )
    out["non_zero_rate_28"] = g[target].shift(1).transform(
        lambda s: s.gt(0).rolling(28, min_periods=1).mean()
    )

    def _days_since_last(grp: pd.DataFrame) -> np.ndarray:
        sales = grp[target].values
        dates = grp["date"].values
        res = np.full(len(grp), 999, dtype=np.float32)
        last_sale_date = None
        for i in range(len(grp)):
            if last_sale_date is not None:
                res[i] = (pd.Timestamp(dates[i]) - pd.Timestamp(last_sale_date)).days
            if sales[i] > 0:
                last_sale_date = dates[i]
        return res

    out["days_since_last_sale"] = out.groupby(
        ["store_id", "item_id"], sort=False, group_keys=False
    ).apply(lambda grp: pd.Series(_days_since_last(grp), index=grp.index), include_groups=False)

    return out


def add_availability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add data-availability features from EDA: in-catalog flag and days since
    first price. Helps separate structural zeros (product not yet introduced)
    from demand zeros (in catalog but no sale). Required for availability-aware
    backtesting and evaluation.
    """
    out = df.copy()
    # Binary: 1 if product was in catalog (had a price) on this date, 0 otherwise
    out["in_catalog"] = out["sell_price"].notna().astype(np.int8)

    # First date per (store_id, item_id) where sell_price is non-null (product introduction)
    first_price_date = (
        out[out["sell_price"].notna()]
        .groupby(["store_id", "item_id"], sort=False)["date"]
        .min()
        .reset_index(name="first_price_date")
    )
    out = out.merge(
        first_price_date, on=["store_id", "item_id"], how="left"
    )
    # Days since introduction; clip at 0 so pre-introduction is 0 (no negative days)
    delta = (out["date"] - out["first_price_date"]).dt.days
    out["days_since_first_price"] = delta.clip(lower=0).fillna(0).astype(np.int32)
    out = out.drop(columns=["first_price_date"])

    return out


def build_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    out = add_time_features(out)
    out = add_lag_features(out, target)
    out = add_price_features(out)
    out = add_availability_features(out)
    out = add_intermittent_features(out, target)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build time-safe features for LightGBM."
    )
    parser.add_argument(
        "--in_path", type=str, default="data/processed/m5_labeled_sample.parquet"
    )
    parser.add_argument(
        "--out_path", type=str, default="data/processed/m5_features_sample.parquet"
    )
    parser.add_argument(
        "--label", type=str, default="y_v2", help="Target label to build lags from"
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_parquet(in_path)

    df_feat = build_features(df, target=args.label)

    # Drop rows where features are not available yet
    feature_cols = [
        c
        for c in df_feat.columns
        if c.startswith(("lag_", "roll_", "price_", "non_zero_", "days_since"))
    ]
    df_feat = df_feat.dropna(subset=feature_cols)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(args.out_path, index=False)

    print(f"[OK] Saved features: {args.out_path}")
    print("Rows after feature build:", len(df_feat))


if __name__ == "__main__":
    main()
