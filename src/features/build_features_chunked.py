import argparse
from pathlib import Path
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["dow"] = out["date"].dt.weekday
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    return out


def add_lag_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # IMPORTANT: ensure sorted for time-safe features
    out = df.sort_values(["store_id", "item_id", "date"]).copy()
    g = out.groupby(["store_id", "item_id"], sort=False)

    for lag in [1, 7, 14, 28]:
        out[f"lag_{lag}"] = g[target].shift(lag)

    out["roll_mean_7"] = g[target].shift(1).rolling(7).mean()
    out["roll_mean_28"] = g[target].shift(1).rolling(28).mean()
    out["roll_std_28"] = g[target].shift(1).rolling(28).std()

    return out


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["store_id", "item_id", "date"]).copy()
    if "sell_price" not in out.columns:
        # If sell_price doesn't exist, skip gracefully
        return out

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
    p = argparse.ArgumentParser(description="Build features for parquet dataset directory (chunked).")
    p.add_argument("--in_dir", required=True, help="Input parquet dataset directory (e.g., m5_labeled_full_ds)")
    p.add_argument("--out_dir", required=True, help="Output parquet dataset directory (e.g., m5_features_full_ds)")
    p.add_argument("--label", default="y_v2", help="Target label column for lag/rolling (e.g., y_v0/y_v1/y_v2)")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"--in_dir must be an existing directory: {in_dir}")

    parts = sorted(in_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found in: {in_dir}")

    for i, part in enumerate(parts, start=1):
        df = pd.read_parquet(part)

        if args.label not in df.columns:
            raise ValueError(f"Label column '{args.label}' not found in {part.name}. موجودها: {list(df.columns)[:30]} ...")

        df_feat = build_features(df, target=args.label)

        # Drop rows where features are not available yet
        feature_cols = [c for c in df_feat.columns if c.startswith(("lag_", "roll_", "price_"))]
        if feature_cols:
            df_feat = df_feat.dropna(subset=feature_cols)

        out_path = out_dir / part.name
        df_feat.to_parquet(out_path, index=False)

        print(f"[OK] {i}/{len(parts)} wrote {out_path} | rows={len(df_feat)}")

    print(f"[DONE] Feature dataset directory: {out_dir}")


if __name__ == "__main__":
    main()

