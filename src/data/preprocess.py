import argparse
from pathlib import Path

import pandas as pd


def load_raw(unzipped_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw M5 files.
    Supports both validation and evaluation variants.
    Returns: (sales_wide, calendar, prices)
    """
    calendar_path = unzipped_dir / "calendar.csv"
    prices_path = unzipped_dir / "sell_prices.csv"

    sales_validation = unzipped_dir / "sales_train_validation.csv"
    sales_evaluation = unzipped_dir / "sales_train_evaluation.csv"

    if sales_validation.exists():
        sales_path = sales_validation
        variant = "validation"
    elif sales_evaluation.exists():
        sales_path = sales_evaluation
        variant = "evaluation"
    else:
        raise FileNotFoundError("Could not find sales_train_validation.csv or sales_train_evaluation.csv")

    if not calendar_path.exists():
        raise FileNotFoundError(f"Missing {calendar_path}")
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing {prices_path}")

    print(f"[INFO] Using sales file: {sales_path.name} ({variant})")

    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)

    return sales, calendar, prices


def to_long_sales(sales_wide: pd.DataFrame, sample_stores: int = 0, sample_items: int = 0) -> pd.DataFrame:
    """
    Convert wide sales format (d_1...d_N) to long:
    id, item_id, dept_id, cat_id, store_id, state_id, d, sales

    Sampling happens BEFORE melt to avoid OOM in small environments.
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    df = sales_wide.copy()

    if sample_stores and sample_stores > 0:
        stores = sorted(df["store_id"].unique())[:sample_stores]
        df = df[df["store_id"].isin(stores)]

    if sample_items and sample_items > 0:
        items = sorted(df["item_id"].unique())[:sample_items]
        df = df[df["item_id"].isin(items)]

    sales_long = df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    return sales_long


def build_canonical_table(sales_long: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Join sales_long with calendar and sell_prices.
    - sales_long joins calendar on 'd'
    - then joins prices on (store_id, item_id, wm_yr_wk)
    """
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")

    df = sales_long.merge(calendar, how="left", on="d")

    # sanity check
    if df["date"].isna().any():
        raise ValueError("Some rows have missing date after joining calendar. Check join keys.")

    df = df.merge(prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess M5 raw files into a canonical daily table (EDA-first).")
    parser.add_argument("--unzipped_dir", type=str, default="data/raw/m5/unzipped", help="Folder with raw CSVs")
    parser.add_argument("--out_path", type=str, default="data/processed/m5_daily_sample.parquet", help="Output parquet path")
    parser.add_argument("--sample_stores", type=int, default=1, help="Number of stores to keep (0 = all)")
    parser.add_argument("--sample_items", type=int, default=200, help="Number of items to keep (0 = all)")
    args = parser.parse_args()

    unzipped_dir = Path(args.unzipped_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sales_wide, calendar, prices = load_raw(unzipped_dir)
    sales_long = to_long_sales(sales_wide, sample_stores=args.sample_stores, sample_items=args.sample_items)
    df = build_canonical_table(sales_long, calendar, prices)

    # Keep a sensible subset of columns (expand later during feature engineering)
    keep_cols = [
        "date",
        "d",
        "wm_yr_wk",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "sales",
        "sell_price",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    df.to_parquet(out_path, index=False)
    print(f"[OK] Saved: {out_path}")
    print("Rows:", len(df), "| Columns:", len(df.columns))
    print("Date range:", df["date"].min(), "->", df["date"].max())
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
