import argparse
from pathlib import Path

import pandas as pd

def load_raw(unzipped_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame,pd.DataFrame]:
    """
    Load raw M5 files.
    Supports both validation and evaluation variants.
    Returns: (sales, calendar, prices)
    """
    calendar_path = unzipped_dir/"calendar.csv"
    prices_path = unzipped_dir/"sell_prices.csv"

    sales_validation = unzipped_dir/"sales_train_validation.csv"
    sales_evaluation = unzipped_dir/"sales_train_evaluation.csv"

    if sales_validation.exists():
        sales_path = sales_validation
        variant = "validation"
    elif sales_evaluation.exists():
        sales_path = sales_evaluation
        variant = "evaluation"
    else:
        raise FileNotFoundError(
            "Could not find sales_train_validation.csv or sales_train_evaluation.csv"
        )
    
    if not calendar_path.exists():
        raise FileNotFoundError(f"Missing {calendar_path}")
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing {prices_path}")

    print(f"[INFO] Using sales file: {sales_path.name} ({variant})")

    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)

    return sales, calendar, prices

def to_long_sales(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert sales data from wide to long format.
    Input: sales_wide with columns [id, item_id, dept_id, cat_id, store_id, state_id, d_1, d_2, ..., d_N]
    Output: sales_long with columns [id, item_id, dept_id, cat_id, store_id, state_id, d, sales]
    """
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [col for col in sales_wide.columns if col.startswith("d_")]

    sales_long = sales_wide.melt(
        id_vars=id_vars,
        value_vars=day_cols,
        var_name="d",
        value_name="sales"
    )

    return sales_long

def build_canonical_table(sales_long: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build canonical sales table by merging sales, calendar, and prices data.
    Returns a DataFrame with columns:
    [id, item_id, dept_id, cat_id, store_id, state_id, d, sales, date, wm_yr_wk, sell_price]
    """
    # Merge sales with calendar
    calendar = calendar.copy()
    calendar['date'] = pd.to_datetime(calendar['date'])

    # Join sales with calendar
    df = sales_long.merge(calendar, how="left", on="d")

    # Join with prices
    df = df.merge(
        prices,
        how="left",
        on=["store_id", "item_id", "wm_yr_wk"],
    )

    # Basic sanity checks
    if df["date"].isna().any():
        raise ValueError("Some rows have missing date after joining calendar. Check join keys.")

    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess M5 raw files into a canonical daily table.")
    parser.add_argument("--unzipped_dir", type=str, default="data/raw/m5/unzipped", help="Folder with raw CSVs")
    parser.add_argument("--out_path", type=str, default="data/processed/m5_daily.parquet", help="Output parquet path")
    args = parser.parse_args()

    unzipped_dir = Path(args.unzipped_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sales_wide, calendar, prices = load_raw(unzipped_dir)
    sales_long = to_long_sales(sales_wide)
    df = build_canonical_table(sales_long, calendar, prices)

    # Keep a sensible subset of columns (you can expand later)
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
    print(df.head(3).to_string(index=False))
    print("\nRows:", len(df), "| Columns:", len(df.columns))
    print("Date range:", df["date"].min(), "->", df["date"].max())


if __name__ == "__main__":
    main()