from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd


ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_calendar(cal_path: Path) -> pd.DataFrame:
    cal = pd.read_csv(cal_path)
    # Keep only what you need (add/remove columns as required by later pipeline)
    keep = [
        "d",
        "date",
        "wm_yr_wk",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    cal = cal[[c for c in keep if c in cal.columns]].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["d"] = cal["d"].astype("string")
    return cal


def _read_prices(prices_path: Path) -> pd.DataFrame:
    prices = pd.read_csv(prices_path)
    # Optimize dtypes
    for c in ["store_id", "item_id"]:
        if c in prices.columns:
            prices[c] = prices[c].astype("category")
    if "wm_yr_wk" in prices.columns:
        prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int32")
    if "sell_price" in prices.columns:
        prices["sell_price"] = prices["sell_price"].astype("float32")
    return prices


def _sales_chunks(
    sales_path: Path,
    chunksize: int,
    sample_stores: int,
    sample_items: int,
) -> tuple[pd.DataFrame, list[str]]:
    # Read only header first to get d_ columns
    header = pd.read_csv(sales_path, nrows=0)
    d_cols = [c for c in header.columns if c.startswith("d_")]

    usecols = ID_COLS + d_cols

    # categories reduce memory a lot
    dtype = {
        "id": "string",
        "item_id": "category",
        "dept_id": "category",
        "cat_id": "category",
        "store_id": "category",
        "state_id": "category",
    }

    reader = pd.read_csv(
        sales_path,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
    )

    for chunk in reader:
        # Optional sampling (0 means no sampling)
        if sample_stores and "store_id" in chunk.columns:
            stores = chunk["store_id"].cat.categories[:sample_stores]
            chunk = chunk[chunk["store_id"].isin(stores)]
        if sample_items and "item_id" in chunk.columns:
            items = chunk["item_id"].cat.categories[:sample_items]
            chunk = chunk[chunk["item_id"].isin(items)]

        yield chunk, d_cols


def preprocess_m5_chunked(
    unzipped_dir: Path,
    out_path: Path,
    sales_file: str = "sales_train_validation.csv",
    chunksize: int = 200,   # <-- tune this
    parts_dir: Path | None = None,
    sample_stores: int = 0,
    sample_items: int = 0,
) -> None:
    unzipped_dir = Path(unzipped_dir)
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    cal = _read_calendar(unzipped_dir / "calendar.csv")
    prices = _read_prices(unzipped_dir / "sell_prices.csv")

    sales_path = unzipped_dir / sales_file
    if not sales_path.exists():
        raise FileNotFoundError(f"Missing sales file: {sales_path}")

    if parts_dir is None:
        parts_dir = out_path.parent / (out_path.stem + "_parts")
    parts_dir = Path(parts_dir)
    _ensure_dir(parts_dir)

    part_idx = 0
    for chunk, d_cols in _sales_chunks(sales_path, chunksize, sample_stores, sample_items):
        # Melt wide -> long for just this chunk
        long = chunk.melt(
            id_vars=ID_COLS,
            value_vars=d_cols,
            var_name="d",
            value_name="sales",
        )

        # Downcast sales
        long["sales"] = pd.to_numeric(long["sales"], errors="coerce").fillna(0).astype("int16")
        long["d"] = long["d"].astype("string")

        # Merge with calendar to get date, wm_yr_wk, etc.
        long = long.merge(cal, on="d", how="left")

        # Merge prices (needs wm_yr_wk + item_id + store_id)
        if {"wm_yr_wk", "store_id", "item_id"}.issubset(long.columns):
            long = long.merge(
                prices,
                on=["store_id", "item_id", "wm_yr_wk"],
                how="left",
            )

        # Save part parquet
        part_path = parts_dir / f"part_{part_idx:05d}.parquet"
        long.to_parquet(part_path, index=False)
        part_idx += 1

        # Free memory aggressively
        del chunk, long
        gc.collect()

    # Combine parts into one parquet
    # Easiest: read each part and write to one file sequentially
    # (still safe because each part is manageable)
    parts = sorted(parts_dir.glob("part_*.parquet"))
    if not parts:
        raise RuntimeError("No parts written; something went wrong.")

    # Write final by concatenation
    # For very large part counts, this is still ok since we read one by one.
    first = True
    for p in parts:
        df = pd.read_parquet(p)
        if first:
            df.to_parquet(out_path, index=False)
            first = False
        else:
            # append using pyarrow dataset style: easiest is to write as partitioned dataset
            # To keep single file without extra deps, we can instead build a dataset folder.
            # So here we switch to dataset output if you want true append.
            raise RuntimeError(
                "Multiple parts created. For robust appending, write a dataset directory instead of single file.\n"
                "Use --out_path data/processed/m5_daily_full_ds (directory) and it will write partitioned parquet."
            )

    print(f"[OK] Wrote first part to: {out_path}")
    print(f"[INFO] Parts are in: {parts_dir}")
    print("[NEXT] Recommended: write as a parquet dataset directory for true append.")


def preprocess_to_dataset_dir(
    unzipped_dir: Path,
    out_dir: Path,
    sales_file: str = "sales_train_validation.csv",
    chunksize: int = 200,
    sample_stores: int = 0,
    sample_items: int = 0,
) -> None:
    """
    Writes a parquet *dataset directory* (many parquet files).
    This is the most memory-safe and append-friendly format.
    """
    unzipped_dir = Path(unzipped_dir)
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    cal = _read_calendar(unzipped_dir / "calendar.csv")
    prices = _read_prices(unzipped_dir / "sell_prices.csv")

    sales_path = unzipped_dir / sales_file
    if not sales_path.exists():
        raise FileNotFoundError(f"Missing sales file: {sales_path}")

    part_idx = 0
    for chunk, d_cols in _sales_chunks(sales_path, chunksize, sample_stores, sample_items):
        long = chunk.melt(
            id_vars=ID_COLS,
            value_vars=d_cols,
            var_name="d",
            value_name="sales",
        )
        long["sales"] = pd.to_numeric(long["sales"], errors="coerce").fillna(0).astype("int16")
        long["d"] = long["d"].astype("string")

        long = long.merge(cal, on="d", how="left")

        if {"wm_yr_wk", "store_id", "item_id"}.issubset(long.columns):
            long = long.merge(
                prices,
                on=["store_id", "item_id", "wm_yr_wk"],
                how="left",
            )

        part_path = out_dir / f"part_{part_idx:05d}.parquet"
        long.to_parquet(part_path, index=False)
        part_idx += 1

        del chunk, long
        gc.collect()

    print(f"[OK] Wrote dataset directory with {part_idx} parts at: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--unzipped_dir", required=True)
    p.add_argument("--out_path", required=True, help="For dataset dir, pass a directory path ending with _ds")
    p.add_argument("--chunksize", type=int, default=200)
    p.add_argument("--sample_stores", type=int, default=0)
    p.add_argument("--sample_items", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_path)

    # If out_path ends with "_ds" or is a directory, write dataset directory
    if str(out).endswith("_ds") or out.suffix == "":
        preprocess_to_dataset_dir(
            unzipped_dir=Path(args.unzipped_dir),
            out_dir=out,
            chunksize=args.chunksize,
            sample_stores=args.sample_stores,
            sample_items=args.sample_items,
        )
    else:
        preprocess_m5_chunked(
            unzipped_dir=Path(args.unzipped_dir),
            out_path=out,
            chunksize=args.chunksize,
            sample_stores=args.sample_stores,
            sample_items=args.sample_items,
        )


if __name__ == "__main__":
    main()

