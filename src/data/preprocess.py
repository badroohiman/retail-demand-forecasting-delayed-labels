def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess M5 raw files into a canonical daily table (EDA-first)."
    )
    parser.add_argument(
        "--unzipped_dir",
        type=str,
        default="data/raw/m5/unzipped",
        help="Folder with raw CSVs",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/processed/m5_daily_sample.parquet",
        help="Output parquet path",
    )

    # Sampling (0 = all)
    parser.add_argument(
        "--sample_stores",
        type=int,
        default=1,
        help="Number of stores to keep (0 = all)",
    )
    parser.add_argument(
        "--sample_items",
        type=int,
        default=200,
        help="Number of items to keep (0 = all)",
    )

    # NEW: convenience flag
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full dataset (sets --sample_stores 0 and --sample_items 0)",
    )

    # NEW (optional): write a run summary artifact
    parser.add_argument(
        "--report_path",
        type=str,
        default="reports/scale_run_preprocess.json",
        help="Where to write a scale-run summary JSON",
    )

    args = parser.parse_args()

    # Apply full override
    if args.full:
        args.sample_stores = 0
        args.sample_items = 0

    unzipped_dir = Path(args.unzipped_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- timing + (optional) memory ---
    import json, time
    try:
        import resource  # unix-only
    except Exception:
        resource = None

    t0 = time.perf_counter()

    sales_wide, calendar, prices = load_raw(unzipped_dir)
    sales_long = to_long_sales(
        sales_wide, sample_stores=args.sample_stores, sample_items=args.sample_items
    )
    df = build_canonical_table(sales_long, calendar, prices)

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
    df = (
        df[keep_cols]
        .sort_values(["store_id", "item_id", "date"])
        .reset_index(drop=True)
    )

    df.to_parquet(out_path, index=False)

    elapsed_s = time.perf_counter() - t0
    max_rss_mb = None
    if resource is not None:
        # ru_maxrss is KB on Linux
        max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    # --- file sizes (raw + processed) ---
    def _bytes(p: Path) -> int:
        return p.stat().st_size if p.exists() else 0

    raw_paths = {
        "calendar_csv": str(unzipped_dir / "calendar.csv"),
        "sell_prices_csv": str(unzipped_dir / "sell_prices.csv"),
        "sales_train_validation_csv": str(unzipped_dir / "sales_train_validation.csv"),
        "sales_train_evaluation_csv": str(unzipped_dir / "sales_train_evaluation.csv"),
    }
    raw_sizes = {k: _bytes(Path(v)) for k, v in raw_paths.items()}

    summary = {
        "stage": "preprocess",
        "full_run": bool(args.full),
        "sample_stores": int(args.sample_stores),
        "sample_items": int(args.sample_items),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "elapsed_seconds": float(elapsed_s),
        "max_rss_mb": None if max_rss_mb is None else float(max_rss_mb),
        "raw_files": raw_paths,
        "raw_file_sizes_bytes": raw_sizes,
        "output_path": str(out_path),
        "output_size_bytes": _bytes(out_path),
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Saved: {out_path}")
    print("Rows:", len(df), "| Columns:", len(df.columns))
    print("Date range:", df["date"].min(), "->", df["date"].max())
    print(f"[REPORT] {report_path}")