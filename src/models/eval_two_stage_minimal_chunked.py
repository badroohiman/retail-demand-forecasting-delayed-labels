import argparse
from pathlib import Path

import pandas as pd

# Reuse your exact logic (and your recent "drop id" fix)
from src.models.tune_two_stage_minimal import run_two_stage


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate two-stage minimal on first N parquet parts.")
    p.add_argument("--in_dir", required=True, help="Input parquet dataset directory (e.g., data/processed/m5_features_full_ds)")
    p.add_argument("--out_path", default="reports/eval_two_stage_minimal_20parts.csv", help="CSV output path")
    p.add_argument("--label", default="y_v1", choices=["y_v0", "y_v1", "y_v2"])
    p.add_argument("--split_date", default="2015-01-01")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_parts", type=int, default=20)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"--in_dir must be an existing directory: {in_dir}")

    parts = sorted(in_dir.glob("part_*.parquet"))
    if not parts:
        parts = sorted(in_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found in: {in_dir}")

    parts = parts[: args.n_parts]

    # Best params based on your single-part tuning result:
    # reg_C_more_cap => reg num_leaves=127, min_child_samples=50; clf stays base (63/50)
    clf_params = dict(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=63,
        min_child_samples=50,
    )
    reg_params = dict(
        objective="poisson",
        n_estimators=900,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=127,
        min_child_samples=50,
    )

    rows = []
    mae_sum = 0.0
    smape_sum = 0.0
    weight_sum = 0

    for i, part in enumerate(parts, start=1):
        df = pd.read_parquet(part)

        # Weight by size of TEST split (so larger shards matter proportionally)
        df["date"] = pd.to_datetime(df["date"])
        test_rows = int((df["date"] >= args.split_date).sum())

        metrics = run_two_stage(
            df=df,
            label=args.label,
            split_date=args.split_date,
            clf_params=clf_params,
            reg_params=reg_params,
            seed=args.seed,
        )

        rows.append(
            {
                "part": part.name,
                "label": args.label,
                "split_date": args.split_date,
                "test_rows": test_rows,
                "mae": metrics["mae"],
                "smape": metrics["smape"],
                "clf_num_leaves": clf_params["num_leaves"],
                "clf_min_child_samples": clf_params["min_child_samples"],
                "reg_num_leaves": reg_params["num_leaves"],
                "reg_min_child_samples": reg_params["min_child_samples"],
            }
        )

        if test_rows > 0:
            mae_sum += metrics["mae"] * test_rows
            smape_sum += metrics["smape"] * test_rows
            weight_sum += test_rows

        print(f"[OK] {i}/{len(parts)} {part.name} | test_rows={test_rows} | MAE={metrics['mae']:.6f} sMAPE={metrics['smape']:.6f}")

    out = pd.DataFrame(rows).sort_values("mae")
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)

    if weight_sum > 0:
        w_mae = mae_sum / weight_sum
        w_smape = smape_sum / weight_sum
    else:
        w_mae = float("nan")
        w_smape = float("nan")

    print(f"\n[OK] Saved: {args.out_path}")
    print(f"[SUMMARY] Parts={len(parts)} | Weighted MAE={w_mae:.6f} | Weighted sMAPE={w_smape:.6f} | Total test rows={weight_sum}")
    print("\nTop 10 parts (lowest MAE):")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

