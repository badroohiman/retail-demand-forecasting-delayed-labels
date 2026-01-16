import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DelayConfig:
    # Probability that a non-zero label is under-reported at this delay
    p_partial: float
    # Minimum fraction of final sales when under-reported (uniform in [min_frac, 1])
    min_frac: float


DEFAULT_CFG = {
    "v0": DelayConfig(p_partial=0.35, min_frac=0.10),  # same-day: more incomplete
    "v1": DelayConfig(p_partial=0.15, min_frac=0.30),  # 7-day: less incomplete
    "v2": DelayConfig(p_partial=0.00, min_frac=1.00),  # final: truth
}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _simulate_partial(final_sales: np.ndarray, cfg: DelayConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate under-reporting for non-zero final_sales.
    - with probability cfg.p_partial: y = floor(final_sales * u), u ~ Uniform[min_frac, 1]
    - else: y = final_sales
    - zeros remain zeros
    """
    y = final_sales.astype(np.int64).copy()

    if cfg.p_partial <= 0:
        return y

    nonzero = y > 0
    n = nonzero.sum()
    if n == 0:
        return y

    mask_partial = np.zeros_like(y, dtype=bool)
    mask_partial[nonzero] = rng.random(n) < cfg.p_partial

    # draw uniform multipliers for partial rows
    u = rng.uniform(cfg.min_frac, 1.0, size=mask_partial.sum())
    y_partial = np.floor(y[mask_partial] * u).astype(np.int64)

    # Ensure partial doesn't exceed final and doesn't go negative
    y_partial = np.clip(y_partial, 0, y[mask_partial])

    y[mask_partial] = y_partial
    return y


def make_label_versions(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Adds y_v0, y_v1, y_v2 columns based on 'sales' as final truth.
    Keeps original 'sales' unchanged.
    """
    required = {"store_id", "item_id", "date", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # Treat 'sales' as final truth for simulation
    y_final = pd.to_numeric(out["sales"], errors="raise").astype(np.int64).to_numpy()

    rng = np.random.default_rng(seed)

    out["y_v2"] = y_final
    out["y_v1"] = _simulate_partial(y_final, DEFAULT_CFG["v1"], rng)
    out["y_v0"] = _simulate_partial(y_final, DEFAULT_CFG["v0"], rng)

    # Sanity checks
    if (out["y_v0"] > out["y_v2"]).any() or (out["y_v1"] > out["y_v2"]).any():
        raise ValueError("Found delayed labels greater than final labels (should not happen).")
    if (out["y_v0"] < 0).any() or (out["y_v1"] < 0).any() or (out["y_v2"] < 0).any():
        raise ValueError("Found negative labels (should not happen).")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create label delay versions v0/v1/v2 for M5 canonical data.")
    parser.add_argument("--in_path", type=str, default="data/processed/m5_daily_sample.parquet", help="Input parquet")
    parser.add_argument("--out_path", type=str, default="data/processed/m5_labeled_sample.parquet", help="Output parquet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing input: {in_path}\n"
            "Run preprocessing first:\n"
            "  python src/data/preprocess.py --sample_stores 1 --sample_items 200 "
            "--out_path data/processed/m5_daily_sample.parquet"
        )

    df = pd.read_parquet(in_path)
    df_labeled = make_label_versions(df, seed=args.seed)

    _ensure_parent(out_path)
    df_labeled.to_parquet(out_path, index=False)

    # Quick report
    print(f"[OK] Saved: {out_path}")
    print("Columns added: y_v0, y_v1, y_v2")
    print(df_labeled[["sales", "y_v0", "y_v1", "y_v2"]].head(10).to_string(index=False))

    # Aggregate check: delayed should be <= final
    agg = df_labeled[["y_v0", "y_v1", "y_v2"]].sum()
    print("\nAggregate totals:")
    print(agg.to_string())
    print("\nRatios vs final:")
    print((agg / agg["y_v2"]).round(3).to_string())


if __name__ == "__main__":
    main()
