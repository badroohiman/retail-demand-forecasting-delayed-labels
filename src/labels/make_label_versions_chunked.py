import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DelayConfig:
    p_partial: float
    min_frac: float


DEFAULT_CFG = {
    "v0": DelayConfig(p_partial=0.35, min_frac=0.10),
    "v1": DelayConfig(p_partial=0.15, min_frac=0.30),
    "v2": DelayConfig(p_partial=0.00, min_frac=1.00),
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _u64_to_unit_float(x: np.ndarray) -> np.ndarray:
    """Map uint64 -> [0,1)."""
    # use top 53 bits for float64 mantissa
    return ((x >> 11) * (1.0 / (1 << 53))).astype(np.float64)


def _simulate_partial_deterministic(
    y_final: np.ndarray,
    cfg: DelayConfig,
    u_decide: np.ndarray,
    u_mult: np.ndarray,
) -> np.ndarray:
    """
    Deterministic under-reporting:
    - for non-zero y_final:
        if u_decide < p_partial: y = floor(y_final * (min_frac + (1-min_frac)*u_mult))
        else: y = y_final
    - zeros remain zeros
    """
    y = y_final.astype(np.int64).copy()
    if cfg.p_partial <= 0:
        return y

    nonzero = y > 0
    if nonzero.sum() == 0:
        return y

    mask_partial = nonzero & (u_decide < cfg.p_partial)
    if mask_partial.sum() == 0:
        return y

    mult = cfg.min_frac + (1.0 - cfg.min_frac) * u_mult
    y_partial = np.floor(y[mask_partial] * mult[mask_partial]).astype(np.int64)
    y_partial = np.clip(y_partial, 0, y[mask_partial])
    y[mask_partial] = y_partial
    return y


def _hash_for_rows(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create two deterministic uniform arrays per row based on stable row identity.
    Uses (store_id, item_id, date) + seed. Works chunk-by-chunk.
    """
    required = ["store_id", "item_id", "date"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # ensure date is datetime for stable hashing
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])

    h = pd.util.hash_pandas_object(
        d[["store_id", "item_id", "date"]],
        index=False,
    ).to_numpy(dtype=np.uint64)

    # mix with seed to get two streams
    s = np.uint64(seed)
    h1 = h ^ (s * np.uint64(0x9E3779B97F4A7C15))
    h2 = (h + (s * np.uint64(0xBF58476D1CE4E5B9))) ^ np.uint64(0x94D049BB133111EB)

    u1 = _u64_to_unit_float(h1)
    u2 = _u64_to_unit_float(h2)
    return u1, u2


def add_label_versions_chunk(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    required = {"store_id", "item_id", "date", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    y_final = pd.to_numeric(out["sales"], errors="raise").astype(np.int64).to_numpy()
    u1, u2 = _hash_for_rows(out, seed=seed)

    out["y_v2"] = y_final
    out["y_v1"] = _simulate_partial_deterministic(y_final, DEFAULT_CFG["v1"], u1, u2)
    out["y_v0"] = _simulate_partial_deterministic(y_final, DEFAULT_CFG["v0"], u1, u2)

    # sanity
    if (out["y_v0"] > out["y_v2"]).any() or (out["y_v1"] > out["y_v2"]).any():
        raise ValueError("Delayed labels > final (should not happen).")
    if (out[["y_v0", "y_v1", "y_v2"]] < 0).any().any():
        raise ValueError("Negative labels found (should not happen).")

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Chunked label versions for parquet dataset directory")
    p.add_argument("--in_path", required=True, help="Input parquet file OR parquet dataset directory")
    p.add_argument("--out_dir", required=True, help="Output parquet dataset directory")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    if in_path.is_dir():
        parts = sorted(in_path.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet parts found in: {in_path}")
        for i, part in enumerate(parts):
            df = pd.read_parquet(part)
            df2 = add_label_versions_chunk(df, seed=args.seed)
            out_part = out_dir / part.name
            df2.to_parquet(out_part, index=False)
            print(f"[OK] {i+1}/{len(parts)} wrote {out_part}")
    else:
        df = pd.read_parquet(in_path)
        df2 = add_label_versions_chunk(df, seed=args.seed)
        out_part = out_dir / "part_00000.parquet"
        df2.to_parquet(out_part, index=False)
        print(f"[OK] wrote {out_part}")

    print(f"[DONE] Labeled dataset directory: {out_dir}")


if __name__ == "__main__":
    main()

