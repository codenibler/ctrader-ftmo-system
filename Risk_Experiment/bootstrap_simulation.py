#!/usr/bin/env python3
"""Bootstrap equity path simulation using observed R multiples."""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from tqdm import tqdm


DEFAULT_F_VALUES: Sequence[float] = (
    0.0000010,
    0.0000015,
    0.0000020
)

DEFAULT_BLOCK_SIZE = 4
DEFAULT_NUM_PATHS = 10_000
DEFAULT_OUTPUT = Path("Risk_Experiment") / "bootstrap_results.csv"
DEFAULT_TRADES_PER_MONTH = 550
DEFAULT_MONTHS = 3
R_VALUES_PATH = Path("Risk_Experiment") / "R_Values.csv"


@dataclass
class SimulationConfig:
    r_values: List[float]
    block_size: int
    path_length: int
    num_paths: int
    fractions: Sequence[float]
    seed: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap drawdown simulation using R multiples.")
    parser.add_argument(
        "--r-values",
        type=Path,
        default=R_VALUES_PATH,
        help="Path to R_Values.csv (default: Risk_Experiment/R_Values.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the simulation summary CSV",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=DEFAULT_NUM_PATHS,
        help="Number of bootstrap paths per risk fraction",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Block size for sampling contiguous R multiples",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=DEFAULT_MONTHS,
        help="Number of months in horizon (default 3)",
    )
    parser.add_argument(
        "--trades-per-month",
        type=int,
        default=DEFAULT_TRADES_PER_MONTH,
        help="Number of trades per month (default 700)",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="*",
        default=list(DEFAULT_F_VALUES),
        help="Risk fractions to evaluate (defaults to predefined grid)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional RNG seed for reproducibility",
    )
    return parser.parse_args()


def load_r_values(path: Path) -> List[float]:
    if not path.exists():
        raise FileNotFoundError(f"R values file not found: {path}")
    r_values: List[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("R values CSV has no header")
        for row in reader:
            value = row.get("r_multiple")
            if not value:
                continue
            try:
                r = float(value)
            except ValueError:
                continue
            r_values.append(r)
    if not r_values:
        raise ValueError("No valid R multiples were loaded")
    return r_values


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    pos = (len(values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[int(pos)]
    lower_val = values[lower]
    upper_val = values[upper]
    weight = pos - lower
    return lower_val * (1 - weight) + upper_val * weight


def simulate_path(r_values: Sequence[float], cfg: SimulationConfig, fraction: float, rng: random.Random) -> float:
    block_size = cfg.block_size
    max_start = len(r_values) - block_size
    if max_start < 0:
        raise ValueError("Block size larger than available R value sequence")

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    trades_generated = 0

    while trades_generated < cfg.path_length:
        start_idx = rng.randint(0, max_start)
        block = r_values[start_idx : start_idx + block_size]
        for r in block:
            trades_generated += 1
            equity *= 1.0 + fraction * r
            if equity <= 0.0:
                equity = 0.0
            if equity > peak:
                peak = equity
            if peak > 0.0:
                drawdown = (peak - equity) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            if trades_generated >= cfg.path_length:
                break
    return max_drawdown


def run_simulations(cfg: SimulationConfig, output_path: Path) -> None:
    rng = random.Random(cfg.seed)
    results: List[dict[str, float]] = []

    for fraction in cfg.fractions:
        dd_samples: List[float] = []
        for _ in tqdm(range(cfg.num_paths), desc=f"f={fraction:.4f}"):
            dd = simulate_path(cfg.r_values, cfg, fraction, rng)
            dd_samples.append(dd)
        dd_samples.sort()

        prob_dd_ge_10 = sum(dd >= 0.10 for dd in dd_samples) / cfg.num_paths
        dd_95 = percentile(dd_samples, 0.95)

        results.append(
            {
                "f": fraction,
                "prob_max_dd_ge_10pct": prob_dd_ge_10,
                "dd_95pct": dd_95,
            }
        )

        print(
            f"f={fraction:.4f}: prob(maxDD>=10%)={prob_dd_ge_10:.4%}, "
            f"95th percentile DD={dd_95:.2%}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["f", "prob_max_dd_ge_10pct", "dd_95pct"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results written to {output_path}")


def main() -> None:
    args = parse_args()
    r_values = load_r_values(args.r_values)

    path_length = args.months * args.trades_per_month
    cfg = SimulationConfig(
        r_values=r_values,
        block_size=args.block_size,
        path_length=path_length,
        num_paths=args.num_paths,
        fractions=args.fractions,
        seed=args.seed,
    )

    run_simulations(cfg, args.output)


if __name__ == "__main__":
    main()
