#!/usr/bin/env python3
"""Compute win/loss outcome autocorrelation for lags 1-10."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List


THIS_DIR = Path(__file__).resolve().parent
INPUT_PATH = THIS_DIR / "R_Values.csv"
OUTPUT_PATH = THIS_DIR / "autocorr_r_vals.csv"


def read_signs(path: Path) -> List[int]:
    signs: List[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            outcome = (row.get("outcome") or "").strip().lower()
            if outcome == "win":
                signs.append(1)
            elif outcome == "loss":
                signs.append(-1)
    return signs


def autocorr(signs: List[int], lag: int) -> float | None:
    n = len(signs)
    if lag <= 0:
        raise ValueError("lag must be positive")
    if lag >= n:
        return None
    mean = sum(signs) / n
    var = sum((s - mean) ** 2 for s in signs) / n
    if var == 0.0:
        return 0.0
    cov = sum((signs[i] - mean) * (signs[i + lag] - mean) for i in range(n - lag)) / (n - lag)
    return cov / var


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError("R_Values.csv not found in Risk_Experiment directory")

    signs = read_signs(INPUT_PATH)
    if not signs:
        raise SystemExit("R_Values.csv contains no win/loss rows")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lag", "autocorrelation"])
        for lag in range(1, 11):
            acf = autocorr(signs, lag)
            writer.writerow([lag, "" if acf is None else f"{acf:.6f}"])

    print(f"Wrote autocorrelation values (lags 1-10) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
