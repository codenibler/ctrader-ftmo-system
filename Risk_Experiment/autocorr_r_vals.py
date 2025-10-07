#!/usr/bin/env python3
"""Compute autocorrelation of R sign series for lags 1-10."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
INPUT_PATH = THIS_DIR / "R_Values.csv"
OUTPUT_PATH = THIS_DIR / "autocorr_r_vals.csv"


def to_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_signs(path: Path) -> List[int]:
    signs: List[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            r_value = to_float(row.get("r_multiple"))
            if r_value is None:
                continue
            signs.append(1 if r_value > 0 else -1)
    return signs


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError("R_Values.csv not found in Risk_Experiment directory")

    signs = read_signs(INPUT_PATH)
    if not signs:
        raise SystemExit("R_Values.csv contains no valid R multiples")

    series = pd.Series(signs)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lag", "autocorrelation"])
        for lag in range(1, 11):
            acf = series.autocorr(lag=lag)
            writer.writerow([lag, f"{acf:.6f}" if acf is not None else ""])

    print(f"Wrote autocorrelation values (lags 1-10) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
