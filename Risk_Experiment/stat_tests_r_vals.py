#!/usr/bin/env python3
"""Compute run-length statistics from R multiples and write summary CSV."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
INPUT_PATH = THIS_DIR / "R_Values.csv"
OUTPUT_PATH = THIS_DIR / "stat_tests_r_vals.csv"


def read_outcomes(path: Path) -> list[str]:
    outcomes: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get("outcome") or "").strip().lower()
            if label in {"win", "loss"}:
                outcomes.append(label)
    return outcomes


def mean_run_length(outcomes: Iterable[str], target: str) -> float:
    runs: list[int] = []
    current = 0
    for outcome in outcomes:
        if outcome == target:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    if not runs:
        return 0.0
    return sum(runs) / len(runs)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError("R_Values.csv not found in Risk_Experiment directory")

    outcomes = read_outcomes(INPUT_PATH)
    mean_win_run = mean_run_length(outcomes, "win")
    mean_loss_run = mean_run_length(outcomes, "loss")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["mean_win_run_length", f"{mean_win_run:.6f}"])
        writer.writerow(["mean_loss_run_length", f"{mean_loss_run:.6f}"])

    print(f"Wrote run-length statistics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
