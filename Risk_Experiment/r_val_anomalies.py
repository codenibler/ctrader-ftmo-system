#!/usr/bin/env python3
"""Scan R multiples for unexpected values and report extremes."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional


THIS_DIR = Path(__file__).resolve().parent
INPUT_CANDIDATES = [
    THIS_DIR / "R_Values.csv",
    THIS_DIR / "r_values.csv",
]
TOLERANCE = 1e-6


def pick_input_path() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("R_Values.csv not found in Risk_Experiment directory")


def read_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        yield from reader


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    path = pick_input_path()
    rows = list(read_rows(path))
    if not rows:
        raise SystemExit("R_Values.csv is empty or unreadable")

    anomalies: list[dict[str, str]] = []
    max_row: Optional[dict[str, str]] = None
    max_r: Optional[float] = None

    for row in rows:
        r_value = to_float(row.get("r_multiple"))
        if r_value is None:
            continue

        if (r_value < 0) and (abs(r_value + 1.0) > TOLERANCE):
            anomalies.append(row)

        if max_r is None or r_value > max_r:
            max_r = r_value
            max_row = row

    if anomalies:
        print(f"Found {len(anomalies)} negative R values not equal to -1:")
        # for row in anomalies:
        #     print(
        #         f"  order_id={row.get('order_id','')}, month={row.get('source_month','')}, "
        #         f"r_multiple={row.get('r_multiple','')}, pnl={row.get('pnl','')}, "
        #         f"stop_loss_distance={row.get('stop_loss_distance','')}"
        #     )
    else:
        print("No negative R values outside -1 detected.")

    if max_row is not None:
        print("\nLargest R value:")
        print(
            f"  order_id={max_row.get('order_id','')}, month={max_row.get('source_month','')}, "
            f"r_multiple={max_row.get('r_multiple','')}, pnl={max_row.get('pnl','')}, "
            f"stop_loss_distance={max_row.get('stop_loss_distance','')}"
        )


if __name__ == "__main__":
    main()
