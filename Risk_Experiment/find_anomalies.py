#!/usr/bin/env python3
"""Identify suspicious trade exits in concatenated diagnostics data."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
INPUT_CANDIDATES = [
    THIS_DIR / "filled_trades_Risk_measurement.csv",
    THIS_DIR / "filled_trades_risk_measurement.csv",
]
OUTPUT_PATH = THIS_DIR / "anomalies.csv"
TOLERANCE = 1e-6


def pick_input_path() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Expected concatenated trades CSV (filled_trades_Risk_measurement.csv) not found."
    )


def read_trades(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        yield from reader


def safe_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def analyze_row(row: dict[str, str]) -> list[str]:
    reasons: list[str] = []

    outcome = (row.get("outcome") or "").lower()
    direction = (row.get("direction") or "").lower()

    pnl = safe_float(row.get("pnl"))
    entry_price = safe_float(row.get("entry_price"))
    stop_price = safe_float(row.get("stop_price"))
    exit_price = safe_float(row.get("exit_price"))

    if outcome == "target" and pnl is not None and pnl < 0:
        reasons.append("target outcome reported but PnL is negative")

    if outcome == "target" and None not in (stop_price, exit_price):
        if direction == "uptrend" and exit_price < stop_price - TOLERANCE:
            reasons.append("long target exit below stop price")
        elif direction == "downtrend" and exit_price > stop_price + TOLERANCE:
            reasons.append("short target exit above stop price")

    if outcome == "stop" and None not in (stop_price, exit_price):
        if abs(exit_price - stop_price) > TOLERANCE:
            reasons.append("stop outcome but exit price differs from stop")

    if outcome == "target" and None not in (entry_price, exit_price):
        if direction == "uptrend" and exit_price < entry_price - TOLERANCE:
            reasons.append("long target exit below entry price")
        elif direction == "downtrend" and exit_price > entry_price + TOLERANCE:
            reasons.append("short target exit above entry price")

    return reasons


def main() -> None:
    input_path = pick_input_path()
    anomalies: list[dict[str, str]] = []

    for row in read_trades(input_path):
        reasons = analyze_row(row)
        if reasons:
            new_row = dict(row)
            new_row["anomaly_reason"] = "; ".join(reasons)
            anomalies.append(new_row)

    if not anomalies:
        print("No anomalies detected.")
        return

    fieldnames = list(anomalies[0].keys())
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(anomalies)

    print(f"Wrote {len(anomalies)} anomalies to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
