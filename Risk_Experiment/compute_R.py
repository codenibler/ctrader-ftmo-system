#!/usr/bin/env python3
"""Compute realised R multiples from trade logs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List


THIS_DIR = Path(__file__).resolve().parent
INPUT_CANDIDATES = [
    THIS_DIR / "new_all_trades.csv",
    THIS_DIR / "filled_trades_Risk_measurement.csv",
    THIS_DIR / "all_trades_10.24_5.25.csv",
]
OUTPUT_PATH = THIS_DIR / "R_Values.csv"


def pick_input_path() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No trade aggregate CSV found in Risk_Experiment directory")


def read_trades(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        for row in reader:
            yield row


def compute_r_multiple(row: dict[str, str]) -> tuple[float | None, float | None, float | None]:
    try:
        entry_price = float(row["entry_price"])
        stop_price = float(row["stop_price"])
        exit_price = float(row["exit_price"])
        contract_value = float(row.get("contract_value", ""))
        size = float(row.get("size", ""))
        pnl = float(row["pnl"])
    except (KeyError, TypeError, ValueError):
        return None, None, None

    if entry_price <= 0 or size <= 0:
        return pnl, None, None

    stop_distance_price = abs(entry_price - stop_price)
    if stop_distance_price == 0:
        return pnl, stop_distance_price, None

    if contract_value and contract_value > 0:
        contract_units = contract_value / entry_price
    else:
        # Default to CL micro contract size (100) if not provided.
        contract_units = 100.0

    risk_per_contract = stop_distance_price * contract_units
    if risk_per_contract == 0:
        return pnl, stop_distance_price, None

    risk_total = risk_per_contract * size
    if risk_total == 0:
        return pnl, stop_distance_price, None

    r_multiple = pnl / risk_total
    return pnl, risk_per_contract, r_multiple


def normalise_outcome(row: dict[str, str], r_multiple: float | None) -> str:
    label = (row.get("outcome") or "").strip().lower()
    if label in {"win", "loss"}:
        return label.title()
    if r_multiple is None:
        return ""
    return "Win" if r_multiple > 0 else "Loss"


def main() -> None:
    input_path = pick_input_path()
    trades = list(read_trades(input_path))
    if not trades:
        raise SystemExit("No trade rows found in input CSV")

    fieldnames: List[str] = [
        "order_id",
        "source_month",
        "leg_id",
        "direction",
        "pnl",
        "risk_per_contract",
        "r_multiple",
        "outcome",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in trades:
            pnl, risk_per_contract, r_multiple = compute_r_multiple(row)
            outcome = normalise_outcome(row, r_multiple)
            writer.writerow(
                {
                    "order_id": row.get("order_id", ""),
                    "source_month": row.get("source_month", ""),
                    "leg_id": row.get("leg_id", ""),
                    "direction": row.get("direction", ""),
                    "pnl": f"{pnl:.10f}" if pnl is not None else "",
                    "risk_per_contract": (
                        f"{risk_per_contract:.10f}" if risk_per_contract is not None else ""
                    ),
                    "r_multiple": f"{r_multiple:.10f}" if r_multiple is not None else "",
                    "outcome": outcome,
                }
            )

    print(f"R values written -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
