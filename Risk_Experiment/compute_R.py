#!/usr/bin/env python3
"""Compute per-trade R multiples from concatenated diagnostics trades."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple


THIS_DIR = Path(__file__).resolve().parent
INPUT_CANDIDATES = [
    THIS_DIR / "new_all_trades.csv",
    THIS_DIR / "all_trades_10.24_5.25.csv",
    THIS_DIR / "all_trades_10.24_5.25.csv",
]
OUTPUT_PATH = THIS_DIR / "R_Values.csv"
STRUCTURE_BASES = [Path("new_diagnostics"), Path("diagnostics")]

MONTH_TO_NUM = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


def month_slug_to_suffix(month_slug: str) -> str | None:
    parts = month_slug.split("_")
    if len(parts) != 2:
        return None
    month_name, year = parts
    month_num = MONTH_TO_NUM.get(month_name.lower())
    if month_num is None:
        return None
    return f"{year}_{month_num}"


def build_leg_target_index() -> Dict[Tuple[str, str], float]:
    index: Dict[Tuple[str, str], float] = {}
    for base in STRUCTURE_BASES:
        if not base.exists():
            continue
        for month_dir in base.iterdir():
            if not month_dir.is_dir():
                continue
            month_slug = month_dir.name
            suffix = month_slug_to_suffix(month_slug)
            if suffix is None:
                continue
            legs_path = month_dir / f"CL_legs_{suffix}.csv"
            if not legs_path.exists():
                continue
            with legs_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    leg_id = row.get("leg_id")
                    target_price = row.get("end_price")
                    if not leg_id or not target_price:
                        continue
                    try:
                        target = float(target_price)
                    except ValueError:
                        continue
                    index[(month_slug, leg_id)] = target
    return index


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


def compute_r_multiple(
    row: dict[str, str],
    target_index: Dict[Tuple[str, str], float],
) -> tuple[float | None, float | None, float | None, float | None]:
    try:
        entry_price = float(row["entry_price"])
        stop_price = float(row["stop_price"])
        pnl = float(row["pnl"])
        direction = row.get("direction", "").lower()
    except (KeyError, TypeError, ValueError):
        return None, None, None, None

    if entry_price <= 0:
        return pnl, None, None, None

    stop_loss_distance_price = abs(entry_price - stop_price)
    if stop_loss_distance_price == 0:
        return pnl, stop_loss_distance_price, None, None

    source_month = row.get("source_month", "")
    leg_id = row.get("leg_id", "")
    target_price = None

    if source_month and leg_id:
        target_price = target_index.get((source_month, leg_id))

    if target_price is None and (row.get("outcome") or "").lower() == "target":
        # Fallback: if the trade actually hit a target, the exit price should match it.
        exit_price_raw = row.get("exit_price")
        if exit_price_raw:
            try:
                target_price = float(exit_price_raw)
            except ValueError:
                target_price = None

    if target_price is None:
        return pnl, stop_loss_distance_price, None, None

    reward_distance = abs(target_price - entry_price)
    r_multiple = reward_distance / stop_loss_distance_price if stop_loss_distance_price else None

    # Preserve sign for clarity by aligning with trade direction and intended target.
    if r_multiple is not None:
        if direction == "downtrend" and target_price > entry_price:
            r_multiple *= -1
        elif direction == "uptrend" and target_price < entry_price:
            r_multiple *= -1

    return pnl, stop_loss_distance_price, r_multiple, target_price


def main() -> None:
    input_path = pick_input_path()
    trades = list(read_trades(input_path))
    if not trades:
        raise SystemExit("No trade rows found in the input CSV.")

    target_index = build_leg_target_index()

    fieldnames = [
        "order_id",
        "source_month",
        "leg_id",
        "direction",
        "pnl",
        "stop_loss_distance",
        "r_multiple",
        "target_price",
        "outcome",
    ]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in trades:
            pnl, stop_loss_distance, r_multiple, target_price = compute_r_multiple(row, target_index)
            raw_outcome = (row.get("outcome") or "").lower()
            outcome_label = ""
            if raw_outcome == "target":
                outcome_label = "Win"
            elif raw_outcome == "stop":
                outcome_label = "Loss"
            elif raw_outcome:
                outcome_label = raw_outcome.title()

            writer.writerow(
                {
                    "order_id": row.get("order_id", ""),
                    "source_month": row.get("source_month", ""),
                    "leg_id": row.get("leg_id", ""),
                    "direction": row.get("direction", ""),
                    "pnl": f"{pnl:.10f}" if pnl is not None else "",
                    "stop_loss_distance": (
                        f"{stop_loss_distance:.10f}" if stop_loss_distance is not None else ""
                    ),
                    "r_multiple": f"{r_multiple:.10f}" if r_multiple is not None else "",
                    "target_price": f"{target_price:.10f}" if target_price is not None else "",
                    "outcome": outcome_label,
                }
            )

    print(f"R values written -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
