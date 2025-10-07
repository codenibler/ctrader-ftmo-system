#!/usr/bin/env python3
"""Concatenate `trades_from_structures.csv` files into a single dataset."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "filled_trades_Risk_measurement.csv"
DEFAULT_INPUT_DIRS = [Path("diagnostics")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate backtest trade CSV exports.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUT_DIRS,
        help="Directories to search (default: diagnostics)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the combined CSV (default: Risk_Experiment/filled_trades_Risk_measurement.csv)",
    )
    return parser.parse_args()


def find_trade_files(directories: Sequence[Path]) -> List[Tuple[datetime, Path]]:
    results: List[Tuple[datetime, Path]] = []
    for base in directories:
        if not base.exists():
            continue
        for csv_path in base.rglob("trades_from_structures.csv"):
            month = csv_path.parent.name
            try:
                dt = datetime.strptime(month.replace("_", " ").title(), "%B %Y")
            except ValueError:
                continue
            results.append((dt, csv_path))
    results.sort(key=lambda item: item[0])
    return results


def read_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        for row in reader:
            # Attach the source directory for traceability if not already present
            row.setdefault("source_month", path.parent.name)
            yield row


def concatenate(trade_files: Sequence[Tuple[datetime, Path]], output_path: Path) -> None:
    if not trade_files:
        raise SystemExit("No trades_from_structures.csv files found.")

    fieldnames: List[str] = []
    rows: List[dict[str, str]] = []

    for _, path in trade_files:
        file_rows = list(read_rows(path))
        if not file_rows:
            continue

        if not fieldnames:
            fieldnames = list(file_rows[0].keys())
            if "source_month" not in fieldnames:
                fieldnames.append("source_month")
        for row in file_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    args = parse_args()
    trade_files = find_trade_files(args.inputs)
    concatenate(trade_files, args.output)


if __name__ == "__main__":
    main()
