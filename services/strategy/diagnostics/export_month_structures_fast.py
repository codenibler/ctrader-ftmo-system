"""Batch export swings, legs, and LVNs from a monthly parquet file."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from services.strategy.indicators.swings.find_all_zigzag_levels import build_all_zigzag_levels, parse_month
from services.strategy.indicators.legs.from_swings import extract_legs, legs_to_frame
from services.strategy.indicators.lvns.from_legs import (
    LowVolumeNode,
    compute_lvns_for_legs,
    load_parquet_candles,
    lvns_to_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export swings, legs, and LVNs for a month from parquet data (batch mode).")
    parser.add_argument("--parquet", type=Path, required=True, help="Path to 1m OHLCV parquet")
    parser.add_argument("--month", type=str, required=True, help="Month to export (YYYY_MM or YYYY-MM)")
    parser.add_argument("--symbol", type=str, default="CL", help="Symbol identifier for naming outputs")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics"), help="Destination directory for CSV outputs")
    parser.add_argument("--pct", type=float, help="Optional zigzag reversal percent override")
    parser.add_argument("--timeframe", type=str, default="minute", help="Timeframe label for LVN profile computation")
    return parser.parse_args()


def export_structures(parquet_path: Path, month: str, pct: float | None, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivots_df = build_all_zigzag_levels(parquet_path, month=month, pct=pct)
    legs = extract_legs(pivots_df)
    legs_df = legs_to_frame(legs)

    candles = load_parquet_candles(parquet_path, month=month)
    nodes = compute_lvns_for_legs(timeframe, candles, legs_df)
    lvns_df = lvns_to_frame(nodes)
    return pivots_df, legs_df, lvns_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pivots_df, legs_df, lvns_df = export_structures(args.parquet, args.month, args.pct, args.timeframe)

    suffix = args.month.replace("-", "_")
    swings_path = args.output_dir / f"{args.symbol}_swings_{suffix}.csv"
    legs_path = args.output_dir / f"{args.symbol}_legs_{suffix}.csv"
    lvns_path = args.output_dir / f"{args.symbol}_lvns_{suffix}.csv"

    pivots_df.to_csv(swings_path, index=False)
    legs_df.to_csv(legs_path, index=False)
    lvns_df.to_csv(lvns_path, index=False)

    print(f"Swings: {len(pivots_df):5d} rows -> {swings_path}")
    print(f"Legs:   {len(legs_df):5d} rows -> {legs_path}")
    print(f"LVNs:   {len(lvns_df):5d} rows -> {lvns_path}")


if __name__ == "__main__":
    main()
