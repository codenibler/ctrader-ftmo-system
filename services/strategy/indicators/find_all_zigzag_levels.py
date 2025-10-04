"""Generate zigzag swing pivots from parquet data using legacy helpers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .swings_extract import (
        PCT_BY_TIMEFRAME,
        confirming_zigzag,
        determine_pct,
        label_structure,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from swings_extract import (
        PCT_BY_TIMEFRAME,
        confirming_zigzag,
        determine_pct,
        label_structure,
    )

DEFAULT_TIMEFRAME = "minute"
DEFAULT_THRESHOLD_PCT = PCT_BY_TIMEFRAME.get(DEFAULT_TIMEFRAME, 0.12)


def load_parquet_prices(
    parquet_path: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load OHLCV data from parquet, normalized like legacy CSV ingestion."""

    df = pd.read_parquet(parquet_path)
    if "timestamp" not in df.columns:
        raise ValueError("Parquet file must contain a 'timestamp' column")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing required columns: {', '.join(missing)}")

    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required_cols)

    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index < end]

    return df


def parse_month(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        year_str, month_str = month.split("-") if "-" in month else month.split("_")
        year = int(year_str)
        month_num = int(month_str)
    except ValueError as exc:
        raise ValueError(f"Invalid month format '{month}'. Expected YYYY_MM or YYYY-MM") from exc

    start = pd.Timestamp(year=year, month=month_num, day=1, tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def build_all_zigzag_levels(
    parquet_path: Path,
    month: str | None = None,
    pct: float | None = None,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    start = end = None
    if month:
        start, end = parse_month(month)

    prices = load_parquet_prices(parquet_path, start=start, end=end)
    if prices.empty:
        raise ValueError("Price DataFrame is empty for the requested range")

    base_pct = pct if pct is not None else PCT_BY_TIMEFRAME.get(timeframe, DEFAULT_THRESHOLD_PCT)
    pct_value = determine_pct(timeframe, prices, base_pct)

    pivots = confirming_zigzag(prices, pct=pct_value)
    pivots = label_structure(pivots)
    pivots = pivots.sort_values("pivot_time").reset_index(drop=True)
    return pivots


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate zigzag swing pivots from a parquet file.")
    parser.add_argument("--parquet", type=Path, required=True, help="Path to 1m OHLCV parquet")
    parser.add_argument("--month", type=str, help="Optional month filter (YYYY_MM or YYYY-MM)")
    parser.add_argument("--pct", type=float, help="Override reversal threshold percent")
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    pivots_df = build_all_zigzag_levels(args.parquet, args.month, pct=args.pct)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pivots_df.to_csv(args.output, index=False)
        print(f"Saved {len(pivots_df)} pivots -> {args.output}")
    else:
        print(pivots_df.head())
        print(f"Generated {len(pivots_df)} pivots (no output path supplied)")


if __name__ == "__main__":
    main()
