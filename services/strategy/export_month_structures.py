"""Export swings, legs, and LVNs for a monthly parquet replay."""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import List

import pandas as pd

from runtime.candle_pipeline import StrategyPipeline
from runtime.parquet_ingestion import ParquetCandleSource
from runtime.strategy_state import StrategyState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LVN diagnostic data for a monthly parquet.")
    parser.add_argument("--parquet", type=Path, required=True, help="Path to monthly parquet file")
    parser.add_argument("--symbol", type=str, default="CL", help="Symbol identifier")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics"), help="Directory to write CSV diagnostics")
    parser.add_argument("--starting-equity", type=float, default=10_000.0, help="Initial equity (for completeness)")
    return parser.parse_args()


async def process_month(parquet_path: Path, symbol: str, starting_equity: float) -> StrategyPipeline:
    state = StrategyState(symbol=symbol, equity=starting_equity)
    pipeline = StrategyPipeline(state)
    source = ParquetCandleSource(parquet_path)
    candles = source.load()
    for candle in candles:
        await pipeline.handle_candle(candle)
    return pipeline


def pivots_to_df(pivots: List) -> pd.DataFrame:
    records = [
        {
            "pivot_time": pivot.pivot_time,
            "pivot_price": pivot.pivot_price,
            "pivot_type": pivot.pivot_type,
            "structure": pivot.structure,
            "pivot_idx": pivot.pivot_idx,
        }
        for pivot in pivots
    ]
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values("pivot_time").reset_index(drop=True)
    return df


def legs_to_df(legs: List) -> pd.DataFrame:
    records = [
        {
            "leg_id": leg.leg_id,
            "start_ts": leg.start_ts,
            "end_ts": leg.end_ts,
            "start_price": leg.start_price,
            "end_price": leg.end_price,
            "leg_pattern": leg.leg_pattern,
            "leg_direction": leg.leg_direction,
            "leg_return_pct": leg.leg_return_pct,
            "leg_bars": leg.leg_bars,
            "leg_duration_sec": leg.leg_duration_sec,
        }
        for leg in legs
    ]
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values("start_ts").reset_index(drop=True)
    return df


def lvns_to_df(lvns: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(lvns)
    if not df.empty:
        df = df.sort_values(["start_ts", "lvn_rank"]).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    pipeline = asyncio.run(process_month(args.parquet, args.symbol, args.starting_equity))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pivots_df = pivots_to_df(list(pipeline.state.pivots))
    legs_df = legs_to_df(list(pipeline.state.legs))
    lvns_df = lvns_to_df(pipeline.all_lvns)

    pivots_path = args.output_dir / f"{args.symbol}_pivots.csv"
    legs_path = args.output_dir / f"{args.symbol}_legs.csv"
    lvns_path = args.output_dir / f"{args.symbol}_lvns.csv"

    pivots_df.to_csv(pivots_path, index=False)
    legs_df.to_csv(legs_path, index=False)
    lvns_df.to_csv(lvns_path, index=False)

    print(f"Pivots saved -> {pivots_path} ({len(pivots_df)} rows)")
    print(f"Legs saved   -> {legs_path} ({len(legs_df)} rows)")
    print(f"LVNs saved   -> {lvns_path} ({len(lvns_df)} rows)")


if __name__ == "__main__":
    main()
