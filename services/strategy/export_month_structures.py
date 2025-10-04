"""Replay a monthly parquet and export swing pivots for validation."""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import plotly.graph_objects as go
from plotly import offline as pyo

from runtime.candle_pipeline import StrategyPipeline
from runtime.parquet_ingestion import ParquetCandleSource
from runtime.strategy_state import StrategyState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a monthly parquet to capture swing pivots.")
    parser.add_argument("--parquet", type=Path, required=True, help="Path to monthly parquet file")
    parser.add_argument("--symbol", type=str, default="CL", help="Symbol identifier")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics"),
        help="Directory to write swing CSV/HTML diagnostics",
    )
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


def pivots_to_df(pivots: Iterable) -> pd.DataFrame:
    records = [
        {
            "pivot_time": pivot.pivot_time,
            "pivot_price": pivot.pivot_price,
            "pivot_type": pivot.pivot_type,
            "structure": pivot.structure,
            "pivot_idx": pivot.pivot_idx,
            "leg_return_pct": pivot.leg_return_pct,
            "leg_bars": pivot.leg_bars,
            "leg_duration_sec": pivot.leg_duration_sec,
        }
        for pivot in pivots
    ]
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values("pivot_time").reset_index(drop=True)
    return df


def build_plot(price_df: pd.DataFrame, pivots_df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig = go.Figure()

    if not price_df.empty:
        candles = price_df.copy()
        candles.index = candles.index.tz_convert(None)
        fig.add_trace(
            go.Candlestick(
                x=candles.index,
                open=candles["open"],
                high=candles["high"],
                low=candles["low"],
                close=candles["close"],
                name="Price",
            )
        )

    if not pivots_df.empty:
        plot_df = pivots_df.copy()
        plot_df["plot_time"] = pd.to_datetime(plot_df["pivot_time"]).dt.tz_convert(None)
        fig.add_trace(
            go.Scatter(
                x=plot_df["plot_time"],
                y=plot_df["pivot_price"],
                mode="lines+markers+text",
                text=plot_df["structure"],
                textposition="top center",
                name="Swings",
                marker=dict(size=6, color="#1f77b4"),
                line=dict(color="#1f77b4", width=1.5),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
        template="plotly_white",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pyo.plot(fig, filename=str(out_path), auto_open=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    pipeline = asyncio.run(process_month(args.parquet, args.symbol, args.starting_equity))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pivots_df = pivots_to_df(pipeline.detected_pivots)
    price_records = [
        {
            "timestamp": candle.timestamp,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in pipeline.state.candles
    ]
    price_df = pd.DataFrame.from_records(price_records)
    if not price_df.empty:
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)
        price_df = price_df.set_index("timestamp").sort_index()

    pivots_path = args.output_dir / f"{args.symbol}_pivots.csv"
    chart_path = args.output_dir / f"{args.symbol}_pivots.html"

    pivots_df.to_csv(pivots_path, index=False)
    build_plot(price_df, pivots_df, title=f"{args.symbol} Swing Pivots", out_path=chart_path)

    print(f"Pivots saved -> {pivots_path} ({len(pivots_df)} rows)")
    print(f"Chart saved  -> {chart_path}")


if __name__ == "__main__":
    main()
