"""Run LVN strategy backtest on a monthly parquet and plot the equity curve."""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from runtime.candle_pipeline import StrategyPipeline
from runtime.parquet_ingestion import ParquetCandleSource
from runtime.strategy_state import StrategyState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest LVN strategy on a monthly parquet file.")
    parser.add_argument("--parquet", type=Path, required=True, help="Path to monthly parquet file")
    parser.add_argument("--symbol", type=str, default="CL", help="Symbol identifier")
    parser.add_argument("--starting-equity", type=float, default=10_000.0, help="Starting account equity")
    parser.add_argument(
        "--output-equity",
        type=Path,
        default=Path("backtests") / "equity_curve.html",
        help="Where to write the equity curve HTML chart",
    )
    parser.add_argument(
        "--output-trades",
        type=Path,
        default=Path("backtests") / "trade_chart.html",
        help="Where to write the trade overlay chart",
    )
    return parser.parse_args()


async def run_backtest(parquet_path: Path, symbol: str, starting_equity: float) -> StrategyPipeline:
    state = StrategyState(symbol=symbol, equity=starting_equity)
    pipeline = StrategyPipeline(state)

    source = ParquetCandleSource(parquet_path)
    candles = source.load()
    for candle in candles:
        await pipeline.handle_candle(candle)
    return pipeline


def build_equity_curve(trades: pd.DataFrame, starting_equity: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame({"timestamp": [], "equity": []})
    trades = trades.sort_values("exit_time").reset_index(drop=True)
    curve = trades[["exit_time", "equity_after"]].rename(columns={"exit_time": "timestamp", "equity_after": "equity"})
    if curve.iloc[0]["equity"] != starting_equity:
        curve = pd.concat([
            pd.DataFrame({"timestamp": [trades.iloc[0]["entry_time"]], "equity": [starting_equity]}),
            curve,
        ], ignore_index=True)
    return curve


def plot_equity_curve(curve: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["timestamp"],
            y=curve["equity"],
            mode="lines+markers",
            name="Equity",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Equity",
    )
    fig.write_html(str(output_path), auto_open=False)


def plot_trade_chart(
    candles: pd.DataFrame,
    trades: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = candles.copy()
    plot_df["plot_ts"] = pd.to_datetime(plot_df["timestamp"]).dt.tz_convert(None)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=plot_df["plot_ts"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Price",
        )
    )

    if not trades.empty:
        entries = trades.copy()
        exits = trades.copy()
        entries["plot_ts"] = pd.to_datetime(entries["entry_time"]).dt.tz_convert(None)
        exits["plot_ts"] = pd.to_datetime(exits["exit_time"]).dt.tz_convert(None)

        fig.add_trace(
            go.Scatter(
                x=entries["plot_ts"],
                y=entries["entry_price"],
                mode="markers",
                name="Entry",
                marker=dict(
                    symbol=entries["direction"].map({"uptrend": "triangle-up", "downtrend": "triangle-down"}),
                    size=10,
                    color=entries["direction"].map({"uptrend": "#2ca02c", "downtrend": "#d62728"}),
                    line=dict(width=1, color="#000000"),
                ),
                customdata=entries[["leg_id", "spread", "slippage"]],
                hovertemplate=(
                    "Entry leg=%{customdata[0]}<br>Price=%{y:.4f}<br>Spread=%{customdata[1]:.4f}<br>Slip=%{customdata[2]:.4f}<br>Time=%{x|%Y-%m-%d %H:%M}"
                ),
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=exits["plot_ts"],
                y=exits["exit_price"],
                mode="markers",
                name="Exit",
                marker=dict(
                    symbol="x",
                    size=8,
                    color=exits["outcome"].map({"target": "#1f77b4", "stop": "#ff7f0e"}).fillna("#7f7f7f"),
                    line=dict(width=1, color="#000000"),
                ),
                customdata=exits[["outcome", "pnl"]],
                hovertemplate=(
                    "Exit (%{customdata[0]})<br>Price=%{y:.4f}<br>PnL=%{customdata[1]:.2f}<br>Time=%{x|%Y-%m-%d %H:%M}"
                ),
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Price",
    )
    fig.write_html(str(output_path), auto_open=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    pipeline = asyncio.run(run_backtest(args.parquet, args.symbol, args.starting_equity))

    trades = pd.DataFrame(pipeline.get_completed_trades())
    if trades.empty:
        print("No trades were completed in the selected month.")
        return

    curve = build_equity_curve(trades, args.starting_equity)
    plot_equity_curve(curve, args.output_equity, title=f"{args.symbol} LVN Strategy Equity")

    candles_df = pd.read_parquet(args.parquet)
    if "timestamp" not in candles_df.columns:
        raise ValueError("Parquet file must contain a 'timestamp' column for plotting")
    candles_df["timestamp"] = pd.to_datetime(candles_df["timestamp"], utc=True, errors="coerce")
    candles_df = candles_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    plot_trade_chart(candles_df, trades, args.output_trades, title=f"{args.symbol} LVN Trades")

    final_equity = curve.iloc[-1]["equity"]
    print(f"Completed trades: {len(trades)}")
    print(f"Final equity: {final_equity:.2f}")
    print(f"Equity chart saved to {args.output_equity}")
    print(f"Trade chart saved to {args.output_trades}")


if __name__ == "__main__":
    main()
