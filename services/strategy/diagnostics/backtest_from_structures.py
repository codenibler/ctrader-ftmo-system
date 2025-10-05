"""Lightweight backtest using pre-exported swings, legs, and LVNs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import plotly.graph_objects as go


Direction = Literal["uptrend", "downtrend"]


@dataclass
class Trade:
    order_id: int
    leg_id: int
    direction: Direction
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    exit_time: pd.Timestamp
    exit_price: float
    outcome: str
    pnl: float
    equity_after: float


def load_prices(parquet_path: Path, month: str | None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "timestamp" not in df.columns:
        raise ValueError("Parquet data must include a 'timestamp' column")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if month:
        start, end = parse_month(month)
        df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    return df


def parse_month(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    year_str, month_str = month.split("_") if "_" in month else month.split("-")
    start = pd.Timestamp(year=int(year_str), month=int(month_str), day=1, tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def first_timestamp_where(df: pd.DataFrame, condition: pd.Series) -> pd.Timestamp | None:
    idx = condition[condition].index
    if len(idx) == 0:
        return None
    return pd.Timestamp(df.loc[idx[0], "timestamp"])


def run_backtest(
    prices: pd.DataFrame,
    swings: pd.DataFrame,
    legs: pd.DataFrame,
    lvns: pd.DataFrame,
    *,
    spread: float,
    starting_equity: float,
) -> tuple[list[Trade], float]:
    trades: list[Trade] = []
    equity = starting_equity

    swings = swings.copy()
    swings["pivot_time"] = pd.to_datetime(swings["pivot_time"], utc=True)

    legs = legs.copy()
    legs["leg_id"] = legs["leg_id"].astype(int)
    legs = legs.set_index("leg_id", drop=False)

    lvns = lvns.copy()
    lvns["leg_id"] = lvns["leg_id"].astype(int)
    lvns = lvns.sort_values(["start_ts", "lvn_rank"]).reset_index(drop=True)
    lvns["start_ts"] = pd.to_datetime(lvns["start_ts"], utc=True)
    lvns["end_ts"] = pd.to_datetime(lvns["end_ts"], utc=True)

    prices = prices.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

    for order_id, node in lvns.iterrows():
        leg_id = int(node["leg_id"])
        if leg_id not in legs.index:
            continue
        leg = legs.loc[leg_id]
        direction: Direction = "uptrend" if node["leg_direction"] == "uptrend" else "downtrend"

        if direction == "uptrend":
            entry_price = float(node["lvn_price"]) - spread
            stop_price = float(leg["start_price"])
            structure_target = "HH"
        else:
            entry_price = float(node["lvn_price"]) + spread
            stop_price = float(leg["start_price"])
            structure_target = "LL"

        start_ts = pd.Timestamp(node["end_ts"])  # leg completed at end_ts
        future_prices = prices[prices["timestamp"] >= start_ts]
        if future_prices.empty:
            continue

        if direction == "uptrend":
            entry_mask = future_prices["low"] <= entry_price
        else:
            entry_mask = future_prices["high"] >= entry_price
        entry_time = first_timestamp_where(future_prices, entry_mask)
        if entry_time is None:
            continue

        swings_future = swings[swings["pivot_time"] > entry_time]
        target_pivot = swings_future[swings_future["structure"].str.upper() == structure_target]
        if target_pivot.empty:
            continue
        target_pivot = target_pivot.iloc[0]
        target_time = pd.Timestamp(target_pivot["pivot_time"])
        target_price = float(target_pivot["pivot_price"])

        after_entry = future_prices[future_prices["timestamp"] >= entry_time]
        window = after_entry[after_entry["timestamp"] <= target_time]
        if window.empty:
            window = after_entry

        if direction == "uptrend":
            stop_hit_time = first_timestamp_where(window, window["low"] <= stop_price)
            target_hit_time = first_timestamp_where(window, window["high"] >= target_price)
        else:
            stop_hit_time = first_timestamp_where(window, window["high"] >= stop_price)
            target_hit_time = first_timestamp_where(window, window["low"] <= target_price)

        outcome: str
        exit_time: pd.Timestamp
        exit_price: float

        if stop_hit_time is not None and (target_hit_time is None or stop_hit_time <= target_hit_time):
            outcome = "stop"
            exit_time = stop_hit_time
            exit_price = stop_price
        elif target_hit_time is not None:
            outcome = "target"
            exit_time = target_hit_time
            exit_price = target_price
        else:
            # neither stop nor target hit within data window; skip trade
            continue

        direction_mult = 1.0 if direction == "uptrend" else -1.0
        pnl = (exit_price - entry_price) * direction_mult
        equity += pnl
        trades.append(
            Trade(
                order_id=order_id,
                leg_id=leg_id,
                direction=direction,
                entry_time=entry_time,
                entry_price=entry_price,
                stop_price=stop_price,
                exit_time=exit_time,
                exit_price=exit_price,
                outcome=outcome,
                pnl=pnl,
                equity_after=equity,
            )
        )

    return trades, equity


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "order_id": trade.order_id,
            "leg_id": trade.leg_id,
            "direction": trade.direction,
            "entry_time": trade.entry_time,
            "entry_price": trade.entry_price,
            "stop_price": trade.stop_price,
            "exit_time": trade.exit_time,
            "exit_price": trade.exit_price,
            "outcome": trade.outcome,
            "pnl": trade.pnl,
            "equity_after": trade.equity_after,
        }
        for trade in trades
    ])


def write_trade_chart(prices: pd.DataFrame, trades: pd.DataFrame, output_path: Path, title: str) -> None:
    if prices.empty:
        raise ValueError("price DataFrame is empty; cannot plot chart")

    prices = prices.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    plot_df = prices.sort_values("timestamp").copy()
    plot_df["plot_ts"] = plot_df["timestamp"].dt.tz_convert(None)

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
        trades = trades.copy()
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
        trades["entry_plot"] = trades["entry_time"].dt.tz_convert(None)
        trades["exit_plot"] = trades["exit_time"].dt.tz_convert(None)

        entry_marker = trades["direction"].map({"uptrend": "triangle-up", "downtrend": "triangle-down"})
        entry_color = trades["direction"].map({"uptrend": "#2ca02c", "downtrend": "#d62728"})

        fig.add_trace(
            go.Scatter(
                x=trades["entry_plot"],
                y=trades["entry_price"],
                mode="markers",
                name="Entry",
                marker=dict(symbol=entry_marker, size=9, color=entry_color, line=dict(width=1, color="#222")),
                customdata=trades[["order_id", "leg_id", "direction", "stop_price"]],
                hovertemplate=(
                    "Entry #%{customdata[0]} (leg %{customdata[1]})<br>"
                    "Dir: %{customdata[2]}<br>Price: %{y:.4f}<br>Stop: %{customdata[3]:.4f}<br>"
                    "Time: %{x|%Y-%m-%d %H:%M:%S}"
                ),
            )
        )

        exit_color = trades["outcome"].map({"target": "#1f77b4", "stop": "#ff7f0e"}).fillna("#7f7f7f")
        fig.add_trace(
            go.Scatter(
                x=trades["exit_plot"],
                y=trades["exit_price"],
                mode="markers",
                name="Exit",
                marker=dict(symbol="x", size=8, color=exit_color, line=dict(width=1, color="#222")),
                customdata=trades[["order_id", "outcome", "pnl"]],
                hovertemplate=(
                    "Exit #%{customdata[0]} (%{customdata[1]})<br>Price: %{y:.4f}<br>PnL: %{customdata[2]:.2f}<br>"
                    "Time: %{x|%Y-%m-%d %H:%M:%S}"
                ),
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", auto_open=False)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight LVN backtest using exported structures.")
    parser.add_argument("--parquet", type=Path, required=True, help="Path to 1m OHLCV parquet")
    parser.add_argument("--month", type=str, required=True, help="Month of data (YYYY_MM or YYYY-MM)")
    parser.add_argument("--symbol", type=str, default="CL", help="Symbol identifier for chart title")
    parser.add_argument("--swings", type=Path, required=True, help="CSV of swings exported by export_month_structures_fast")
    parser.add_argument("--legs", type=Path, required=True, help="CSV of legs exported by export_month_structures_fast")
    parser.add_argument("--lvns", type=Path, required=True, help="CSV of LVNs exported by export_month_structures_fast")
    parser.add_argument("--spread", type=float, default=0.02, help="Spread adjustment applied to LVN entry price")
    parser.add_argument("--starting-equity", type=float, default=10_000.0, help="Starting equity for cumulative PnL")
    parser.add_argument("--trades-csv", type=Path, help="Optional path to write trade results")
    parser.add_argument("--trades-html", type=Path, help="Optional path to write trade table as HTML")
    parser.add_argument("--chart-html", type=Path, help="Optional path to write price chart with trades")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prices = load_prices(args.parquet, args.month)
    swings = pd.read_csv(args.swings)
    legs = pd.read_csv(args.legs)
    lvns = pd.read_csv(args.lvns)

    trades, final_equity = run_backtest(
        prices,
        swings,
        legs,
        lvns,
        spread=args.spread,
        starting_equity=args.starting_equity,
    )

    trades_df = trades_to_dataframe(trades)
    print(f"Generated {len(trades_df)} trades; final equity = {final_equity:.2f}")
    if args.trades_csv:
        args.trades_csv.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(args.trades_csv, index=False)
        print(f"Trades written -> {args.trades_csv}")

    if args.trades_html:
        args.trades_html.parent.mkdir(parents=True, exist_ok=True)
        html = trades_df.to_html(index=False, justify="center", border=0, classes="table table-striped")
        args.trades_html.write_text("<html><head><meta charset='utf-8'><title>Trades</title>"
                                  "<style>body{font-family:sans-serif;margin:20px;} table{border-collapse:collapse;width:100%;} "
                                  "th,td{padding:8px;border:1px solid #ddd;text-align:center;} tr:nth-child(even){background:#f7f7f7;}"
                                  "th{background:#222;color:#fff;}"
                                  "</style></head><body><h1>Backtest Trades</h1>" + html + "</body></html>")
        print(f"Trades HTML written -> {args.trades_html}")

    if args.chart_html:
        args.chart_html.parent.mkdir(parents=True, exist_ok=True)
        write_trade_chart(prices, trades_df, args.chart_html, title=f"{args.symbol} Trades ({args.month})")
        print(f"Chart written -> {args.chart_html}")


if __name__ == "__main__":
    main()
