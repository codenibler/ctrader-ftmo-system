"""Lightweight backtest using pre-exported swings, legs, and LVNs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from datetime import time

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..risk import (
    DEFAULT_CONTRACT_UNITS,
    DEFAULT_RISK_PCT,
    compute_position_size,
    contract_value_from_price,
)


Direction = Literal["uptrend", "downtrend"]

FRIDAY = 4
FRIDAY_ORDER_CUTOFF = time(16, 50)
SESSION_START = time(3, 0)
SESSION_END = time(16, 0)
FRIDAY_SESSION_CLOSE = time(16, 59)


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
    contract_value: float
    size: float
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
    risk_pct: float = DEFAULT_RISK_PCT,
) -> tuple[list[Trade], float, list[dict[str, object]]]:
    trades: list[Trade] = []
    equity = starting_equity
    geometry_debug: list[dict[str, object]] = []

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
            if stop_price >= entry_price:
                geometry_debug.append(
                    {
                        "stage": "order_creation",
                        "leg_id": leg_id,
                        "direction": direction,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "target_price": float(leg.get("end_price", float("nan"))),
                        "message": "Long entry is at/below stop; trade skipped.",
                        "logged_at": pd.Timestamp.utcnow(),
                    }
                )
                continue
        else:
            entry_price = float(node["lvn_price"]) + spread
            stop_price = float(leg["start_price"])
            structure_target = "LL"
            if stop_price <= entry_price:
                geometry_debug.append(
                    {
                        "stage": "order_creation",
                        "leg_id": leg_id,
                        "direction": direction,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "target_price": float(leg.get("end_price", float("nan"))),
                        "message": "Short entry is at/above stop; trade skipped.",
                        "logged_at": pd.Timestamp.utcnow(),
                    }
                )
                continue

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

        local_time = entry_time.time()
        if not (SESSION_START <= local_time <= SESSION_END):
            continue
        if entry_time.weekday() == FRIDAY and local_time > FRIDAY_ORDER_CUTOFF:
            continue

        swings_future = swings[swings["pivot_time"] > entry_time]
        target_pivot = swings_future[swings_future["structure"].str.upper() == structure_target]
        if target_pivot.empty:
            continue
        target_pivot = target_pivot.iloc[0]
        target_time = pd.Timestamp(target_pivot["pivot_time"])
        target_price = float(target_pivot["pivot_price"])

        # Discard targets that are not in the profitable direction so the stop remains authoritative.
        price_is_favorable = (
            (direction == "uptrend" and target_price > entry_price)
            or (direction == "downtrend" and target_price < entry_price)
        )
        if not price_is_favorable:
            target_time = None
            target_price = None
        else:
            if direction == "uptrend":
                if not (stop_price < entry_price < target_price):
                    geometry_debug.append(
                        {
                            "stage": "target_check",
                            "leg_id": leg_id,
                            "direction": direction,
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "message": "Long target geometry invalid; proceeding with stop logic.",
                            "logged_at": pd.Timestamp.utcnow(),
                        }
                    )
            else:
                if not (stop_price > entry_price > target_price):
                    geometry_debug.append(
                        {
                            "stage": "target_check",
                            "leg_id": leg_id,
                            "direction": direction,
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "message": "Short target geometry invalid; proceeding with stop logic.",
                            "logged_at": pd.Timestamp.utcnow(),
                        }
                    )

        after_entry = future_prices[future_prices["timestamp"] >= entry_time]
        cutoff_time = target_time if target_time is not None else after_entry.iloc[-1]["timestamp"]

        forced_rows = after_entry[after_entry["timestamp"].dt.weekday.eq(FRIDAY) & (after_entry["timestamp"].dt.time >= FRIDAY_SESSION_CLOSE)]
        forced_exit_time = forced_rows.iloc[0]["timestamp"] if not forced_rows.empty else None
        forced_exit_price = float(forced_rows.iloc[0]["close"]) if forced_exit_time is not None else None

        if forced_exit_time is not None and forced_exit_time < cutoff_time:
            cutoff_time = forced_exit_time

        window = after_entry[after_entry["timestamp"] <= cutoff_time]
        if window.empty:
            window = after_entry

        if direction == "uptrend":
            stop_hit_time = first_timestamp_where(window, window["low"] <= stop_price)
            target_hit_time = (
                first_timestamp_where(window, window["high"] >= target_price)
                if target_price is not None
                else None
            )
        else:
            stop_hit_time = first_timestamp_where(window, window["high"] >= stop_price)
            target_hit_time = (
                first_timestamp_where(window, window["low"] <= target_price)
                if target_price is not None
                else None
            )

        events = []
        if stop_hit_time is not None:
            events.append((stop_hit_time, "stop", stop_price))
        if target_hit_time is not None:
            events.append((target_hit_time, "target", target_price))
        if forced_exit_time is not None:
            events.append((forced_exit_time, "close", forced_exit_price if forced_exit_price is not None else target_price))

        if not events:
            continue

        exit_time, outcome, exit_price = min(events, key=lambda item: item[0])

        if entry_price <= 0:
            continue
        contract_value = contract_value_from_price(entry_price)
        size = compute_position_size(
            equity=equity,
            entry_price=entry_price,
            stop_price=stop_price,
            risk_pct=risk_pct,
            contract_units=DEFAULT_CONTRACT_UNITS,
        )
        if size <= 0:
            continue
        direction_mult = 1.0 if direction == "uptrend" else -1.0
        price_change = (exit_price - entry_price) / entry_price
        pnl = price_change * direction_mult * contract_value * size
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
                contract_value=contract_value,
                size=size,
                pnl=pnl,
                equity_after=equity,
            )
        )

    return trades, equity, geometry_debug


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
            "contract_value": trade.contract_value,
            "size": trade.size,
            "pnl": trade.pnl,
            "equity_after": trade.equity_after,
        }
        for trade in trades
    ])


def write_equity_chart(trades: pd.DataFrame, output_path: Path) -> None:
    if trades.empty:
        raise ValueError("No trades available to build equity curve.")

    equity_df = trades[["equity_after"]].copy().reset_index(drop=True)
    equity_df["trade_number"] = equity_df.index + 1

    fig = go.Figure(
        go.Scatter(
            x=equity_df["trade_number"],
            y=equity_df["equity_after"],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5, color="#1f77b4"),
            name="Equity",
            hovertemplate="Trade #%{x}<br>Equity: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Equity Curve",
        template="plotly_white",
        xaxis_title="Trade Number",
        yaxis_title="Equity",
        hovermode="x unified",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", auto_open=False)
    return fig


def write_trade_chart(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    output_path: Path,
    title: str,
    starting_equity: float,
) -> go.Figure:
    if prices.empty:
        raise ValueError("price DataFrame is empty; cannot plot chart")

    prices = prices.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    plot_df = prices.sort_values("timestamp").copy()
    plot_df["plot_ts"] = plot_df["timestamp"].dt.tz_convert(None)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, "Equity Curve"),
    )
    fig.add_trace(
        go.Candlestick(
            x=plot_df["plot_ts"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    equity_curve = pd.DataFrame(columns=["timestamp", "equity"])

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
            ),
            row=1,
            col=1,
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
            ),
            row=1,
            col=1,
        )

        equity_points = trades.sort_values("exit_time").copy()
        equity_curve = pd.DataFrame(columns=["timestamp", "equity"])
        if not equity_points.empty:
            equity_curve = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(equity_points["exit_time"], utc=True),
                    "equity": equity_points["equity_after"],
                }
            )

    if not equity_curve.empty:
        equity_curve = equity_curve.sort_values("timestamp")
        equity_curve["plot_ts"] = equity_curve["timestamp"].dt.tz_convert(None)
        fig.add_trace(
            go.Scatter(
                x=equity_curve["plot_ts"],
                y=equity_curve["equity"],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6, color="#1f77b4"),
                name="Equity",
                hovertemplate="Equity: %{y:.2f}<br>Time: %{x|%Y-%m-%d %H:%M:%S}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", auto_open=False)
    return fig

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
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT, help="Risk percentage per trade")
    parser.add_argument("--trades-csv", type=Path, help="Optional path to write trade results")
    parser.add_argument("--trades-html", type=Path, help="Optional path to write trade table as HTML")
    parser.add_argument("--chart-html", type=Path, help="Optional path to write price & equity chart")
    parser.add_argument("--equity-html", type=Path, help="Optional path to write standalone equity curve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prices = load_prices(args.parquet, args.month)
    swings = pd.read_csv(args.swings)
    legs = pd.read_csv(args.legs)
    lvns = pd.read_csv(args.lvns)

    trades, final_equity, geometry_debug = run_backtest(
        prices,
        swings,
        legs,
        lvns,
        spread=args.spread,
        starting_equity=args.starting_equity,
        risk_pct=args.risk_pct,
    )

    trades_df = trades_to_dataframe(trades)
    print(f"Generated {len(trades_df)} trades; final equity = {final_equity:.2f}")
    if geometry_debug:
        print(f"Geometry issues recorded: {len(geometry_debug)}")
    if args.trades_csv:
        args.trades_csv.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(args.trades_csv, index=False)
        print(f"Trades written -> {args.trades_csv}")
        if geometry_debug:
            debug_path = args.trades_csv.parent / "geometry_debug.csv"
            pd.DataFrame(geometry_debug).to_csv(debug_path, index=False)
            print(f"Geometry debug written -> {debug_path}")

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
        fig_trade = write_trade_chart(
            prices,
            trades_df,
            args.chart_html,
            title=f"{args.symbol} Trades ({args.month})",
            starting_equity=args.starting_equity,
        )
        print(f"Chart written -> {args.chart_html}")
        png_chart = args.chart_html.with_suffix('.png')
        fig_trade.write_image(str(png_chart), engine='kaleido')
        print(f"Chart PNG written -> {png_chart}")

    if args.equity_html:
        args.equity_html.parent.mkdir(parents=True, exist_ok=True)
        fig_equity = write_equity_chart(trades_df, args.equity_html)
        print(f"Equity curve written -> {args.equity_html}")
        png_equity = args.equity_html.with_suffix('.png')
        fig_equity.write_image(str(png_equity), engine='kaleido')
        print(f"Equity PNG written -> {png_equity}")


if __name__ == "__main__":
    main()

def write_trade_chart_png(prices: pd.DataFrame, trades: pd.DataFrame, output_path: Path, title: str, starting_equity: float) -> None:
    write_trade_chart(prices, trades, output_path, title, starting_equity)
    fig = go.Figure(go.Scatter())  # placeholder to load HTML
    fig.write_image(str(output_path.with_suffix(".png")), engine="kaleido")


def write_equity_chart_png(trades: pd.DataFrame, output_path: Path) -> None:
    write_equity_chart(trades, output_path)
    fig = go.Figure(go.Scatter())
    fig.write_image(str(output_path.with_suffix(".png")), engine="kaleido")
