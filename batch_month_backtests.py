"""Batch export structures and run backtests for multiple months."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# Ensure project root on sys.path when executed directly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.strategy.diagnostics.export_month_structures_fast import export_structures
from services.strategy.diagnostics.backtest_from_structures import (
    load_prices,
    run_backtest,
    trades_to_dataframe,
    write_trade_chart,
    write_equity_chart,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export structures and run lightweight backtests for multiple months.")
    parser.add_argument("--parquet", type=Path, required=True, help="Path to CL 1m parquet file")
    parser.add_argument("--months", nargs="+", required=True, help="Months to process (YYYY_MM or YYYY-MM)")
    parser.add_argument("--symbol", type=str, default="CL", help="Symbol identifier for output naming")
    parser.add_argument("--output-base", type=Path, default=Path("diagnostics"), help="Base directory for outputs")
    parser.add_argument("--spread", type=float, default=0.02, help="Spread adjustment applied to LVN entry price")
    parser.add_argument("--starting-equity", type=float, default=10_000.0, help="Starting equity for PnL accumulation")
    return parser.parse_args()


def normalise_month(month: str) -> str:
    return month.replace('-', '_')


def month_folder_name(month: str) -> str:
    ts = pd.Timestamp(month.replace('_', '-') + '-01')
    return ts.strftime('%B_%Y').lower()


def write_structures(symbol: str, month: str, out_dir: Path,
                     swings: pd.DataFrame, legs: pd.DataFrame, lvns: pd.DataFrame) -> tuple[Path, Path, Path]:
    swings_path = out_dir / f"{symbol}_swings_{month}.csv"
    legs_path = out_dir / f"{symbol}_legs_{month}.csv"
    lvns_path = out_dir / f"{symbol}_lvns_{month}.csv"

    swings.to_csv(swings_path, index=False)
    legs.to_csv(legs_path, index=False)
    lvns.to_csv(lvns_path, index=False)

    return swings_path, legs_path, lvns_path


def write_backtest_outputs(
    prices: pd.DataFrame,
    trades_df: pd.DataFrame,
    out_dir: Path,
    month_label: str,
    symbol: str,
    starting_equity: float,
    spread: float,
) -> None:
    trades_csv = out_dir / "trades_from_structures.csv"
    trades_html = out_dir / "trades_from_structures.html"
    chart_html = out_dir / "trades_chart.html"
    equity_html = out_dir / "equity_curve.html"

    trades_df.to_csv(trades_csv, index=False)
    trades_html.parent.mkdir(parents=True, exist_ok=True)
    html = trades_df.to_html(index=False, justify="center", border=0, classes="table table-striped")
    trades_html.write_text(
        "<html><head><meta charset='utf-8'><title>Trades</title>"
        "<style>body{font-family:sans-serif;margin:20px;} table{border-collapse:collapse;width:100%;} "
        "th,td{padding:8px;border:1px solid #ddd;text-align:center;} tr:nth-child(even){background:#f7f7f7;}"
        "th{background:#222;color:#fff;}"
        "</style></head><body><h1>Backtest Trades</h1>" + html + "</body></html>"
    )

    fig_trade = write_trade_chart(
        prices,
        trades_df,
        chart_html,
        title=f"{symbol} Trades ({month_label})",
        starting_equity=starting_equity,
    )
    fig_trade.write_image(str(chart_html.with_suffix('.png')), engine='kaleido')

    fig_equity = write_equity_chart(trades_df, equity_html)
    fig_equity.write_image(str(equity_html.with_suffix('.png')), engine='kaleido')

    print(f"  Trades CSV      -> {trades_csv}")
    print(f"  Trades HTML     -> {trades_html}")
    print(f"  Trades chart    -> {chart_html}")
    print(f"  Trades PNG      -> {chart_html.with_suffix('.png')}")
    print(f"  Equity curve    -> {equity_html}")
    print(f"  Equity PNG      -> {equity_html.with_suffix('.png')}")


def main() -> None:
    args = parse_args()

    for month in args.months:
        norm_month = normalise_month(month)
        out_dir = args.output_base / month_folder_name(norm_month)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {norm_month} ...")
        swings_df, legs_df, lvns_df = export_structures(args.parquet, norm_month, pct=None, timeframe="minute")
        swings_path, legs_path, lvns_path = write_structures(args.symbol, norm_month, out_dir, swings_df, legs_df, lvns_df)
        print(f"  Structures saved -> {swings_path}, {legs_path}, {lvns_path}")

        prices = load_prices(args.parquet, norm_month)
        trades, final_equity = run_backtest(
            prices,
            swings_df,
            legs_df,
            lvns_df,
            spread=args.spread,
            starting_equity=args.starting_equity,
        )
        trades_df = trades_to_dataframe(trades)
        print(f"  Trades: {len(trades_df)}; final equity = {final_equity:.2f}")

        write_backtest_outputs(
            prices,
            trades_df,
            out_dir,
            norm_month,
            args.symbol,
            args.starting_equity,
            args.spread,
        )

    print("Done.")


if __name__ == "__main__":
    main()
