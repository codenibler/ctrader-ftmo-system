from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly import offline as pyo

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from findings.swings.swings_extract import (  # type: ignore[attr-defined]
        PCT_BY_TIMEFRAME,
        confirming_zigzag,
        determine_pct,
        label_structure,
        prepare_df,
    )
except ModuleNotFoundError:
    fallback_paths = [
        PROJECT_ROOT / 'findings' / 'swings' / 'swings_extract.py',
        PROJECT_ROOT / 'formula' / 'findings' / 'swings' / 'swings_extract.py',
    ]
    swings_path = None
    for candidate in fallback_paths:
        if candidate.exists():
            swings_path = candidate
            break
    if swings_path is None:
        raise
    spec = importlib.util.spec_from_file_location('_swings_extract_module', swings_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PCT_BY_TIMEFRAME = module.PCT_BY_TIMEFRAME
    confirming_zigzag = module.confirming_zigzag
    determine_pct = module.determine_pct
    label_structure = module.label_structure
    prepare_df = module.prepare_df


def plot_all_zigzag_levels(price_df: pd.DataFrame, pivots: pd.DataFrame, out_path: Path, title: str) -> None:
    fig = go.Figure()

    if not price_df.empty:
        candles = price_df.copy()
        if isinstance(candles.index, pd.DatetimeIndex) and candles.index.tz is not None:
            candles.index = candles.index.tz_convert(None)
        fig.add_trace(
            go.Candlestick(
                x=candles.index,
                open=candles['open'],
                high=candles['high'],
                low=candles['low'],
                close=candles['close'],
                name='Price',
            )
        )

    if not pivots.empty:
        plot_df = pivots.copy()
        plot_df['plot_time'] = pd.to_datetime(plot_df['pivot_time']).dt.tz_convert(None)
        fig.add_trace(
            go.Scatter(
                x=plot_df['plot_time'],
                y=plot_df['pivot_price'],
                mode='lines+markers+text',
                text=plot_df['structure'],
                textposition='top center',
                textfont=dict(size=9, color='#1f77b4'),
                marker=dict(size=6, color='#1f77b4'),
                line=dict(color='#1f77b4', width=1.5),
                name='ZigZag',
                hovertemplate=(
                    'Structure: %{text}<br>'
                    'Pivot: %{y:.4f}<br>'
                    'Type: %{customdata[0]}<br>'
                    'Time: %{x|%Y-%m-%d %H:%M:%S}<br>'
                    'Leg Return: %{customdata[1]:.2f}%<br>'
                    'Bars: %{customdata[2]}<br>'
                    'Duration: %{customdata[3]:.0f}s<extra></extra>'
                ),
                customdata=plot_df[['pivot_type', 'leg_return_pct', 'leg_bars', 'leg_duration_sec']].to_numpy(),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=True,
        hovermode='x unified',
        template='plotly_white',
        yaxis_title='Price',
        dragmode='zoom',
        uirevision='zigzag_levels',
        modebar_add=['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d', 'toggleSpikelines'],
        modebar_remove=['select2d', 'lasso2d'],
        margin=dict(l=70, r=60, t=60, b=40),
    )

    fig.update_yaxes(
        fixedrange=False,
        autorange=True,
        rangemode='normal',
        automargin=True,
        spikemode='across+toaxis',
        spikesnap='cursor',
        showspikes=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pyo.plot(fig, filename=str(out_path), auto_open=False)
def build_all_zigzag_levels(month: str) -> None:
    try:
        year_str, month_str = month.split('_')
        year = int(year_str)
        month_num = int(month_str)
    except ValueError as exc:
        raise ValueError(f'invalid month format: {month!r}, expected YYYY_MM') from exc

    start = pd.Timestamp(year=year, month=month_num, day=1, tz='UTC')
    end = start + pd.offsets.MonthBegin(1)

    data_path = PROJECT_ROOT / 'data' / 'minute_months' / month / 'CL_1m.csv'
    if not data_path.exists():
        raise FileNotFoundError(f'missing monthly data file: {data_path}')

    source_df = prepare_df(data_path)
    if source_df.empty:
        raise ValueError(f'{month}: no rows loaded from {data_path}')

    month_df = source_df.loc[(source_df.index >= start) & (source_df.index < end)]
    if month_df.empty:
        raise ValueError(f'{month}: no rows after filtering timestamps')

    pct = PCT_BY_TIMEFRAME.get('minute', 0.12)
    pct = determine_pct('minute', month_df, pct)
    print(f'Processing zigzag {month}: {len(month_df)} rows @ {pct:.3f}% threshold')

    pivots = confirming_zigzag(month_df, pct=pct)
    pivots = label_structure(pivots)

    pivots['swing_label'] = pivots['structure'] + '_' + pivots['pivot_type'].str.upper()

    suffix = 'october' if month == '2024_10' else month
    out_csv = SCRIPT_DIR / f'zigzag_levels_minute_{suffix}.csv'
    out_html = SCRIPT_DIR / f'zigzag_levels_minute_{suffix}.html'

    pivots.to_csv(out_csv, index=False)

    plot_df = month_df.copy()
    plot_all_zigzag_levels(plot_df, pivots, out_html, title=f'All ZigZag Levels {month}')

    print(f'Saved zigzag pivots to {out_csv}')
    print(f'Saved zigzag chart to {out_html}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Find all ZigZag levels for a given month (YYYY_MM).')
    parser.add_argument('month', nargs='?', default='2024_10', help='Month in YYYY_MM format (default: 2024_10).')
    args = parser.parse_args()
    build_all_zigzag_levels(args.month)


if __name__ == '__main__':
    main()










