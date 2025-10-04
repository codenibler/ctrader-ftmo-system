"""Extract HL->HH and LH->LL swing legs from existing swing outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go
from plotly import offline as pyo

import pandas as pd

from findings.lvns.volume_profile import load_ohlcv_csv

TIMEFRAMES: Dict[str, str] = {
    'minute': 'swings_minute.csv',
    '5minute': 'swings_5minute.csv',
    'hour': 'swings_hour.csv',
}

REQUIRED_COLUMNS = {
    'pivot_time',
    'pivot_price',
    'pivot_type',
    'structure',
    'leg_return_pct',
    'leg_bars',
    'leg_duration_sec',
}


# Load swing pivots for a timeframe

def load_swings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
    df['pivot_time'] = pd.to_datetime(df['pivot_time'], utc=True, errors='coerce')
    df = df.dropna(subset=['pivot_time']).sort_values('pivot_time').reset_index(drop=True)
    return df




# Plot price with leg overlays
def plot_legs_chart(tf_name: str, ohlcv: pd.DataFrame, legs: pd.DataFrame, out_path: Path) -> None:
    fig = go.Figure()

    if not ohlcv.empty:
        plot_df = ohlcv.copy()
        plot_df['plot_ts'] = pd.to_datetime(plot_df['timestamp']).dt.tz_convert(None)
        fig.add_trace(
            go.Candlestick(
                x=plot_df['plot_ts'],
                open=plot_df['open'],
                high=plot_df['high'],
                low=plot_df['low'],
                close=plot_df['close'],
                name='Price',
            )
        )

    if not legs.empty:
        legs_plot = legs.copy()
        legs_plot['start_ts'] = pd.to_datetime(legs_plot['start_ts'], utc=True, errors='coerce').dt.tz_convert(None)
        legs_plot['end_ts'] = pd.to_datetime(legs_plot['end_ts'], utc=True, errors='coerce').dt.tz_convert(None)
        legs_plot = legs_plot.dropna(subset=['start_ts', 'end_ts']).sort_values('start_ts')
        for row in legs_plot.itertuples(index=False):
            color = '#2ca02c' if getattr(row, 'leg_direction', '') == 'uptrend' else '#d62728'
            fig.add_trace(
                go.Scatter(
                    x=[row.start_ts, row.end_ts],
                    y=[row.start_price, row.end_price],
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    name='Leg',
                    customdata=[[row.leg_id, row.leg_pattern, row.leg_return_pct]] * 2,
                    hovertemplate=(
                        'Leg %{customdata[0]} (%{customdata[1]})<br>'
                        'Start: %{x|%Y-%m-%d %H:%M:%S}<br>'
                        'Price: %{y:.4f}<br>'
                        'Return: %{customdata[2]:.2f}%<br>'
                        '<extra></extra>'
                    ),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"{tf_name} Legs",
        xaxis_rangeslider_visible=True,
        hovermode='x unified',
        template='plotly_white',
        yaxis_title='Price',
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pyo.plot(fig, filename=str(out_path), auto_open=False)

# Build qualifying legs

def extract_legs(pivots: pd.DataFrame) -> pd.DataFrame:
    records: List[dict] = []
    meta_cols = ['leg_return_pct', 'leg_bars', 'leg_duration_sec']
    for idx in range(len(pivots) - 1):
        start = pivots.iloc[idx]
        end = pivots.iloc[idx + 1]
        start_struct = str(start['structure']).upper()
        end_struct = str(end['structure']).upper()
        start_type = str(start['pivot_type']).lower()
        end_type = str(end['pivot_type']).lower()

        pattern = None
        direction = None
        if start_type == 'low' and end_type == 'high' and start_struct == 'HL' and end_struct == 'HH':
            pattern = 'HL->HH'
            direction = 'uptrend'
        elif start_type == 'high' and end_type == 'low' and start_struct == 'LH' and end_struct == 'LL':
            pattern = 'LH->LL'
            direction = 'downtrend'
        if pattern is None:
            continue

        record = {
            'leg_id': idx,
            'start_ts': start['pivot_time'],
            'end_ts': end['pivot_time'],
            'start_price': start['pivot_price'],
            'end_price': end['pivot_price'],
            'start_structure': start_struct,
            'end_structure': end_struct,
            'start_type': start_type,
            'end_type': end_type,
            'leg_pattern': pattern,
            'leg_direction': direction,
        }
        for col in meta_cols:
            record[col] = end.get(col)
        records.append(record)
    return pd.DataFrame(records)


# Main driver

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    swings_dir = script_dir.parent / 'swings'
    legs_dir = script_dir
    legs_dir.mkdir(parents=True, exist_ok=True)

    ohlcv_map = {
        'minute': project_root / 'data' / '1minute.csv',
        '5minute': project_root / 'data' / '5minute.csv',
        'hour': project_root / 'data' / 'hour.csv',
    }

    for tf, filename in TIMEFRAMES.items():
        swings_path = swings_dir / filename
        if not swings_path.exists():
            print(f'[{tf}] skipping, missing {swings_path}')
            continue
        pivots = load_swings(swings_path)
        legs = extract_legs(pivots)
        out_csv = legs_dir / f'legs_{tf}.csv'
        legs.to_csv(out_csv, index=False)

        ohlcv_path = ohlcv_map.get(tf)
        if ohlcv_path and ohlcv_path.exists():
            ohlcv = load_ohlcv_csv(ohlcv_path)
            plot_path = legs_dir / f'legs_{tf}.html'
            plot_legs_chart(tf, ohlcv, legs, plot_path)
            print(f'[{tf}] {len(legs)} legs -> {out_csv} (plot {plot_path})')
        else:
            print(f'[{tf}] {len(legs)} legs -> {out_csv}')


if __name__ == '__main__':
    main()
