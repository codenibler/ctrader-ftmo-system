"""LVN extraction on HL->HH and LH->LL swing legs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import offline as pyo

from findings.legs.extract_legs import extract_legs as build_legs, load_swings
from findings.lvns.volume_profile import infer_bin_size, load_ohlcv_csv, profile_between

# Tuning notes:
# - Increase EDGE_BIN_RATIO to suppress edge LVNs.
# - Relax PROMINENCE_RATIO or MIN_PCT_TOTAL_VOL for more LVNs.
# - Lower VALUE_AREA_TARGET to focus on most traded prices.
# - Provide fixed bin sizes per instrument/timeframe for cleaner bins.

EDGE_BIN_RATIO = 0.15  # fraction of bins removed from both price edges (filters start/end nodes)
PROMINENCE_RATIO = 0.50  # max LVN volume as share of nearest HVN volume (depth requirement)
MIN_PCT_TOTAL_VOL = 0.005  # minimum LVN smoothed volume as share of total leg volume
VALUE_AREA_TARGET = 0.70  # target fraction of smoothed volume in value area around POC
MAX_LVNS_PER_LEG = 3  # cap on LVNs per leg (sorted by depth)
SMOOTH_WINDOW = 3  # rolling window for smoothing volume profile
MIN_LEG_BARS = 10  # default minimum number of bars per leg to consider

EDGE_BIN_RATIO_BY_TF = {'hour': EDGE_BIN_RATIO, '5minute': EDGE_BIN_RATIO, 'minute': EDGE_BIN_RATIO}
PROMINENCE_RATIO_BY_TF = {'hour': PROMINENCE_RATIO, '5minute': PROMINENCE_RATIO, 'minute': PROMINENCE_RATIO}
MIN_PCT_TOTAL_VOL_BY_TF = {'hour': MIN_PCT_TOTAL_VOL, '5minute': MIN_PCT_TOTAL_VOL, 'minute': MIN_PCT_TOTAL_VOL}
VALUE_AREA_TARGET_BY_TF = {'hour': VALUE_AREA_TARGET, '5minute': VALUE_AREA_TARGET, 'minute': VALUE_AREA_TARGET}
MAX_LVNS_PER_LEG_BY_TF = {'hour': MAX_LVNS_PER_LEG, '5minute': MAX_LVNS_PER_LEG, 'minute': MAX_LVNS_PER_LEG}
MIN_LEG_BARS_BY_TF = {'hour': 5, '5minute': 10, 'minute': 10}

# --- LVN detection helpers ---

def _local_minima(values):
    mins = []
    for i in range(1, len(values) - 1):
        left, mid, right = values[i - 1], values[i], values[i + 1]
        if mid <= left and mid <= right and (mid < left or mid < right):
            mins.append(i)
    return mins

def _local_maxima(values):
    maxima = []
    for i in range(1, len(values) - 1):
        left, mid, right = values[i - 1], values[i], values[i + 1]
        if mid >= left and mid >= right and (mid > left or mid > right):
            maxima.append(i)
    if maxima == [] and len(values) > 0:
        maxima.append(int(values.argmax()))
    return maxima


def _value_area_bounds(volumes, target):
    total = float(volumes.sum())
    n = len(volumes)
    if total <= 0 or n == 0 or target is None:
        return 0, max(0, n - 1)
    poc_idx = int(volumes.argmax())
    left = right = poc_idx
    covered = volumes[poc_idx]
    while covered / total < target and (left > 0 or right < n - 1):
        left_candidate = (volumes[left - 1], left - 1) if left > 0 else (-1, None)
        right_candidate = (volumes[right + 1], right + 1) if right < n - 1 else (-1, None)
        candidates = [c for c in (left_candidate, right_candidate) if c[1] is not None]
        if not candidates:
            break
        value, idx = max(candidates, key=lambda item: (item[0], -item[1]))
        covered += max(value, 0.0)
        if idx < left:
            left = idx
        if idx > right:
            right = idx
    return left, right




def find_lvns(
    profile: pd.DataFrame,
    *,
    edge_ratio: float = EDGE_BIN_RATIO,
    prominence_ratio: float = PROMINENCE_RATIO,
    min_pct_total: float = MIN_PCT_TOTAL_VOL,
    value_area_target: float = VALUE_AREA_TARGET,
    max_lvns: int = MAX_LVNS_PER_LEG,
    smooth_window: int = SMOOTH_WINDOW,
) -> List[Dict[str, float]]:
    """Return significant LVNs that pass prominence/value-area filters."""
    if profile.empty:
        return []

    p = profile.sort_values('price').reset_index(drop=True).copy()
    if len(p) < 3:
        return []

    if smooth_window > 1:
        p['v_smooth'] = p['volume'].rolling(smooth_window, center=True, min_periods=1).mean()
    else:
        p['v_smooth'] = p['volume'].astype(float)

    v_smooth = p['v_smooth'].to_numpy(dtype=float)
    volumes = p['volume'].to_numpy(dtype=float)
    prices = p['price'].to_numpy(dtype=float)

    total_volume = float(volumes.sum())
    total_smooth = float(v_smooth.sum())
    if total_volume <= 0 or total_smooth <= 0:
        return []

    n_bins = len(p)
    max_volume = float(v_smooth.max()) or 1.0
    poc_idx = int(v_smooth.argmax())
    if value_area_target is None:
        value_area_left, value_area_right = 0, n_bins - 1
    else:
        value_area_left, value_area_right = _value_area_bounds(v_smooth, value_area_target)
    poc_price = float(prices[poc_idx])

    minima_idxs = _local_minima(v_smooth)
    maxima_idxs = _local_maxima(v_smooth)
    if poc_idx not in maxima_idxs:
        maxima_idxs.append(poc_idx)
    if not minima_idxs:
        return []

    edge_bins = max(1, int(np.floor(n_bins * edge_ratio)))
    upper_edge = n_bins - edge_bins - 1

    candidates: List[Dict[str, float]] = []
    for idx in minima_idxs:
        if idx < edge_bins or idx > upper_edge:
            continue

        lvn_volume_smooth = float(v_smooth[idx])
        lvn_volume_raw = float(volumes[idx])
        if lvn_volume_smooth < min_pct_total * total_smooth:
            continue

        nearest_hvn_idx = min(maxima_idxs, key=lambda m: (abs(m - idx), m))
        hvn_volume = float(v_smooth[nearest_hvn_idx]) or 1.0
        if hvn_volume <= 0 or lvn_volume_smooth > prominence_ratio * hvn_volume:
            continue

        within_value_area = True
        if value_area_target is not None:
            within_value_area = value_area_left <= idx <= value_area_right
        if not within_value_area:
            continue

        pct_of_max = float(lvn_volume_smooth / max_volume) if max_volume else 0.0
        depth_pct = float(1.0 - pct_of_max)
        pct_of_total = float(lvn_volume_smooth / total_smooth) if total_smooth else 0.0
        distance_to_poc = float(abs(prices[idx] - poc_price))

        candidates.append(
            {
                'price': float(prices[idx]),
                'volume': lvn_volume_smooth,
                'raw_volume': lvn_volume_raw,
                'pct_of_max': pct_of_max,
                'depth_pct': depth_pct,
                'pct_of_total': pct_of_total,
                'within_value_area': bool(within_value_area),
                'distance_to_poc': distance_to_poc,
                'poc_price': poc_price,
                'value_area_low': float(prices[value_area_left]),
                'value_area_high': float(prices[value_area_right]),
            }
        )

    candidates.sort(key=lambda item: (-item['depth_pct'], -item['distance_to_poc']))
    return candidates[:max_lvns]

# --- Leg helpers ---

LEG_COLUMNS = [
    'leg_id',
    'start_ts',
    'end_ts',
    'start_price',
    'end_price',
    'leg_pattern',
    'leg_direction',
    'leg_return_pct',
    'leg_bars',
    'leg_duration_sec',
]


def load_legs_csv(path: Path) -> pd.DataFrame:
    """Load pre-computed legs CSV and ensure timestamps are parsed."""
    df = pd.read_csv(path)
    missing = [col for col in ['start_ts', 'end_ts', 'leg_pattern'] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(missing)}")
    df['start_ts'] = pd.to_datetime(df['start_ts'], utc=True, errors='coerce')
    df['end_ts'] = pd.to_datetime(df['end_ts'], utc=True, errors='coerce')
    df = df.dropna(subset=['start_ts', 'end_ts']).reset_index(drop=True)
    return df


# --- Core processing ---



def scan_timeframe(
    tf_name: str,
    ohlcv: pd.DataFrame,
    legs: pd.DataFrame,
    bin_size: float | None = None,
) -> pd.DataFrame:
    """Build profiles per qualified leg, detect LVNs, attach leg metadata."""
    columns = [
        'timeframe',
        'leg_id',
        'start_ts',
        'end_ts',
        'bin_size',
        'lvn_price',
        'lvn_volume',
        'lvn_raw_volume',
        'pct_of_max',
        'depth_pct',
        'pct_of_total',
        'within_value_area',
        'distance_to_poc',
        'poc_price',
        'value_area_low',
        'value_area_high',
        'lvn_rank',
        'leg_pattern',
        'leg_direction',
        'leg_return_pct',
        'leg_bars',
        'leg_duration_sec',
    ]
    if legs.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, object]] = []
    min_required = MIN_LEG_BARS_BY_TF.get(tf_name, MIN_LEG_BARS)
    edge_ratio = EDGE_BIN_RATIO_BY_TF.get(tf_name, EDGE_BIN_RATIO)
    prominence_ratio = PROMINENCE_RATIO_BY_TF.get(tf_name, PROMINENCE_RATIO)
    min_pct_total = MIN_PCT_TOTAL_VOL_BY_TF.get(tf_name, MIN_PCT_TOTAL_VOL)
    value_area_target = VALUE_AREA_TARGET_BY_TF.get(tf_name, VALUE_AREA_TARGET)
    max_lvns_per_leg = MAX_LVNS_PER_LEG_BY_TF.get(tf_name, MAX_LVNS_PER_LEG)

    for row in legs.itertuples(index=False):
        ts0 = row.start_ts
        ts1 = row.end_ts
        if pd.isna(ts0) or pd.isna(ts1):
            continue

        window = ohlcv[(ohlcv['timestamp'] >= ts0) & (ohlcv['timestamp'] <= ts1)]
        if window.shape[0] < min_required:
            continue

        leg_bars = getattr(row, 'leg_bars', None)
        if pd.notna(leg_bars) and leg_bars < min_required:
            continue

        leg_bin = float(bin_size) if bin_size else infer_bin_size(window)
        profile = profile_between(window, ts0, ts1, bin_size=leg_bin)
        lvns = find_lvns(
            profile,
            edge_ratio=edge_ratio,
            prominence_ratio=prominence_ratio,
            min_pct_total=min_pct_total,
            value_area_target=value_area_target,
            max_lvns=max_lvns_per_leg,
            smooth_window=SMOOTH_WINDOW,
        )
        if not lvns:
            continue

        meta_cols = {
            'leg_return_pct': getattr(row, 'leg_return_pct', None),
            'leg_bars': getattr(row, 'leg_bars', None),
            'leg_duration_sec': getattr(row, 'leg_duration_sec', None),
        }

        for rank, lvn in enumerate(lvns, start=1):
            record: Dict[str, object] = {
                'timeframe': tf_name,
                'leg_id': int(getattr(row, 'leg_id', len(rows))),
                'start_ts': ts0,
                'end_ts': ts1,
                'bin_size': leg_bin,
                'lvn_price': lvn['price'],
                'lvn_volume': lvn['volume'],
                'lvn_raw_volume': lvn.get('raw_volume'),
                'pct_of_max': lvn['pct_of_max'],
                'depth_pct': lvn['depth_pct'],
                'pct_of_total': lvn['pct_of_total'],
                'within_value_area': bool(lvn['within_value_area']),
                'distance_to_poc': lvn['distance_to_poc'],
                'poc_price': lvn['poc_price'],
                'value_area_low': lvn['value_area_low'],
                'value_area_high': lvn['value_area_high'],
                'lvn_rank': rank,
                'leg_pattern': getattr(row, 'leg_pattern', ''),
                'leg_direction': getattr(row, 'leg_direction', ''),
            }
            record.update(meta_cols)
            rows.append(record)

    return pd.DataFrame(rows, columns=columns)

# --- Plotting ---

def plot_lvns(df: pd.DataFrame, lvns: pd.DataFrame, title: str, out_path: Path) -> None:
    """Render candlestick chart with horizontal LVN segments."""
    fig = go.Figure()

    if not df.empty:
        plot_df = df.copy()
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

    if not lvns.empty:
        lvns_plot = lvns.copy()
        lvns_plot['start_ts'] = pd.to_datetime(lvns_plot['start_ts'], utc=True, errors='coerce').dt.tz_convert(None)
        lvns_plot['end_ts'] = pd.to_datetime(lvns_plot['end_ts'], utc=True, errors='coerce').dt.tz_convert(None)
        lvns_plot = lvns_plot.dropna(subset=['start_ts', 'end_ts']).sort_values('start_ts')
        for row in lvns_plot.itertuples(index=False):
            color = '#2ca02c' if getattr(row, 'leg_direction', '') == 'uptrend' else '#d62728'
            within_label = 'Yes' if getattr(row, 'within_value_area', False) else 'No'
            custom = [[
                row.leg_id,
                row.lvn_rank,
                row.leg_pattern,
                row.depth_pct,
                row.pct_of_total,
                row.distance_to_poc,
                within_label,
            ]] * 2
            fig.add_trace(
                go.Scatter(
                    x=[row.start_ts, row.end_ts],
                    y=[row.lvn_price, row.lvn_price],
                    mode='lines',
                    line=dict(color=color, width=2, dash='dot'),
                    name='LVN',
                    customdata=custom,
                    hovertemplate=(
                        'Leg %{customdata[0]} - LVN %{customdata[1]} (%{customdata[2]})<br>'
                        'Price: %{y:.4f}<br>'
                        'Depth: %{customdata[3]:.2%}<br>'
                        'Pct of Total: %{customdata[4]:.2%}<br>'
                        'Dist to POC: %{customdata[5]:.4f}<br>'
                        'In Value Area: %{customdata[6]}<br>'
                        'Time: %{x|%Y-%m-%d %H:%M:%S}'
                    ),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=True,
        hovermode='x unified',
        template='plotly_white',
        yaxis_title='Price',
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pyo.plot(fig, filename=str(out_path), auto_open=False)


# --- Entrypoint ---

def main() -> None:
    """Scan configured timeframes and persist LVN CSV/HTML reports."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir = project_root / 'data'

    swings_dir = script_dir.parent / 'swings'
    legs_dir = script_dir.parent / 'legs'
    lvns_dir = script_dir
    for subdir in (swings_dir, legs_dir, lvns_dir):
        subdir.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Dict[str, Path]] = {
        'minute': {
            'ohlcv': data_dir / '1minute.csv',
            'swings': swings_dir / 'swings_minute.csv',
            'legs': legs_dir / 'legs_minute.csv',
            'lvn_csv': lvns_dir / 'lvns_minute.csv',
            'lvn_html': lvns_dir / 'lvns_minute.html',
        },
        '5minute': {
            'ohlcv': data_dir / '5minute.csv',
            'swings': swings_dir / 'swings_5minute.csv',
            'legs': legs_dir / 'legs_5minute.csv',
            'lvn_csv': lvns_dir / 'lvns_5minute.csv',
            'lvn_html': lvns_dir / 'lvns_5minute.html',
        },
        'hour': {
            'ohlcv': data_dir / 'hour.csv',
            'swings': swings_dir / 'swings_hour.csv',
            'legs': legs_dir / 'legs_hour.csv',
            'lvn_csv': lvns_dir / 'lvns_hour.csv',
            'lvn_html': lvns_dir / 'lvns_hour.html',
        },
    }

    outputs: List[pd.DataFrame] = []

    for tf, paths in cfg.items():
        if not paths['ohlcv'].exists() or not paths['swings'].exists():
            print(f"[{tf}] skipping, missing OHLCV or swings data")
            continue

        if not paths['legs'].exists():
            swings = load_swings(paths['swings'])
            legs = build_legs(swings)
            legs.to_csv(paths['legs'], index=False)

        legs = load_legs_csv(paths['legs'])
        if legs.empty:
            print(f"[{tf}] no qualifying legs found")
            continue

        ohlcv = load_ohlcv_csv(paths['ohlcv'])
        result = scan_timeframe(tf, ohlcv, legs)
        if result.empty:
            print(f"[{tf}] no LVNs detected")
            continue

        csv_path = paths['lvn_csv']
        html_path = paths['lvn_html']
        result.to_csv(csv_path, index=False)
        plot_lvns(ohlcv, result, title=f"{tf} LVNs", out_path=html_path)
        print(f"[{tf}] {len(result)} LVNs -> {csv_path}")
        outputs.append(result)

    if outputs:
        all_out = pd.concat(outputs, ignore_index=True)
        all_out.to_csv(lvns_dir / 'lvns_all.csv', index=False)


if __name__ == '__main__':
    main()
