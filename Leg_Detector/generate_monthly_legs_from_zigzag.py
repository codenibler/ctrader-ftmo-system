from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_swings_module() -> None:
    target = 'findings.swings.swings_extract'
    try:
        import importlib
        spec = importlib.util.find_spec(target)
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        return

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
        raise ModuleNotFoundError(f'Unable to locate swings_extract.py in {fallback_paths}')

    spec = importlib.util.spec_from_file_location(target, swings_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to create spec for {swings_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Ensure package namespace objects exist
    pkg_name = 'findings'
    if pkg_name not in sys.modules:
        pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader(pkg_name, loader=None))
        pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg
    subpkg_name = 'findings.swings'
    if subpkg_name not in sys.modules:
        swings_pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader(subpkg_name, loader=None))
        swings_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules[subpkg_name] = swings_pkg

    sys.modules[target] = module


ensure_swings_module()

from findings.legs.extract_legs import extract_legs, plot_legs_chart
from findings.lvns.volume_profile import load_ohlcv_csv


def load_zigzag_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=['pivot_time', 'pivot_price', 'pivot_type', 'structure'])
    df['pivot_time'] = pd.to_datetime(df['pivot_time'], utc=True, errors='coerce')
    df = df.dropna(subset=['pivot_time']).sort_values('pivot_time').reset_index(drop=True)
    return df


def build_legs_from_zigzag(month: str = '2024_10') -> None:
    try:
        year_str, month_str = month.split('_')
        year = int(year_str)
        month_num = int(month_str)
    except ValueError as exc:
        raise ValueError(f'invalid month format: {month!r}, expected YYYY_MM') from exc

    findings_dir = SCRIPT_DIR.parent
    start = pd.Timestamp(year=year, month=month_num, day=1, tz='UTC')
    end = start + pd.offsets.MonthBegin(1)

    zigzag_suffix = 'october' if month == '2024_10' else month
    zigzag_path = findings_dir / 'swings' / f'zigzag_levels_minute_{zigzag_suffix}.csv'
    if not zigzag_path.exists():
        raise FileNotFoundError(f'missing zigzag pivots file: {zigzag_path}')

    pivots = load_zigzag_csv(zigzag_path)
    pivots = pivots[(pivots['pivot_time'] >= start) & (pivots['pivot_time'] < end)].reset_index(drop=True)

    legs = extract_legs(pivots)

    out_suffix = 'october' if month == '2024_10' else month
    out_csv = SCRIPT_DIR / f'legs_minute_{out_suffix}.csv'
    out_html = SCRIPT_DIR / f'legs_minute_{out_suffix}.html'

    legs.to_csv(out_csv, index=False)

    ohlcv_path = PROJECT_ROOT / 'data' / 'minute_months' / month / 'CL_1m.csv'
    if ohlcv_path.exists():
        ohlcv = load_ohlcv_csv(ohlcv_path)
        ohlcv = ohlcv[(ohlcv['timestamp'] >= start) & (ohlcv['timestamp'] < end)].reset_index(drop=True)
    else:
        ohlcv = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    plot_legs_chart('minute', ohlcv, legs, out_html)

    print(f'Saved {len(legs)} legs to {out_csv}')
    print(f'Saved chart to {out_html}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract legs from ZigZag pivots for a month (YYYY_MM).')
    parser.add_argument('month', nargs='?', default='2024_10', help='Month in YYYY_MM format (default: 2024_10).')
    args = parser.parse_args()
    build_legs_from_zigzag(args.month)


if __name__ == '__main__':
    main()
