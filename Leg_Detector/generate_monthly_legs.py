from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from findings.legs.extract_legs import extract_legs, load_swings, plot_legs_chart
from findings.lvns.volume_profile import load_ohlcv_csv


def build_minute_legs_for_month(month: str) -> None:
    findings_dir = SCRIPT_DIR.parent

    swings_path = findings_dir / 'swings' / 'swings_minute.csv'
    if not swings_path.exists():
        raise FileNotFoundError(f'missing swings file: {swings_path}')

    pivots = load_swings(swings_path)

    try:
        year_str, month_str = month.split('_')
        year = int(year_str)
        month_num = int(month_str)
    except ValueError as exc:
        raise ValueError(f'invalid month format: {month!r}, expected YYYY_MM') from exc

    start = pd.Timestamp(year=year, month=month_num, day=1, tz='UTC')
    end = start + pd.offsets.MonthBegin(1)

    pivots_month = pivots[(pivots['pivot_time'] >= start) & (pivots['pivot_time'] < end)].reset_index(drop=True)

    legs = extract_legs(pivots_month)

    suffix = 'october' if month == '2024_10' else month
    out_csv = SCRIPT_DIR / f'legs_minute_{suffix}.csv'
    out_html = SCRIPT_DIR / f'legs_minute_{suffix}.html'

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
    parser = argparse.ArgumentParser(description='Build minute legs for a specified month (YYYY_MM).')
    parser.add_argument('month', nargs='?', default='2024_10', help='Month in YYYY_MM format (default: 2024_10).')
    args = parser.parse_args()
    build_minute_legs_for_month(args.month)


if __name__ == '__main__':
    main()
