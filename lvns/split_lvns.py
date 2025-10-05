"""Split lvns_all.csv into per-timeframe CSV files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LVNS_DIR = Path(__file__).resolve().parent


def main() -> None:
    master_path = LVNS_DIR / 'lvns_all.csv'
    if not master_path.exists():
        raise SystemExit(f"missing {master_path}")

    df = pd.read_csv(master_path)
    if 'timeframe' not in df.columns:
        raise SystemExit('lvns_all.csv missing timeframe column')

    written = 0
    for tf, group in df.groupby('timeframe', sort=False):
        out_path = LVNS_DIR / f'lvns_{tf}.csv'
        group.to_csv(out_path, index=False)
        print(f"[{tf}] {len(group)} rows -> {out_path}")
        written += len(group)

    print(f"total rows written: {written}")


if __name__ == '__main__':
    main()
