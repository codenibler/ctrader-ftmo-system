#!/usr/bin/env python3
"""
Reformat a 1‑minute OHLCV Parquet with separate Date and Time columns.

- Merges the first two columns (Date, Time) into a single 'timestamp' column.
- Renames the next four numeric columns to: open, high, low, close
- Renames the final numeric column to: volume
- Keeps row order and writes a clean Parquet by default (can also write CSV).

Usage:
  python reformat_cl_parquet.py INPUT.parquet [-o OUTPUT.parquet] [--csv OUTPUT.csv] [--tz TZ]

Notes:
- Expects the first two columns to be Date and Time (strings) like "08/13/2024" and "00:00".
- Attempts robust datetime parsing; you can force a timezone with --tz (e.g., 'UTC', 'America/New_York').
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


if __package__ is None or __package__ == "":
    CURRENT_DIR = Path(__file__).resolve().parent
    STRATEGY_ROOT = CURRENT_DIR.parent
    if str(STRATEGY_ROOT) not in sys.path:
        sys.path.insert(0, str(STRATEGY_ROOT))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="Input Parquet file")
    p.add_argument("-o", "--output", type=Path, help="Output Parquet file (default: <input>_reformatted.parquet)")
    p.add_argument("--csv", type=Path, help="Optional: also write a CSV to this path")
    p.add_argument("--tz", type=str, default=None, help="Optional timezone to localize/convert timestamps (e.g., 'UTC')")
    return p.parse_args()

def reformat(df: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    cols = list(df.columns)

    if len(cols) < 6:
        raise ValueError(f"Expected at least 6 columns (Date, Time, O, H, L, C, V). Found: {len(cols)} -> {cols}")

    date_col, time_col = cols[0], cols[1]

    # Build timestamp
    # Try a strict common format first; if it fails, fall back to generic parsing
    dt_str = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()

    # Try common MM/DD/YYYY HH:MM first for speed; fallback to pandas flexible parser
    try:
        ts = pd.to_datetime(dt_str, format="%m/%d/%Y %H:%M", errors="raise")
    except Exception:
        # Fallback: flexible parser; if dayfirst matters, pandas will infer in most cases
        ts = pd.to_datetime(dt_str, errors="raise")

    # Timezone handling
    if tz:
        # If naive, localize; if aware, convert
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(tz)
        else:
            ts = ts.dt.tz_convert(tz)

    # Assign 'timestamp' then drop original date/time columns
    df = df.copy()
    df.insert(0, "timestamp", ts)
    df = df.drop(columns=[date_col, time_col])

    # Now rename remaining columns
    remaining = list(df.columns[1:])  # after 'timestamp'
    if len(remaining) < 5:
        raise ValueError(f"Expected at least 5 data columns after timestamp (open, high, low, close, volume). Found: {remaining}")

    rename_map = {}
    # Map in order: open, high, low, close, volume
    target = ["open", "high", "low", "close", "volume"]
    for i, tgt in enumerate(target):
        rename_map[remaining[i]] = tgt

    df = df.rename(columns=rename_map)

    # Keep only the first 6 columns if extras exist beyond volume
    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    extra = [c for c in df.columns if c not in keep]
    if extra:
        # Keep extras but place them after the standard columns
        ordered = keep + extra
        df = df[ordered]
    else:
        df = df[keep]

    return df

def main():
    args = parse_args()
    if not args.output:
        args.output = args.input.with_name(args.input.stem + "_reformatted.parquet")

    df = pd.read_parquet(args.input)
    out = reformat(df, tz=args.tz)

    out.to_parquet(args.output, index=False)

    if args.csv:
        out.to_csv(args.csv, index=False)

    print("Wrote:", args.output)
    if args.csv:
        print("Also wrote CSV:", args.csv)

if __name__ == "__main__":
    main()
