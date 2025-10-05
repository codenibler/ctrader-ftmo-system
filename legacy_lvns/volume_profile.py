"""Volume profile utilities used by LVN scanners."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from findings.swings.swings_extract import prepare_df

# Load OHLCV data and enforce UTC ordering

def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Return OHLCV rows sorted by timestamp using swings_extract.prepare_df."""
    df = prepare_df(path)
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    result = df.reset_index().rename(columns={'index': 'timestamp'})
    result['timestamp'] = pd.to_datetime(result['timestamp'], utc=True, errors='coerce')
    result = result.dropna(subset=['timestamp'])
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    return result[cols]

# Estimate profile bin size from recent volatility

def infer_bin_size(df: pd.DataFrame) -> float:
    """Heuristic bin size from median high-low range (keeps at least 1e-6)."""
    hl = (df['high'] - df['low']).replace([np.inf, -np.inf], np.nan).dropna()
    if hl.empty:
        return 1.0
    return max(float(hl.tail(200).median()) / 20.0, 1e-6)

# Build a fixed-range volume profile between timestamps

def profile_between(
    df: pd.DataFrame,
    ts0: pd.Timestamp,
    ts1: pd.Timestamp,
    bin_size: float,
    *,
    price_col: str = 'close',
) -> pd.DataFrame:
    """Return volume-at-price profile between `ts0` and `ts1`.

    Attempts to use `marketprofile` when available; falls back to a simple
    histogram that spreads each bar's volume across overlapping bins.
    """
    sl = df[(df['timestamp'] >= ts0) & (df['timestamp'] <= ts1)]
    if sl.empty:
        return pd.DataFrame(columns=['price', 'volume'])
    try:
        from marketprofile import MarketProfile  # type: ignore

        mp = MarketProfile(
            sl,
            price=price_col,
            high='high',
            low='low',
            volume='volume',
            tick_size=bin_size,
        )
        prof = mp.profile.rename(columns={'Price': 'price', 'Volume': 'volume'})
        return prof[['price', 'volume']].sort_values('price').reset_index(drop=True)
    except Exception:
        lo, hi = sl['low'].min(), sl['high'].max()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.DataFrame(columns=['price', 'volume'])
        edges = np.arange(lo, hi + bin_size, bin_size)
        if len(edges) < 2:
            return pd.DataFrame(columns=['price', 'volume'])
        vols = np.zeros(len(edges) - 1, dtype=float)
        lows = sl['low'].to_numpy(dtype=float)
        highs = sl['high'].to_numpy(dtype=float)
        volumes = sl['volume'].to_numpy(dtype=float)
        for low, high, vol in zip(lows, highs, volumes):
            if not np.isfinite(low) or not np.isfinite(high) or high <= low or vol <= 0:
                continue
            s = max(0, np.searchsorted(edges, low, side='right') - 1)
            e = min(len(vols) - 1, np.searchsorted(edges, high, side='right') - 1)
            span = high - low
            for idx in range(s, e + 1):
                blo, bhi = edges[idx], edges[idx + 1]
                overlap = max(0.0, min(high, bhi) - max(low, blo))
                if overlap > 0:
                    vols[idx] += vol * (overlap / span)
        centers = (edges[:-1] + edges[1:]) / 2.0
        prof = pd.DataFrame({'price': centers, 'volume': vols})
        prof = prof[prof['volume'] > 0]
        return prof.sort_values('price').reset_index(drop=True)
