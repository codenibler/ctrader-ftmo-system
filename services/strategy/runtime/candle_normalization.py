"""Data normalization utilities for incoming candles."""
from __future__ import annotations

from typing import Iterable
import math
import pandas as pd


EXPECTED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized dataframe with UTC timestamps and sorted rows."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required candle columns: {', '.join(missing)}")
    clean = df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    clean = clean.dropna(subset=["timestamp"]).sort_values("timestamp")
    clean = clean.drop_duplicates(subset=["timestamp"], keep="last")
    return clean.reset_index(drop=True)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rows_to_candles(rows: Iterable[dict]) -> Iterable[dict]:
    for row in rows:
        bid = _safe_float(row.get("bid")) if "bid" in row else float("nan")
        ask = _safe_float(row.get("ask")) if "ask" in row else float("nan")
        yield {
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "bid": None if math.isnan(bid) else bid,
            "ask": None if math.isnan(ask) else ask,
        }


__all__ = ["normalize_dataframe", "rows_to_candles"]
