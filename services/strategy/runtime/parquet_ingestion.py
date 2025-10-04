"""Candle ingestion utilities."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator

import pandas as pd

from .candle_normalization import normalize_dataframe, rows_to_candles
from .strategy_state import Candle

log = logging.getLogger(__name__)


class ParquetCandleSource:
    """Stream candles from a parquet file sequentially."""

    def __init__(self, path: Path, *, replay_speed: float = 0.0) -> None:
        self.path = path
        self.replay_speed = replay_speed

    def load(self) -> list[Candle]:
        if not self.path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.path}")
        log.info("Loading candles from %s", self.path)
        df = pd.read_parquet(self.path)
        df = normalize_dataframe(df)
        candles = [Candle(**payload) for payload in rows_to_candles(df.to_dict("records"))]
        log.info("Loaded %d candles", len(candles))
        return candles

    async def stream(self) -> AsyncIterator[Candle]:
        delay = max(0.0, self.replay_speed)
        for candle in self.load():
            yield candle
            if delay:
                await asyncio.sleep(delay)


__all__ = ["ParquetCandleSource"]
