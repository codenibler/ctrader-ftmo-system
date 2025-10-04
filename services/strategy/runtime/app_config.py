"""
Configuration helpers for strategy runtime. Contains: 
    - Parquet data path
    - Max candles in cache memory
    - Symbol being traded
    - Replay speed. 
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "CL_1m.parquet"


@dataclass
class AppConfig:
    symbol: str
    parquet_path: Path
    replay_speed: float
    queue_maxsize: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        symbol = os.getenv("STRATEGY_SYMBOL", "CL")
        data_path = Path(os.getenv("STRATEGY_PARQUET", str(DEFAULT_DATA_PATH))).expanduser()
        replay_speed = float(os.getenv("STRATEGY_REPLAY_SPEED", "0.0"))
        queue_maxsize = int(os.getenv("STRATEGY_QUEUE_MAXSIZE", "1000"))
        return cls(
            symbol=symbol,
            parquet_path=data_path,
            replay_speed=replay_speed,
            queue_maxsize=queue_maxsize,
        )


__all__ = ["AppConfig"]
