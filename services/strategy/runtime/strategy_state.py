"""State containers for streaming LVN strategy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Dict, List, Optional
from collections import deque
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing helper only
    try:
        from ..indicators.swings import SwingPivot  # type: ignore
    except ImportError:  # legacy module optional
        SwingPivot = object  # type: ignore
    from ..indicators.generate_swing_legs_from_parquet import Leg
else:  # pragma: no cover - runtime fallback types
    from typing import Any

    SwingPivot = Any  # type: ignore
    Leg = Any  # type: ignore


@dataclass
class Candle:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    indicators: Dict[str, float] = field(default_factory=dict)


@dataclass
class EMAState:
    period: int
    value: Optional[float] = None

    @property
    def alpha(self) -> float:
        return 2.0 / (self.period + 1)

    def update(self, price: float) -> float:
        if self.value is None:
            self.value = price
        else:
            self.value = price * self.alpha + self.value * (1 - self.alpha)
        return self.value


@dataclass
class StrategyState:
    symbol: str
    equity: float
    ema: Dict[int, EMAState] = field(default_factory=dict)
    candles: Deque[Candle] = field(default_factory=deque)
    pivots: Deque["SwingPivot"] = field(default_factory=deque)
    legs: Deque["Leg"] = field(default_factory=deque)
    pending_lvns: List[dict] = field(default_factory=list)

    def ensure_ema(self, period: int) -> EMAState:
        if period not in self.ema:
            self.ema[period] = EMAState(period=period)
        return self.ema[period]

    def update_equity(self, new_equity: float) -> None:
        self.equity = new_equity

    def push_candle(self, candle: Candle) -> None:
        self.candles.append(candle)


__all__ = ["Candle", "EMAState", "StrategyState"]
