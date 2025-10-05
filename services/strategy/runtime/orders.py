"""Order data structures for the LVN strategy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import pandas as pd

Direction = Literal["uptrend", "downtrend"]
OrderStatus = Literal["pending", "filled", "closed"]
OrderOutcome = Literal["target", "stop", "cancelled"]


@dataclass
class PendingOrder:
    order_id: str
    leg_key: Tuple[pd.Timestamp, pd.Timestamp]
    direction: Direction
    entry_price: float
    stop_price: float
    lvn_price: float
    lvn_rank: int
    spread: float
    created_at: pd.Timestamp
    status: OrderStatus = "pending"
    filled_at: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    outcome: Optional[OrderOutcome] = None
    pnl: Optional[float] = None
    equity_after: Optional[float] = None

    @property
    def is_long(self) -> bool:
        return self.direction == "uptrend"

    @property
    def is_short(self) -> bool:
        return self.direction == "downtrend"


__all__ = ["PendingOrder", "Direction", "OrderOutcome", "OrderStatus"]
