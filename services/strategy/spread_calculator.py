"""Spread and slippage adjustment helpers for order simulation."""
from __future__ import annotations

from typing import Optional, Tuple

LONG_BASELINE_SPREAD = 0.02


def compute_spread(bid: Optional[float], ask: Optional[float]) -> float:
    if bid is None or ask is None:
        return 0.0
    spread = ask - bid
    return spread if spread > 0 else 0.0


def adjust_entry_price(
    direction: str,
    desired_price: float,
    bid: Optional[float],
    ask: Optional[float],
) -> Tuple[float, float]:
    """Return adjusted order price and total spread applied."""
    spread = compute_spread(bid, ask)
    if direction == "uptrend":
        spread += LONG_BASELINE_SPREAD
        adjusted = desired_price - spread
    else:
        adjusted = desired_price
    return adjusted, spread


__all__ = ["compute_spread", "adjust_entry_price", "LONG_BASELINE_SPREAD"]
