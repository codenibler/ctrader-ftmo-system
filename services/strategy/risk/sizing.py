"""Simple position sizing utilities."""
from __future__ import annotations

from typing import Optional

DEFAULT_RISK_PCT: float = 0.01  # risk 1% of equity by default
DEFAULT_CONTRACT_VALUE: float = 1000.0  # dollars per 1.0 price move (CL futures)


def compute_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    *,
    risk_pct: float = DEFAULT_RISK_PCT,
    contract_value: float = DEFAULT_CONTRACT_VALUE,
    min_size: float = 0.0,
) -> float:
    """Return contracts to trade based on equity, risk%, and stop distance.

    Args:
        equity: current account equity in dollars.
        entry_price: planned entry price.
        stop_price: protective stop price.
        risk_pct: fraction of equity to risk per trade (default 1%).
        contract_value: dollar value of a full 1.0 price move for one contract.
        min_size: optional floor applied to the resulting size.
    """

    if contract_value <= 0:
        raise ValueError("contract_value must be positive")

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0 or equity <= 0 or risk_pct <= 0:
        return 0.0

    risk_amount = equity * risk_pct
    size = risk_amount / (stop_distance * contract_value)
    return max(min_size, size)
