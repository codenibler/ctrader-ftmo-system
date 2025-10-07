"""Simple position sizing utilities."""
from __future__ import annotations

from typing import Optional

DEFAULT_RISK_PCT: float = 0.00025  # % of equity risked per trade. 
DEFAULT_CONTRACT_UNITS: float = 100.0  # CL micro contract is 100 barrels


def contract_value_from_price(
    price: float,
    *,
    contract_units: float = DEFAULT_CONTRACT_UNITS,
) -> float:
    """Return the notional dollar value for one contract at ``price``.

    Args:
        price: quoted CL price (dollars per barrel).
        contract_units: barrels represented by one futures contract.
    """

    if price <= 0:
        raise ValueError("price must be positive")
    if contract_units <= 0:
        raise ValueError("contract_units must be positive")
    return price * contract_units


def compute_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    *,
    risk_pct: float = DEFAULT_RISK_PCT,
    contract_value: Optional[float] = None,
    contract_units: float = DEFAULT_CONTRACT_UNITS,
    min_size: float = 0.0,
) -> float:
    """Return contracts to trade based on equity, risk%, and stop distance.

    Args:
        equity: current account equity in dollars.
        entry_price: planned entry price.
        stop_price: protective stop price.
        risk_pct: fraction of equity to risk per trade (default 1%).
        contract_value: optional override for the notional per contract. When
            omitted we derive it using ``entry_price`` and ``contract_units``.
        contract_units: barrels per contract (default 100 for CL micro).
        min_size: optional floor applied to the resulting size.
    """

    if contract_units <= 0:
        raise ValueError("contract_units must be positive")

    if contract_value is None:
        if entry_price <= 0:
            return 0.0
        contract_value = contract_value_from_price(entry_price, contract_units=contract_units)
    if contract_value <= 0:
        raise ValueError("contract_value must be positive")
    if entry_price <= 0:
        return 0.0

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0 or equity <= 0 or risk_pct <= 0:
        return 0.0

    risk_amount = equity * risk_pct
    per_contract_risk = (stop_distance / entry_price) * contract_value
    if per_contract_risk <= 0:
        return 0.0
    size = risk_amount / per_contract_risk
    return max(min_size, size)
