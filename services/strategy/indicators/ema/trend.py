"""Streaming EMA helpers for the LVN strategy."""
from __future__ import annotations

from typing import Iterable

# Import from sibling module using project-relative path for script execution support.
from runtime.strategy_state import Candle, StrategyState


DEFAULT_EMA_PERIODS: tuple[int, ...] = (9, 20)


def update_emas(state: StrategyState, candle: Candle, periods: Iterable[int] | None = None) -> None:
    """Update tracked EMA periods using the candle close price."""

    for period in periods or DEFAULT_EMA_PERIODS:
        ema_state = state.ensure_ema(period)
        value = ema_state.update(candle.close)
        candle.indicators[f"ema_{period}"] = value


def current_trend(state: StrategyState, fast: int = 9, slow: int = 20) -> str | None:
    """Return trend direction based on fast/slow EMA relationship."""

    fast_state = state.ema.get(fast)
    slow_state = state.ema.get(slow)
    if not fast_state or not slow_state:
        return None

    fast_value = fast_state.value
    slow_value = slow_state.value
    if fast_value is None or slow_value is None:
        return None

    if fast_value > slow_value:
        return "uptrend"
    if fast_value < slow_value:
        return "downtrend"
    return None


__all__ = ["DEFAULT_EMA_PERIODS", "current_trend", "update_emas"]
