from __future__ import annotations

import pytest

try:
    import pandas as pd
except ImportError:  # pragma: no cover - environment without pandas
    pd = None  # type: ignore[assignment]
    pytest.skip("pandas not available", allow_module_level=True)

from services.strategy.diagnostics.backtest_from_structures import run_backtest


def make_prices() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-10-01 09:59:00+00:00",
            "2024-10-01 10:00:00+00:00",
            "2024-10-01 10:01:00+00:00",
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 100.0, 99.0],
            "high": [100.0, 100.0, 99.0],
            "low": [100.0, 99.0, 98.5],
            "close": [100.0, 99.5, 98.5],
        }
    )


def make_swings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pivot_time": pd.to_datetime(
                [
                    "2024-10-01 10:02:00+00:00",
                ],
                utc=True,
            ),
            "structure": ["HH"],
            "pivot_price": [98.0],
        }
    )


def make_legs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "leg_id": [1],
            "leg_direction": ["uptrend"],
            "start_price": [99.0],
        }
    )


def make_lvns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "leg_id": [1],
            "leg_direction": ["uptrend"],
            "lvn_price": [100.0],
            "start_ts": pd.to_datetime(["2024-10-01 09:50:00+00:00"], utc=True),
            "end_ts": pd.to_datetime(["2024-10-01 09:59:00+00:00"], utc=True),
            "lvn_rank": [1],
        }
    )


def test_unfavorable_target_is_ignored_in_backtest() -> None:
    prices = make_prices()
    swings = make_swings()
    legs = make_legs()
    lvns = make_lvns()

    trades, _, debug_rows = run_backtest(
        prices,
        swings,
        legs,
        lvns,
        spread=0.0,
        starting_equity=10_000.0,
    )

    assert len(trades) == 1
    trade = trades[0]

    # With the new guard the stop should fire and we should not record a target loss.
    assert trade.outcome == "stop"
    assert trade.exit_price == 99.0
    assert trade.pnl < 0
    assert trade.size == pytest.approx(1.0)
    assert trade.pnl == pytest.approx(-100.0)
    assert not debug_rows
