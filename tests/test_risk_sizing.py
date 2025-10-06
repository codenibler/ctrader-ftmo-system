from __future__ import annotations

import unittest

from services.strategy.risk.sizing import (
    DEFAULT_CONTRACT_VALUE,
    DEFAULT_RISK_PCT,
    compute_position_size,
)


class RiskSizingTests(unittest.TestCase):
    def test_basic_position_size(self) -> None:
        size = compute_position_size(
            equity=10_000.0,
            entry_price=70.0,
            stop_price=69.5,
            risk_pct=DEFAULT_RISK_PCT,
            contract_value=DEFAULT_CONTRACT_VALUE,
        )
        self.assertGreater(size, 0.0)

    def test_zero_stop_distance_returns_zero(self) -> None:
        self.assertEqual(
            compute_position_size(10_000.0, 70.0, 70.0),
            0.0,
        )

    def test_negative_or_zero_equity_returns_zero(self) -> None:
        self.assertEqual(
            compute_position_size(0.0, 70.0, 69.0),
            0.0,
        )

    def test_contract_value_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            compute_position_size(10_000.0, 70.0, 69.0, contract_value=0.0)


if __name__ == "__main__":
    unittest.main()
