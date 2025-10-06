from __future__ import annotations

import unittest

from services.strategy.risk.sizing import (
    DEFAULT_CONTRACT_UNITS,
    DEFAULT_RISK_PCT,
    compute_position_size,
    contract_value_from_price,
)


class RiskSizingTests(unittest.TestCase):
    def test_basic_position_size(self) -> None:
        entry_price = 70.0
        size = compute_position_size(
            equity=10_000.0,
            entry_price=entry_price,
            stop_price=69.5,
            risk_pct=DEFAULT_RISK_PCT,
            contract_units=DEFAULT_CONTRACT_UNITS,
        )
        self.assertAlmostEqual(size, 2.0)

    def test_position_size_with_explicit_contract_value(self) -> None:
        entry_price = 70.0
        size_with_default = compute_position_size(
            equity=10_000.0,
            entry_price=entry_price,
            stop_price=69.5,
        )
        manual_contract_value = contract_value_from_price(entry_price)
        size_with_manual = compute_position_size(
            equity=10_000.0,
            entry_price=entry_price,
            stop_price=69.5,
            contract_value=manual_contract_value,
        )
        self.assertAlmostEqual(size_with_default, size_with_manual)

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

    def test_contract_value_from_price(self) -> None:
        self.assertAlmostEqual(contract_value_from_price(80.0), 8000.0)


if __name__ == "__main__":
    unittest.main()
