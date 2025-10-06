"""Processing pipeline wiring candles to indicators and order logic."""
from __future__ import annotations

import uuid
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from ..indicators import (
    Leg,
    LowVolumeNode,
    compute_lvns_for_legs,
    confirming_zigzag,
    current_trend,
    determine_pct,
    extract_legs,
    label_structure,
    legs_to_frame,
    update_emas,
)
from ..risk import contract_value_from_price
from .orders import PendingOrder
from .strategy_state import Candle, StrategyState

DEFAULT_TIMEFRAME = "minute"
DEFAULT_SPREAD = 0.02


class StrategyPipeline:
    """Stream candles, maintain geometry, and generate trade intents."""

    def __init__(self, state: StrategyState, *, spread: float = DEFAULT_SPREAD) -> None:
        self.state = state
        self.spread = spread
        self._known_pivots: Set[pd.Timestamp] = set()
        self._known_leg_keys: Set[Tuple[pd.Timestamp, pd.Timestamp]] = set()
        self._known_lvn_keys: Set[Tuple[pd.Timestamp, pd.Timestamp, float]] = set()
        self._leg_index: Dict[Tuple[pd.Timestamp, pd.Timestamp], Leg] = {}

    async def handle_candle(self, candle: Candle) -> None:
        self.state.push_candle(candle)
        update_emas(self.state, candle)
        self.state.current_trend = current_trend(self.state)

        self._check_pending_orders_for_fill(candle)
        self._check_stop_hits(candle)

        candles_df = self._candles_dataframe()
        if candles_df is None:
            return

        pivots_df = self._recompute_pivots(candles_df)
        if pivots_df is None or pivots_df.empty:
            return

        legs = self._recompute_legs(pivots_df)
        if legs:
            self._process_new_legs(legs, candles_df)

    # ------------------------------------------------------------------
    # Candle helpers
    def _candles_dataframe(self) -> Optional[pd.DataFrame]:
        if not self.state.candles:
            return None
        records = [
            {
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in self.state.candles
        ]
        df = pd.DataFrame.from_records(records)
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df.set_index("timestamp")
        if len(df) < 5:
            return None
        return df

    # ------------------------------------------------------------------
    # Swing processing
    def _recompute_pivots(self, candles_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        pct_hint = determine_pct(DEFAULT_TIMEFRAME, candles_df, base_pct=0.12)
        pivots = confirming_zigzag(candles_df, pct=pct_hint)
        pivots = label_structure(pivots)
        if pivots.empty:
            return None
        for row in pivots.itertuples(index=False):
            pivot_time = pd.Timestamp(row.pivot_time)
            if pivot_time not in self._known_pivots:
                self._known_pivots.add(pivot_time)
                self.state.pivots.append(row._asdict())
                self._handle_new_pivot(row)
        return pivots

    def _recompute_legs(self, pivots_df: pd.DataFrame) -> List[Leg]:
        legs = extract_legs(pivots_df)
        new_legs: List[Leg] = []
        for leg in legs:
            key = (pd.Timestamp(leg.start_ts), pd.Timestamp(leg.end_ts))
            if key in self._known_leg_keys:
                continue
            self._known_leg_keys.add(key)
            self._leg_index[key] = leg
            self.state.legs.append(leg)
            new_legs.append(leg)
        return new_legs

    def _process_new_legs(self, legs: Iterable[Leg], candles_df: pd.DataFrame) -> None:
        candles_for_profiles = candles_df.reset_index().rename(columns={"index": "timestamp"})
        for leg in legs:
            trend = self.state.current_trend
            if trend not in ("uptrend", "downtrend"):
                continue
            if leg.leg_direction != trend:
                continue
            leg_frame = legs_to_frame([leg])
            nodes = compute_lvns_for_legs(
                DEFAULT_TIMEFRAME,
                candles_for_profiles,
                leg_frame,
            )
            for node in nodes:
                key = (
                    pd.Timestamp(node.start_ts),
                    pd.Timestamp(node.end_ts),
                    float(node.lvn_price),
                )
                if key in self._known_lvn_keys:
                    continue
                self._known_lvn_keys.add(key)
                self.state.pending_lvns.append(node)
                self._create_order(node)

    # ------------------------------------------------------------------
    # Order lifecycle
    def _create_order(self, node: LowVolumeNode) -> None:
        leg_key = (pd.Timestamp(node.start_ts), pd.Timestamp(node.end_ts))
        leg = self._leg_index.get(leg_key)
        if leg is None:
            return
        direction = "uptrend" if node.leg_direction == "uptrend" else "downtrend"
        spread = self.spread
        if direction == "uptrend":
            entry_price = float(node.lvn_price) - spread
            stop_price = float(leg.start_price)
        else:
            entry_price = float(node.lvn_price) + spread
            stop_price = float(leg.start_price)
        if entry_price <= 0:
            return
        contract_value = contract_value_from_price(entry_price)
        order = PendingOrder(
            order_id=uuid.uuid4().hex,
            leg_key=leg_key,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            lvn_price=float(node.lvn_price),
            lvn_rank=int(node.lvn_rank),
            spread=spread,
            contract_value=contract_value,
            created_at=pd.Timestamp(node.start_ts),
        )
        self.state.orders[order.order_id] = order

    def _check_pending_orders_for_fill(self, candle: Candle) -> None:
        for order in list(self.state.orders.values()):
            if order.status != "pending":
                continue
            if order.is_long and candle.low <= order.entry_price:
                order.status = "filled"
                order.filled_at = candle.timestamp
            elif order.is_short and candle.high >= order.entry_price:
                order.status = "filled"
                order.filled_at = candle.timestamp

    def _check_stop_hits(self, candle: Candle) -> None:
        for order in list(self.state.orders.values()):
            if order.status != "filled":
                continue
            if order.is_long and candle.low <= order.stop_price:
                self._close_order(order, price=order.stop_price, time=candle.timestamp, outcome="stop")
            elif order.is_short and candle.high >= order.stop_price:
                self._close_order(order, price=order.stop_price, time=candle.timestamp, outcome="stop")

    def _handle_new_pivot(self, pivot_row) -> None:
        structure = getattr(pivot_row, "structure", "").upper()
        pivot_price = float(getattr(pivot_row, "pivot_price", 0.0))
        pivot_time = pd.Timestamp(getattr(pivot_row, "pivot_time"))
        if structure == "HH":
            for order in list(self.state.orders.values()):
                if order.status == "filled" and order.is_long:
                    self._close_order(order, price=pivot_price, time=pivot_time, outcome="target")
        elif structure == "LL":
            for order in list(self.state.orders.values()):
                if order.status == "filled" and order.is_short:
                    self._close_order(order, price=pivot_price, time=pivot_time, outcome="target")

    def _close_order(self, order: PendingOrder, *, price: float, time: pd.Timestamp, outcome: str) -> None:
        direction_mult = 1.0 if order.is_long else -1.0
        if order.entry_price > 0:
            price_change = (price - order.entry_price) / order.entry_price
        else:
            price_change = 0.0
        pnl = price_change * direction_mult * order.contract_value
        order.status = "closed"
        order.exit_time = time
        order.exit_price = price
        order.outcome = outcome  # type: ignore[assignment]
        order.pnl = pnl
        new_equity = self.state.equity + pnl
        order.equity_after = new_equity
        self.state.update_equity(new_equity)
        self.state.completed_orders.append(order)
        del self.state.orders[order.order_id]

    # ------------------------------------------------------------------
    # Diagnostics
    def get_completed_trades(self) -> List[dict]:
        results = []
        for order in self.state.completed_orders:
            results.append(
                {
                    "order_id": order.order_id,
                    "direction": order.direction,
                    "entry_price": order.entry_price,
                    "entry_time": order.filled_at,
                    "exit_price": order.exit_price,
                    "exit_time": order.exit_time,
                    "outcome": order.outcome,
                    "contract_value": order.contract_value,
                    "pnl": order.pnl,
                    "equity_after": order.equity_after,
                    "leg_start": order.leg_key[0],
                    "leg_end": order.leg_key[1],
                }
            )
        return results


__all__ = ["StrategyPipeline"]
