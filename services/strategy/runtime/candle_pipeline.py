"""Processing pipeline for streaming 1m candles."""
from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from indicators.ema import current_trend, update_emas
from indicators.legs import StreamingLegBuilder, StreamingSwingDetector
from indicators.lvns import compute_lvns_for_legs
from risk import AdvancedSLIntakeBuffer, SwingNavigator
from spread_calculator import adjust_entry_price
from .strategy_state import Candle, StrategyState

log = logging.getLogger(__name__)

POSITION_RISK_PCT = 0.01


class StrategyPipeline:
    """Processes candles, manages trades, and records results."""

    def __init__(self, state: StrategyState) -> None:
        self.state = state
        self.swing_detector = StreamingSwingDetector()
        self.leg_builder = StreamingLegBuilder()
        self._pending_entries: List[dict] = []
        self._open_trades: List[dict] = []
        self.completed_trades: List[dict] = []
        self.all_lvns: List[dict] = []

    async def handle_candle(self, candle: Candle) -> None:
        self.state.push_candle(candle)
        update_emas(self.state, candle)

        pivots = self.swing_detector.update(candle)
        for pivot in pivots:
            self.state.pivots.append(pivot)
            log.debug(
                "Swing pivot detected at %s type=%s structure=%s price=%.4f idx=%d",
                pivot.pivot_time,
                pivot.pivot_type,
                pivot.structure,
                pivot.pivot_price,
                pivot.pivot_idx,
            )

        legs = self.leg_builder.update(pivots)
        for leg in legs:
            self.state.legs.append(leg)
            log.debug(
                "Leg %d %s pattern=%s start=%s %.4f end=%s %.4f return=%.2f%% bars=%d dur=%.0fs",
                leg.leg_id,
                leg.leg_direction,
                leg.leg_pattern,
                leg.start_ts,
                leg.start_price,
                leg.end_ts,
                leg.end_price,
                leg.leg_return_pct,
                leg.leg_bars,
                leg.leg_duration_sec,
            )

        lvns = compute_lvns_for_legs(legs, self.state.candles, timeframe="minute")
        enriched_lvns: List[dict] = []
        for lvn in lvns:
            self.all_lvns.append(lvn.copy())
            enriched = {**lvn, "status": "pending", "earliest_entry_ts": lvn["end_ts"]}
            self.state.pending_lvns.append(enriched)
            enriched_lvns.append(enriched)
            log.debug(
                "LVN leg=%d dir=%s price=%.4f raw_vol=%.2f pct_max=%.3f pct_total=%.3f VA[%0.4f,%0.4f] POC=%.4f",
                enriched["leg_id"],
                enriched["leg_direction"],
                enriched["lvn_price"],
                enriched["lvn_raw_volume"],
                enriched["pct_of_max"],
                enriched["pct_of_total"],
                enriched["value_area_low"],
                enriched["value_area_high"],
                enriched["poc_price"],
            )

        self._evaluate_trade_opportunities(candle, enriched_lvns)
        self._update_open_trades(candle)

    def _get_swing_navigator(self) -> Optional[SwingNavigator]:
        if len(self.state.pivots) < 2:
            return None
        data = [
            {
                "pivot_time": pivot.pivot_time,
                "pivot_price": pivot.pivot_price,
                "pivot_type": pivot.pivot_type,
                "structure": pivot.structure,
            }
            for pivot in self.state.pivots
        ]
        df = pd.DataFrame(data)
        if df.empty:
            return None
        return SwingNavigator(df)

    def _evaluate_trade_opportunities(self, candle: Candle, new_lvns: List[dict]) -> None:
        if new_lvns:
            self._pending_entries.extend(new_lvns)

        if not self._pending_entries:
            return

        trend = current_trend(self.state)
        if trend is None:
            return

        navigator = self._get_swing_navigator()
        remaining: List[dict] = []
        for lvn in self._pending_entries:
            if candle.timestamp <= lvn["earliest_entry_ts"]:
                remaining.append(lvn)
                continue

            direction = lvn.get("leg_direction")
            if direction != trend:
                remaining.append(lvn)
                continue

            price = float(lvn["lvn_price"])
            if not (candle.low <= price <= candle.high):
                remaining.append(lvn)
                continue

            targets = AdvancedSLIntakeBuffer(lvn, navigator)
            if targets is None:
                remaining.append(lvn)
                continue

            adjusted_price, spread = adjust_entry_price(direction, price, candle.bid, candle.ask)
            if adjusted_price <= 0:
                remaining.append(lvn)
                continue

            size = (self.state.equity * POSITION_RISK_PCT) / adjusted_price
            if size <= 0:
                remaining.append(lvn)
                continue

            trade = {
                "leg_id": lvn.get("leg_id"),
                "lvn": lvn,
                "direction": direction,
                "entry_time": candle.timestamp,
                "entry_price": adjusted_price,
                "raw_entry_price": price,
                "stop_price": float(targets.stop_price),
                "target_price": float(targets.target_price),
                "size": size,
                "spread": spread,
                "slippage": 0.0,
                "equity_before": self.state.equity,
            }
            self._open_trades.append(trade)
            lvn["status"] = "open"
            log.info(
                "Opened %s trade at %.4f (stop %.4f, target %.4f, spread %.4f, slip %.4f) leg %s",
                direction,
                adjusted_price,
                trade["stop_price"],
                trade["target_price"],
                spread,
                0.0,
                lvn.get("leg_id"),
            )

        self._pending_entries = [lvn for lvn in remaining if lvn.get("status") not in {"open", "filled"}]
        self.state.pending_lvns = [lvn for lvn in self.state.pending_lvns if lvn.get("status") not in {"open", "filled"}]

    def _update_open_trades(self, candle: Candle) -> None:
        if not self._open_trades:
            return

        still_open: List[dict] = []
        for trade in self._open_trades:
            exit_info = self._check_exit(trade, candle)
            if exit_info is None:
                still_open.append(trade)
                continue

            exit_time, exit_price, outcome = exit_info
            direction = trade["direction"]
            size = trade["size"]
            entry_price = trade["entry_price"]
            if direction == "uptrend":
                pnl = (exit_price - entry_price) * size
            else:
                pnl = (entry_price - exit_price) * size

            new_equity = self.state.equity + pnl
            self.state.update_equity(new_equity)

            result = {
                **trade,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "outcome": outcome,
                "pnl": pnl,
                "equity_after": new_equity,
            }
            self.completed_trades.append(result)
            log.info(
                "Closed %s trade leg %s at %.4f outcome=%s pnl=%.2f equity=%.2f",
                direction,
                trade.get("leg_id"),
                exit_price,
                outcome,
                pnl,
                new_equity,
            )

        self._open_trades = still_open

    @staticmethod
    def _check_exit(trade: dict, candle: Candle) -> Optional[tuple[pd.Timestamp, float, str]]:
        high = float(candle.high)
        low = float(candle.low)
        direction = trade["direction"]
        stop = trade["stop_price"]
        target = trade["target_price"]
        ts = candle.timestamp

        if direction == "uptrend":
            if low <= stop:
                return ts, float(stop), "stop"
            if high >= target:
                return ts, float(target), "target"
        else:
            if high >= stop:
                return ts, float(stop), "stop"
            if low <= target:
                return ts, float(target), "target"
        return None

    def get_completed_trades(self) -> List[dict]:
        return list(self.completed_trades)


__all__ = ["StrategyPipeline", "POSITION_RISK_PCT"]
