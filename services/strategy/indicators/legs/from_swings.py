"""Generate swing legs from parquet-backed zigzag pivots."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, TYPE_CHECKING

import pandas as pd

try:
    from ..swings import build_all_zigzag_levels
except ImportError:
    import sys
    CURRENT_DIR = Path(__file__).resolve().parent
    STRATEGY_ROOT = CURRENT_DIR.parent
    if str(STRATEGY_ROOT) not in sys.path:
        sys.path.insert(0, str(STRATEGY_ROOT))
    from swings import build_all_zigzag_levels  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing helper only
    from ...runtime.strategy_state import StrategyState


@dataclass
class Leg:
    """Directional swing leg derived from consecutive pivots."""

    leg_id: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    start_price: float
    end_price: float
    start_structure: str
    end_structure: str
    start_type: str
    end_type: str
    leg_pattern: str
    leg_direction: str
    leg_return_pct: float
    leg_bars: int
    leg_duration_sec: float

    def as_record(self) -> dict[str, object]:
        return {
            "leg_id": self.leg_id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "start_price": self.start_price,
            "end_price": self.end_price,
            "start_structure": self.start_structure,
            "end_structure": self.end_structure,
            "start_type": self.start_type,
            "end_type": self.end_type,
            "leg_pattern": self.leg_pattern,
            "leg_direction": self.leg_direction,
            "leg_return_pct": self.leg_return_pct,
            "leg_bars": self.leg_bars,
            "leg_duration_sec": self.leg_duration_sec,
        }


PATTERN_UP = ("low", "high", "HL", "HH")
PATTERN_DOWN = ("high", "low", "LH", "LL")


def _normalize_row(row: pd.Series) -> tuple[str, str]:
    pivot_type = str(row["pivot_type"]).lower()
    structure = str(row["structure"]).upper()
    return pivot_type, structure


def extract_legs(pivots: pd.DataFrame) -> List[Leg]:
    """Return HL→HH and LH→LL legs for the provided pivot DataFrame."""

    if pivots.empty:
        return []

    pivots = pivots.sort_values("pivot_time").reset_index(drop=True)

    legs: List[Leg] = []
    for idx in range(len(pivots) - 1):
        start = pivots.iloc[idx]
        end = pivots.iloc[idx + 1]

        start_type, start_struct = _normalize_row(start)
        end_type, end_struct = _normalize_row(end)

        pattern = direction = None
        if (start_type, end_type, start_struct, end_struct) == PATTERN_UP:
            pattern = "HL->HH"
            direction = "uptrend"
        elif (start_type, end_type, start_struct, end_struct) == PATTERN_DOWN:
            pattern = "LH->LL"
            direction = "downtrend"
        if pattern is None or direction is None:
            continue

        leg = Leg(
            leg_id=len(legs),
            start_ts=pd.Timestamp(start["pivot_time"]),
            end_ts=pd.Timestamp(end["pivot_time"]),
            start_price=float(start["pivot_price"]),
            end_price=float(end["pivot_price"]),
            start_structure=start_struct,
            end_structure=end_struct,
            start_type=start_type,
            end_type=end_type,
            leg_pattern=pattern,
            leg_direction=direction,
            leg_return_pct=float(end.get("leg_return_pct", 0.0) or 0.0),
            leg_bars=int(end.get("leg_bars", 0) or 0),
            leg_duration_sec=float(end.get("leg_duration_sec", 0.0) or 0.0),
        )
        legs.append(leg)
    return legs


def legs_to_frame(legs: Sequence[Leg]) -> pd.DataFrame:
    records = [leg.as_record() for leg in legs]
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values("start_ts").reset_index(drop=True)
    return df


def build_legs_from_parquet(
    parquet_path: Path,
    month: str | None = None,
    pct: float | None = None,
) -> pd.DataFrame:
    pivots = build_all_zigzag_levels(parquet_path, month=month, pct=pct)
    legs = extract_legs(pivots)
    return legs_to_frame(legs)


def update_state_with_legs(state: "StrategyState", legs: Iterable[Leg]) -> None:
    """Append legs to the strategy state's rolling collection."""

    for leg in legs:
        state.legs.append(leg)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate swing legs from a parquet file.")
    parser.add_argument("--parquet", type=Path, required=True, help="Path to 1m OHLCV parquet")
    parser.add_argument("--month", type=str, help="Optional month filter (YYYY_MM or YYYY-MM)")
    parser.add_argument("--pct", type=float, help="Override zigzag reversal threshold percent")
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    legs_df = build_legs_from_parquet(args.parquet, month=args.month, pct=args.pct)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        legs_df.to_csv(args.output, index=False)
        print(f"Saved {len(legs_df)} legs -> {args.output}")
    else:
        print(legs_df.head())
        print(f"Generated {len(legs_df)} legs (no output path supplied)")


if __name__ == "__main__":
    main()
