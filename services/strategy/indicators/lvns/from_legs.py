"""Generate Low Volume Nodes (LVNs) from parquet-backed swing legs."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    from ..legs import build_legs_from_parquet
    from ..swings import parse_month
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    STRATEGY_ROOT = CURRENT_DIR.parent
    if str(STRATEGY_ROOT) not in sys.path:
        sys.path.insert(0, str(STRATEGY_ROOT))
    from legs import build_legs_from_parquet  # type: ignore
    from swings import parse_month  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing helper only
    from ...runtime.strategy_state import StrategyState

# --- LVN tuning defaults (mirrors legacy lvn_scan.py) ---

EDGE_BIN_RATIO = 0.15
PROMINENCE_RATIO = 0.60
MIN_PCT_TOTAL_VOL = 0.005
VALUE_AREA_TARGET = 0.90
MAX_LVNS_PER_LEG = 5
SMOOTH_WINDOW = 1
MIN_LEG_BARS = 5


@dataclass
class LowVolumeNode:
    timeframe: str
    leg_id: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    bin_size: float
    lvn_price: float
    lvn_volume: float
    lvn_raw_volume: float | None
    pct_of_max: float
    depth_pct: float
    pct_of_total: float
    within_value_area: bool
    distance_to_poc: float
    poc_price: float
    value_area_low: float
    value_area_high: float
    lvn_rank: int
    leg_pattern: str
    leg_direction: str
    leg_return_pct: float | None
    leg_bars: int | None
    leg_duration_sec: float | None

    def as_record(self) -> dict[str, object]:
        return {
            "timeframe": self.timeframe,
            "leg_id": self.leg_id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "bin_size": self.bin_size,
            "lvn_price": self.lvn_price,
            "lvn_volume": self.lvn_volume,
            "lvn_raw_volume": self.lvn_raw_volume,
            "pct_of_max": self.pct_of_max,
            "depth_pct": self.depth_pct,
            "pct_of_total": self.pct_of_total,
            "within_value_area": self.within_value_area,
            "distance_to_poc": self.distance_to_poc,
            "poc_price": self.poc_price,
            "value_area_low": self.value_area_low,
            "value_area_high": self.value_area_high,
            "lvn_rank": self.lvn_rank,
            "leg_pattern": self.leg_pattern,
            "leg_direction": self.leg_direction,
            "leg_return_pct": self.leg_return_pct,
            "leg_bars": self.leg_bars,
            "leg_duration_sec": self.leg_duration_sec,
        }


# --- Data loading helpers ---


def load_parquet_candles(parquet_path: Path, month: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "timestamp" not in df.columns:
        raise ValueError("Parquet file must contain a 'timestamp' column")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing required columns: {', '.join(missing)}")

    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required_cols)

    if month:
        start, end = parse_month(month)
        df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# --- Volume profile utilities (mirrors legacy volume_profile.py) ---


def infer_bin_size(df: pd.DataFrame) -> float:
    hl = (df["high"] - df["low"]).replace([np.inf, -np.inf], np.nan).dropna()
    if hl.empty:
        return 1.0
    return max(float(hl.tail(200).median()) / 20.0, 1e-6)


def profile_between(
    df: pd.DataFrame,
    ts0: pd.Timestamp,
    ts1: pd.Timestamp,
    bin_size: float,
    *,
    price_col: str = "close",
) -> pd.DataFrame:
    segment = df[(df["timestamp"] >= ts0) & (df["timestamp"] <= ts1)]
    if segment.empty:
        return pd.DataFrame(columns=["price", "volume"])
    try:
        from marketprofile import MarketProfile  # type: ignore

        mp = MarketProfile(
            segment,
            price=price_col,
            high="high",
            low="low",
            volume="volume",
            tick_size=bin_size,
        )
        prof = mp.profile.rename(columns={"Price": "price", "Volume": "volume"})
        return prof[["price", "volume"]].sort_values("price").reset_index(drop=True)
    except Exception:
        low_min = segment["low"].min()
        high_max = segment["high"].max()
        if not np.isfinite(low_min) or not np.isfinite(high_max) or high_max <= low_min:
            return pd.DataFrame(columns=["price", "volume"])
        edges = np.arange(low_min, high_max + bin_size, bin_size)
        if len(edges) < 2:
            return pd.DataFrame(columns=["price", "volume"])
        volumes = np.zeros(len(edges) - 1, dtype=float)
        lows = segment["low"].to_numpy(dtype=float)
        highs = segment["high"].to_numpy(dtype=float)
        vols = segment["volume"].to_numpy(dtype=float)
        for low, high, vol in zip(lows, highs, vols):
            if not np.isfinite(low) or not np.isfinite(high) or high <= low or vol <= 0:
                continue
            start_idx = max(0, np.searchsorted(edges, low, side="right") - 1)
            end_idx = min(len(volumes) - 1, np.searchsorted(edges, high, side="right") - 1)
            span = max(high - low, 1e-12)
            for idx in range(start_idx, end_idx + 1):
                bin_low, bin_high = edges[idx], edges[idx + 1]
                overlap = max(0.0, min(high, bin_high) - max(low, bin_low))
                if overlap > 0:
                    volumes[idx] += vol * (overlap / span)
        centers = (edges[:-1] + edges[1:]) / 2.0
        prof = pd.DataFrame({"price": centers, "volume": volumes})
        prof = prof[prof["volume"] > 0]
        return prof.sort_values("price").reset_index(drop=True)


# --- LVN detection helpers ---


def _local_minima(values: np.ndarray) -> List[int]:
    mins: List[int] = []
    for i in range(1, len(values) - 1):
        left, mid, right = values[i - 1], values[i], values[i + 1]
        if mid <= left and mid <= right and (mid < left or mid < right):
            mins.append(i)
    return mins


def _local_maxima(values: np.ndarray) -> List[int]:
    maxima: List[int] = []
    for i in range(1, len(values) - 1):
        left, mid, right = values[i - 1], values[i], values[i + 1]
        if mid >= left and mid >= right and (mid > left or mid > right):
            maxima.append(i)
    if not maxima and len(values) > 0:
        maxima.append(int(np.argmax(values)))
    return maxima


def _value_area_bounds(volumes: np.ndarray, target: float | None) -> tuple[int, int]:
    total = float(volumes.sum())
    n = len(volumes)
    if total <= 0 or n == 0 or target is None:
        return 0, max(0, n - 1)
    poc_idx = int(np.argmax(volumes))
    left = right = poc_idx
    covered = volumes[poc_idx]
    while covered / total < target and (left > 0 or right < n - 1):
        left_candidate = (volumes[left - 1], left - 1) if left > 0 else (-1.0, None)
        right_candidate = (volumes[right + 1], right + 1) if right < n - 1 else (-1.0, None)
        candidates = [c for c in (left_candidate, right_candidate) if c[1] is not None]
        if not candidates:
            break
        value, idx = max(candidates, key=lambda item: (item[0], -item[1]))
        covered += max(value, 0.0)
        if idx < left:
            left = idx
        if idx > right:
            right = idx
    return left, right


def find_low_volume_nodes(
    profile: pd.DataFrame,
    *,
    edge_ratio: float,
    prominence_ratio: float,
    min_pct_total: float,
    value_area_target: float | None,
    max_lvns: int,
    smooth_window: int,
) -> List[dict[str, float]]:
    if profile.empty:
        return []

    ordered = profile.sort_values("price").reset_index(drop=True).copy()
    if len(ordered) < 3:
        return []

    if smooth_window > 1:
        ordered["v_smooth"] = ordered["volume"].rolling(smooth_window, center=True, min_periods=1).mean()
    else:
        ordered["v_smooth"] = ordered["volume"].astype(float)

    v_smooth = ordered["v_smooth"].to_numpy(dtype=float)
    raw_volumes = ordered["volume"].to_numpy(dtype=float)
    prices = ordered["price"].to_numpy(dtype=float)

    total_volume = float(raw_volumes.sum())
    total_smooth = float(v_smooth.sum())
    if total_volume <= 0 or total_smooth <= 0:
        return []

    n_bins = len(ordered)
    max_volume = float(np.max(v_smooth)) or 1.0
    poc_idx = int(np.argmax(v_smooth))
    value_area_left, value_area_right = _value_area_bounds(v_smooth, value_area_target)
    poc_price = float(prices[poc_idx])

    minima = _local_minima(v_smooth)
    maxima = _local_maxima(v_smooth)
    if poc_idx not in maxima:
        maxima.append(poc_idx)
    if not minima:
        return []

    edge_bins = max(1, int(np.floor(n_bins * edge_ratio)))
    upper_edge = n_bins - edge_bins - 1

    candidates: List[dict[str, float]] = []
    for idx in minima:
        if idx < edge_bins or idx > upper_edge:
            continue

        lvn_volume_smooth = float(v_smooth[idx])
        lvn_volume_raw = float(raw_volumes[idx])
        if lvn_volume_smooth < min_pct_total * total_smooth:
            continue

        nearest_hvn_idx = min(maxima, key=lambda m: (abs(m - idx), m))
        hvn_volume = float(v_smooth[nearest_hvn_idx]) or 1.0
        if hvn_volume <= 0 or lvn_volume_smooth > prominence_ratio * hvn_volume:
            continue

        within_value_area = True
        if value_area_target is not None:
            within_value_area = value_area_left <= idx <= value_area_right
        if not within_value_area:
            continue

        pct_of_max = float(lvn_volume_smooth / max_volume) if max_volume else 0.0
        depth_pct = float(1.0 - pct_of_max)
        pct_of_total = float(lvn_volume_smooth / total_smooth) if total_smooth else 0.0
        distance_to_poc = float(abs(prices[idx] - poc_price))

        candidates.append(
            {
                "price": float(prices[idx]),
                "volume": lvn_volume_smooth,
                "raw_volume": lvn_volume_raw,
                "pct_of_max": pct_of_max,
                "depth_pct": depth_pct,
                "pct_of_total": pct_of_total,
                "within_value_area": bool(within_value_area),
                "distance_to_poc": distance_to_poc,
                "poc_price": poc_price,
                "value_area_low": float(prices[value_area_left]),
                "value_area_high": float(prices[value_area_right]),
            }
        )

    candidates.sort(key=lambda item: (-item["depth_pct"], -item["distance_to_poc"]))
    return candidates[:max_lvns]


# --- Core LVN scanning ---


def compute_lvns_for_legs(
    timeframe: str,
    candles: pd.DataFrame,
    legs: pd.DataFrame,
    *,
    bin_size: float | None = None,
) -> List[LowVolumeNode]:
    if legs.empty:
        return []

    min_required = MIN_LEG_BARS
    edge_ratio = EDGE_BIN_RATIO
    prominence_ratio = PROMINENCE_RATIO
    min_pct_total = MIN_PCT_TOTAL_VOL
    value_area_target = VALUE_AREA_TARGET
    max_lvns_per_leg = MAX_LVNS_PER_LEG

    lvns: List[LowVolumeNode] = []
    for leg in legs.itertuples(index=False):
        start_ts = pd.Timestamp(getattr(leg, "start_ts"))
        end_ts = pd.Timestamp(getattr(leg, "end_ts"))
        if pd.isna(start_ts) or pd.isna(end_ts):
            continue

        window = candles[(candles["timestamp"] >= start_ts) & (candles["timestamp"] <= end_ts)]
        if window.shape[0] < min_required:
            continue

        leg_bars = getattr(leg, "leg_bars", None)
        if pd.notna(leg_bars) and leg_bars < min_required:
            continue

        leg_bin = float(bin_size) if bin_size else infer_bin_size(window)
        profile = profile_between(window, start_ts, end_ts, bin_size=leg_bin)
        nodes = find_low_volume_nodes(
            profile,
            edge_ratio=edge_ratio,
            prominence_ratio=prominence_ratio,
            min_pct_total=min_pct_total,
            value_area_target=value_area_target,
            max_lvns=max_lvns_per_leg,
            smooth_window=SMOOTH_WINDOW,
        )
        if not nodes:
            continue

        for rank, node in enumerate(nodes, start=1):
            lvns.append(
                LowVolumeNode(
                    timeframe=timeframe,
                    leg_id=int(getattr(leg, "leg_id", len(lvns))),
                    start_ts=start_ts,
                    end_ts=end_ts,
                    bin_size=leg_bin,
                    lvn_price=node["price"],
                    lvn_volume=node["volume"],
                    lvn_raw_volume=node.get("raw_volume"),
                    pct_of_max=node["pct_of_max"],
                    depth_pct=node["depth_pct"],
                    pct_of_total=node["pct_of_total"],
                    within_value_area=bool(node["within_value_area"]),
                    distance_to_poc=node["distance_to_poc"],
                    poc_price=node["poc_price"],
                    value_area_low=node["value_area_low"],
                    value_area_high=node["value_area_high"],
                    lvn_rank=rank,
                    leg_pattern=str(getattr(leg, "leg_pattern", "")),
                    leg_direction=str(getattr(leg, "leg_direction", "")),
                    leg_return_pct=float(getattr(leg, "leg_return_pct", 0.0) or 0.0)
                    if getattr(leg, "leg_return_pct", None) is not None
                    else None,
                    leg_bars=int(getattr(leg, "leg_bars", 0) or 0)
                    if getattr(leg, "leg_bars", None) is not None
                    else None,
                    leg_duration_sec=float(getattr(leg, "leg_duration_sec", 0.0) or 0.0)
                    if getattr(leg, "leg_duration_sec", None) is not None
                    else None,
                )
            )
    return lvns


def lvns_to_frame(nodes: Sequence[LowVolumeNode]) -> pd.DataFrame:
    records = [node.as_record() for node in nodes]
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values(["start_ts", "lvn_rank"]).reset_index(drop=True)
    return df


def build_lvns_from_parquet(
    parquet_path: Path,
    month: str | None = None,
    pct: float | None = None,
    timeframe: str = "minute",
    *,
    bin_size: float | None = None,
) -> pd.DataFrame:
    candles = load_parquet_candles(parquet_path, month=month)
    legs = build_legs_from_parquet(parquet_path, month=month, pct=pct)
    nodes = compute_lvns_for_legs(timeframe, candles, legs, bin_size=bin_size)
    return lvns_to_frame(nodes)


def update_state_with_lvns(state: "StrategyState", nodes: Iterable[LowVolumeNode]) -> None:
    for node in nodes:
        state.pending_lvns.append(node)


# --- CLI ---


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LVNs from a parquet file.")
    parser.add_argument("--parquet", type=Path, required=True, help="Path to 1m OHLCV parquet")
    parser.add_argument("--month", type=str, help="Optional month filter (YYYY_MM or YYYY-MM)")
    parser.add_argument("--pct", type=float, help="Override zigzag reversal threshold percent")
    parser.add_argument("--bin-size", type=float, dest="bin_size", help="Optional fixed volume-profile bin size")
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    lvns_df = build_lvns_from_parquet(
        args.parquet,
        month=args.month,
        pct=args.pct,
        bin_size=args.bin_size,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lvns_df.to_csv(args.output, index=False)
        print(f"Saved {len(lvns_df)} LVNs -> {args.output}")
    else:
        print(lvns_df.head())
        print(f"Generated {len(lvns_df)} LVNs (no output path supplied)")


if __name__ == "__main__":
    main()
