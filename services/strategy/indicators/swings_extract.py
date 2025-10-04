"""Swing extraction via ZigZag for multiple timeframes."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import offline as pyo

def _progress(label: str, current: int, total: int, interval: int | None = None) -> None:
    if total <= 0:
        return
    if interval is None:
        interval = max(1, total // 20)
    if current == total - 1 or current % interval == 0:
        pct = (current + 1) / total * 100
        print(f'[{label}] {pct:5.1f}% ({current + 1}/{total})')

DEFAULT_PCT = 0.5
PCT_BY_TIMEFRAME: Dict[str, float] = {
    "hour": 0.5,
    "5minute": 0.30,
    "minute": 0.12,
}

# Optional per-timeframe hints to auto-tune pct based on typical intrabar ranges.
AUTO_PCT_SETTINGS: Dict[str, Dict[str, float]] = {
    "minute": {"percentile": 0.5, "multiplier": 1.8, "min_pct": 0.04},
    "5minute": {"percentile": 0.5, "multiplier": 1.7, "min_pct": 0.06},
    "1minute": {"percentile": 0.5, "multiplier": 1.8, "min_pct": 0.04},
}

LABEL_ALIASES: Dict[str, str] = {
    "1minute": "minute",
    "01minute": "minute",
    "1m": "minute",
    "05minute": "5minute",
    "5m": "5minute",
}

ANNOTATION_LIMITS: Dict[str, int] = {
    "minute": 800,
    "5minute": 600,
}


def _normalize_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


def determine_pct(label: str, df: pd.DataFrame, base_pct: float) -> float:
    normalized = _normalize_label(label)
    settings = AUTO_PCT_SETTINGS.get(normalized)
    if not settings or df.empty:
        return base_pct

    percentile = float(settings.get("percentile", 0.5))
    percentile = min(max(percentile, 0.0), 1.0)
    multiplier = float(settings.get("multiplier", 1.0))
    min_pct = float(settings.get("min_pct", 0.0))
    max_pct = float(settings.get("max_pct", base_pct))

    prices = df["close"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    lows = df["low"].astype(float).to_numpy()

    ranges = highs - lows
    mask = np.isfinite(prices) & np.isfinite(ranges) & (prices > 0)
    if not np.any(mask):
        return base_pct

    pct_moves = (ranges[mask] / prices[mask]) * 100.0
    if pct_moves.size == 0:
        return base_pct

    quantile = float(np.nanquantile(pct_moves, percentile))
    if not np.isfinite(quantile):
        return base_pct

    candidate = quantile * multiplier
    candidate = max(min_pct, candidate)
    candidate = min(max_pct, candidate)
    return max(candidate, 1e-6)


# Load OHLCV CSV data and standardize timestamps/numeric fields.
def prepare_df(path: Path) -> pd.DataFrame:
    expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="utf-16")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-16-le")
    except FileNotFoundError:
        print(f"Missing data file: {path}")
        empty = pd.DataFrame(columns=expected_cols)
        empty.index = pd.DatetimeIndex([], name="timestamp")
        return empty

    if df.empty:
        df.columns = expected_cols[: len(df.columns)]
        df.index = pd.DatetimeIndex([], name="timestamp")
        return df

    normalized = [str(col).strip().lower() for col in df.columns]
    if "timestamp" in normalized:
        rename_map = {old: new for old, new in zip(df.columns, normalized)}
        df = df.rename(columns=rename_map)
    elif {"date", "time"}.issubset(set(normalized)):
        rename_map = {old: new for old, new in zip(df.columns, normalized)}
        df = df.rename(columns=rename_map)
    elif len(df.columns) == 7:
        df.columns = ["date", "time", "open", "high", "low", "close", "volume"]
    else:
        df.columns = expected_cols[: len(df.columns)]

    if "timestamp" not in df.columns and {"date", "time"}.issubset(df.columns):
        df["timestamp"] = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        df = df.drop(columns=["date", "time"])

    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} missing required columns: {', '.join(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp")
    df = df.sort_index()

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    return df


# Confirm ZigZag pivots without look-ahead using a percent threshold.
def confirming_zigzag(df: pd.DataFrame, pct: float = DEFAULT_PCT) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["pivot_idx", "pivot_time", "pivot_price", "pivot_type"])

    threshold = pct / 100.0
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df.index.to_numpy(copy=True)

    pivots: List[dict] = []

    last_pivot_idx = 0
    last_pivot_price = float(lows[0])
    last_pivot_type = "low"
    pivots.append(
        {
            "pivot_idx": last_pivot_idx,
            "pivot_time": pd.Timestamp(times[last_pivot_idx]),
            "pivot_price": last_pivot_price,
            "pivot_type": last_pivot_type,
        }
    )

    trend = 1  # 1 -> tracking highs, -1 -> tracking lows
    extreme_idx = 0
    extreme_price = float(highs[0])

    total = len(df)
    for idx in range(1, total):
        high = float(highs[idx])
        low = float(lows[idx])

        if trend == 1:
            if high >= extreme_price or extreme_idx <= last_pivot_idx:
                extreme_price = high
                extreme_idx = idx
            drawdown = (extreme_price - low) / extreme_price if extreme_price else 0.0
            if idx > last_pivot_idx and drawdown >= threshold and extreme_idx > last_pivot_idx:
                pivots.append(
                    {
                        "pivot_idx": extreme_idx,
                        "pivot_time": pd.Timestamp(times[extreme_idx]),
                        "pivot_price": extreme_price,
                        "pivot_type": "high",
                    }
                )
                last_pivot_idx = extreme_idx
                last_pivot_price = extreme_price
                last_pivot_type = "high"

                trend = -1
                extreme_idx = idx
                extreme_price = low
        else:
            if low <= extreme_price or extreme_idx <= last_pivot_idx:
                extreme_price = low
                extreme_idx = idx
            drawup = (high - extreme_price) / extreme_price if extreme_price else 0.0
            if idx > last_pivot_idx and drawup >= threshold and extreme_idx > last_pivot_idx:
                pivots.append(
                    {
                        "pivot_idx": extreme_idx,
                        "pivot_time": pd.Timestamp(times[extreme_idx]),
                        "pivot_price": extreme_price,
                        "pivot_type": "low",
                    }
                )
                last_pivot_idx = extreme_idx
                last_pivot_price = extreme_price
                last_pivot_type = "low"

                trend = 1
                extreme_idx = idx
                extreme_price = high
        _progress("zigzag", idx, total)

    if extreme_idx > pivots[-1]["pivot_idx"]:
        pivots.append(
            {
                "pivot_idx": extreme_idx,
                "pivot_time": pd.Timestamp(times[extreme_idx]),
                "pivot_price": extreme_price,
                "pivot_type": "high" if trend == 1 else "low",
            }
        )

    return pd.DataFrame(pivots)


# Label market structure and leg statistics for pivots.
def label_structure(pivots_df: pd.DataFrame) -> pd.DataFrame:
    if pivots_df.empty:
        columns = [
            "pivot_time",
            "pivot_price",
            "pivot_type",
            "structure",
            "leg_return_pct",
            "leg_bars",
            "leg_duration_sec",
            "pivot_idx",
        ]
        return pd.DataFrame(columns=columns)

    pivots_df = pivots_df.sort_values("pivot_idx").reset_index(drop=True)

    structures: List[str] = []
    leg_returns: List[float] = []
    leg_bars: List[int] = []
    leg_durations: List[float] = []

    last_high_price: float | None = None
    last_low_price: float | None = None
    prev_idx: int | None = None
    prev_price: float | None = None
    prev_time: pd.Timestamp | None = None

    for row in pivots_df.itertuples(index=False):
        pivot_price = float(row.pivot_price)
        pivot_type = row.pivot_type

        if pivot_type == "high":
            structure = "HH" if last_high_price is not None and pivot_price > last_high_price else "LH"
            if last_high_price is None:
                structure = "HH"
            last_high_price = pivot_price
        else:
            structure = "HL" if last_low_price is not None and pivot_price > last_low_price else "LL"
            if last_low_price is None:
                structure = "HL"
            last_low_price = pivot_price

        if prev_idx is None:
            leg_returns.append(0.0)
            leg_bars.append(0)
            leg_durations.append(0.0)
        else:
            change = (pivot_price / prev_price - 1.0) * 100.0 if prev_price else 0.0
            leg_returns.append(change)
            leg_bars.append(row.pivot_idx - prev_idx)

            delta = row.pivot_time - prev_time
            if isinstance(delta, pd.Timedelta):
                duration = delta.total_seconds()
            else:
                duration = float(delta / np.timedelta64(1, "s"))
            leg_durations.append(duration)

        structures.append(structure)
        prev_idx = row.pivot_idx
        prev_price = pivot_price
        prev_time = row.pivot_time

    result = pivots_df.copy()
    result["structure"] = structures
    result["leg_return_pct"] = leg_returns
    result["leg_bars"] = leg_bars
    result["leg_duration_sec"] = leg_durations

    return result


# Persist pivot data to CSV.
def export_csv(pivots_df: pd.DataFrame, out_path: Path) -> None:
    to_save = pivots_df.drop(columns=["pivot_idx"], errors="ignore")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_save.to_csv(out_path, index=False)


# Plot candlesticks with ZigZag overlays and annotations.
def plot_swings(
    df: pd.DataFrame,
    pivots_df: pd.DataFrame,
    title: str,
    out_path: Path,
    label: str | None = None,
) -> None:
    fig = go.Figure()

    if not df.empty:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
            )
        )

    if not pivots_df.empty:
        plot_df = pivots_df.sort_values("pivot_idx").copy()
        plot_df["plot_time"] = pd.to_datetime(plot_df["pivot_time"]).dt.tz_convert(None)
        fig.add_trace(
            go.Scatter(
                x=plot_df["plot_time"],
                y=plot_df["pivot_price"],
                mode="lines+markers",
                name="Swings",
                line=dict(color="#d62728", width=2),
                marker=dict(size=6, color="#d62728"),
            )
        )

        normalized = _normalize_label(label) if label else None
        annotation_limit = ANNOTATION_LIMITS.get(normalized, 0) if normalized else 0
        annotation_df = plot_df
        if annotation_limit and len(plot_df) > annotation_limit:
            step = max(1, math.ceil(len(plot_df) / annotation_limit))
            annotation_df = plot_df.iloc[::step].copy()
            if annotation_df.iloc[-1]["pivot_idx"] != plot_df.iloc[-1]["pivot_idx"]:
                annotation_df = pd.concat([annotation_df, plot_df.iloc[[-1]]])
            annotation_df = annotation_df.drop_duplicates("pivot_idx", keep="last")
            msg_label = label or normalized or title
            print(f"{msg_label}: thinning annotations from {len(plot_df)} to {len(annotation_df)} (step {step})")
        for row in annotation_df.itertuples(index=False):
            fig.add_annotation(
                x=row.plot_time,
                y=row.pivot_price,
                text=row.structure,
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-18,
                font=dict(size=10, color="#d62728"),
                bgcolor="rgba(255,255,255,0.65)",
                bordercolor="#d62728",
                borderwidth=1,
            )
    else:
        fig.add_annotation(
            text="No confirmed pivots",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(title_text="Price")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pyo.plot(fig, filename=str(out_path), auto_open=False)


def resolve_data_dir(base_dir: Path) -> Path:
    for name in ("datat", "data"):
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return base_dir / "datat"


def process_timeframe(label: str, csv_path: Path, swings_dir: Path, base_pct: float) -> None:
    df = prepare_df(csv_path)
    if df.empty:
        print(f"{label}: no data loaded")
        return

    pct = determine_pct(label, df, base_pct)
    if abs(pct - base_pct) >= 1e-4:
        print(
            f"Processing {label}: {len(df)} rows @ {pct:.3f}% (auto from {base_pct:.3f}%)..."
        )
    else:
        print(f"Processing {label}: {len(df)} rows @ {pct:.3f}%...")

    pivots = confirming_zigzag(df, pct=pct)
    pivots = label_structure(pivots)

    out_csv = swings_dir / f'swings_{label}.csv'
    out_html = swings_dir / f'swings_{label}.html'

    export_csv(pivots, out_csv)
    plot_swings(
        df,
        pivots,
        title=f"{label.capitalize()} Swings",
        out_path=out_html,
        label=label,
    )

    if not pivots.empty:
        first = pivots.iloc[0]
        last = pivots.iloc[-1]
        print(
            f"Processed {label} ({pct:.3f}% threshold): {len(pivots)} pivots, "
            f"range {first['pivot_time']} -> {last['pivot_time']}"
        )
    else:
        print(f"Processed {label} ({pct:.3f}% threshold): 0 pivots")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir = resolve_data_dir(project_root)
    swings_dir = script_dir
    swings_dir.mkdir(parents=True, exist_ok=True)

    timeframe_sources: Dict[str, Sequence[str]] = {
        "hour": ("hour.csv", "1hour.csv", "hourly.csv"),
        "minute": ("minute.csv", "1minute.csv", "01minute.csv", "1m.csv"),
        "5minute": ("5minute.csv", "05minute.csv", "5m.csv"),
    }

    for label, candidates in timeframe_sources.items():
        csv_path: Path | None = None
        for filename in candidates:
            candidate_path = data_dir / filename
            if candidate_path.exists():
                csv_path = candidate_path
                break
        if csv_path is None:
            print(f"Skipping {label}: none of {candidates} found")
            continue

        base_label = _normalize_label(label)
        pct = PCT_BY_TIMEFRAME.get(base_label, DEFAULT_PCT)
        process_timeframe(label, csv_path, swings_dir, base_pct=pct)


if __name__ == "__main__":
    main()
