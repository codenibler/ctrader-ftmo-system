# cTrader FTMO System

## Overview

This repository contains the codebase for an automated trading stack designed to
drive live strategies against cTrader accounts while retaining compatibility with
our legacy research tooling. The new architecture is containerised (Docker) and
organised as microservices so that market data ingestion, strategy execution,
order routing, and post-trade analytics can evolve independently. Alongside the
modern services lives `Legacy_strategy`, the original Python backtesting suite we
use as the source of truth for indicator calculations and strategy rules.

The current focus is the LVN (Low Volume Node) strategy that trades CL (WTI
crude) on the 1-minute timeframe. A streaming implementation mirrors the legacy
backtest logic candle-for-candle, enabling direct comparison before wiring the
system to the live cTrader Open API feed.

## Repository Structure

```
Legacy_strategy/               # Original research sandbox with backtests, indicators, reports
├── LVN_Strategy/              # Legacy LVN backtest project
│   ├── findings/              # Swing/lvn/EMA CSVs and logic used in historical runs
│   ├── Minutetrades/          # Post-processing for fills, spread adjustment, equity charts
│   └── orchestrator.py        # Drives the multi-step legacy pipeline
services/                      # Modern microservices
├── md-gateway/                # Placeholder: will stream cTrader data into NATS
├── order-router/              # Placeholder: will receive trade instructions & persist to Postgres
├── strategy/                  # New LVN streaming strategy service + tooling
│   ├── indicators/            # EMA, zigzag legs, LVN calculators (streaming friendly)
│   ├── risk/                  # Advanced stop/target logic matching legacy
│   ├── runtime/               # Async orchestration: parquet ingestion, state, pipelines
│   ├── backtest_month.py      # Monthly replay utility (equity & trade charts)
│   ├── split_parquet_by_month.py
│   ├── export_month_structures.py
│   ├── spread_calculator.py
│   └── CL_1m.parquet          # Sample minute data seed (replay until Open API is live)
backtests/                     # Generated equity/trade charts from streaming backtests
diagnostics/                   # CSV exports of pivots/legs/LVNs for comparison with legacy outputs
references/                    # Documentation, API specs, notes
sql/                           # Database schema migrations for order persistence
```

## Services & Responsibilities

### Market-Data Gateway (`services/md-gateway`)

- **Current state**: placeholder that keeps the container alive.
- **Planned**: connect to the cTrader Open API WebSocket, subscribe to CL 1m
  candles, and publish them via NATS. Once live data is flowing, the strategy
  service simply replaces its parquet reader with a NATS subscriber.

### Strategy Service (`services/strategy`)

An asyncio application that currently replays candles from `CL_1m.parquet` but is
designed to consume live streams. Major components:

- `runtime/strategy_state.py`: persistent state (candles, pivots, legs, LVNs,
  EMA caches, equity, open trades).
- `runtime/candle_pipeline.py`: orchestrates per-candle processing—update EMAs,
  stream zigzag pivots, build legs, compute LVNs, evaluate trade entries,
  manage open trades using the advanced TP/SL, and track completed trades.
- `indicators/legs.py`: streaming zigzag detector that replicates the legacy
  swing logic (same thresholds, structures, leg metrics).
- `indicators/lvns.py`: volume-profile based LVN discovery ported from the
  legacy `lvn_scan.py`, including value-area filters, prominence, and adaptive
  bin sizing.
- `indicators/ema.py`: incremental EMA9/EMA20 updates plus trend detection.
- `risk/advanced_sl.py`: imported from the legacy code to compute stop/targets
  based on surrounding pivots.
- `spread_calculator.py`: applies live spread + a 0.02 baseline for long trades
  before sizing the position.

The service publishes detailed logs during replays, records trades in memory,
and exposes helper scripts for analysis.

### Order Router (`services/order-router`)

- **Current state**: placeholder container. Once the live strategy is ready,
  this service will receive order instructions (likely via NATS or REST), send
  them to cTrader’s trading endpoint, and log fills in Postgres (see `sql/`).

## Tooling & Workflow

### Strategy Replay & Diagnostics

You can run the full streaming pipeline against a parquet file without live
connectivity:

```bash
# Replay and generate equity/trade charts for a month
python services/strategy/backtest_month.py \
  --parquet services/strategy/data/monthly/CL_1m_2024-08.parquet \
  --output-equity backtests/CL_2024-08_equity.html \
  --output-trades backtests/CL_2024-08_trades.html
```

The script leverages the same pipeline as live trading, then:
- saves `CL_2024-08_equity.html` (Plotly equity curve), and
- saves `CL_2024-08_trades.html` (candlestick chart with entry/exit markers).

### Parquet Utilities

```bash
# Split master parquet into per-month files
default_out="services/strategy/data/monthly"
python services/strategy/split_parquet_by_month.py \
  --input services/strategy/CL_1m.parquet --output "$default_out" --symbol CL

# Optional helper to reform/reindex raw parquet data (if needed)
python services/strategy/reformat_cl_parquet.py --input raw.parquet --output CL_1m.parquet
```

### Diagnostics & Legacy Comparison

To compare the streaming implementation against the legacy CSVs:

```bash
python services/strategy/export_month_structures.py \
  --parquet services/strategy/data/monthly/CL_1m_2024-12.parquet \
  --output-dir diagnostics/2024-12 --symbol CL
```

This produces three CSV files (`*_pivots.csv`, `*_legs.csv`, `*_lvns.csv`) that
mirror the legacy `findings/` outputs, making it easier to diff swings, legs,
and LVNs between the codebases.

## Strategy Logic Summary

1. **Deploy EMA trend**: EMA9/EMA20 updated per candle; a trade can fire only
   when EMA9 > EMA20 for longs or EMA9 < EMA20 for shorts.
2. **Swings → Legs**: Streaming zigzag calculates pivots (HH, HL, LH, LL) with
   a 0.12% reversal threshold, exactly as the legacy script. Consecutive HL→HH
   or LH→LL pivots form directional legs.
3. **Volume Profiles**: For each leg, a volume profile is built, smoothed, and
   scanned for significant LVNs (value area, prominence, depth filters). Each
   LVN carries metadata (rank, raw volume, POC, VA bounds).
4. **Entries**: When price trades through an LVN after its leg finishes and the
   EMA trend aligns, we open a position. Long entries subtract live spread plus
   the 0.02 baseline; shorts are entered at the raw LVN price.
5. **Stops & Targets**: `AdvancedSLIntakeBuffer` matches the legacy methodology,
   referencing prior pivots when available and falling back to configured
   offsets otherwise (0.25 uptrend, 0.20 downtrend).
6. **Risk & Sizing**: Each trade risks 1% of current equity, compounding over
   time. (Constraint to one trade at a time can be toggled later.)
7. **Trade Management**: The pipeline monitors each open trade candle-by-candle;
   whichever level is touched first (stop vs target) exits the trade. PnL and
   equity updates are recorded for analysis.

## Legacy Integration Notes

- **Legacy_strategy/LVN_Strategy** remains your canonical reference. The new
  streaming code ports the exact leg/LVN/EMA logic so you can diff outputs.
- `Legacy_strategy/LVN_Strategy/orchestrator.py` demonstrates the old batch
  workflow (apply time rules, adjust spread, build monthly reports). The new
  stack replicates the same calculations in real-time so we no longer depend on
  pre-generated CSVs.

## Next Steps

1. Wire `md-gateway` to cTrader’s Open API WebSocket and publish candles via NATS.
2. Update the strategy service to subscribe to NATS rather than replaying the
   parquet on start-up.
3. Implement the order-router to deliver orders to the live account and log
   fills in Postgres (using migrations from `sql/`).
4. Harden the backtest harness with regression tests comparing against legacy
   outputs (e.g., trade-by-trade diffs for sample months).
5. Add configuration for multi-symbol support and runtime feature toggles.

## Getting Started

1. **Replay locally**: run `python services/strategy/strategy.py` to see the
   streaming pipeline consume the sample parquet.
2. **Explore charts**: open the generated HTML files under `backtests/`.
3. **Compare diagnostics**: diff the CSVs under `diagnostics/` with the legacy
   outputs in `Legacy_strategy/LVN_Strategy/findings/`.
4. **Prepare for live**: keep `STRATEGY_PARQUET` as a configurable knob—once
   `md-gateway` publishes real candles, switch the strategy service to consume
   them directly.

## Dependencies & Environment

- Python 3.11 (with pandas, numpy, plotly). Ensure NumPy wheels match your CPU
  architecture (Arm vs x86) when running on macOS.
- Docker & docker-compose to orchestrate services locally (see `Dockerfile`s and
  `docker-compose.yaml`).
- Postgres (for order persistence) when the order-router is implemented.

## Support & Contribution

- Use the diagnostics scripts to verify that any changes keep parity with the
  legacy calculations.
- Keep spread and risk parameters configurable via environment variables to
  simplify live tuning.
- Before deploying to production, integrate unit/integration tests for EMA
  warm-up, swing classification, LVN detection, and edge-case trade flows.

