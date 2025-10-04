"""Async strategy entrypoint wiring ingestion to processing pipeline."""
from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import asynccontextmanager

from runtime.app_config import AppConfig
from runtime.parquet_ingestion import ParquetCandleSource
from runtime.candle_pipeline import StrategyPipeline
from runtime.strategy_state import StrategyState


log = logging.getLogger(__name__)


@asynccontextmanager
async def _shutdown_signal(loop: asyncio.AbstractEventLoop) -> asyncio.Queue[None]:
    queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)

    def _handle_sig(*_: int) -> None:
        log.info("Shutdown signal received")
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            # add_signal_handler is unavailable on some platforms (e.g. Windows)
            pass
    try:
        yield queue
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.remove_signal_handler(sig)
            except NotImplementedError:
                pass


async def _ingest_candles(source: ParquetCandleSource, queue: asyncio.Queue) -> None:
    try:
        async for candle in source.stream():
            await queue.put(candle)
    finally:
        await queue.put(None)


async def _process_candles(queue: asyncio.Queue, pipeline: StrategyPipeline) -> None:
    while True:
        candle = await queue.get()
        if candle is None:
            queue.task_done()
            break
        await pipeline.handle_candle(candle)
        queue.task_done()


async def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    config = AppConfig.from_env()
    log.info("Starting strategy for %s using %s", config.symbol, config.parquet_path)

    state = StrategyState(symbol=config.symbol, equity=10_000.0)
    pipeline = StrategyPipeline(state)
    queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_maxsize)
    source = ParquetCandleSource(config.parquet_path, replay_speed=config.replay_speed)

    loop = asyncio.get_running_loop()
    async with _shutdown_signal(loop) as shutdown_queue:
        ingest_task = asyncio.create_task(_ingest_candles(source, queue), name="ingest")
        process_task = asyncio.create_task(_process_candles(queue, pipeline), name="process")
        shutdown_task = asyncio.create_task(shutdown_queue.get(), name="shutdown-wait")

        done, pending = await asyncio.wait(
            {ingest_task, process_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await queue.join()

        for task in done:
            if task is shutdown_task:
                continue
            exc = task.exception()
            if exc:
                raise exc


if __name__ == "__main__":
    asyncio.run(main())
