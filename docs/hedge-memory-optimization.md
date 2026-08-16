# Hedge Backtest Memory Management

The Hedge backtest path is designed to keep long 1m histories bounded by the
analyzed dataframe and current simulation state instead of by the number of
historical events/snapshots.

## Freqtrade-derived lifecycle principles

The implementation intentionally follows the upstream backtesting lifecycle:

- indicator population is vectorized before the execution loop;
- `reduce_df_footprint` is enabled by default for Hedge backtesting, reusing
  Freqtrade's float32/int32 downcast of non-OHLCV numeric columns;
- the execution surface is narrowed before replay;
- one-shot Hedge backtests release analyzed/informative DataProvider caches once
  the replay columns are detached;
- Hyperopt prepares indicators/data once and reuses the detached replay surface
  across planner/paper epochs;
- large temporary object graphs are released at phase boundaries, never by
  forcing collection in every candle.

The default `DataProvider.clear_cache()` behavior is unchanged for upstream
Freqtrade/Hyperopt. Hedge one-shot consumers explicitly request
`clear_cache(include_backtesting=True)` after detaching their data.

## Compact ordered stream

Normal Hedge backtesting does not retain the full `SignalEvent + BarEvent`
history. `COMPACT_ORDERED_STREAM_V2` consumes one canonical input event at a
time, validates only the current timestamp slot set, and immediately applies it
to the same planner/matcher/wallet implementation.

The stream does not:

- call `list(enumerate(events))`;
- sort a multi-year event list;
- retain all historical input slots;
- create a snapshot object for every bar;
- retain normal input events in the result ledger.

Exact max drawdown and dual-leg duration remain online wallet statistics.
Equity snapshots are sampled approximately daily and bounded to 2048 points.

## Wallet history compaction

Compact historical replay releases data whose idempotency scope ended with the
committed deterministic bar:

- `realized_by_fill` scratch values;
- generated fill/liquidation id sets;
- fully closed tactical-lot objects.

Closed tactical lots are archived into scalar realized-PnL/funding/fee totals and
a closed-lot count before their objects are removed. Durable Paper/live replay
keeps the original unbounded idempotency history and does not use this trusted
backtest-only compaction.

## Prepared optimization data

`prepare_freqtrade_hedge_backtest()` performs the expensive Freqtrade
OHLCV/informative/indicator analysis once. It keeps only the narrow immutable
replay dataframe and funding columns, releases the Backtesting/DataProvider/
Exchange object graph, and allows Native Hedge Hyperopt trials to reuse that
prepared surface without loading and calculating indicators for every epoch.

Research optimization uses an O(1)-memory regular timestamp sequence for
gap-free compact datasets instead of materializing a million `datetime` objects.

## Allocator trimming

At large phase boundaries Hedge calls `release_phase_memory()`: Python cyclic GC
runs first, and Linux containers perform a best-effort `malloc_trim(0)` to return
freed glibc pages to the container RSS. This is deliberately not used in the
per-bar hot loop. Set `HEDGE_MEMORY_TRIM=0` to disable allocator trimming.

## Detailed mode

Explicit event export remains the detailed/audit path and may materialize the
full event history. It is not used by long-range Hyperopt or the default compact
backtest path.
