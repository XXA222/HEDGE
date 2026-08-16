# Hedge adaptive CPU and performance architecture

V1.5 targets CPU throughput after the V1.4 bounded-memory redesign.  A chronological
single-symbol replay remains serial because fills, active orders, Cross wallet state,
funding, liquidation and planner state are causally dependent on preceding bars.
The performance design therefore has two layers:

1. Remove avoidable Python work from the serial hot path.
2. Use process-level parallelism for work that is genuinely independent: parameter
   trials, backtest candidates and research evaluations.  Vectorized indicator phases
   may use native NumPy/BLAS threads.

## Serial hot path

The conservative matcher first proves whether a bar can touch any active order.  Flat
or non-touching bars avoid full wallet cloning and two-path order simulation.  Bars
that can execute orders continue to evaluate both conservative OHLC paths.

CrossWallet computes unrealized PnL and risk scalars directly from mutable leg fields.
It no longer builds immutable LegPosition objects merely to calculate scalar risk.
Matcher clones use a dedicated lightweight exact state copy instead of generic
`copy.deepcopy`.  Planner projections and effective planner configuration are cached
until the underlying state changes.  Compact replay suppresses diagnostic string/list
allocation and no-fill state evolution.

## Adaptive parallel work

`workers=0` means adaptive execution, `workers=-1` means use the resource-aware maximum,
`workers=1` forces serial execution, and a positive value greater than one is an upper
bound.  Linux Docker uses fork/COW process workers so prepared historical data can be
shared efficiently while Python CPU work bypasses the GIL.

A small Windows PowerShell 5.1 host broker writes total Windows CPU and available RAM
to `user_data/runtime/host-resource-snapshot.json`.  The Docker governor consumes only
fresh snapshots.  When other desktop applications are busy, the scheduler stops
feeding replacement workers.  When the desktop becomes idle, it increases independent
worker count in bounded steps.  Worker processes limit BLAS/NumExpr to one thread to
avoid process x thread oversubscription.

For a single vectorized indicator phase, the governor instead grants a dynamic native
thread budget.  This lets NumPy/OpenBLAS/NumExpr use spare cores while preserving CPU
headroom for interactive applications.

## Memory coordination

Worker count is capped by both CPU and available-memory budgets.  V1.4 bounded replay,
cache release, compact snapshots, prepared-dataset reuse, and pressure-aware GC remain
active.  Optimization trial paths skip unnecessary per-trial JSON artifacts.

## Host broker

The broker queries CPU and memory with the Windows `GetSystemTimes` and
`GlobalMemoryStatusEx` kernel APIs inside the sampling loop.  WMI/CIM is used only once
at startup for static CPU topology, keeping the monitoring overhead low.

## Safety

No resource scheduler may parallelize bars inside one stateful symbol replay.  No CPU
optimization changes exchange-write gates, Hedge-mode identity, Cross-account risk,
NEXT_BAR_NO_LOOKAHEAD ordering, or conservative dual-path execution for fillable bars.
