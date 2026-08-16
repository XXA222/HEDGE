# Hedge Risk-Level RL integrated mainline

This mainline absorbs the complete `risklevel-rl-v3-actionreward` source surface into the
current adaptive-CPU / global-memory Hedge source without replacing the existing 21-action
Hedge RL environment.

## Two parallel RL contracts

The existing `HedgeReinforcementLearner` remains available with its stable discrete 21-action
catalog.  The imported `HedgeRiskLevelReinforcementLearner` is an independent FreqAI model
whose action contract is factorized:

```text
MultiDiscrete([5, 5]) -> [LONG risk level, SHORT risk level]
```

The default per-leg margin ladder is `0 / 5% / 12% / 25% / 40%`.  A level is a margin-risk
cap, not an order instruction.  Leverage is applied separately.  LONG and SHORT can coexist.
The HEAVY level is deliberately below all-in and the default two-leg combined cap keeps a
20% margin reserve.

## Integration boundary

The integrated path is:

```text
FreqAI features + canonical CentralRuntimeProjection
    -> HedgeRiskPolicyContext
    -> HedgeRiskLevelReinforcementLearner
    -> [long_level, short_level]
    -> HedgeRiskLevelPlannerAdapter
    -> SignalSnapshot / target-risk intent
    -> PureHedgePlanner
    -> Hedge risk engine
    -> OrderIntent
    -> execution engine
```

The model never submits exchange orders.  Missing/stale account facts, stale user-stream
facts, invalid risk data or unconverged reconciliation force the inference bridge to return
`[0, 0]`.

## Runtime projection integration

`risk_projection_adapter.py` converts `CentralRuntimeProjection` into the exact risk-level
policy context.  Observed live margin usage is mapped to the smallest configured level that
is not below the observed risk.  This ceiling rule is intentionally conservative and avoids
understating current risk near a level boundary.

The state-aware planner adapter preserves the source package rule:

```text
RISK_CAP_NO_SAME_LEVEL_SCALE_IN
```

A same-level action cannot add risk merely because price/equity drift changed the notional
required by the nominal level.  New risk requires an explicit upward level transition.

## Memory integration

The imported V2 memory lifecycle is retained:

- normalized feature matrix defaults to float32;
- price/accounting arrays remain float64;
- the environment does not retain the original price DataFrame;
- observations are written into one float32 output buffer;
- pending delayed rewards have a hard bound;
- full reward component dictionaries are emitted sparsely;
- train/eval environments are closed/detached after fitting;
- V1.4 phase-boundary memory release is reused;
- no heavy GC or allocator trim is executed in the candle hot loop.

## Adaptive CPU integration

The current mainline adds `risk_runtime.py`.  The risk-level learner remains one causal
chronological environment by default, but its PyTorch numerical thread budget is refreshed
from the project-wide `AdaptiveResourceGovernor` at coarse training intervals.  It can consume
the Windows host resource broker snapshot when training in Docker.  The default cap is the
smaller of physical CPU count, the host-aware numerical budget and 16 threads.

The formal project training baseline is CPU-only.  This learner therefore forces SB3 model
construction and best-model loading to `device="cpu"` even if a stale user configuration
contains an accelerator device.

Independent Hyperopt trials, research candidates, pairs and walk-forward folds continue to
use the project-level adaptive process scheduler rather than attempting unsafe per-bar
parallelism inside one account trajectory.

## Imported source coverage

All runtime/support files present only in the uploaded full source package are retained,
including `risk_gym.py`, `risk_memory.py`, the FreqAI prediction model, both risk-level
validation generations, and both documentation files.  The smaller uploaded overlay is a
strict subset of the full package and is treated as a cross-check, not as the authoritative
file inventory.
