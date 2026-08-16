# Hedge Risk-Level RL V2 - Memory Optimization

## Scope

This revision applies the GlobalMemoryOpt V1.4 lifecycle principles to the independent
FreqAI Hedge risk-level learner. It does not merge the module with HPRL and it does not
allow RL to bypass planner/risk/execution boundaries.

The policy contract remains `MultiDiscrete([5, 5])` with LONG and SHORT target margin
levels `0 / 5% / 12% / 25% / 40%` by default. The changes are memory-lifecycle and
allocation optimizations around training, observations, rewards and inference.

## V1 memory costs found

The V1 risk-level environment had several avoidable memory costs:

1. Every environment converted the full feature DataFrame into another float64 matrix.
2. It retained a pandas price DataFrame even though the hot loop only needs OPEN, CLOSE,
   funding and uncertainty after OHLC validation.
3. Every observation converted the full feature matrix through `np.asarray(..., float64)`,
   allocated a clipped market array, allocated an account array, concatenated both, and
   then converted to float32.
4. Every candle materialized a full reward-component dictionary even when no diagnostic
   consumer needed it.
5. The action mask and TensorBoard metric dictionaries were recreated in the hot path.
6. Episode reset rebuilt simulator and reward-model object graphs instead of resetting them.
7. Delayed reward outcomes rebuilt a `remaining` list each step and had no explicit hard
   memory cap.
8. The upstream FreqAI RL training method deep-copied all training features into `df_raw`,
   although `HedgeRiskLevelEnv` does not use `df_raw`.
9. Train/eval environment references could remain attached to the learner/model after
   training, retaining compact feature arrays longer than necessary.
10. Inference created an intermediate numeric DataFrame plus a float64 ndarray and emitted
    two int64 columns although the only valid level values are 0..4.

## V2 implementation

### Compact feature and market retention

`HedgeRLMemoryConfig` defaults normalized market features to float32. Price and accounting
precision remains float64. The environment retains only:

- one compact C-contiguous feature matrix;
- OPEN float64;
- CLOSE float64;
- funding float32;
- uncertainty float32.

HIGH/LOW/VOLUME are validated during environment construction and then released. The input
pandas price DataFrame is not retained.

### Single-output observation construction

`HedgeRiskObservationBuilder.build_into()` writes directly into one caller-supplied float32
buffer. The standard `build()` allocates only that final observation vector. There is no
per-step full-matrix dtype conversion and no `np.concatenate()`.

### Bounded reward state

Delayed scale/probe outcomes retain a tiny bounded list with an explicit hard cap. Resolution
compacts that list in place. `reset()` clears the same list object. Full reward breakdown
materialization is sparse by default (every 64 steps and episode end), while the scalar reward
is calculated every step exactly as before.

### Episode lifecycle

Simulator and reward model objects are reused across resets. A read-only static MaskablePPO
action mask is reused. TensorBoard metric dictionaries are updated in place. Optional GC is
performed only at coarse episode boundaries, never in the candle hot loop.

### Training lifecycle

`HedgeRiskLevelReinforcementLearner.train()` follows the official FreqAI training pipeline but
removes the unused `copy.deepcopy(train_features)` into `df_raw`. Transformed RL feature
DataFrames are downcast to float32 by default. After `model.learn()` train/eval environments
are closed and the model->environment retention link is removed. Callback references are
cleared before the V1.4 phase-boundary GC/allocator trim helper is called.

The retained `dk.data_dictionary["train_features"]` is deliberately not destroyed during
training because FreqAI's `DataDrawer.save_data()` persists that DataFrame after model fitting.
It is therefore compacted rather than illegally freed early.

### Inference lifecycle

Prediction features are converted directly to one compact numeric matrix (float32 by default)
without `DataFrame.apply(pd.to_numeric)` and an additional float64 matrix. Output levels are
stored as int8 because the domain is 0..4. Missing/stale account projection still fails closed
to `[0, 0]`.

## Measured synthetic benchmark

Benchmark shape: 60,000 rows, 32 features, 1,000 environment steps. Input DataFrames are
created before measurement; measurements therefore isolate the additional environment/hot-loop
footprint.

| Metric | V1 | V2 | Change |
|---|---:|---:|---:|
| Python tracemalloc peak | 48,554,480 B | 10,105,917 B | -79.19% |
| RSS delta | 24,887,296 B | 8,355,840 B | -66.43% |
| elapsed | 0.7165 s | 0.3516 s | -50.93% |
| retained feature dtype | float64 | float32 | 50% per element |
| retained price DataFrame | yes | no | removed |

These are synthetic measurements in the delivery environment, not promises of identical RSS
reductions on every Windows/Docker workload. Actual savings depend strongly on feature count,
SB3 rollout settings, multiprocessing and pandas allocator state.

## Repeated-episode retention check

A 200-episode / 16-step repeated-reset test finished with zero pending reward outcomes. The
Python tracemalloc current-allocation change between episode 20 and episode 200 was only 2,360
bytes; the sampling list itself contributes to that figure. This is consistent with bounded
runtime state rather than episode-proportional retention.

## Validation

- Risk-level V1 + V2 focused tests: 45/45 PASS.
- GlobalMemoryOpt V1.4 focused tests used with this integration: 18/18 PASS.
- Combined focused suite: 63/63 PASS.
- Risk-Level RL V2 Memory development matrix: 400/400 PASS across 20 themes.
- Existing GlobalMemoryOpt V1.4 global matrix: 400/400 PASS after integration.

The environment used for packaging does not contain Stable-Baselines3/sb3-contrib, so a real
PPO training run is not falsely reported as executed here. The learner integration is compile-
checked and source-contract checked; final Windows acceptance should execute the provided
installer against the project's `.venv`, where FreqAI RL dependencies are installed.
