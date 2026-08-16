# Hedge Risk-Level RL V1.7 dual-runtime integration

V1.7 keeps the V3 Action/Reward contract and the V1.6 memory/adaptive-CPU lifecycle, then
closes the missing runtime and deployment gaps for Windows native execution and Docker.

## Preserved V3 trading semantics

- `MultiDiscrete([5, 5])`: independent LONG/SHORT levels.
- Margin-risk ladder `0 / 5% / 12% / 25% / 40%`; HEAVY is a cap, not all-in.
- Leverage stays independent from the level ladder.
- Same-level actions never grant implicit scale-in permission.
- Primary reward is equity log-return; accounting costs are not deducted twice.
- Side-specific delayed credit prevents one profitable leg from hiding the other leg's failure.
- Heavy losses are penalized asymmetrically; positive shaping remains bounded.
- Reward-history state remains observable in the training environment.

## Preserved V1.6 engineering strengths

- Compact float32 feature memory and bounded delayed-reward state.
- Gym/SB3-compatible learner lifecycle.
- CPU-only model construction/loading and adaptive PyTorch thread control.
- Windows host resource broker for Docker host CPU/RAM telemetry.
- Existing 21-action Hedge RL remains available in parallel.
- Risk-Level RL remains in the semantic mainline, not a versioned runtime package tree.

## V1.7 runtime corrections

### Actual HedgeRuntime -> FreqAI binding

Earlier source exposed `set_hedge_context_provider()` but did not call it from the bot
lifecycle. With no provider, inference deliberately failed closed and returned FLAT/FLAT.
V1.7 passes `HedgeRuntime` into `StrategyInterface.ft_bot_start()`, binds the Risk-Level
FreqAI model immediately after model loading, and does so before the user's `bot_start()`
callback.

### Source-separated inference

`HedgeRiskRuntimeContextProvider` consumes the effective `HedgeRuntimeView`:

- `readonly` / `shadow`: EXCHANGE projection;
- `paper`: PAPER projection;
- `live`: LIVE projection when the surrounding safety model eventually enables it.

PAPER no longer incorrectly requires exchange user-stream/reconciliation check names.
Stale, halted, incomplete, risk-invalid, or leverage-incompatible projections still fail
closed.

### Stateful account observation without duplicate-row distortion

The runtime provider advances account history once per `(projection source, sequence)`.
FreqAI may evaluate several feature rows against one current account snapshot; those rows
must not each count as a new account return. Peak equity and downside semideviation are
therefore updated only when the runtime publishes a new sequence.

### Leverage contract safety

A level is a margin budget, so inference is only valid when observed LONG/SHORT leverage
matches the configured Risk-Level profile. A mismatch now fails closed instead of mapping
an exchange position to the wrong risk level.

### Continual-learning CPU enforcement

New models were already constructed with `device="cpu"`. V1.7 additionally migrates a
reused continual-learning policy and optimizer state to CPU before `learn()`, so a model
artifact created in a different device environment cannot silently violate this project's
CPU-only training contract.

## Windows native contract

- Project-local `.venv\Scripts\python.exe` remains authoritative.
- No editable install is required; the existing local-source `.pth` mechanism is preserved.
- PowerShell scripts are ASCII and CRLF for Windows PowerShell 5.1 validation.
- `Test-Freqtrade-Hedge-RiskLevelRL-DualRuntime.ps1` runs source authority, dependencies,
  deterministic matrices, focused pytest, real SB3 CPU smoke, `pip check`, and PS5.1 AST.

## Docker contract

- Root `docker-compose.yml` now builds the current Hedge source instead of launching the
  unrelated official stable image.
- Runtime image: `freqtrade-hedge:1.7-risklevel-rl-cpu`.
- The root Dockerfile installs FreqAI + Hedge RL dependencies and explicitly installs a
  CPU PyTorch wheel.
- Image build asserts CUDA is unavailable in this CPU-only image.
- `user_data` is consistently mounted at `/opt/freqtrade-hedge/user_data`.
- `Start-Hedge-Docker.ps1` builds the image when missing, validates source authority and
  RL dependencies, starts the Windows host resource broker, and invokes `freqtrade trade`
  through the custom image's non-prefixing entrypoint.

## Validation

`tools/validate_hedge_risklevel_rl_dual_runtime_400.py` executes 400 deterministic
checks across ten families: action lattice, explicit-increase semantics, PAPER/readonly
fresh and fail-closed projections, runtime history de-duplication, Windows/Docker static
contracts, risk-budget semantics, and subsystem/device independence.
