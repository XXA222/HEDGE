# Hedge numeric primitive audit

This document records the A-01 migration boundary. `freqtrade.hedge.numeric`
owns strict finite-decimal validation and the shared `ZERO` identity. Other
packages may keep a thin adapter only when they must translate the exception
type or preserve a public legacy API.

## Canonical primitive

- `freqtrade/hedge/numeric.py::to_decimal` rejects `None` unless
  `allow_none=True`, empty strings, booleans, malformed values, and non-finite
  values with `HedgeDataError`.
- `freqtrade/hedge/numeric.py::ZERO` is the only project-level `ZERO` constant.
- `require_nonnegative`, `require_positive`, and `require_unit_interval` remain
  the canonical range validators.

The following modules now import `ZERO` from the canonical primitive rather than
creating a second `Decimal(0)` object:

`backtesting/advanced_metrics.py`, `backtesting/decimal_utils.py`,
`backtesting/execution_realism.py`, `backtesting/quality.py`,
`integration/paper_runtime.py`, `integration/production_context.py`,
`native/models.py`, `operations/common.py`, `optimization/aggregation.py`,
`optimization/metrics.py`, `optimization/robust_selection.py`,
`planning/context.py`, `strategies/contract.py`,
`strategies/simple_ma_hedge.py`, and `telemetry/dryrun.py`.

## Deliberate adapters

- `contracts/types.py::finite_decimal` keeps the frozen contract exception
  surface and delegates only after its public input contract is checked.
- `native/models.py::finite_decimal` remains a compatibility adapter for the
  native convergence API; its public `TypeError`/`ValueError` behavior is
  migrated separately under A-02.
- `exchange/binance_normalizer.py` uses `MissingPolicy` and
  `optional_decimal`. Missing exchange fields are now an explicit decision:
  `REJECT`, `ZERO`, or `NONE`; no function-level default silently rewrites a
  missing field.

## Exchange missing-value policy

The current intentional choices are:

| Boundary | Policy | Reason |
| --- | --- | --- |
| Account aggregate and asset balance fields | `ZERO` | Binance may omit zero-valued balance rows; the normalized fact is defined with a zero balance. |
| Income history numeric value | `ZERO` | Income rows may omit a zero amount while the row identity remains useful. |
| Historical account exposure probes | `ZERO` | Missing position/order-margin values mean no observed exposure for that probe. |
| Liquidation price | `NONE` | A missing or zero liquidation price means the exchange did not provide a usable liquidation level. |
| Required position/order/fill quantities and prices | `REJECT` | Missing execution facts must never be converted to a valid trading quantity or price. |

Each policy is named at the call site and covered by exchange normalizer tests.
Future fields must choose a policy explicitly instead of adding a default
argument to `finite_decimal` or `nonnegative_decimal`.
