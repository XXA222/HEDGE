# FreqAI Hedge Risk-Level Reinforcement Learning

This module extends the standard FreqAI reinforcement-learning lifecycle for a Binance-style
Hedge account where LONG and SHORT can coexist. It is independent from HPRL.

## Action contract

The policy emits a factorized Gymnasium action:

```text
MultiDiscrete([5, 5]) -> [long_level, short_level]
```

Each leg uses the same configurable margin-budget ladder. The default profile is:

| Level | Name | Margin budget |
|---:|---|---:|
| 0 | FLAT | 0% |
| 1 | VERY_LOW | 5% |
| 2 | LIGHT | 12% |
| 3 | MEDIUM | 25% |
| 4 | HEAVY | 40% |

A level is a risk bucket / margin budget, not notional exposure. Leverage is a separate
parameter:

```text
level -> margin budget -> leverage -> notional cap
```

With 1000 USDT equity, LONG level 1 and 3x leverage means a 50 USDT margin budget and a
150 USDT notional cap.

### No implicit same-level scale-in

A level is also a risk cap. Maintaining the same level must never buy more only because the
position lost money or account equity changed. New risk is allowed only when the policy
explicitly raises that leg's level. Same-level and lower-level decisions may keep or reduce
risk, but may not average down implicitly.

The training simulator enforces this rule. Dry-run/live planner integrations must use the
state-aware `HedgeRiskLevelPlannerAdapter.from_account_action()` contract, which emits
`long_increase_allowed`, `short_increase_allowed`, and
`RISK_CAP_NO_SAME_LEVEL_SCALE_IN` semantics.

## Reward contract

Account equity log-return is the only primary profit source:

```text
R_primary = equity_log_return_scale * log(equity_t / equity_t-1)
```

Equity already contains realized/unrealized PnL, fees, funding and slippage. Those accounting
items are therefore not subtracted a second time. `accounting_cost_ratio` is diagnostic only.

Small shaping terms price risk-management quality:

- incremental convex drawdown deterioration;
- downside return times gross margin exposure;
- observable downside semideviation memory;
- uncertainty times squared gross margin exposure;
- leverage-adjusted notional excess risk;
- reserve-margin pressure and liquidation-buffer shortfall;
- asymmetric level loss multipliers;
- adverse scale-in, large upward level jumps, level churn and turnover;
- side-specific delayed probe and scale credit;
- side-specific risk reduction and profit lock;
- hedge efficiency or hedge drag based on actual step PnL attribution.

Positive shaping is capped so it cannot dominate the equity objective.

## Credit assignment

Delayed probe/scale outcomes are attributed to the leg that created them. A profitable SHORT
cannot make a losing LONG probe appear successful. Probe baselines are captured before the
entry action, so fee/slippage costs count toward whether the probe succeeded. Closing or
reducing a pending probe early resolves it immediately instead of waiting for the original
horizon.

Reward history that can affect future rewards is visible to the policy observation:

- per-side failed probe counts;
- downside semideviation;
- fraction of delayed reward slots currently pending.

This keeps the reward process observable rather than hiding reward-relevant state from the
agent.

## Default configuration

```json
{
  "freqai": {
    "rl_config": {
      "model_type": "PPO",
      "policy_type": "MlpPolicy",
      "hedge_action_space": {
        "enabled": true,
        "position_levels": [0.0, 0.05, 0.12, 0.25, 0.40],
        "long_leverage": 3.0,
        "short_leverage": 3.0,
        "max_combined_margin_fraction": 0.80,
        "minimum_reserve_margin_fraction": 0.20,
        "hard_max_margin_fraction_per_leg": 0.50,
        "rebalance_deadband_fraction": 0.0025
      },
      "hedge_reward": {
        "loss_level_multipliers": [0.0, 1.0, 1.10, 1.30, 1.65],
        "win_level_multipliers": [0.0, 1.0, 1.03, 1.06, 1.10],
        "wrong_level_loss_weight": 0.15,
        "uncertainty_exposure_weight": 0.12,
        "leverage_exposure_weight": 0.01,
        "adverse_scale_in_weight": 0.25,
        "repeated_probe_weight": 0.02,
        "max_positive_shaping": 0.25,
        "reward_clip": 10.0
      }
    }
  }
}
```

The numerical reward weights are defaults for controlled experiments, not universally optimal
trading parameters. They must be calibrated by walk-forward training/backtests and held-out
regime evaluation before production use.

## Safety boundary

The RL policy does not submit exchange orders. Its output remains:

```text
market/account observation
  -> [LONG level, SHORT level]
  -> state-aware target-risk adapter
  -> Hedge planner
  -> risk governor
  -> OrderIntent
  -> execution engine
```

A stale/missing account projection fails closed to `[0, 0]` in the inference bridge.
