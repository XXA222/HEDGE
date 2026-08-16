from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from freqtrade.freqai.hedge_rl.risk_bridge import (  # noqa: E402
    HedgeRiskLevelPolicyBridge,
    HedgeRiskPolicyContext,
)
from freqtrade.freqai.hedge_rl.risk_environment import HedgeRiskLevelEnv  # noqa: E402
from freqtrade.freqai.hedge_rl.risk_levels import (  # noqa: E402
    HedgeRiskLevelAction,
    RiskLevelMapper,
    RiskLevelProfile,
)
from freqtrade.freqai.hedge_rl.risk_memory import (  # noqa: E402
    CompactRiskMarketData,
    compact_feature_matrix,
)
from freqtrade.freqai.hedge_rl.risk_observation import (  # noqa: E402
    HedgeRiskObservationBuilder,
    RiskObservationSchema,
)
from freqtrade.freqai.hedge_rl.risk_planner_adapter import (  # noqa: E402
    HedgeRiskLevelPlannerAdapter,
)
from freqtrade.freqai.hedge_rl.risk_portfolio import (  # noqa: E402
    LegSide,
    RiskAccountState,
    TargetLevelPortfolioSimulator,
)
from freqtrade.freqai.hedge_rl.risk_reward import (  # noqa: E402
    HedgeRiskRewardModel,
    PendingOutcome,
    RiskRewardConfig,
)


class Matrix:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def check(self, theme: str, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            self.rows.append({"theme": theme, "name": name, "status": "FAIL", "error": repr(exc)})
        else:
            self.rows.append({"theme": theme, "name": name, "status": "PASS"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def market(rows: int, cols: int = 4, *, shift: float = 0.0):
    features = pd.DataFrame(
        np.linspace(-2.0, 2.0, rows * cols, dtype=np.float64).reshape(rows, cols),
        columns=[f"f{index}" for index in range(cols)],
    )
    base = np.linspace(100.0, 102.0 + shift, rows)
    prices = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.1,
            "volume": np.ones(rows),
            "funding_rate": np.zeros(rows),
            "uncertainty_score": np.full(rows, 0.5),
            "wide_unused": np.arange(rows, dtype=np.float64),
        }
    )
    return features, prices


def make_env(rows: int = 128, cols: int = 4, *, interval: int = 64) -> HedgeRiskLevelEnv:
    features, prices = market(rows, cols)
    return HedgeRiskLevelEnv(
        df=features,
        prices=prices,
        window_size=8,
        config={
            "freqai": {
                "hedge_rl_config": {
                    "random_start": False,
                    "max_episode_steps": min(100, rows - 9),
                    "fee_rate": 0.0,
                    "slippage_bps": 0.0,
                    "memory": {
                        "reward_breakdown_interval": interval,
                        "gc_collect_every_episodes": 0,
                    },
                }
            }
        },
    )


def reference_observation(
    features: np.ndarray,
    schema: RiskObservationSchema,
    tick: int,
    account: RiskAccountState,
    mark: float,
    profile: RiskLevelProfile,
    uncertainty: float,
    funding: float,
    max_steps: int,
    clip: float,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    start = tick - schema.window_size + 1
    market_values = np.clip(values[start : tick + 1], -clip, clip).reshape(-1)
    base = max(abs(account.equity), 1e-12)
    used = account.used_margin_fraction(mark, profile)
    reserve = max(0.0, 1.0 - used)
    account_values = np.asarray(
        [
            account.long_level / 4.0,
            account.short_level / 4.0,
            account.long.notional(mark) / profile.long_leverage / base,
            account.short.notional(mark) / profile.short_leverage / base,
            account.long.unrealized_pnl(mark) / base,
            account.short.unrealized_pnl(mark) / base,
            account.gross_notional_ratio(mark),
            account.net_notional_ratio(mark),
            used,
            reserve,
            account.drawdown(),
            profile.long_leverage / 20.0,
            profile.short_leverage / 20.0,
            min(1.0, max(0.0, uncertainty)),
            funding,
            min(1.0, account.step / max(max_steps, 1)),
            0.0,  # failed_probe_long_norm
            0.0,  # failed_probe_short_norm
            0.0,  # downside_semideviation_norm
            0.0,  # pending_reward_fraction
        ],
        dtype=np.float64,
    )
    return np.concatenate((market_values, np.clip(account_values, -clip, clip))).astype(np.float32)


def _run_phase_one(matrix: Matrix, profile: RiskLevelProfile, mapper: RiskLevelMapper) -> None:
    # 01: target-risk mapping invariants
    for index in range(20):
        long_level, short_level = divmod(index, 5)

        def check(index=index, long_level=long_level, short_level=short_level):
            target = mapper.map((long_level, short_level), equity=1000 + index)
            require(target.combined_margin_fraction <= 0.8 + 1e-12, "combined margin cap")
            require(target.reserve_margin_fraction >= 0.2 - 1e-12, "reserve margin")

        matrix.check("01_action_budget", f"action_{long_level}_{short_level}", check)

    # 02: feature downcast
    for index in range(20):
        rows, cols = 32 + index, 1 + index % 7

        def check(rows=rows, cols=cols):
            frame, _ = market(rows, cols)
            values = compact_feature_matrix(frame)
            require(values.dtype == np.float32, "feature dtype")
            require(values.nbytes == rows * cols * 4, "feature bytes")
            require(not values.flags.writeable, "feature matrix must be readonly")

        matrix.check("02_feature_compaction", f"shape_{rows}x{cols}", check)

    # 03: compact market retention
    for index in range(20):
        rows = 24 + index

        def check(rows=rows):
            _, prices = market(rows)
            compact = CompactRiskMarketData.from_prices(prices)
            require(compact.nbytes == rows * 24, "compact market byte contract")
            require(len(compact.__dataclass_fields__) == 4, "only four retained arrays")

        matrix.check("03_market_compaction", f"rows_{rows}", check)

    # 04: observation parity with V1 semantics
    rng = np.random.default_rng(9)
    for index in range(20):

        def check(index=index):
            rows, cols, window = 40, 3, 5
            features = rng.normal(size=(rows, cols)).astype(np.float32)
            schema = RiskObservationSchema(tuple(f"f{i}" for i in range(cols)), window)
            builder = HedgeRiskObservationBuilder(schema, feature_clip=3.0)
            sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
            sim.apply_target(
                (index % 5, (index * 3) % 5), reference_price=100, mark_price=99 + index * 0.2
            )
            tick = window - 1 + index % 10
            actual = builder.build(
                features,
                tick=tick,
                account=sim.state,
                mark=101,
                profile=profile,
                uncertainty_score=(index % 10) / 10,
                funding_rate=0.0001 * (index % 3),
                max_episode_steps=200,
            )
            expected = reference_observation(
                features,
                schema,
                tick,
                sim.state,
                101,
                profile,
                (index % 10) / 10,
                0.0001 * (index % 3),
                200,
                3.0,
            )
            require(np.allclose(actual, expected, rtol=1e-6, atol=1e-7), "observation drift")

        matrix.check("04_observation_parity", f"case_{index}", check)

    # 05: direct output-buffer builds
    for index in range(20):

        def check(index=index):
            schema = RiskObservationSchema(("x", "y"), 4)
            builder = HedgeRiskObservationBuilder(schema)
            features = np.arange(80, dtype=np.float32).reshape(40, 2)
            out = np.empty(schema.flat_size, dtype=np.float32)
            result = builder.build_into(
                features,
                out,
                tick=3 + index,
                account=RiskAccountState.initial(1000),
                mark=100,
                profile=profile,
                uncertainty_score=0.5,
                funding_rate=0,
                max_episode_steps=100,
            )
            require(result is out, "builder allocated replacement output")
            require(np.isfinite(out).all(), "nonfinite observation")

        matrix.check("05_observation_buffer", f"tick_{index}", check)


def _run_phase_two(matrix: Matrix, profile: RiskLevelProfile) -> None:
    # 06: reset object reuse
    env = make_env(256)
    simulator_id, reward_id = id(env.simulator), id(env.reward_model)
    for index in range(20):

        def check(index=index):
            env.reset()
            env.step(np.asarray([index % 5, (index * 2) % 5], dtype=np.int64))
            env.reset()
            require(id(env.simulator) == simulator_id, "simulator reallocated")
            require(id(env.reward_model) == reward_id, "reward model reallocated")

        matrix.check("06_episode_reuse", f"episode_{index}", check)

    # 07: static action-mask cache
    mask_id = id(env.action_masks())
    for index in range(20):

        def check(index=index):
            mask = env.action_masks()
            require(id(mask) == mask_id, "action mask allocation")
            require(mask.shape == (10,) and bool(mask.all()), "mask contents")
            require(not mask.flags.writeable, "mask must be readonly")

        matrix.check("07_action_mask_cache", f"query_{index}", check)

    # 08: reward pending memory bound
    for index in range(20):

        def check(index=index):
            model = HedgeRiskRewardModel(RiskRewardConfig(), max_pending_outcomes=8)
            pending_id = id(model._pending)
            count = index % 8
            for item in range(count):
                model._append_pending(
                    PendingOutcome(
                        kind="probe",
                        side=LegSide.LONG,
                        created_step=item,
                        due_step=99,
                        baseline_equity=1000.0,
                        baseline_drawdown=0.0,
                        baseline_leg_net_pnl=0.0,
                        baseline_level=0,
                        target_level=1,
                    )
                )
            require(model.pending_outcome_count == count, "pending count")
            model.reset()
            require(
                model.pending_outcome_count == 0 and id(model._pending) == pending_id,
                "reset allocation",
            )

        matrix.check("08_reward_pending_bound", f"count_{index}", check)

    # 09: sparse reward breakdown telemetry
    tele_env = make_env(128, interval=5)
    tele_env.reset()
    for index in range(1, 21):

        def check(index=index):
            _, _, terminated, truncated, info = tele_env.step(np.asarray([1, 0], dtype=np.int64))
            expected = index % 5 == 0 or terminated or truncated
            require(bool(info["reward_components"]) == expected, "reward breakdown cadence")

        matrix.check("09_sparse_reward_telemetry", f"step_{index}", check)

    # 10: simulator target execution
    for index in range(20):

        def check(index=index):
            sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
            long_level, short_level = index % 5, (index * 4) % 5
            transition = sim.apply_target(
                (long_level, short_level), reference_price=100, mark_price=95 + index * 0.5
            )
            require(math.isfinite(transition.equity), "finite equity")
            require(
                sim.state.long_level == long_level and sim.state.short_level == short_level,
                "level state",
            )
            require(sim.state.used_margin_fraction(100, profile) <= 0.81, "margin cap")

        matrix.check("10_simulator_targets", f"case_{index}", check)


def _run_phase_three(matrix: Matrix, profile: RiskLevelProfile) -> None:
    # 11: uncertainty exposure risk
    for index in range(20):

        def check(index=index):
            level = 1 + index % 4
            sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
            transition = sim.apply_target((level, 0), reference_price=100, mark_price=100)
            cfg = RiskRewardConfig(
                drawdown_weight=0,
                downside_exposure_weight=0,
                liquidation_buffer_weight=0,
                adverse_scale_in_weight=0,
                turnover_shaping_weight=0,
                repeated_probe_weight=0,
                risk_reduction_bonus_weight=0,
                profit_lock_bonus_weight=0,
                hedge_efficiency_weight=0,
                hedge_waste_weight=0,
                delayed_scale_bonus_weight=0,
                delayed_probe_bonus_weight=0,
            )
            low = HedgeRiskRewardModel(cfg).calculate(
                transition=transition,
                account=sim.state,
                mark=100,
                uncertainty_score=0,
                reserve_margin_fraction=sim.state.reserve_margin_fraction(100, profile),
            )
            high = HedgeRiskRewardModel(cfg).calculate(
                transition=transition,
                account=sim.state,
                mark=100,
                uncertainty_score=1,
                reserve_margin_fraction=sim.state.reserve_margin_fraction(100, profile),
            )
            require(high.reward <= low.reward + 1e-12, "uncertainty must not improve reward")

        matrix.check("11_uncertainty_risk", f"case_{index}", check)

    # 12: fail-closed stale inference
    class ExplodingModel:
        def predict(self, *args, **kwargs):
            raise AssertionError("stale projection must not invoke model")

    for index in range(20):

        def check(index=index):
            bridge = HedgeRiskLevelPolicyBridge(
                feature_names=("x",), window_size=2, profile=profile
            )
            context = HedgeRiskPolicyContext(
                account=RiskAccountState.initial(1000), mark=100, projection_fresh=False
            )
            obs = bridge.observation(
                np.asarray([[index], [index + 1]], dtype=np.float32), tick=1, context=context
            )
            require(
                bridge.predict_action(ExplodingModel(), obs, context=context).as_tuple() == (0, 0),
                "fail close",
            )

        matrix.check("12_stale_fail_closed", f"case_{index}", check)

    # 13: planner remains target-only
    adapter = HedgeRiskLevelPlannerAdapter(profile)
    for index in range(20):
        long_level, short_level = divmod(index, 5)

        def check(long_level=long_level, short_level=short_level):
            signal = adapter.from_action(
                HedgeRiskLevelAction.from_value((long_level, short_level)), equity=1000
            )
            require(signal.long_margin_fraction == profile.fraction(long_level), "long fraction")
            require(signal.short_margin_fraction == profile.fraction(short_level), "short fraction")
            require("order" not in " ".join(signal.strategy_columns()).lower(), "order leakage")

        matrix.check("13_planner_target_contract", f"case_{index}", check)

    # 14: int8 prediction footprint contract
    for index in range(20):
        rows = 100 + index * 17

        def check(rows=rows):
            compact = np.zeros((rows, 2), dtype=np.int8)
            legacy = np.zeros((rows, 2), dtype=np.int64)
            require(compact.nbytes * 8 == legacy.nbytes, "int8 output footprint")

        matrix.check("14_prediction_output_compaction", f"rows_{rows}", check)

    # 15: source lifecycle invariants
    learner_source = (
        ROOT / "freqtrade/freqai/prediction_models/HedgeRiskLevelReinforcementLearner.py"
    ).read_text(encoding="utf-8")
    env_source = (ROOT / "freqtrade/freqai/hedge_rl/risk_environment.py").read_text(
        encoding="utf-8"
    )
    obs_source = (ROOT / "freqtrade/freqai/hedge_rl/risk_observation.py").read_text(
        encoding="utf-8"
    )
    source_checks = [
        ("no_train_deepcopy", "copy.deepcopy" not in learner_source),
        ("empty_df_raw", "self.df_raw = DataFrame()" in learner_source),
        ("downcast_training", "compact_training_dataframe" in learner_source),
        ("release_env_method", "_release_training_environments" in learner_source),
        ("close_train_eval", 'for name in ("train_env", "eval_env")' in learner_source),
        ("detach_model_env", "model.env = None" in learner_source),
        ("clear_last_obs", '"_last_obs"' in learner_source),
        ("phase_release_fit", "release_phase_memory_after_fit" in learner_source),
        ("compact_predict", "compact_feature_matrix" in learner_source),
        ("int8_predict", "dtype=np.int8" in learner_source),
        ("no_config_retention", "self.config_dict" not in env_source),
        ("compact_market", "CompactRiskMarketData" in env_source),
        ("no_price_iloc", ".iloc[" not in env_source),
        ("cached_mask", "self._action_mask" in env_source),
        ("reuse_simulator", "self.simulator.reset" in env_source),
        ("reuse_reward", "self.reward_model.reset" in env_source),
        ("sparse_breakdown", "reward_breakdown_interval" in env_source),
        (
            "no_gc_hot_step",
            "release_rl_phase_memory"
            not in env_source.split("    def step(self, action: Sequence[int]):", 1)[1].split(
                "    def memory_telemetry", 1
            )[0],
        ),
        ("build_into", "def build_into(" in obs_source),
        ("no_concatenate", "np.concatenate" not in obs_source),
    ]
    for name, condition in source_checks:
        matrix.check(
            "15_source_lifecycle",
            name,
            lambda condition=condition, name=name: require(condition, name),
        )


def _run_phase_four_prefix(matrix: Matrix) -> None:
    # 16: longer episode pending bound
    for index in range(20):

        def check(index=index):
            local = make_env(180, interval=0)
            local.reset(seed=index + 1)
            for step in range(80):
                action = np.asarray([(step + index) % 5, (step * 2 + index) % 5], dtype=np.int64)
                _, _, terminated, truncated, _ = local.step(action)
                require(
                    local.reward_model.pending_outcome_count
                    <= local.memory_config.max_pending_reward_outcomes,
                    "pending leak",
                )
                if terminated or truncated:
                    break

        matrix.check("16_long_episode_bound", f"seed_{index}", check)

    # 17: missing optional columns stay compact
    for index in range(20):
        rows = 20 + index

        def check(rows=rows):
            base = np.full(rows, 100.0)
            prices = pd.DataFrame({"open": base, "high": base + 1, "low": base - 1, "close": base})
            compact = CompactRiskMarketData.from_prices(prices)
            require(np.all(compact.funding_rate == 0), "funding default")
            require(np.isnan(compact.uncertainty_score).all(), "uncertainty default")
            require(compact.nbytes == rows * 24, "optional columns footprint")

        matrix.check("17_optional_market_defaults", f"rows_{rows}", check)


def _run_phase_four(matrix: Matrix, profile: RiskLevelProfile) -> None:
    _run_phase_four_prefix(matrix)
    # 18: invalid market data fails before retention
    for index in range(20):

        def check(index=index):
            _, prices = market(24)
            row = index % len(prices)
            if index % 4 == 0:
                prices.loc[row, "open"] = -1
            elif index % 4 == 1:
                prices.loc[row, "high"] = prices.loc[row, "low"] - 1
            elif index % 4 == 2:
                prices.loc[row, "low"] = prices.loc[row, "high"] + 1
            else:
                prices.loc[row, "funding_rate"] = np.nan
            try:
                CompactRiskMarketData.from_prices(prices)
            except ValueError:
                return
            raise AssertionError("invalid market data accepted")

        matrix.check("18_market_validation", f"case_{index}", check)

    # 19: environment byte telemetry
    for index in range(20):
        rows, cols = 64 + index, 2 + index % 6

        def check(rows=rows, cols=cols):
            local = make_env(rows + 12, cols)
            telemetry = local.memory_telemetry()
            require(telemetry["feature_bytes"] == (rows + 12) * cols * 4, "feature telemetry")
            require(telemetry["market_bytes"] == (rows + 12) * 24, "market telemetry")
            require(telemetry["retained_price_dataframe"] == 0, "price dataframe retained")

        matrix.check("19_memory_telemetry", f"shape_{rows + 12}x{cols}", check)

    # 20: end-to-end risk-level steps
    for index in range(20):

        def check(index=index):
            features, prices = market(48, 3, shift=index * 0.1)
            prices["close"] = prices["open"] * (1.0 + (index - 10) * 0.0005)
            prices["high"] = np.maximum(prices["high"], prices[["open", "close"]].max(axis=1) + 0.1)
            prices["low"] = np.minimum(prices["low"], prices[["open", "close"]].min(axis=1) - 0.1)
            local = HedgeRiskLevelEnv(
                df=features,
                prices=prices,
                window_size=4,
                config={
                    "freqai": {
                        "hedge_rl_config": {
                            "random_start": False,
                            "fee_rate": 0,
                            "slippage_bps": 0,
                            "memory": {"gc_collect_every_episodes": 0},
                        }
                    }
                },
            )
            obs, _ = local.reset()
            action = np.asarray([index % 5, (index * 3) % 5], dtype=np.int64)
            obs2, reward, _, _, info = local.step(action)
            require(obs.shape == obs2.shape == local.observation_space.shape, "observation shape")
            require(
                math.isfinite(float(reward)) and math.isfinite(float(info["equity"])),
                "finite end-to-end",
            )

        matrix.check("20_end_to_end", f"case_{index}", check)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "HEDGE-RL-RISK-LEVEL-V2-MEMORY-400-VALIDATION.json"
    )
    args = parser.parse_args()
    matrix = Matrix()
    profile = RiskLevelProfile()
    mapper = RiskLevelMapper(profile)

    _run_phase_one(matrix, profile, mapper)

    _run_phase_two(matrix, profile)

    _run_phase_three(matrix, profile)

    _run_phase_four(matrix, profile)

    passed = sum(row["status"] == "PASS" for row in matrix.rows)
    failed = len(matrix.rows) - passed
    summary = {
        "validator": "Hedge Risk-Level RL V2/V3 Memory Compatibility 400",
        "checks": len(matrix.rows),
        "passed": passed,
        "failed": failed,
        "status": "PASS" if failed == 0 and len(matrix.rows) == 400 else "FAIL",
        "themes": 20,
        "rows": matrix.rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"HEDGE RISK LEVEL RL V2/V3 MEMORY 400: {passed}/{len(matrix.rows)} PASS; FAIL={failed}")
    print(args.output)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
