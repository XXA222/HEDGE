#!/usr/bin/env python3
"""400-round consolidation gate for Hedge Risk-Level RL.

This validator focuses on merge-sensitive contracts that span the V3 action/reward overlay
and the V1.6 adaptive/memory mainline. It is deterministic and does not require exchange I/O.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freqtrade.freqai.hedge_rl.risk_environment import HedgeRiskLevelEnv  # noqa: E402
from freqtrade.freqai.hedge_rl.risk_levels import (  # noqa: E402
    HedgeRiskLevelAction,
    RiskActionTopology,
    RiskLevelProfile,
)
from freqtrade.freqai.hedge_rl.risk_memory import HedgeRLMemoryConfig  # noqa: E402
from freqtrade.freqai.hedge_rl.risk_portfolio import TargetLevelPortfolioSimulator  # noqa: E402
from freqtrade.freqai.hedge_rl.risk_reward import (  # noqa: E402
    HedgeRiskRewardModel,
    RiskRewardConfig,
)
from freqtrade.hedge.config_schema_extension import extend_config_schema  # noqa: E402


@dataclass(slots=True)
class Result:
    index: int
    category: str
    name: str
    status: str
    detail: str = ""


class Matrix:
    def __init__(self) -> None:
        self.results: list[Result] = []

    def check(self, category: str, name: str, fn: Callable[[], Any]) -> None:
        index = len(self.results) + 1
        try:
            detail = fn()
        except Exception as exc:  # validator boundary: aggregate all 400 rounds
            self.results.append(
                Result(index, category, name, "FAIL", f"{type(exc).__name__}: {exc}")
            )
        else:
            self.results.append(
                Result(index, category, name, "PASS", "" if detail is None else str(detail))
            )


def require(condition: bool, detail: Any = "") -> str:
    if not condition:
        raise AssertionError(detail or "condition failed")
    return str(detail)


def make_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"freqai": {"$ref": "#/definitions/freqai"}},
        "definitions": {
            "freqai": {
                "type": "object",
                "properties": {
                    "rl_config": {"type": "object", "properties": {}},
                },
            }
        },
    }


def reward_step(
    sim: TargetLevelPortfolioSimulator,
    reward: HedgeRiskRewardModel,
    action: tuple[int, int],
    *,
    ref: float,
    mark: float,
    uncertainty: float,
    funding: float,
):
    transition = sim.apply_target(
        action,
        reference_price=ref,
        mark_price=mark,
        funding_rate=funding,
    )
    reserve = sim.state.reserve_margin_fraction(mark, sim.profile)
    breakdown = reward.calculate(
        transition=transition,
        account=sim.state,
        mark=mark,
        uncertainty_score=uncertainty,
        reserve_margin_fraction=reserve,
    )
    return transition, breakdown


def _run_action_portfolio_environment(
    matrix: Matrix,
    profile: RiskLevelProfile,
    topology: RiskActionTopology,
) -> None:
    # 1-50: complete action encoding plus representative topology geometry.
    for joint_id in range(25):
        matrix.check(
            "action-contract",
            f"joint roundtrip {joint_id}",
            lambda joint_id=joint_id: require(
                HedgeRiskLevelAction.from_joint_id(joint_id).joint_id == joint_id,
                joint_id,
            ),
        )
    for joint_id in range(25):
        before = HedgeRiskLevelAction.from_joint_id(joint_id)
        after = HedgeRiskLevelAction.from_joint_id(24 - joint_id)
        matrix.check(
            "action-topology",
            f"topology mirror {joint_id}",
            lambda before=before, after=after: require(
                topology.transition(before, after).manhattan_distance
                == abs(int(after.long_level) - int(before.long_level))
                + abs(int(after.short_level) - int(before.short_level)),
                (before.as_tuple(), after.as_tuple()),
            ),
        )

    # 51-150: randomized accounting/reward paths across 100 deterministic seeds.
    for seed in range(100):

        def portfolio_case(seed=seed):
            rng = np.random.default_rng(seed)
            sim = TargetLevelPortfolioSimulator(
                1000.0,
                profile=profile,
                fee_rate=0.0004,
                slippage_bps=float(seed % 17),
            )
            reward = HedgeRiskRewardModel(RiskRewardConfig(), max_pending_outcomes=64)
            price = 100.0
            for _ in range(12):
                ref = max(price * (1.0 + float(rng.normal(0.0, 0.003))), 1.0)
                mark = max(ref * (1.0 + float(rng.normal(0.0, 0.006))), 1.0)
                action = (int(rng.integers(0, 5)), int(rng.integers(0, 5)))
                transition, breakdown = reward_step(
                    sim,
                    reward,
                    action,
                    ref=ref,
                    mark=mark,
                    uncertainty=float(rng.random()),
                    funding=float(rng.normal(0.0, 0.00002)),
                )
                values = (
                    sim.state.cash_balance,
                    sim.state.equity,
                    transition.traded_notional,
                    breakdown.reward,
                    breakdown.unclipped_reward,
                )
                require(all(math.isfinite(float(v)) for v in values), values)
                require(0 <= sim.state.long_level <= 4 and 0 <= sim.state.short_level <= 4)
                require(reward.pending_outcome_count <= reward.max_pending_outcomes)
                price = mark
            return f"equity={sim.state.equity:.6f};pending={reward.pending_outcome_count}"

        matrix.check("portfolio-reward", f"seed {seed}", portfolio_case)

    # 151-250: environment lifecycle; cap=8 must safely cover simultaneous dual-leg ramp.
    for seed in range(100):

        def environment_case(seed=seed):
            rows = 48 + (seed % 8)
            x = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
            features = pd.DataFrame({"f1": x, "f2": x * x, "f3": np.sin(x)})
            base = 100.0 + np.linspace(0.0, 1.0 + seed / 1000.0, rows)
            prices = pd.DataFrame(
                {
                    "open": base,
                    "high": base + 1.0,
                    "low": base - 1.0,
                    "close": base + 0.1,
                    "volume": np.ones(rows),
                }
            )
            env = HedgeRiskLevelEnv(
                df=features,
                prices=prices,
                window_size=8,
                config={
                    "freqai": {
                        "hedge_rl_config": {
                            "random_start": False,
                            "max_episode_steps": 16,
                            "slippage_bps": float(seed % 11),
                            "memory": {
                                "max_pending_reward_outcomes": 8,
                                "reward_breakdown_interval": 0,
                                "gc_collect_every_episodes": 0,
                            },
                        }
                    }
                },
            )
            obs, _ = env.reset(seed=seed)
            require(obs.dtype == np.float32)
            for level in (1, 2, 3, 4):
                obs, reward_value, terminated, truncated, _ = env.step(
                    np.asarray([level, level], dtype=np.int64)
                )
                require(np.isfinite(obs).all())
                require(math.isfinite(float(reward_value)))
                if terminated or truncated:
                    break
            require(env.reward_model.pending_outcome_count <= 8)
            return env.memory_telemetry()

        matrix.check("environment-memory", f"dual-leg ramp {seed}", environment_case)


def _invalid_extreme_slippage_cases(
    profile: RiskLevelProfile,
) -> list[tuple[str, Callable[[], Any]]]:
    cases: list[tuple[str, Callable[[], Any]]] = []
    for value in range(10002, 10029):

        def invalid_slippage_extreme(value=value):
            try:
                TargetLevelPortfolioSimulator(1000, profile=profile, slippage_bps=float(value))
            except ValueError:
                return value
            raise AssertionError(f"accepted invalid slippage: {value}")

        cases.append((f"slippage extreme {value}", invalid_slippage_extreme))
    return cases


def _invalid_boundary_cases(profile: RiskLevelProfile) -> list[tuple[str, Callable[[], Any]]]:
    invalids: list[tuple[str, Callable[[], Any]]] = []
    for value in (True, -1, 25, 25.0, 1.5, float("nan"), float("inf"), "1.5", "nan", None):

        def invalid_joint(value=value):
            try:
                HedgeRiskLevelAction.from_joint_id(value)  # type: ignore[arg-type]
            except (TypeError, ValueError, OverflowError):
                return value
            raise AssertionError(f"accepted invalid joint id: {value!r}")

        invalids.append((f"joint {value!r}", invalid_joint))
    for value in (10000.0, 10001.0, 20000.0, float("inf"), float("nan")):

        def invalid_slippage(value=value):
            try:
                TargetLevelPortfolioSimulator(1000, profile=profile, slippage_bps=value)
            except ValueError:
                return value
            raise AssertionError(f"accepted invalid slippage: {value!r}")

        invalids.append((f"slippage {value!r}", invalid_slippage))
    for value in range(8):

        def invalid_cap(value=value):
            try:
                HedgeRLMemoryConfig(max_pending_reward_outcomes=value)
            except ValueError:
                return value
            raise AssertionError(f"accepted unsafe production pending cap: {value}")

        invalids.append((f"pending cap {value}", invalid_cap))
    invalids.extend(_invalid_extreme_slippage_cases(profile))
    return invalids


def _run_invalid_boundaries(matrix: Matrix, profile: RiskLevelProfile) -> None:
    invalids = _invalid_boundary_cases(profile)
    require(len(invalids) == 50, len(invalids))
    for name, fn in invalids:
        matrix.check("fail-closed-boundaries", name, fn)


def _run_boundary_and_integration(matrix: Matrix, profile: RiskLevelProfile) -> None:
    _run_invalid_boundaries(matrix, profile)
    # 301-350: schema/idempotence and merged-source contract.
    root = Path(__file__).resolve().parents[1]
    required_files = (
        "freqtrade/freqai/hedge_rl/risk_levels.py",
        "freqtrade/freqai/hedge_rl/risk_reward.py",
        "freqtrade/freqai/hedge_rl/risk_memory.py",
        "freqtrade/freqai/hedge_rl/risk_runtime.py",
        "freqtrade/freqai/hedge_rl/risk_projection_adapter.py",
        "freqtrade/freqai/prediction_models/HedgeRiskLevelReinforcementLearner.py",
    )
    for i in range(50):

        def integration_case(i=i):
            schema = make_schema()
            extend_config_schema(schema)
            extend_config_schema(schema)
            freqai = schema["definitions"]["freqai"]["properties"]
            runtime = freqai["hedge_rl_config"]["properties"]
            require(runtime["slippage_bps"]["exclusiveMaximum"] == 10000.0)
            require(runtime["memory"]["properties"]["max_pending_reward_outcomes"]["minimum"] == 8)
            require("adaptive_cpu" in runtime)
            for relative in required_files:
                require((root / relative).is_file(), relative)
            if i % 2 == 0:
                source = (root / required_files[-1]).read_text(encoding="utf-8")
                require('parameters["device"] = "cpu"' in source)
                lowered = source.lower()
                require("freqtrade.freqai.hprl" not in lowered and "\nimport hprl" not in lowered)
            return "schema+source"

        matrix.check("merge-integration", f"integration {i}", integration_case)


def _run_deterministic_replay(matrix: Matrix, profile: RiskLevelProfile) -> None:
    # 351-400: paired deterministic replay must be bitwise-stable at the state/action level.
    for seed in range(50):

        def deterministic_case(seed=seed):
            rng = np.random.default_rng(10_000 + seed)
            actions = [(int(rng.integers(0, 5)), int(rng.integers(0, 5))) for _ in range(10)]
            refs = [100.0 * (1.0 + 0.001 * i) for i in range(10)]
            marks = [ref * (1.0 + float(rng.normal(0, 0.002))) for ref in refs]
            outputs = []
            for _copy in range(2):
                sim = TargetLevelPortfolioSimulator(
                    1000.0, profile=profile, fee_rate=0.0004, slippage_bps=3.0
                )
                reward = HedgeRiskRewardModel(RiskRewardConfig())
                trace = []
                for action, ref, mark in zip(actions, refs, marks, strict=True):
                    transition, breakdown = reward_step(
                        sim,
                        reward,
                        action,
                        ref=ref,
                        mark=mark,
                        uncertainty=0.4,
                        funding=0.0,
                    )
                    trace.append(
                        (
                            sim.state.cash_balance,
                            sim.state.equity,
                            transition.traded_notional,
                            breakdown.reward,
                            sim.state.long_level,
                            sim.state.short_level,
                        )
                    )
                outputs.append(trace)
            require(outputs[0] == outputs[1], seed)
            return f"steps={len(outputs[0])}"

        matrix.check("determinism", f"replay {seed}", deterministic_case)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    matrix = Matrix()
    profile = RiskLevelProfile()
    topology = RiskActionTopology(profile)

    _run_action_portfolio_environment(matrix, profile, topology)

    _run_boundary_and_integration(matrix, profile)
    _run_deterministic_replay(matrix, profile)

    if len(matrix.results) != 400:
        raise RuntimeError(f"validator programming error: {len(matrix.results)} rounds")
    passed = sum(item.status == "PASS" for item in matrix.results)
    failed = len(matrix.results) - passed
    payload = {
        "schema": "freqtrade-hedge-risklevel-rl-consolidated-400-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "rounds": len(matrix.results),
        "passed": passed,
        "failed": failed,
        "status": "PASS" if failed == 0 else "FAIL",
        "results": [asdict(item) for item in matrix.results],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    print(
        f"HEDGE RISKLEVEL RL CONSOLIDATED 400: {passed}/{len(matrix.results)} PASS; FAIL={failed}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
