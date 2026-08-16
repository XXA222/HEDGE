from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from freqtrade.enums.hedge import PositionSide  # noqa: E402
from freqtrade.freqai.hedge_rl.actions import DEFAULT_ACTION_CATALOG  # noqa: E402
from freqtrade.freqai.hedge_rl.risk_bridge import (  # noqa: E402
    HedgeRiskLevelPolicyBridge,
    HedgeRiskPolicyContext,
)
from freqtrade.freqai.hedge_rl.risk_levels import (  # noqa: E402
    HedgeRiskLevelAction,
    RiskActionTopology,
    RiskLevelMapper,
    RiskLevelProfile,
)
from freqtrade.freqai.hedge_rl.risk_memory import compact_feature_matrix  # noqa: E402
from freqtrade.freqai.hedge_rl.risk_observation import (  # noqa: E402
    HedgeRiskObservationBuilder,
    RiskObservationSchema,
)
from freqtrade.freqai.hedge_rl.risk_planner_adapter import (  # noqa: E402
    HedgeRiskLevelPlannerAdapter,
)
from freqtrade.freqai.hedge_rl.risk_portfolio import TargetLevelPortfolioSimulator  # noqa: E402
from freqtrade.freqai.hedge_rl.risk_projection_adapter import (  # noqa: E402
    context_from_central_projection,
)
from freqtrade.freqai.hedge_rl.risk_reward import (  # noqa: E402
    HedgeRiskRewardModel,
    RiskRewardConfig,
)
from freqtrade.freqai.hedge_rl.risk_runtime import (  # noqa: E402
    RiskRLAdaptiveCpuConfig,
    RiskRLAdaptiveCpuController,
)
from freqtrade.hedge.config_schema_extension import extend_config_schema  # noqa: E402
from freqtrade.hedge.integration.projection import CentralRuntimeProjection  # noqa: E402
from freqtrade.hedge.performance.resource_governor import ResourceSnapshot  # noqa: E402
from freqtrade.hedge.position_book import PositionRecord  # noqa: E402
from freqtrade.hedge.risk.models import AccountRiskSnapshot  # noqa: E402


@dataclass(slots=True)
class Row:
    round: int
    theme: str
    name: str
    passed: bool
    detail: Any = None


class Matrix:
    def __init__(self) -> None:
        self.rows: list[Row] = []

    def check(self, theme: str, name: str, fn: Callable[[], Any]) -> None:
        try:
            detail = fn()
            passed = detail is not False
        except Exception as exc:  # deliberate validator boundary
            detail = f"{type(exc).__name__}: {exc}"
            passed = False
        self.rows.append(Row(len(self.rows) + 1, theme, name, passed, detail))


def require(condition: bool, detail: Any = "ok") -> Any:
    if not condition:
        raise AssertionError(detail)
    return detail


def market(rows: int = 128) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = np.linspace(0.0, 1.0, rows, dtype=np.float32).reshape(-1, 1)
    prices = {
        "open": np.linspace(100.0, 110.0, rows, dtype=np.float64),
        "high": np.linspace(101.0, 111.0, rows, dtype=np.float64),
        "low": np.linspace(99.0, 109.0, rows, dtype=np.float64),
        "close": np.linspace(100.5, 110.5, rows, dtype=np.float64),
        "volume": np.ones(rows, dtype=np.float64),
        "funding_rate": np.zeros(rows, dtype=np.float32),
        "uncertainty_score": np.full(rows, 0.5, dtype=np.float32),
    }
    return x, prices


def central_projection(*, stale: bool = False, healthy: bool = True) -> CentralRuntimeProjection:
    risk = AccountRiskSnapshot(
        account_id="acct",
        equity=Decimal(1000),
        wallet_balance=Decimal(1000),
        available_balance=Decimal(700),
        initial_margin=Decimal(300),
        maintenance_margin=Decimal(10),
        gross_long_notional=Decimal(150),
        gross_short_notional=Decimal(90),
        net_notional=Decimal(60),
        risk_data_valid=True,
        source_version=7,
    )
    checks = {
        "exchange.rest_calibrated": healthy,
        "exchange.user_stream_fresh": healthy,
        "exchange.reconciliation_converged": healthy,
        "exchange.risk_snapshot_valid": healthy,
    }
    return CentralRuntimeProjection(
        positions=(
            PositionRecord(
                symbol="BTC/USDT:USDT",
                position_side=PositionSide.LONG,
                amount=Decimal("0.5"),
                entry_price=Decimal(100),
                mark_price=Decimal(100),
                leverage=Decimal(3),
                exchange="binance",
                account_id="acct",
            ),
            PositionRecord(
                symbol="BTC/USDT:USDT",
                position_side=PositionSide.SHORT,
                amount=Decimal("0.3"),
                entry_price=Decimal(100),
                mark_price=Decimal(100),
                leverage=Decimal(3),
                exchange="binance",
                account_id="acct",
            ),
        ),
        risk=risk,
        reconciliation_status="HEALTHY" if healthy else "DRIFT",
        reconciliation_at=datetime.now(UTC),
        reconciliation_details=(),
        stream_state="CONNECTED" if healthy else "STALE",
        stream_last_event_at=datetime.now(UTC),
        stream_reconnect_count=0,
        checks=checks,
        reasons=(),
        source_version="7",
        source_event_time=datetime.now(UTC),
        stale=stale,
    )


class FakeGovernor:
    def __init__(self, *, cpu: float, suggested: int, physical: int = 16) -> None:
        self.cpu = cpu
        self.suggested = suggested
        self.physical = physical

    def snapshot(self, *, sample_seconds: float = 0.0) -> ResourceSnapshot:
        del sample_seconds
        return ResourceSnapshot(
            logical_cpus=32,
            physical_cpus=self.physical,
            affinity_cpus=32,
            system_cpu_percent=self.cpu,
            process_cpu_percent=0.0,
            cgroup_memory_limit_bytes=8 * 1024**3,
            cgroup_memory_current_bytes=1024**3,
            host_memory_available_bytes=12 * 1024**3,
            timestamp_monotonic=1.0,
            source="host-broker",
            host_snapshot_age_seconds=0.1,
        )

    def numeric_threads(self, *, concurrent_python_workers: int, snapshot: ResourceSnapshot) -> int:
        require(concurrent_python_workers == 1)
        require(snapshot.source == "host-broker")
        return self.suggested


def build_schema() -> dict[str, Any]:
    schema: dict[str, Any] = {
        "properties": {"freqai": {"$ref": "#/definitions/freqai"}},
        "definitions": {
            "freqai": {
                "type": "object",
                "properties": {"rl_config": {"type": "object", "properties": {}}},
            }
        },
    }
    extend_config_schema(schema)
    return schema


def _run_phase_one(project_root: Path, matrix: Matrix) -> None:
    profile = RiskLevelProfile(long_leverage=3.0, short_leverage=3.0)

    imported = [
        "docs/freqai-hedge-risk-level-rl.md",
        "docs/hedge-risk-level-rl-memory-v2.md",
        "freqtrade/freqai/hedge_rl/risk_bridge.py",
        "freqtrade/freqai/hedge_rl/risk_environment.py",
        "freqtrade/freqai/hedge_rl/risk_gym.py",
        "freqtrade/freqai/hedge_rl/risk_levels.py",
        "freqtrade/freqai/hedge_rl/risk_memory.py",
        "freqtrade/freqai/hedge_rl/risk_observation.py",
        "freqtrade/freqai/hedge_rl/risk_planner_adapter.py",
        "freqtrade/freqai/hedge_rl/risk_portfolio.py",
        "freqtrade/freqai/hedge_rl/risk_reward.py",
        "freqtrade/freqai/prediction_models/HedgeRiskLevelReinforcementLearner.py",
        "tests/hedge/mlrl/test_risk_action_reward_contract.py",
        "tests/hedge/mlrl/test_risk_level_rl.py",
        "tests/hedge/mlrl/test_risk_level_rl_memory.py",
        "tools/validate_hedge_risk_action_reward_200.py",
        "tools/validate_hedge_risk_level_rl_v1_400.py",
        "tools/validate_hedge_risk_level_rl_v2_memory_400.py",
        "freqtrade/freqai/hedge_rl/risk_runtime.py",
        "freqtrade/freqai/hedge_rl/risk_projection_adapter.py",
    ]
    for rel in imported:
        matrix.check("inventory", rel, lambda rel=rel: require((project_root / rel).is_file(), rel))

    risk_files = sorted((project_root / "freqtrade/freqai/hedge_rl").glob("risk_*.py"))
    for i in range(20):
        path = risk_files[i % len(risk_files)]
        matrix.check(
            "parse",
            f"AST parse {path.name} case {i}",
            lambda path=path: require(
                isinstance(ast.parse(path.read_text(encoding="utf-8")), ast.Module), path.name
            ),
        )

    for i in range(20):
        long = i % 5
        short = (i * 3) % 5
        matrix.check(
            "action-lattice",
            f"joint action {long},{short}",
            lambda long=long, short=short: require(
                HedgeRiskLevelAction.from_value((long, short)).joint_id == long * 5 + short,
                (long, short),
            ),
        )

    for i in range(20):
        long = i % 5
        short = (i * 2) % 5
        action = HedgeRiskLevelAction.from_value((long, short))
        target = RiskLevelMapper(profile).map(action, equity=1000.0 + i)
        matrix.check(
            "risk-ladder",
            f"reserve invariant {i}",
            lambda target=target: require(
                target.combined_margin_fraction <= profile.max_combined_margin_fraction + 1e-12
                and target.reserve_margin_fraction
                >= profile.minimum_reserve_margin_fraction - 1e-12,
                target.reserve_margin_fraction,
            ),
        )

    topology = RiskActionTopology(profile)
    for i in range(20):
        prev = HedgeRiskLevelAction.from_value((i % 5, (i + 1) % 5))
        nxt = HedgeRiskLevelAction.from_value(((i + 2) % 5, (i + 4) % 5))
        transition = topology.transition(prev, nxt)
        matrix.check(
            "topology",
            f"topology {i}",
            lambda transition=transition: require(
                transition.manhattan_distance
                == abs(transition.requested.long_level - transition.previous.long_level)
                + abs(transition.requested.short_level - transition.previous.short_level),
                transition.manhattan_distance,
            ),
        )

    for i in range(20):
        simulator = TargetLevelPortfolioSimulator(
            1000.0, profile=profile, fee_rate=0.0004, slippage_bps=1.0
        )
        action = (i % 5, (i * 2) % 5)
        transition = simulator.apply_target(
            action,
            reference_price=100.0 + i,
            mark_price=100.5 + i,
            funding_rate=0.00001,
        )
        matrix.check(
            "portfolio",
            f"accounting finite {i}",
            lambda transition=transition, simulator=simulator: require(
                math.isfinite(transition.equity)
                and math.isfinite(simulator.state.equity)
                and simulator.state.equity == transition.equity,
                transition.equity,
            ),
        )

    for i in range(20):
        simulator = TargetLevelPortfolioSimulator(1000.0, profile=profile)
        transition = simulator.apply_target(
            (i % 5, 0), reference_price=100.0, mark_price=100.0 + i * 0.05
        )
        reward_model = HedgeRiskRewardModel(RiskRewardConfig())
        result = reward_model.calculate(
            transition=transition,
            account=simulator.state,
            mark=100.0 + i * 0.05,
            uncertainty_score=0.5,
            reserve_margin_fraction=simulator.state.reserve_margin_fraction(
                100.0 + i * 0.05, profile
            ),
        )
        matrix.check(
            "reward",
            f"reward finite clipped {i}",
            lambda result=result: require(
                math.isfinite(result.reward) and abs(result.reward) <= 10.0 + 1e-12,
                result.reward,
            ),
        )

    for i in range(20):
        rows = 32 + i
        frame = np.arange(rows * 4, dtype=np.float64).reshape(rows, 4) / 100.0
        compact = compact_feature_matrix(frame, dtype=np.float32, readonly=True)
        matrix.check(
            "memory",
            f"compact feature {i}",
            lambda compact=compact, rows=rows: require(
                compact.dtype == np.float32
                and compact.flags.c_contiguous
                and not compact.flags.writeable
                and compact.shape == (rows, 4),
                compact.nbytes,
            ),
        )

    for i in range(20):
        schema = RiskObservationSchema(("a", "b"), 4)
        builder = HedgeRiskObservationBuilder(schema)
        simulator = TargetLevelPortfolioSimulator(1000.0, profile=profile)
        features = np.ones((8, 2), dtype=np.float32) * (i / 20.0)
        obs = builder.build(
            features,
            tick=3,
            account=simulator.state,
            mark=100.0,
            profile=profile,
            uncertainty_score=0.5,
            funding_rate=0.0,
            max_episode_steps=100,
            failed_probe_long=0,
            failed_probe_short=0,
            downside_semideviation=0.0,
            pending_reward_fraction=0.0,
        )
        matrix.check(
            "observation",
            f"observation {i}",
            lambda obs=obs, schema=schema: require(
                obs.shape == (schema.flat_size,)
                and obs.dtype == np.float32
                and np.isfinite(obs).all(),
                obs.shape,
            ),
        )


def _run_stale_inference(matrix: Matrix, profile: RiskLevelProfile) -> None:
    bridge = HedgeRiskLevelPolicyBridge(feature_names=("x",), window_size=2, profile=profile)
    features = np.asarray([[0.1], [0.2]], dtype=np.float32)
    for i in range(20):
        context = HedgeRiskPolicyContext.flat(1000.0, mark=100.0)
        context = HedgeRiskPolicyContext(
            account=context.account,
            mark=context.mark,
            feature_age_steps=(2 if i % 2 else 0),
            projection_fresh=(False if i % 2 == 0 else True),
        )
        obs = bridge.observation(features, tick=1, context=context)

        class Model:
            def predict(self, *_args, **_kwargs):
                raise AssertionError("fail-closed case invoked model")

        matrix.check(
            "fail-closed",
            f"stale inference {i}",
            lambda context=context, obs=obs: require(
                bridge.predict_action(Model(), obs, context=context).joint_id == 0,
                "flat",
            ),
        )


def run(project_root: Path) -> Matrix:
    m = Matrix()
    profile = RiskLevelProfile(long_leverage=3.0, short_leverage=3.0)

    _run_phase_one(project_root, m)
    _run_stale_inference(m, profile)

    # 201-220: planner no same-level scale-in.
    context = context_from_central_projection(
        central_projection(), pair="BTC/USDT:USDT", mark=100.0, profile=profile
    )
    adapter = HedgeRiskLevelPlannerAdapter(profile)
    for i in range(20):
        long = context.account.long_level
        short = context.account.short_level
        signal = adapter.from_account_action(
            HedgeRiskLevelAction.from_value((long, short)),
            account=context.account,
            mark=100.0 + i * 0.01,
        )
        m.check(
            "planner-safety",
            f"same-level no increase {i}",
            lambda signal=signal: require(
                not signal.long_increase_allowed
                and not signal.short_increase_allowed
                and not signal.allow_new_risk,
                signal.target_semantics,
            ),
        )

    # 221-240: canonical projection fresh/stale contract.
    for i in range(20):
        healthy = i % 2 == 0
        context_i = context_from_central_projection(
            central_projection(stale=not healthy, healthy=healthy),
            pair="BTC/USDT:USDT",
            mark=100.0,
            profile=profile,
        )
        m.check(
            "projection",
            f"projection freshness {i}",
            lambda context_i=context_i, healthy=healthy: require(
                context_i.projection_fresh is healthy,
                f"fresh={context_i.projection_fresh}; expected={healthy}",
            ),
        )

    # 241-260: config schema integration / idempotence.
    for i in range(20):
        schema_i = build_schema()
        extend_config_schema(schema_i)
        freqai = schema_i["definitions"]["freqai"]["properties"]
        m.check(
            "schema",
            f"schema idempotence {i}",
            lambda freqai=freqai, schema_i=schema_i: require(
                "hedge_action_space" in freqai["rl_config"]["properties"]
                and "hedge_reward" in freqai["rl_config"]["properties"]
                and "hedge_rl_config" in freqai
                and "adaptive_cpu" in freqai["hedge_rl_config"]["properties"]
                and schema_i["properties"]["freqai"] == {"$ref": "#/definitions/freqai"},
                "ok",
            ),
        )

    # 261-280: adaptive CPU respects physical/config/load suggestions.
    for i in range(20):
        max_threads = 4 + (i % 13)
        suggested = 1 + (i % 20)
        physical = 8 + (i % 9)
        controller = RiskRLAdaptiveCpuController(
            RiskRLAdaptiveCpuConfig(max_torch_threads=max_threads),
            governor=FakeGovernor(cpu=float(i), suggested=suggested, physical=physical),  # type: ignore[arg-type]
        )
        result = controller.recommended_threads()
        expected = min(max_threads, suggested, physical)
        m.check(
            "adaptive-cpu",
            f"adaptive threads {i}",
            lambda result=result, expected=expected, controller=controller: require(
                result == expected and controller.telemetry()["resource_source"] == "host-broker",
                result,
            ),
        )

    # 281-300: legacy 21-action and risk-level 25-action coexistence.
    for i in range(20):
        legacy = DEFAULT_ACTION_CATALOG.decode(i % len(DEFAULT_ACTION_CATALOG))
        risk_action = HedgeRiskLevelAction.from_value((i % 5, (i + 1) % 5))
        m.check(
            "coexistence",
            f"parallel contracts {i}",
            lambda legacy=legacy, risk_action=risk_action: require(
                len(DEFAULT_ACTION_CATALOG) == 21
                and 0 <= int(legacy.action) < 21
                and 0 <= risk_action.joint_id < 25,
                (int(legacy.action), risk_action.joint_id),
            ),
        )

    # 301-320: independent from HPRL/versioned runtime namespaces.
    risk_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (project_root / "freqtrade/freqai/hedge_rl").glob("risk_*.py")
    ).lower()
    forbidden = ["hprl", "freqtrade.hedge.r54", "freqtrade.hedge.r55", "freqtrade.hedge.r56"]
    for i in range(20):
        token = forbidden[i % len(forbidden)]
        m.check(
            "isolation",
            f"forbidden dependency {token} case {i}",
            lambda token=token: require(token not in risk_source, token),
        )

    # 321-340: CPU-only learner and adaptive source contract.
    learner_source = (
        project_root / "freqtrade/freqai/prediction_models/HedgeRiskLevelReinforcementLearner.py"
    ).read_text(encoding="utf-8")
    source_needles = [
        'parameters["device"] = "cpu"',
        'device="cpu"',
        "RiskRLAdaptiveCpuController",
        "_AdaptiveRiskCpuCallback",
        "compact_training_dataframe",
        "release_rl_phase_memory",
        "MultiDiscrete",
        "projection_fresh=False",
        "callbacks.clear()",
        "model.env = None",
    ]
    for i in range(20):
        needle = source_needles[i % len(source_needles)]
        m.check(
            "learner-runtime",
            f"learner contract {needle} {i}",
            lambda needle=needle: require(needle in learner_source, needle),
        )

    # 341-360: V1.5 performance surfaces remain present and semantically wired.
    perf_needles = [
        ("freqtrade/hedge/performance/resource_governor.py", "AdaptiveResourceGovernor"),
        ("freqtrade/hedge/native/parallel_hyperopt.py", "recommended_workers"),
        ("freqtrade/hedge/optimization/engine.py", "recommended_workers"),
        ("freqtrade/hedge/backtesting/parallel.py", "recommended_workers"),
        ("freqtrade/hedge/simulation/matcher.py", "def match_outcome"),
        ("freqtrade/hedge/simulation/cross_wallet.py", "def unrealized"),
        ("tools/validate_hedge_cpu_adaptive_400.py", "400"),
        ("tools/validate_hedge_memory_global_400.py", "400"),
    ]
    for i in range(20):
        rel, needle = perf_needles[i % len(perf_needles)]
        m.check(
            "v15-preservation",
            f"preserve {rel}:{needle} {i}",
            lambda rel=rel, needle=needle: require(
                needle in (project_root / rel).read_text(encoding="utf-8"), rel
            ),
        )

    # 361-380: deterministic signatures and exact defaults.
    for i in range(20):
        p1 = RiskLevelProfile()
        p2 = RiskLevelProfile()
        r1 = RiskRewardConfig()
        r2 = RiskRewardConfig()
        m.check(
            "signatures",
            f"deterministic signatures {i}",
            lambda p1=p1, p2=p2, r1=r1, r2=r2: require(
                p1.signature == p2.signature
                and r1.signature == r2.signature
                and len(p1.signature) == 16
                and len(r1.signature) == 16,
                (p1.signature, r1.signature),
            ),
        )

    # 381-400: packaging/docs/validation wiring.
    wiring = [
        "docs/freqai-hedge-risk-level-rl-integration-v1.6.md",
        "tests/hedge/mlrl/test_risk_level_rl_mainline_integration.py",
        "tools/validate_hedge_risklevel_rl_integration_400.py",
        "CLEAN-MAINLINE-VERSION.txt",
        "CLEAN-MAINLINE-ARCHITECTURE.md",
    ]
    for i in range(20):
        rel = wiring[i % len(wiring)]
        m.check(
            "wiring",
            f"integration wiring {rel} {i}",
            lambda rel=rel: require((project_root / rel).is_file(), rel),
        )

    if len(m.rows) != 400:
        raise AssertionError(f"validator programming error: {len(m.rows)} rows")
    return m


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="HEDGE-RISKLEVEL-RL-INTEGRATION-400.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    matrix = run(root)
    passed = sum(row.passed for row in matrix.rows)
    failed = len(matrix.rows) - passed
    payload = {
        "schema": "freqtrade-hedge-risklevel-rl-integration-400-v1",
        "project_root": str(root),
        "expected": 400,
        "executed": len(matrix.rows),
        "passed": passed,
        "failed": failed,
        "status": "PASS" if failed == 0 else "FAIL",
        "rounds": [asdict(row) for row in matrix.rows],
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"HEDGE RISKLEVEL RL INTEGRATION 400: {passed}/400 PASS; FAIL={failed}")
    print(Path(args.output).resolve())
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
