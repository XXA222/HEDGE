"""Run exactly 200 deterministic checks for Hedge Risk-Level action/reward V3.

The matrix is intentionally action/reward specific: 10 themes x 20 checks.  It is not
claimed as 200 independent training runs.  Each row executes a deterministic invariant
or scenario and records PASS/FAIL for delivery auditing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.freqai.hedge_rl.risk_levels import (  # noqa: E402
    HedgeRiskLevelAction,
    RiskActionTopology,
    RiskLevelMapper,
    RiskLevelProfile,
)
from freqtrade.freqai.hedge_rl.risk_observation import (  # noqa: E402
    HedgeRiskObservationBuilder,
    RiskObservationSchema,
)
from freqtrade.freqai.hedge_rl.risk_planner_adapter import (  # noqa: E402
    HedgeRiskLevelPlannerAdapter,
)
from freqtrade.freqai.hedge_rl.risk_portfolio import TargetLevelPortfolioSimulator  # noqa: E402
from freqtrade.freqai.hedge_rl.risk_reward import (  # noqa: E402
    HedgeRiskRewardModel,
    RiskRewardConfig,
)


@dataclass(slots=True)
class Row:
    round: int
    theme: str
    case: str
    status: str
    detail: str


class Matrix:
    def __init__(self) -> None:
        self.rows: list[Row] = []

    def check(self, theme: str, case: str, condition: bool, detail: str = "") -> None:
        self.rows.append(
            Row(
                round=len(self.rows) + 1,
                theme=theme,
                case=case,
                status="PASS" if bool(condition) else "FAIL",
                detail=detail,
            )
        )


def reward(model, sim, transition, *, mark, uncertainty=0.0, reserve=None):
    if reserve is None:
        reserve = sim.state.reserve_margin_fraction(mark, sim.profile)
    return model.calculate(
        transition=transition,
        account=sim.state,
        mark=mark,
        uncertainty_score=uncertainty,
        reserve_margin_fraction=reserve,
    )


def base_only(**overrides):
    values = {
        "drawdown_weight": 0,
        "downside_exposure_weight": 0,
        "downside_ewma_weight": 0,
        "uncertainty_exposure_weight": 0,
        "reserve_pressure_weight": 0,
        "liquidation_buffer_weight": 0,
        "wrong_level_loss_weight": 0,
        "position_success_bonus_weight": 0,
        "adverse_scale_in_weight": 0,
        "upward_jump_weight": 0,
        "level_churn_weight": 0,
        "turnover_shaping_weight": 0,
        "repeated_probe_weight": 0,
        "risk_reduction_bonus_weight": 0,
        "profit_lock_bonus_weight": 0,
        "hedge_efficiency_weight": 0,
        "hedge_waste_weight": 0,
        "delayed_scale_bonus_weight": 0,
        "delayed_probe_bonus_weight": 0,
    }
    values.update(overrides)
    return RiskRewardConfig(**values)


def _add_asymmetric_level_checks(matrix: Matrix, profile: RiskLevelProfile) -> None:
    adverse_moves = [0.999 - i * 0.001 for i in range(5)]
    round_index = 0
    for level in range(1, 5):
        for ratio in adverse_moves:
            sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
            transition = sim.apply_target((level, 0), reference_price=100, mark_price=100 * ratio)
            result = reward(
                HedgeRiskRewardModel(RiskRewardConfig(uncertainty_exposure_weight=0)),
                sim,
                transition,
                mark=100 * ratio,
            )
            matrix.check(
                "06-asymmetric-level-risk",
                f"L{level}-loss-{round_index:02d}",
                result.wrong_level_loss_penalty >= 0
                and (level <= 1 or result.wrong_level_loss_penalty > 0),
                f"penalty={result.wrong_level_loss_penalty:.8f}",
            )
            round_index += 1


def _add_observation_checks(matrix: Matrix, profile: RiskLevelProfile) -> None:
    schema = RiskObservationSchema(("x", "y"), 4)
    builder = HedgeRiskObservationBuilder(schema)
    features = np.linspace(-20, 20, 80, dtype=np.float32).reshape(40, 2)
    for index in range(20):
        sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
        vector = builder.build(
            features,
            tick=3 + index,
            account=sim.state,
            mark=100,
            profile=profile,
            uncertainty_score=(index % 5) / 4,
            funding_rate=0.0001 * (index - 10),
            max_episode_steps=100,
            failed_probe_long=index % 5,
            failed_probe_short=(index + 2) % 5,
            downside_semideviation=index / 10,
            pending_reward_fraction=index / 20,
        )
        matrix.check(
            "10-observation-stability",
            f"obs-{index:02d}",
            vector.shape == (schema.flat_size,)
            and vector.dtype == np.float32
            and np.isfinite(vector).all()
            and np.max(np.abs(vector)) <= 10.0 + 1e-6,
            f"size={vector.size},max={float(np.max(np.abs(vector))):.4f}",
        )


def run() -> Matrix:
    matrix = Matrix()
    profile = RiskLevelProfile()
    mapper = RiskLevelMapper(profile)
    topology = RiskActionTopology(profile)

    # Theme 1: action encoding (20 checks covering all 25 joint states)
    for joint_id in range(20):
        ids = (joint_id, 20 + joint_id) if joint_id < 5 else (joint_id,)
        decoded = tuple(HedgeRiskLevelAction.from_joint_id(item) for item in ids)
        matrix.check(
            "01-action-encoding",
            "joint-roundtrip-" + "-".join(str(item) for item in ids),
            all(
                action.joint_id == item and all(0 <= x <= 4 for x in action.as_tuple())
                for item, action in zip(ids, decoded, strict=True)
            ),
            str(tuple(action.as_tuple() for action in decoded)),
        )

    # Theme 2: margin ladder / leverage separation (20)
    equities = (500.0, 1000.0, 2500.0, 10000.0)
    cases = [(1, 0), (2, 1), (3, 2), (4, 0), (4, 4)]
    for equity in equities:
        for long_level, short_level in cases:
            target = mapper.map((long_level, short_level), equity=equity)
            expected = profile.fraction(long_level) + profile.fraction(short_level)
            matrix.check(
                "02-margin-budget",
                f"eq-{equity:g}-L{long_level}-S{short_level}",
                math.isclose(target.combined_margin_fraction, expected, abs_tol=1e-12)
                and target.reserve_margin_fraction
                >= profile.minimum_reserve_margin_fraction - 1e-12
                and math.isclose(
                    target.long_target_notional,
                    equity * profile.fraction(long_level) * profile.long_leverage,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    target.short_target_notional,
                    equity * profile.fraction(short_level) * profile.short_leverage,
                    abs_tol=1e-12,
                ),
                f"combined={target.combined_margin_fraction:.6f}",
            )

    # Theme 3: action topology / jump geometry (20)
    transitions = [
        ((0, 0), (0, 0)),
        ((0, 0), (1, 0)),
        ((0, 0), (4, 0)),
        ((4, 0), (0, 0)),
        ((1, 1), (2, 1)),
        ((2, 1), (3, 2)),
        ((4, 4), (3, 3)),
        ((3, 1), (1, 3)),
        ((0, 4), (4, 0)),
        ((2, 2), (4, 4)),
        ((4, 2), (2, 1)),
        ((1, 3), (1, 1)),
        ((1, 0), (1, 4)),
        ((3, 3), (4, 3)),
        ((3, 3), (3, 4)),
        ((4, 1), (4, 1)),
        ((2, 4), (4, 2)),
        ((0, 2), (3, 2)),
        ((4, 3), (1, 0)),
        ((1, 4), (0, 1)),
    ]
    planner = HedgeRiskLevelPlannerAdapter(profile)
    for before, after in transitions:
        t = topology.transition(before, after)
        expected_distance = abs(after[0] - before[0]) + abs(after[1] - before[1])
        sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
        sim.apply_target(before, reference_price=100, mark_price=100)
        current_long = sim.state.long.notional(100)
        current_short = sim.state.short.notional(100)
        signal = planner.from_account_action(
            HedgeRiskLevelAction.from_value(after), account=sim.state, mark=100
        )
        planner_ok = (
            signal.long_increase_allowed == (after[0] > before[0])
            and signal.short_increase_allowed == (after[1] > before[1])
            and (after[0] > before[0] or signal.long_target_notional <= current_long + 1e-12)
            and (after[1] > before[1] or signal.short_target_notional <= current_short + 1e-12)
            and signal.target_semantics == "RISK_CAP_NO_SAME_LEVEL_SCALE_IN"
        )
        matrix.check(
            "03-action-topology",
            f"{before}->{after}",
            t.manhattan_distance == expected_distance
            and math.isclose(
                t.gross_margin_delta,
                t.requested_gross_margin_fraction - t.previous_gross_margin_fraction,
                abs_tol=1e-12,
            )
            and planner_ok,
            f"distance={t.manhattan_distance},jump={t.upward_jump_excess}",
        )

    # Theme 4: target simulator / no hidden same-level scale-in (20)
    for index in range(20):
        level = 1 + index % 4
        side_short = bool(index % 2)
        p = RiskLevelProfile(rebalance_deadband_fraction=0.0025)
        sim = TargetLevelPortfolioSimulator(1000, profile=p, fee_rate=0, slippage_bps=0)
        action = (0, level) if side_short else (level, 0)
        sim.apply_target(action, reference_price=100, mark_price=100)
        before_qty = sim.state.short.quantity if side_short else sim.state.long.quantity
        # Include both tiny drift and substantial adverse moves.  Same level may trim but
        # must never increase risk merely to refill the percentage target.
        if index < 10:
            next_price = 100 * (1 + (index + 1) * 0.000001)
        else:
            adverse = 0.90 + (index - 10) * 0.005
            next_price = 100 / adverse if side_short else 100 * adverse
        t = sim.apply_target(action, reference_price=next_price, mark_price=next_price)
        after_qty = sim.state.short.quantity if side_short else sim.state.long.quantity
        deadband_ok = index >= 10 or t.traded_notional == 0.0
        matrix.check(
            "04-target-simulator",
            f"same-level-{index:02d}",
            after_qty <= before_qty + 1e-12 and deadband_ok and math.isfinite(t.equity),
            f"qty={before_qty:.8g}->{after_qty:.8g},turnover={t.traded_notional:.8g}",
        )

    # Theme 5: primary reward accounting (20)
    moves = [0.98 + i * 0.002 for i in range(20)]
    for index, ratio in enumerate(moves):
        sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0.001, slippage_bps=2)
        t = sim.apply_target(
            (3, 1), reference_price=100, mark_price=100 * ratio, funding_rate=0.0001
        )
        r = reward(HedgeRiskRewardModel(base_only()), sim, t, mark=100 * ratio)
        expected = 100 * math.log(t.equity / t.previous_equity)
        matrix.check(
            "05-primary-accounting",
            f"move-{ratio:.3f}",
            math.isclose(r.unclipped_reward, expected, rel_tol=1e-10, abs_tol=1e-10)
            and math.isfinite(r.accounting_cost_ratio),
            f"reward={r.unclipped_reward:.8f},cost={r.accounting_cost_ratio:.8g}",
        )

    # Theme 6: asymmetric risk-level loss curve (20)
    _add_asymmetric_level_checks(matrix, profile)

    # Theme 7: uncertainty / reserve / drawdown (20)
    for index in range(20):
        level = 1 + index % 4
        uncertainty = (index % 5) / 4
        sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
        mark = 100 * (1 - 0.002 * (index % 4))
        t = sim.apply_target(
            (level, level if index % 3 == 0 else 0), reference_price=100, mark_price=mark
        )
        reserve = sim.state.reserve_margin_fraction(mark, profile)
        r = reward(
            HedgeRiskRewardModel(RiskRewardConfig()), sim, t, mark=mark, uncertainty=uncertainty
        )
        condition = (
            r.uncertainty_exposure_penalty >= 0
            and r.drawdown_penalty >= 0
            and r.reserve_pressure_penalty >= 0
            and r.liquidation_buffer_penalty >= 0
            and reserve >= 0
        )
        matrix.check(
            "07-continuous-risk",
            f"risk-{index:02d}",
            condition,
            f"u={uncertainty:.2f},dd={t.drawdown:.5f},reserve={reserve:.5f}",
        )

    # Theme 8: side-specific probe attribution (20)
    for index in range(20):
        long_probe = index % 2 == 0
        cfg = base_only(repeated_probe_weight=0.02, probe_confirmation_steps=1)
        sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
        model = HedgeRiskRewardModel(cfg)
        first = (1, 3) if long_probe else (3, 1)
        t1 = sim.apply_target(first, reference_price=100, mark_price=100)
        reward(model, sim, t1, mark=100)
        mark = 99 if long_probe else 101
        t2 = sim.apply_target(first, reference_price=100, mark_price=mark)
        r2 = reward(model, sim, t2, mark=mark)
        failed = (
            r2.consecutive_failed_probes_long if long_probe else r2.consecutive_failed_probes_short
        )
        other = (
            r2.consecutive_failed_probes_short if long_probe else r2.consecutive_failed_probes_long
        )
        matrix.check(
            "08-side-probe-credit",
            f"probe-{index:02d}",
            failed == 1 and other == 0 and r2.repeated_probe_penalty > 0,
            f"long={r2.consecutive_failed_probes_long},short={r2.consecutive_failed_probes_short}",
        )

    # Theme 9: hedge efficiency / management shaping (20)
    for index in range(20):
        rising = index % 2 == 0
        sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
        model = HedgeRiskRewardModel(base_only(hedge_efficiency_weight=1, hedge_waste_weight=1))
        t1 = sim.apply_target((3, 1), reference_price=100, mark_price=100)
        reward(model, sim, t1, mark=100)
        mark = 105 if rising else 95
        t2 = sim.apply_target((3, 1), reference_price=100, mark_price=mark)
        r2 = reward(model, sim, t2, mark=mark)
        condition = r2.hedge_waste_penalty > 0 if rising else r2.hedge_efficiency_bonus > 0
        matrix.check(
            "09-hedge-management",
            f"hedge-{index:02d}",
            condition,
            f"bonus={r2.hedge_efficiency_bonus:.8f},waste={r2.hedge_waste_penalty:.8f}",
        )

    # Theme 10: observation / numerical / exploit resistance (20)
    _add_observation_checks(matrix, profile)

    if len(matrix.rows) != 200:
        raise AssertionError(f"validator matrix drifted: expected 200 rows, got {len(matrix.rows)}")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "HEDGE-RL-ACTION-REWARD-200-VALIDATION.json",
    )
    args = parser.parse_args()
    matrix = run()
    passed = sum(row.status == "PASS" for row in matrix.rows)
    failed = len(matrix.rows) - passed
    payload = {
        "validator": "Hedge Risk-Level Action Reward 200",
        "checks": len(matrix.rows),
        "passed": passed,
        "failed": failed,
        "status": "PASS" if failed == 0 else "FAIL",
        "rows": [asdict(row) for row in matrix.rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"HEDGE RISK ACTION REWARD 200: {passed}/{len(matrix.rows)} PASS; FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
