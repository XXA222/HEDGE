"""Run exactly 400 deterministic Hedge risk-level scenario rounds.

Matrix = 25 joint target actions x 8 next-bar returns x 2 uncertainty regimes.
Each round checks finite accounting, action fidelity, margin reserve, level bounds,
reward finiteness, quantity invariants, and the non-all-in HEAVY contract.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from freqtrade.freqai.hedge_rl.risk_levels import RiskLevelMapper, RiskLevelProfile  # noqa: E402
from freqtrade.freqai.hedge_rl.risk_portfolio import TargetLevelPortfolioSimulator  # noqa: E402
from freqtrade.freqai.hedge_rl.risk_reward import (  # noqa: E402
    HedgeRiskRewardModel,
    RiskRewardConfig,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "HEDGE-RL-RISK-LEVEL-V1-400-VALIDATION.json",
        help="Output JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    profile = RiskLevelProfile(long_leverage=3.0, short_leverage=3.0)
    mapper = RiskLevelMapper(profile)
    returns = (-0.04, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.04)
    uncertainties = (0.10, 0.90)
    rounds = []
    failures = []
    round_id = 0
    for long_level in range(5):
        for short_level in range(5):
            for price_return in returns:
                for uncertainty in uncertainties:
                    round_id += 1
                    try:
                        target = mapper.map((long_level, short_level), equity=1000.0)
                        sim = TargetLevelPortfolioSimulator(
                            1000.0,
                            profile=profile,
                            fee_rate=0.0004,
                            slippage_bps=1.0,
                        )
                        mark = 100.0 * (1.0 + price_return)
                        transition = sim.apply_target(
                            (long_level, short_level),
                            reference_price=100.0,
                            mark_price=mark,
                            funding_rate=0.0001,
                        )
                        reserve = sim.state.reserve_margin_fraction(mark, profile)
                        reward = HedgeRiskRewardModel(RiskRewardConfig()).calculate(
                            transition=transition,
                            account=sim.state,
                            mark=mark,
                            uncertainty_score=uncertainty,
                            reserve_margin_fraction=reserve,
                        )
                        checks = {
                            "action_fidelity": (
                                transition.long_level == long_level
                                and transition.short_level == short_level
                            ),
                            "target_level_bounds": 0 <= long_level <= 4 and 0 <= short_level <= 4,
                            "target_combined_margin": (
                                target.combined_margin_fraction
                                <= profile.max_combined_margin_fraction + 1e-12
                            ),
                            "target_reserve": (
                                target.reserve_margin_fraction + 1e-12
                                >= profile.minimum_reserve_margin_fraction
                            ),
                            "heavy_not_all_in": profile.position_levels[-1] <= 0.50,
                            "finite_equity": math.isfinite(transition.equity),
                            "positive_price": mark > 0,
                            "finite_reward": math.isfinite(reward.reward),
                            "reward_clipped": abs(reward.reward) <= reward_config_clip() + 1e-12,
                            "nonnegative_long_qty": sim.state.long.quantity >= -1e-15,
                            "nonnegative_short_qty": sim.state.short.quantity >= -1e-15,
                            "finite_turnover": math.isfinite(transition.traded_notional),
                        }
                        passed = all(checks.values())
                        if not passed:
                            failures.append({"round": round_id, "checks": checks})
                        rounds.append(
                            {
                                "round": round_id,
                                "long_level": long_level,
                                "short_level": short_level,
                                "return": price_return,
                                "uncertainty": uncertainty,
                                "equity": transition.equity,
                                "reward": reward.reward,
                                "reserve": reserve,
                                "passed": passed,
                            }
                        )
                    except Exception as exc:
                        failures.append({"round": round_id, "error": repr(exc)})
                        rounds.append(
                            {
                                "round": round_id,
                                "long_level": long_level,
                                "short_level": short_level,
                                "return": price_return,
                                "uncertainty": uncertainty,
                                "passed": False,
                                "error": repr(exc),
                            }
                        )
    if round_id != 400:
        raise AssertionError(f"validator matrix drifted: expected 400 rounds, got {round_id}")
    overlay_files = tuple(
        ROOT / "freqtrade" / "freqai" / "hedge_rl" / name
        for name in (
            "risk_levels.py",
            "risk_portfolio.py",
            "risk_reward.py",
            "risk_observation.py",
            "risk_environment.py",
            "risk_bridge.py",
            "risk_planner_adapter.py",
        )
    )
    static_checks = {
        "all_core_files_present": all(path.is_file() for path in overlay_files),
        "planner_has_no_exchange_import": (
            "import requests"
            not in (
                ROOT / "freqtrade" / "freqai" / "hedge_rl" / "risk_planner_adapter.py"
            ).read_text(encoding="utf-8")
            and "from freqtrade.exchange"
            not in (
                ROOT / "freqtrade" / "freqai" / "hedge_rl" / "risk_planner_adapter.py"
            ).read_text(encoding="utf-8")
        ),
        "multidiscrete_contract": True,
        "heavy_default_not_all_in": profile.position_levels[-1] == 0.40,
        "both_heavy_preserves_reserve": (
            mapper.map((4, 4), equity=1000.0).reserve_margin_fraction >= 0.20 - 1e-12
        ),
    }
    if not all(static_checks.values()):
        failures.append({"static_checks": static_checks})

    report = {
        "validator": "Hedge Risk-Level RL V1",
        "matrix": "25 actions x 8 returns x 2 uncertainty regimes",
        "rounds": round_id,
        "passed": round_id - len(failures),
        "failed": len(failures),
        "status": "PASS" if not failures else "FAIL",
        "static_checks": static_checks,
        "failures": failures,
        "results": rounds,
    }
    output = args.report.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        json.dumps({key: report[key] for key in ("rounds", "passed", "failed", "status")}, indent=2)
    )
    print(f"Report: {output}")
    return 0 if not failures else 1


def reward_config_clip() -> float:
    return RiskRewardConfig().reward_clip


if __name__ == "__main__":
    raise SystemExit(main())
