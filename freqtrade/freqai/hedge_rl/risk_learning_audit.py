"""Causal OOS evidence for Hedge Risk-Level RL dynamic position sizing.

The core attribution rule is strict: counterfactuals keep the learned LONG/SHORT
active/flat path fixed and alter only non-zero risk-level magnitudes.  Permutation
counterfactuals go one step further and apply a bijection over levels 1..4, which
preserves every active/flat decision and every level-change timestamp while changing
only which concrete margin budget is assigned to each learned rank.

This module is isolated from unrelated reinforcement-learning subsystems and
does not change trading policy.
It produces evidence about a trained policy.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .risk_environment import HedgeRiskLevelEnv
from .risk_levels import RiskLevelProfile
from .risk_observation import ACCOUNT_FEATURE_NAMES


RiskAction = tuple[int, int]
ActionTransform = Callable[[Sequence[RiskAction]], list[RiskAction]]
_IDENTITY_LEVEL_MAPPING = (1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class RiskLearningAuditThresholds:
    """Fail-closed evidence thresholds for the dynamic-sizing claim."""

    drawdown_weight: float = 1.0
    min_sizing_edge: float = 0.0005
    max_active_joint_action_share: float = 0.90
    min_distinct_nonzero_levels: int = 3
    min_active_fraction: float = 0.02
    min_nonzero_level_entropy: float = 0.20
    min_magnitude_change_fraction: float = 0.005
    shuffle_trials: int = 8
    shuffle_quantile: float = 0.75
    permutation_trials: int = 23
    permutation_quantile: float = 0.75
    max_permutation_exceedance: float = 0.25
    segment_count: int = 4
    min_segment_steps: int = 128
    min_segments: int = 2
    min_segment_pass_ratio: float = 0.50

    def __post_init__(self) -> None:
        bounded = {
            "max_active_joint_action_share": self.max_active_joint_action_share,
            "min_active_fraction": self.min_active_fraction,
            "min_nonzero_level_entropy": self.min_nonzero_level_entropy,
            "min_magnitude_change_fraction": self.min_magnitude_change_fraction,
            "shuffle_quantile": self.shuffle_quantile,
            "permutation_quantile": self.permutation_quantile,
            "max_permutation_exceedance": self.max_permutation_exceedance,
            "min_segment_pass_ratio": self.min_segment_pass_ratio,
        }
        if not math.isfinite(float(self.drawdown_weight)) or self.drawdown_weight < 0:
            raise ValueError("drawdown_weight must be finite and non-negative")
        if not math.isfinite(float(self.min_sizing_edge)) or self.min_sizing_edge < 0:
            raise ValueError("min_sizing_edge must be finite and non-negative")
        for name, value in bounded.items():
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if self.max_active_joint_action_share <= 0:
            raise ValueError("max_active_joint_action_share must be positive")
        if not 1 <= self.min_distinct_nonzero_levels <= 4:
            raise ValueError("min_distinct_nonzero_levels must be within [1, 4]")
        if self.shuffle_trials < 1:
            raise ValueError("shuffle_trials must be positive")
        if self.permutation_trials < 1 or self.permutation_trials > 23:
            raise ValueError("permutation_trials must be within [1, 23]")
        if self.segment_count < 1:
            raise ValueError("segment_count must be positive")
        if self.min_segment_steps < 2:
            raise ValueError("min_segment_steps must be at least 2")
        if self.min_segments < 1 or self.min_segments > self.segment_count:
            raise ValueError("min_segments must be within [1, segment_count]")

    @classmethod
    def from_freqtrade_config(
        cls,
        config: Mapping[str, Any],
    ) -> RiskLearningAuditThresholds:
        freqai = config.get("freqai", {}) if isinstance(config, Mapping) else {}
        hedge = freqai.get("hedge_rl_config", {}) if isinstance(freqai, Mapping) else {}
        audit = hedge.get("learning_audit", {}) if isinstance(hedge, Mapping) else {}
        if not isinstance(audit, Mapping):
            audit = {}
        defaults = cls()
        return cls(
            drawdown_weight=float(audit.get("drawdown_weight", defaults.drawdown_weight)),
            min_sizing_edge=float(audit.get("min_sizing_edge", defaults.min_sizing_edge)),
            max_active_joint_action_share=float(
                audit.get(
                    "max_active_joint_action_share",
                    defaults.max_active_joint_action_share,
                )
            ),
            min_distinct_nonzero_levels=int(
                audit.get(
                    "min_distinct_nonzero_levels",
                    defaults.min_distinct_nonzero_levels,
                )
            ),
            min_active_fraction=float(
                audit.get("min_active_fraction", defaults.min_active_fraction)
            ),
            min_nonzero_level_entropy=float(
                audit.get(
                    "min_nonzero_level_entropy",
                    defaults.min_nonzero_level_entropy,
                )
            ),
            min_magnitude_change_fraction=float(
                audit.get(
                    "min_magnitude_change_fraction",
                    defaults.min_magnitude_change_fraction,
                )
            ),
            shuffle_trials=int(audit.get("shuffle_trials", defaults.shuffle_trials)),
            shuffle_quantile=float(audit.get("shuffle_quantile", defaults.shuffle_quantile)),
            permutation_trials=int(audit.get("permutation_trials", defaults.permutation_trials)),
            permutation_quantile=float(
                audit.get("permutation_quantile", defaults.permutation_quantile)
            ),
            max_permutation_exceedance=float(
                audit.get(
                    "max_permutation_exceedance",
                    defaults.max_permutation_exceedance,
                )
            ),
            segment_count=int(audit.get("segment_count", defaults.segment_count)),
            min_segment_steps=int(audit.get("min_segment_steps", defaults.min_segment_steps)),
            min_segments=int(audit.get("min_segments", defaults.min_segments)),
            min_segment_pass_ratio=float(
                audit.get("min_segment_pass_ratio", defaults.min_segment_pass_ratio)
            ),
        )


@dataclass(frozen=True, slots=True)
class RiskLearningRollout:
    name: str
    steps: int
    initial_equity: float
    final_equity: float
    total_return: float
    max_drawdown: float
    total_reward: float
    sizing_score: float
    turnover: float
    mean_used_margin_fraction: float
    max_used_margin_fraction: float
    terminated_early: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RiskActionDiagnostics:
    steps: int
    active_steps: int
    active_fraction: float
    joint_action_counts: dict[str, int]
    active_joint_action_counts: dict[str, int]
    long_level_counts: dict[str, int]
    short_level_counts: dict[str, int]
    max_joint_action_share: float
    max_active_joint_action_share: float
    normalized_joint_entropy: float
    normalized_nonzero_level_entropy: float
    distinct_nonzero_levels: int
    magnitude_change_steps: int
    magnitude_transition_opportunities: int
    magnitude_change_fraction: float
    mean_gross_margin_fraction: float
    gross_margin_fraction_std: float
    uncertainty_exposure_correlation: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RiskAuditSegment:
    segment: int
    start_tick: int
    end_tick: int
    passed: bool
    adaptive: RiskLearningRollout
    best_fixed_score: float
    permutation_score_quantile: float
    adaptive_vs_best_fixed_edge: float
    adaptive_vs_permutation_quantile_edge: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["adaptive"] = self.adaptive.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class RiskLearningAuditReport:
    passed: bool
    claim: str
    gates: Mapping[str, bool]
    thresholds: RiskLearningAuditThresholds
    adaptive: RiskLearningRollout
    fixed_counterfactuals: tuple[RiskLearningRollout, ...]
    permutation_counterfactuals: tuple[RiskLearningRollout, ...]
    shuffled_counterfactuals: tuple[RiskLearningRollout, ...]
    best_fixed_name: str
    best_fixed_score: float
    permutation_score_quantile: float
    permutation_exceedance_fraction: float
    shuffled_score_quantile: float
    diagnostics: RiskActionDiagnostics
    segments: tuple[RiskAuditSegment, ...]
    segment_pass_ratio: float
    evidence: Mapping[str, float]
    action_signature: str
    observation_signature: str
    reward_signature: str
    observation_window: int
    feature_count: int
    policy_fingerprint: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "hedge-risk-level-learning-audit-v4",
            "status": "PASS" if self.passed else "FAIL",
            "claim": self.claim,
            "gates": dict(self.gates),
            "thresholds": asdict(self.thresholds),
            "adaptive": self.adaptive.to_dict(),
            "fixed_counterfactuals": [item.to_dict() for item in self.fixed_counterfactuals],
            "permutation_counterfactuals": [
                item.to_dict() for item in self.permutation_counterfactuals
            ],
            "shuffled_counterfactuals": [item.to_dict() for item in self.shuffled_counterfactuals],
            "best_fixed_name": self.best_fixed_name,
            "best_fixed_score": self.best_fixed_score,
            "permutation_score_quantile": self.permutation_score_quantile,
            "permutation_exceedance_fraction": self.permutation_exceedance_fraction,
            "shuffled_score_quantile": self.shuffled_score_quantile,
            "diagnostics": self.diagnostics.to_dict(),
            "segments": [item.to_dict() for item in self.segments],
            "segment_pass_ratio": self.segment_pass_ratio,
            "evidence": dict(self.evidence),
            "action_signature": self.action_signature,
            "observation_signature": self.observation_signature,
            "reward_signature": self.reward_signature,
            "observation_window": self.observation_window,
            "feature_count": self.feature_count,
            "policy_fingerprint": self.policy_fingerprint,
            "metadata": dict(self.metadata),
        }


def audit_enabled(config: Mapping[str, Any]) -> bool:
    freqai = config.get("freqai", {}) if isinstance(config, Mapping) else {}
    hedge = freqai.get("hedge_rl_config", {}) if isinstance(freqai, Mapping) else {}
    audit = hedge.get("learning_audit", {}) if isinstance(hedge, Mapping) else {}
    return bool(audit.get("enabled", False)) if isinstance(audit, Mapping) else False


def audit_required(config: Mapping[str, Any]) -> bool:
    freqai = config.get("freqai", {}) if isinstance(config, Mapping) else {}
    hedge = freqai.get("hedge_rl_config", {}) if isinstance(freqai, Mapping) else {}
    audit = hedge.get("learning_audit", {}) if isinstance(hedge, Mapping) else {}
    return bool(audit.get("fail_training_on_gate", False)) if isinstance(audit, Mapping) else False


def sizing_score(
    *,
    initial_equity: float,
    final_equity: float,
    max_drawdown: float,
    drawdown_weight: float,
) -> float:
    """Risk-adjusted score using net equity and exact observed drawdown."""
    initial = float(initial_equity)
    final = float(final_equity)
    drawdown = float(max_drawdown)
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 0 or final <= 0:
        return -1e12
    if not math.isfinite(drawdown) or drawdown < 0:
        return -1e12
    return math.log(final / initial) - float(drawdown_weight) * drawdown


def fixed_level_transform(level: int) -> ActionTransform:
    value = int(level)
    if value not in (1, 2, 3, 4):
        raise ValueError("fixed risk level must be within [1, 4]")

    def transform(actions: Sequence[RiskAction]) -> list[RiskAction]:
        return [
            (value if long_level > 0 else 0, value if short_level > 0 else 0)
            for long_level, short_level in actions
        ]

    return transform


def level_permutation_transform(mapping: Sequence[int]) -> ActionTransform:
    """Apply a bijection over levels 1..4 while keeping zero and change times exact."""
    values = tuple(int(item) for item in mapping)
    if len(values) != 4 or set(values) != {1, 2, 3, 4}:
        raise ValueError("level permutation must be a bijection of (1, 2, 3, 4)")
    lookup = (0, *values)

    def transform(actions: Sequence[RiskAction]) -> list[RiskAction]:
        return [
            (lookup[int(long_level)], lookup[int(short_level)])
            for long_level, short_level in actions
        ]

    return transform


def shuffled_level_transform(seed: int) -> ActionTransform:
    """Shuffle learned magnitudes while preserving active/flat timing per leg."""

    def transform(actions: Sequence[RiskAction]) -> list[RiskAction]:
        rng = np.random.default_rng(int(seed))
        long_levels = [int(action[0]) for action in actions if int(action[0]) > 0]
        short_levels = [int(action[1]) for action in actions if int(action[1]) > 0]
        rng.shuffle(long_levels)
        rng.shuffle(short_levels)
        long_index = 0
        short_index = 0
        output: list[RiskAction] = []
        for raw_long, raw_short in actions:
            long_level = int(raw_long)
            short_level = int(raw_short)
            if long_level > 0:
                long_level = long_levels[long_index]
                long_index += 1
            if short_level > 0:
                short_level = short_levels[short_index]
                short_index += 1
            output.append((long_level, short_level))
        return output

    return transform


def active_change_signature(
    actions: Sequence[RiskAction],
) -> tuple[tuple[bool, bool, bool, bool], ...]:
    """Return active masks and same-leg level-change flags for attribution tests."""
    normalized = [(int(a), int(b)) for a, b in actions]
    signature: list[tuple[bool, bool, bool, bool]] = []
    previous: RiskAction | None = None
    for current in normalized:
        if previous is None:
            long_changed = False
            short_changed = False
        else:
            long_changed = current[0] != previous[0]
            short_changed = current[1] != previous[1]
        signature.append((current[0] > 0, current[1] > 0, long_changed, short_changed))
        previous = current
    return tuple(signature)


def _normalized_entropy(counter: Counter[int] | Counter[RiskAction], cardinality: int) -> float:
    total = sum(counter.values())
    if total <= 0 or cardinality <= 1:
        return 0.0
    probabilities = np.asarray(list(counter.values()), dtype=np.float64) / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / math.log(float(cardinality)) if entropy > 0 else 0.0


def action_diagnostics(
    actions: Sequence[RiskAction],
    *,
    profile: RiskLevelProfile,
    uncertainties: Sequence[float] | None = None,
) -> RiskActionDiagnostics:
    if not actions:
        raise ValueError("risk policy produced no OOS actions")
    normalized = [(int(item[0]), int(item[1])) for item in actions]
    if any(not 0 <= value <= 4 for item in normalized for value in item):
        raise ValueError("risk actions must stay within [0, 4]")
    joint = Counter(normalized)
    active_joint = Counter(item for item in normalized if item != (0, 0))
    longs = Counter(item[0] for item in normalized)
    shorts = Counter(item[1] for item in normalized)
    nonzero_levels = Counter(level for action in normalized for level in action if level > 0)
    count = len(normalized)
    active_steps = sum(action != (0, 0) for action in normalized)

    magnitude_changes = 0
    magnitude_opportunities = 0
    last_nonzero: list[int | None] = [None, None]
    for action in normalized:
        for side in (0, 1):
            level = action[side]
            if level <= 0:
                continue
            previous = last_nonzero[side]
            if previous is not None:
                magnitude_opportunities += 1
                if previous != level:
                    magnitude_changes += 1
            last_nonzero[side] = level

    gross = np.asarray(
        [
            profile.fraction(long_level) + profile.fraction(short_level)
            for long_level, short_level in normalized
        ],
        dtype=np.float64,
    )
    correlation: float | None = None
    if uncertainties is not None and len(uncertainties) == count:
        uncertainty_array = np.asarray(uncertainties, dtype=np.float64)
        valid = np.isfinite(uncertainty_array) & np.isfinite(gross)
        if valid.sum() >= 3:
            left = uncertainty_array[valid]
            right = gross[valid]
            if float(np.std(left)) > 0 and float(np.std(right)) > 0:
                correlation = float(np.corrcoef(left, right)[0, 1])

    return RiskActionDiagnostics(
        steps=count,
        active_steps=active_steps,
        active_fraction=active_steps / count,
        joint_action_counts={
            f"{long_level},{short_level}": occurrences
            for (long_level, short_level), occurrences in sorted(joint.items())
        },
        active_joint_action_counts={
            f"{long_level},{short_level}": occurrences
            for (long_level, short_level), occurrences in sorted(active_joint.items())
        },
        long_level_counts={str(level): occurrences for level, occurrences in sorted(longs.items())},
        short_level_counts={
            str(level): occurrences for level, occurrences in sorted(shorts.items())
        },
        max_joint_action_share=max(joint.values()) / count,
        max_active_joint_action_share=(
            max(active_joint.values()) / active_steps if active_steps else 1.0
        ),
        normalized_joint_entropy=_normalized_entropy(joint, 25),
        normalized_nonzero_level_entropy=_normalized_entropy(nonzero_levels, 4),
        distinct_nonzero_levels=len(nonzero_levels),
        magnitude_change_steps=magnitude_changes,
        magnitude_transition_opportunities=magnitude_opportunities,
        magnitude_change_fraction=(
            magnitude_changes / magnitude_opportunities if magnitude_opportunities else 0.0
        ),
        mean_gross_margin_fraction=float(gross.mean()),
        gross_margin_fraction_std=float(gross.std()),
        uncertainty_exposure_correlation=correlation,
    )


def infer_observation_window(
    model: Any,
    *,
    feature_count: int,
    fallback_window: int,
) -> int:
    """Infer the exact training window from the model observation shape when possible."""
    if feature_count < 1:
        raise ValueError("feature_count must be positive")
    observation_space = getattr(model, "observation_space", None)
    shape = getattr(observation_space, "shape", None)
    if shape and len(shape) == 1:
        flat_size = int(shape[0])
        market_size = flat_size - len(ACCOUNT_FEATURE_NAMES)
        if market_size > 0 and market_size % feature_count == 0:
            inferred = market_size // feature_count
            if inferred >= 2:
                return inferred
    fallback = int(fallback_window)
    if fallback < 2:
        raise ValueError("fallback observation window must be at least 2")
    return fallback


def policy_fingerprint(model: Any) -> str:
    """Hash policy parameters without depending on an on-disk SB3 archive."""
    policy = getattr(model, "policy", None)
    state_dict = getattr(policy, "state_dict", None)
    if not callable(state_dict):
        return hashlib.sha256(type(model).__qualname__.encode("utf-8")).hexdigest()
    hasher = hashlib.sha256()
    tensors = state_dict()
    for name in sorted(tensors):
        value = tensors[name]
        hasher.update(str(name).encode("utf-8"))
        try:
            array = value.detach().cpu().contiguous().numpy()
            hasher.update(str(array.dtype).encode("ascii"))
            hasher.update(repr(tuple(array.shape)).encode("ascii"))
            hasher.update(memoryview(array).cast("B"))
        except Exception:
            hasher.update(repr(value).encode("utf-8"))
    return hasher.hexdigest()


def _audit_config(config: Mapping[str, Any], *, rows: int) -> dict[str, Any]:
    output = deepcopy(dict(config))
    freqai = output.setdefault("freqai", {})
    if not isinstance(freqai, dict):
        raise TypeError("freqai configuration must be an object")
    hedge = freqai.setdefault("hedge_rl_config", {})
    if not isinstance(hedge, dict):
        raise TypeError("freqai.hedge_rl_config must be an object")
    hedge["random_start"] = False
    hedge["max_episode_steps"] = max(
        int(rows) + 16,
        int(hedge.get("max_episode_steps", 0) or 0),
    )
    return output


def _unwrap_env(env: Any) -> HedgeRiskLevelEnv:
    current = env
    seen: set[int] = set()
    while not isinstance(current, HedgeRiskLevelEnv):
        identity = id(current)
        if identity in seen:
            break
        seen.add(identity)
        next_env = getattr(current, "env", None)
        if next_env is None:
            next_env = getattr(current, "unwrapped", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    if not isinstance(current, HedgeRiskLevelEnv):
        raise TypeError("Risk-Level audit requires HedgeRiskLevelEnv or a simple wrapper around it")
    return current


def _reset_for_range(env: HedgeRiskLevelEnv, start_tick: int, end_tick: int):
    return env.reset(
        seed=1,
        options={
            "start_tick": int(start_tick),
            "end_tick": int(end_tick),
            "max_episode_steps": max(1, int(end_tick) - int(start_tick)),
        },
    )


def _rollout_actions_on_env(
    *,
    env: HedgeRiskLevelEnv,
    name: str,
    actions: Sequence[RiskAction],
    start_tick: int,
    end_tick: int,
    thresholds: RiskLearningAuditThresholds,
) -> RiskLearningRollout:
    """Replay a counterfactual through the real portfolio simulator only.

    Counterfactual attribution does not need model observations or reward shaping.
    Reusing the environment's compact market arrays and deterministic simulator keeps
    fees, slippage, funding and risk termination identical while avoiding millions of
    unnecessary observation/reward allocations across permutation and shuffle trials.
    """
    expected_steps = max(0, int(end_tick) - int(start_tick))
    if len(actions) != expected_steps:
        raise ValueError(
            f"counterfactual action count {len(actions)} does not match OOS steps {expected_steps}"
        )
    simulator = env.simulator
    state = simulator.reset(env.env_config.starting_balance)
    initial_equity = float(state.equity)
    max_drawdown = 0.0
    used_margin_sum = 0.0
    max_used_margin = 0.0
    terminated = False
    consumed = 0

    for offset, (long_level, short_level) in enumerate(actions, start=1):
        if terminated:
            break
        tick = int(start_tick) + offset
        transition = simulator.apply_target(
            (int(long_level), int(short_level)),
            reference_price=float(env.market.open[tick]),
            mark_price=float(env.market.close[tick]),
            funding_rate=env._funding(tick),
        )
        state = simulator.state
        mark = float(env.market.close[tick])
        reserve = state.reserve_margin_fraction(mark, env.profile)
        drawdown = float(state.drawdown())
        used = float(state.used_margin_fraction(mark, env.profile))
        used_margin_sum += used
        max_used_margin = max(max_used_margin, used)
        max_drawdown = max(max_drawdown, drawdown)
        consumed += 1
        terminated = bool(
            state.equity <= 0
            or drawdown >= env.env_config.drawdown_stop
            or reserve <= env.env_config.maintenance_margin_fraction
        )
        # Keep a hard invariant between simulator transition and committed state.
        if not math.isclose(
            float(transition.equity), float(state.equity), rel_tol=0.0, abs_tol=1e-10
        ):
            raise AssertionError("counterfactual simulator transition/state equity diverged")

    final_equity = float(state.equity)
    incomplete = consumed != expected_steps
    return RiskLearningRollout(
        name=name,
        steps=consumed,
        initial_equity=initial_equity,
        final_equity=final_equity,
        total_return=final_equity / initial_equity - 1.0,
        max_drawdown=max_drawdown,
        total_reward=0.0,
        sizing_score=sizing_score(
            initial_equity=initial_equity,
            final_equity=final_equity,
            max_drawdown=max_drawdown,
            drawdown_weight=thresholds.drawdown_weight,
        ),
        turnover=float(state.turnover),
        mean_used_margin_fraction=(used_margin_sum / consumed if consumed else 0.0),
        max_used_margin_fraction=max_used_margin,
        terminated_early=bool(terminated or incomplete),
    )


def _rollout_model_on_env(
    *,
    env: HedgeRiskLevelEnv,
    model: Any,
    start_tick: int,
    end_tick: int,
    thresholds: RiskLearningAuditThresholds,
    name: str = "ADAPTIVE",
) -> tuple[RiskLearningRollout, list[RiskAction], list[float]]:
    observation, info = _reset_for_range(env, start_tick, end_tick)
    initial_equity = float(info["equity"])
    total_reward = 0.0
    max_drawdown = 0.0
    used_margin_sum = 0.0
    max_used_margin = 0.0
    terminated = False
    truncated = False
    actions: list[RiskAction] = []
    uncertainties: list[float] = []
    recurrent_state = None
    episode_start = np.asarray([True], dtype=np.bool_)
    while not (terminated or truncated):
        current_tick = int(info["tick"])
        uncertainties.append(float(env._row_uncertainty(current_tick)))
        raw_action, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        action_array = np.asarray(raw_action).reshape(-1)
        if action_array.size != 2:
            raise ValueError("Risk-Level model must predict exactly [long_level, short_level]")
        action = (int(action_array[0]), int(action_array[1]))
        if any(value < 0 or value > 4 for value in action):
            raise ValueError(f"Risk-Level model predicted invalid action {action!r}")
        actions.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_start[...] = False
        used = float(info["used_margin_fraction"])
        used_margin_sum += used
        max_used_margin = max(max_used_margin, used)
        total_reward += float(reward)
        max_drawdown = max(max_drawdown, float(info["drawdown"]))
    final_equity = float(info["equity"])
    expected_steps = max(0, int(end_tick) - int(start_tick))
    natural_end = bool(int(info["tick"]) >= int(end_tick))
    incomplete = len(actions) != expected_steps
    rollout = RiskLearningRollout(
        name=name,
        steps=len(actions),
        initial_equity=initial_equity,
        final_equity=final_equity,
        total_return=final_equity / initial_equity - 1.0,
        max_drawdown=max_drawdown,
        total_reward=total_reward,
        sizing_score=sizing_score(
            initial_equity=initial_equity,
            final_equity=final_equity,
            max_drawdown=max_drawdown,
            drawdown_weight=thresholds.drawdown_weight,
        ),
        turnover=float(env.simulator.state.turnover),
        mean_used_margin_fraction=(used_margin_sum / len(actions) if actions else 0.0),
        max_used_margin_fraction=max_used_margin,
        terminated_early=bool(incomplete or (terminated and not natural_end)),
    )
    return rollout, actions, uncertainties


def _permutation_mappings(trials: int, seed: int) -> tuple[tuple[int, int, int, int], ...]:
    available = tuple(
        (item[0], item[1], item[2], item[3])
        for item in permutations((1, 2, 3, 4))
        if item != _IDENTITY_LEVEL_MAPPING
    )
    requested = int(trials)
    if requested >= len(available):
        return available
    rng = np.random.default_rng(int(seed))
    indexes = np.sort(rng.choice(len(available), size=requested, replace=False))
    return tuple(available[int(index)] for index in indexes)


def _counterfactual_rollouts(
    *,
    env: HedgeRiskLevelEnv,
    learned_actions: Sequence[RiskAction],
    start_tick: int,
    end_tick: int,
    thresholds: RiskLearningAuditThresholds,
    seed: int,
    include_shuffle: bool,
) -> tuple[
    tuple[RiskLearningRollout, ...],
    tuple[RiskLearningRollout, ...],
    tuple[RiskLearningRollout, ...],
]:
    counterfactual_end_tick = int(start_tick) + len(learned_actions)
    if counterfactual_end_tick > int(end_tick):
        raise ValueError("learned action path exceeds requested OOS range")
    fixed = tuple(
        _rollout_actions_on_env(
            env=env,
            name=f"FIXED_L{level}",
            actions=fixed_level_transform(level)(learned_actions),
            start_tick=start_tick,
            end_tick=counterfactual_end_tick,
            thresholds=thresholds,
        )
        for level in range(1, 5)
    )
    permutation_rollouts: list[RiskLearningRollout] = []
    original_signature = active_change_signature(learned_actions)
    for index, mapping in enumerate(
        _permutation_mappings(thresholds.permutation_trials, seed),
        start=1,
    ):
        transformed = level_permutation_transform(mapping)(learned_actions)
        if active_change_signature(transformed) != original_signature:
            raise AssertionError("level permutation changed active/change timing")
        permutation_rollouts.append(
            _rollout_actions_on_env(
                env=env,
                name=(f"PERM_{index:02d}_" + "".join(str(value) for value in mapping)),
                actions=transformed,
                start_tick=start_tick,
                end_tick=counterfactual_end_tick,
                thresholds=thresholds,
            )
        )
    shuffled: list[RiskLearningRollout] = []
    if include_shuffle:
        for trial in range(thresholds.shuffle_trials):
            trial_seed = int(seed) + 104729 + trial * 7919
            shuffled.append(
                _rollout_actions_on_env(
                    env=env,
                    name=f"SHUFFLED_{trial + 1:02d}",
                    actions=shuffled_level_transform(trial_seed)(learned_actions),
                    start_tick=start_tick,
                    end_tick=counterfactual_end_tick,
                    thresholds=thresholds,
                )
            )
    return fixed, tuple(permutation_rollouts), tuple(shuffled)


def _score_quantile(rollouts: Sequence[RiskLearningRollout], quantile: float) -> float:
    if not rollouts:
        return -1e12
    scores = np.asarray([item.sizing_score for item in rollouts], dtype=np.float64)
    return float(np.quantile(scores, float(quantile)))


def _segment_ranges(
    start_tick: int,
    end_tick: int,
    thresholds: RiskLearningAuditThresholds,
) -> tuple[tuple[int, int], ...]:
    total_steps = int(end_tick) - int(start_tick)
    if total_steps < thresholds.min_segment_steps:
        return ()
    count = min(
        thresholds.segment_count,
        total_steps // thresholds.min_segment_steps,
    )
    if count < 1:
        return ()
    base = total_steps // count
    remainder = total_steps % count
    ranges: list[tuple[int, int]] = []
    cursor = int(start_tick)
    for index in range(count):
        length = base + (1 if index < remainder else 0)
        next_cursor = cursor + length
        ranges.append((cursor, next_cursor))
        cursor = next_cursor
    return tuple(ranges)


def _segment_audits(
    *,
    env: HedgeRiskLevelEnv,
    model: Any,
    thresholds: RiskLearningAuditThresholds,
    seed: int,
) -> tuple[RiskAuditSegment, ...]:
    output: list[RiskAuditSegment] = []
    for index, (start_tick, end_tick) in enumerate(
        _segment_ranges(env._start_tick, env._end_tick, thresholds),
        start=1,
    ):
        adaptive, actions, _ = _rollout_model_on_env(
            env=env,
            model=model,
            start_tick=start_tick,
            end_tick=end_tick,
            thresholds=thresholds,
            name=f"ADAPTIVE_SEGMENT_{index:02d}",
        )
        fixed, permuted, _ = _counterfactual_rollouts(
            env=env,
            learned_actions=actions,
            start_tick=start_tick,
            end_tick=end_tick,
            thresholds=thresholds,
            seed=int(seed) + index * 65537,
            include_shuffle=False,
        )
        best_fixed = max(item.sizing_score for item in fixed)
        permutation_quantile = _score_quantile(
            permuted,
            thresholds.permutation_quantile,
        )
        fixed_edge = adaptive.sizing_score - best_fixed
        permutation_edge = adaptive.sizing_score - permutation_quantile
        passed = bool(
            not adaptive.terminated_early
            and fixed_edge > thresholds.min_sizing_edge
            and permutation_edge > thresholds.min_sizing_edge
        )
        output.append(
            RiskAuditSegment(
                segment=index,
                start_tick=start_tick,
                end_tick=end_tick,
                passed=passed,
                adaptive=adaptive,
                best_fixed_score=best_fixed,
                permutation_score_quantile=permutation_quantile,
                adaptive_vs_best_fixed_edge=fixed_edge,
                adaptive_vs_permutation_quantile_edge=permutation_edge,
            )
        )
    return tuple(output)


def run_risk_level_learning_audit_on_env(
    *,
    model: Any,
    env: Any,
    thresholds: RiskLearningAuditThresholds | None = None,
    shuffle_seed: int = 20260815,
    metadata: Mapping[str, Any] | None = None,
) -> RiskLearningAuditReport:
    """Audit a trained policy against direction-preserving sizing counterfactuals."""
    limits = thresholds or RiskLearningAuditThresholds()
    risk_env = _unwrap_env(env)
    if risk_env._end_tick <= risk_env._start_tick:
        raise ValueError("OOS audit environment has no executable next-bar steps")

    adaptive, learned_actions, uncertainties = _rollout_model_on_env(
        env=risk_env,
        model=model,
        start_tick=risk_env._start_tick,
        end_tick=risk_env._end_tick,
        thresholds=limits,
    )
    diagnostics = action_diagnostics(
        learned_actions,
        profile=risk_env.profile,
        uncertainties=uncertainties,
    )
    fixed, permuted, shuffled = _counterfactual_rollouts(
        env=risk_env,
        learned_actions=learned_actions,
        start_tick=risk_env._start_tick,
        end_tick=risk_env._end_tick,
        thresholds=limits,
        seed=int(shuffle_seed),
        include_shuffle=True,
    )
    best_fixed = max(fixed, key=lambda item: item.sizing_score)
    permutation_quantile = _score_quantile(permuted, limits.permutation_quantile)
    permutation_exceedance = (
        sum(item.sizing_score >= adaptive.sizing_score for item in permuted) / len(permuted)
        if permuted
        else 1.0
    )
    shuffled_quantile = _score_quantile(shuffled, limits.shuffle_quantile)
    segments = _segment_audits(
        env=risk_env,
        model=model,
        thresholds=limits,
        seed=int(shuffle_seed),
    )
    segment_pass_ratio = sum(item.passed for item in segments) / len(segments) if segments else 0.0

    fixed_edge = adaptive.sizing_score - best_fixed.sizing_score
    permutation_edge = adaptive.sizing_score - permutation_quantile
    shuffled_edge = adaptive.sizing_score - shuffled_quantile
    evidence = {
        "adaptive_vs_best_fixed_edge": fixed_edge,
        "adaptive_vs_permutation_quantile_edge": permutation_edge,
        "adaptive_vs_shuffled_quantile_edge": shuffled_edge,
        "permutation_exceedance_fraction": permutation_exceedance,
        "segment_pass_ratio": segment_pass_ratio,
    }
    gates = {
        "adaptive_completed_oos": not adaptive.terminated_early,
        "policy_is_materially_active": diagnostics.active_fraction >= limits.min_active_fraction,
        "no_active_action_collapse": (
            diagnostics.max_active_joint_action_share <= limits.max_active_joint_action_share
        ),
        "uses_multiple_nonzero_risk_levels": (
            diagnostics.distinct_nonzero_levels >= limits.min_distinct_nonzero_levels
        ),
        "nonzero_level_distribution_has_entropy": (
            diagnostics.normalized_nonzero_level_entropy >= limits.min_nonzero_level_entropy
        ),
        "risk_magnitude_changes_over_time": (
            diagnostics.magnitude_change_fraction >= limits.min_magnitude_change_fraction
        ),
        "beats_best_fixed_level_sizing": fixed_edge > limits.min_sizing_edge,
        "beats_level_permutation_quantile": permutation_edge > limits.min_sizing_edge,
        "specific_level_mapping_is_not_exchangeable": (
            permutation_exceedance <= limits.max_permutation_exceedance
        ),
        "beats_shuffled_sizing_quantile": shuffled_edge > limits.min_sizing_edge,
        "enough_oos_segments": len(segments) >= limits.min_segments,
        "oos_segment_stability": (
            len(segments) >= limits.min_segments
            and segment_pass_ratio >= limits.min_segment_pass_ratio
        ),
    }
    passed = all(gates.values())
    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault("oos_rows", len(risk_env.features))
    merged_metadata.setdefault("oos_start_tick", risk_env._start_tick)
    merged_metadata.setdefault("oos_end_tick", risk_env._end_tick)
    merged_metadata.setdefault("counterfactual_execution", "DIRECT_SIMULATOR_V1")
    return RiskLearningAuditReport(
        passed=passed,
        claim=(
            "OOS counterfactual evidence supports learned dynamic position sizing."
            if passed
            else "OOS evidence is insufficient to claim learned dynamic position sizing."
        ),
        gates=gates,
        thresholds=limits,
        adaptive=adaptive,
        fixed_counterfactuals=fixed,
        permutation_counterfactuals=permuted,
        shuffled_counterfactuals=shuffled,
        best_fixed_name=best_fixed.name,
        best_fixed_score=best_fixed.sizing_score,
        permutation_score_quantile=permutation_quantile,
        permutation_exceedance_fraction=permutation_exceedance,
        shuffled_score_quantile=shuffled_quantile,
        diagnostics=diagnostics,
        segments=segments,
        segment_pass_ratio=segment_pass_ratio,
        evidence=evidence,
        action_signature=risk_env.profile.signature,
        observation_signature=risk_env.schema.signature,
        reward_signature=risk_env.reward_config.signature,
        observation_window=risk_env.env_config.observation_window,
        feature_count=len(risk_env.feature_names),
        policy_fingerprint=policy_fingerprint(model),
        metadata=merged_metadata,
    )


def _environment(
    *,
    features: pd.DataFrame,
    prices: pd.DataFrame,
    config: Mapping[str, Any],
    window_size: int,
) -> HedgeRiskLevelEnv:
    return HedgeRiskLevelEnv(
        df=features,
        prices=prices,
        config=dict(config),
        window_size=int(window_size),
        seed=1,
    )


def run_risk_level_learning_audit(
    *,
    model: Any,
    features: pd.DataFrame,
    prices: pd.DataFrame,
    config: Mapping[str, Any],
    thresholds: RiskLearningAuditThresholds | None = None,
    shuffle_seed: int = 20260815,
    metadata: Mapping[str, Any] | None = None,
) -> RiskLearningAuditReport:
    """Create one compact env, then reuse it for every OOS counterfactual replay."""
    if len(features) != len(prices):
        raise ValueError("features and prices must contain the same number of rows")
    if not features.index.equals(prices.index):
        raise ValueError("features and prices must have identical indexes and row order")
    if len(features) < 4:
        raise ValueError("OOS audit requires at least four rows")
    if features.shape[1] < 1:
        raise ValueError("OOS audit requires at least one model feature")

    prepared_config = _audit_config(config, rows=len(features))
    hedge_config = prepared_config.get("freqai", {}).get("hedge_rl_config", {})
    fallback_window = int(hedge_config.get("observation_window", 32))
    window_size = infer_observation_window(
        model,
        feature_count=features.shape[1],
        fallback_window=fallback_window,
    )
    if len(features) <= window_size:
        raise ValueError("OOS audit dataset must be longer than the model observation window")

    env = _environment(
        features=features,
        prices=prices,
        config=prepared_config,
        window_size=window_size,
    )
    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault("oos_start", str(features.index[0]))
    merged_metadata.setdefault("oos_end", str(features.index[-1]))
    merged_metadata.setdefault("alignment", "IDENTICAL_INDEX")
    try:
        return run_risk_level_learning_audit_on_env(
            model=model,
            env=env,
            thresholds=thresholds,
            shuffle_seed=shuffle_seed,
            metadata=merged_metadata,
        )
    finally:
        env.close()


def write_risk_learning_audit(
    report: RiskLearningAuditReport,
    path: str | Path,
) -> Path:
    import json

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
