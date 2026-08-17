"""Behavioral evidence that an RL policy learned position management."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from freqtrade.hedge.contracts import finite_decimal
from freqtrade.hedge.contracts.types import required_text


@dataclass(frozen=True, slots=True)
class PositionDecision:
    requested_long_level: int
    requested_short_level: int
    projected_long_level: int
    projected_short_level: int
    realized_pnl: Decimal = Decimal(0)
    risk_rejected: bool = False
    risk_event: bool = False

    def __post_init__(self) -> None:
        for name in (
            "requested_long_level",
            "requested_short_level",
            "projected_long_level",
            "projected_short_level",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
                raise ValueError(f"{name} must be an int in [0, 4]")
        object.__setattr__(
            self,
            "realized_pnl",
            finite_decimal(self.realized_pnl, field_name="realized_pnl"),
        )
        if not isinstance(self.risk_rejected, bool) or not isinstance(self.risk_event, bool):
            raise TypeError("risk flags must be bool")


@dataclass(frozen=True, slots=True)
class PositionManagementMetrics:
    observation_count: int
    level_occupancy: tuple[tuple[str, int], ...]
    transition_matrix: tuple[tuple[str, str, int], ...]
    mean_hold_duration: tuple[tuple[str, Decimal], ...]
    scale_up_frequency: Decimal
    scale_down_frequency: Decimal
    same_level_churn_rate: Decimal
    projection_rate: Decimal
    mean_projection_distance: Decimal
    risk_reject_rate: Decimal
    profit_lock_rate: Decimal
    mean_de_risk_latency: Decimal
    probe_rate: Decimal
    gross_exposure_distribution: tuple[tuple[int, int], ...]
    net_exposure_distribution: tuple[tuple[int, int], ...]


def _state(long_level: int, short_level: int) -> str:
    return f"L{long_level}:S{short_level}"


def _ratio(numerator: int | Decimal, denominator: int) -> Decimal:
    return Decimal(numerator) / Decimal(denominator) if denominator else Decimal(0)


def position_management_metrics(
    decisions: tuple[PositionDecision, ...],
) -> PositionManagementMetrics:
    if not decisions:
        raise ValueError("at least one position decision is required")

    occupancy: Counter[str] = Counter()
    transitions: Counter[tuple[str, str]] = Counter()
    gross: Counter[int] = Counter()
    net: Counter[int] = Counter()
    runs: defaultdict[str, list[int]] = defaultdict(list)
    current_state: str | None = None
    current_run = 0
    scale_up = scale_down = churn = projected = risk_rejects = 0
    projection_distance = 0
    profit_opportunities = profit_locks = probes = 0
    pending_risk: list[tuple[int, int]] = []
    de_risk_latencies: list[int] = []

    previous: PositionDecision | None = None
    for index, decision in enumerate(decisions):
        state = _state(decision.projected_long_level, decision.projected_short_level)
        occupancy[state] += 1
        projected_gross = decision.projected_long_level + decision.projected_short_level
        gross[projected_gross] += 1
        net[decision.projected_long_level - decision.projected_short_level] += 1
        distance = abs(decision.requested_long_level - decision.projected_long_level) + abs(
            decision.requested_short_level - decision.projected_short_level
        )
        if distance:
            projected += 1
            projection_distance += distance
        risk_rejects += int(decision.risk_rejected)

        if state == current_state:
            current_run += 1
        else:
            if current_state is not None:
                runs[current_state].append(current_run)
            current_state = state
            current_run = 1

        if decision.risk_event:
            pending_risk.append((index, projected_gross))
        still_pending: list[tuple[int, int]] = []
        for started_at, starting_gross in pending_risk:
            if index > started_at and projected_gross < starting_gross:
                de_risk_latencies.append(index - started_at)
            else:
                still_pending.append((started_at, starting_gross))
        pending_risk = still_pending

        if previous is not None:
            previous_state = _state(previous.projected_long_level, previous.projected_short_level)
            transitions[(previous_state, state)] += 1
            previous_gross = previous.projected_long_level + previous.projected_short_level
            scale_up += int(projected_gross > previous_gross)
            scale_down += int(projected_gross < previous_gross)
            requested_changed = (
                decision.requested_long_level != previous.requested_long_level
                or decision.requested_short_level != previous.requested_short_level
            )
            churn += int(requested_changed and projected_gross == previous_gross)
            if previous.realized_pnl > 0:
                profit_opportunities += 1
                profit_locks += int(projected_gross < previous_gross)
            probes += int(previous_gross == 0 and 0 < projected_gross <= 1)
        previous = decision
    if current_state is None:  # Defensive: the nonempty input guard makes this unreachable.
        raise RuntimeError("position state was not initialized")
    runs[current_state].append(current_run)

    transition_count = max(0, len(decisions) - 1)
    mean_hold = tuple(
        (state, _ratio(sum(lengths), len(lengths)))
        for state, lengths in sorted(runs.items())
    )
    return PositionManagementMetrics(
        observation_count=len(decisions),
        level_occupancy=tuple(sorted(occupancy.items())),
        transition_matrix=tuple(
            (source, target, count)
            for (source, target), count in sorted(transitions.items())
        ),
        mean_hold_duration=mean_hold,
        scale_up_frequency=_ratio(scale_up, transition_count),
        scale_down_frequency=_ratio(scale_down, transition_count),
        same_level_churn_rate=_ratio(churn, transition_count),
        projection_rate=_ratio(projected, len(decisions)),
        mean_projection_distance=_ratio(projection_distance, len(decisions)),
        risk_reject_rate=_ratio(risk_rejects, len(decisions)),
        profit_lock_rate=_ratio(profit_locks, profit_opportunities),
        mean_de_risk_latency=_ratio(sum(de_risk_latencies), len(de_risk_latencies)),
        probe_rate=_ratio(probes, transition_count),
        gross_exposure_distribution=tuple(sorted(gross.items())),
        net_exposure_distribution=tuple(sorted(net.items())),
    )


class AblationAxis(StrEnum):
    LEVEL_MAPPING = "LEVEL_MAPPING"
    REWARD = "REWARD"
    PROJECTOR = "PROJECTOR"


@dataclass(frozen=True, slots=True)
class AblationEvidence:
    axis: AblationAxis
    variant: str
    net_return_delta: Decimal
    max_drawdown_delta: Decimal
    projection_rate_delta: Decimal
    behavior_preserved: bool

    def __post_init__(self) -> None:
        if not isinstance(self.axis, AblationAxis):
            raise TypeError("axis must be AblationAxis")
        object.__setattr__(
            self,
            "variant",
            required_text(self.variant, field_name="variant", max_length=128),
        )
        for name in ("net_return_delta", "max_drawdown_delta", "projection_rate_delta"):
            object.__setattr__(self, name, finite_decimal(getattr(self, name), field_name=name))
        if not isinstance(self.behavior_preserved, bool):
            raise TypeError("behavior_preserved must be bool")


def qualify_position_management(
    metrics: PositionManagementMetrics,
    ablations: tuple[AblationEvidence, ...],
    *,
    maximum_projection_rate: Decimal,
    maximum_churn_rate: Decimal,
    maximum_de_risk_latency: Decimal,
) -> tuple[bool, tuple[str, ...]]:
    limits = {
        "maximum_projection_rate": finite_decimal(
            maximum_projection_rate, field_name="maximum_projection_rate"
        ),
        "maximum_churn_rate": finite_decimal(maximum_churn_rate, field_name="maximum_churn_rate"),
        "maximum_de_risk_latency": finite_decimal(
            maximum_de_risk_latency, field_name="maximum_de_risk_latency"
        ),
    }
    if any(value < 0 for value in limits.values()):
        raise ValueError("position-management limits must be nonnegative")
    reasons: list[str] = []
    if metrics.projection_rate > limits["maximum_projection_rate"]:
        reasons.append("PROJECTOR_DEPENDENCE_TOO_HIGH")
    if metrics.same_level_churn_rate > limits["maximum_churn_rate"]:
        reasons.append("SAME_LEVEL_CHURN_TOO_HIGH")
    if metrics.mean_de_risk_latency > limits["maximum_de_risk_latency"]:
        reasons.append("DE_RISK_LATENCY_TOO_HIGH")
    axes = {evidence.axis for evidence in ablations}
    reasons.extend(
        f"ABLATION_MISSING:{axis.value}"
        for axis in sorted(set(AblationAxis) - axes, key=lambda item: item.value)
    )
    if any(not evidence.behavior_preserved for evidence in ablations):
        reasons.append("ABLATION_BEHAVIOR_UNSTABLE")
    return not reasons, tuple(reasons)
