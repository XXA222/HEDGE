"""Causal HPRL action adapter for ETH dual-leg Hedge strategies.

The model is deliberately outside this adapter: training/inference owns tensors and checkpoints,
while this module is the narrow, deterministic boundary from a policy action to a canonical HEDGE
strategy directive.  This prevents a strategy callback from silently bypassing HEDGE risk limits.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.contracts.events import BarEvent, SignalEvent, SimulationInputEvent
from freqtrade.hedge.hprl.config import HPRLActionConfig
from freqtrade.hedge.strategies.contract import StrategyDirective


ZERO = Decimal(0)
ONE = Decimal(1)


def _unit(value: object) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("HPRL policy actions must be finite")
    return min(ONE, max(ZERO, result))


@dataclass(frozen=True, slots=True)
class HprlEthDualLegConfig:
    action: HPRLActionConfig = HPRLActionConfig()
    model_version: str = "hprl-eth-two-year"
    confidence_floor: Decimal = Decimal("0.20")

    def __post_init__(self) -> None:
        if not self.model_version or len(self.model_version) > 128:
            raise ValueError("model_version must contain 1..128 characters")
        if not self.confidence_floor.is_finite() or not ZERO <= self.confidence_floor <= ONE:
            raise ValueError("confidence_floor must be within [0, 1]")


class HprlEthDualLegStrategy:
    """Adapt one causal HPRL LONG/SHORT action to the Hedge strategy contract."""

    def __init__(self, config: HprlEthDualLegConfig | None = None) -> None:
        self.config = config or HprlEthDualLegConfig()

    def directive_from_policy_action(self, action: Sequence[object]) -> StrategyDirective:
        if len(action) != 2:
            raise ValueError("ETH dual-leg HPRL strategy requires exactly LONG and SHORT actions")
        long_code, short_code = (_unit(value) for value in action)
        levels = self.config.action.position_levels
        maximum_index = len(levels) - 1
        long_index = min(maximum_index, max(0, int(long_code * maximum_index + Decimal("0.5"))))
        short_index = min(maximum_index, max(0, int(short_code * maximum_index + Decimal("0.5"))))
        long_margin = Decimal(str(levels[long_index]))
        short_margin = Decimal(str(levels[short_index]))
        confidence = max(self.config.confidence_floor, abs(long_code - short_code))
        risk_scale = min(ONE, (long_margin + short_margin) / Decimal(str(self.config.action.max_gross_margin_ratio)))
        return StrategyDirective(
            long_score=long_code,
            short_score=short_code,
            target_net_ratio=long_margin - short_margin,
            confidence=confidence,
            risk_scale=risk_scale,
            long_exposure_scale=min(ONE, long_margin / Decimal(str(self.config.action.max_leg_margin_ratio))),
            short_exposure_scale=min(ONE, short_margin / Decimal(str(self.config.action.max_leg_margin_ratio))),
            allow_new_risk=bool(long_margin or short_margin),
            regime="HPRL",
            reason=f"tiered-policy:{long_index}/{short_index}",
            model_version=self.config.model_version,
        )

    def events(
        self,
        bars: Iterable[BarEvent],
        policy_actions: Iterable[Sequence[object]],
    ) -> Iterator[SimulationInputEvent]:
        """Emit a precomputed causal action before its matching bar.

        ``policy_actions[t]`` must have been generated only from information available before
        ``bars[t]``.  The two-year research runner enforces this by shifting every feature row.
        """
        for bar, action in zip(bars, policy_actions, strict=True):
            directive = self.directive_from_policy_action(action)
            yield SignalEvent(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                long_signal=directive.long_score,
                short_signal=directive.short_score,
            )
            yield bar
