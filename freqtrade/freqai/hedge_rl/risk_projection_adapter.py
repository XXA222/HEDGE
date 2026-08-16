"""Runtime projection adapters for Hedge risk-level RL inference.

The risk-level policy is intentionally isolated from other experimental RL subsystems.
This module bridges both legacy and source-separated runtime projections into one context.
legacy ``CentralRuntimeProjection`` snapshots and the current source-separated
``HedgeRuntime`` into the exact policy context.  The live runtime bridge is stateful so
peak equity/downside history survive across bot loops while repeated prediction rows for
the same runtime sequence do not double-count account returns.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from threading import RLock

from freqtrade.enums.hedge import PositionSide
from freqtrade.hedge.integration.projection import CentralRuntimeProjection
from freqtrade.hedge.position_book import PositionRecord
from freqtrade.hedge.runtime import HedgeProjectionSource, HedgeRuntime, HedgeRuntimeView
from freqtrade.hedge.symbols import canonicalize_symbol

from .risk_bridge import HedgeRiskPolicyContext
from .risk_levels import RiskLevelProfile
from .risk_portfolio import LegSide, RiskAccountState, RiskLegState


def _level_for_observed_margin_fraction(profile: RiskLevelProfile, fraction: float) -> int:
    """Return a conservative level that never understates observed margin risk."""

    value = max(0.0, float(fraction))
    for level, cap in enumerate(profile.position_levels):
        if value <= cap + 1e-12:
            return level
    return len(profile.position_levels) - 1


def _leg_state(
    positions: Iterable[PositionRecord],
    *,
    symbol: str,
    side: PositionSide,
) -> RiskLegState:
    leg_side = LegSide.LONG if side is PositionSide.LONG else LegSide.SHORT
    for position in positions:
        if position.symbol == symbol and position.position_side is side:
            return RiskLegState(
                leg_side,
                float(position.amount),
                float(position.entry_price),
            )
    return RiskLegState(leg_side)


def _leverage_compatible(
    positions: Iterable[PositionRecord],
    *,
    pair: str,
    profile: RiskLevelProfile,
) -> bool:
    """Reject account facts whose actual leverage disagrees with the risk profile."""

    symbol = canonicalize_symbol(pair)
    for position in positions:
        if position.symbol != symbol or position.amount <= 0:
            continue
        expected = (
            profile.long_leverage
            if position.position_side is PositionSide.LONG
            else profile.short_leverage
        )
        actual = float(position.leverage)
        tolerance = max(1e-9, abs(expected) * 1e-6)
        if not math.isfinite(actual) or actual <= 0 or abs(actual - expected) > tolerance:
            return False
    return True


def _account_from_facts(
    *,
    positions: Iterable[PositionRecord],
    risk,
    pair: str,
    mark: float,
    profile: RiskLevelProfile,
    step: int,
    peak_equity: float | None = None,
) -> RiskAccountState:
    symbol = canonicalize_symbol(pair)
    equity = float(risk.equity)
    wallet = float(risk.wallet_balance)
    long = _leg_state(positions, symbol=symbol, side=PositionSide.LONG)
    short = _leg_state(positions, symbol=symbol, side=PositionSide.SHORT)

    long_notional = long.notional(mark)
    short_notional = short.notional(mark)
    long_margin_fraction = (
        long_notional / max(profile.long_leverage, 1e-12) / equity if equity > 0 else 1.0
    )
    short_margin_fraction = (
        short_notional / max(profile.short_leverage, 1e-12) / equity if equity > 0 else 1.0
    )
    peak = max(equity, wallet) if peak_equity is None else max(float(peak_equity), equity)
    return RiskAccountState(
        cash_balance=wallet,
        equity=equity,
        peak_equity=peak,
        long=long,
        short=short,
        long_level=_level_for_observed_margin_fraction(profile, long_margin_fraction),
        short_level=_level_for_observed_margin_fraction(profile, short_margin_fraction),
        step=max(0, int(step)),
        turnover=0.0,
    )


def _empty_context(
    *,
    mark: float,
    funding_rate: float,
    feature_age_steps: int,
    failed_probe_long: int,
    failed_probe_short: int,
    downside_semideviation: float,
    pending_reward_fraction: float,
) -> HedgeRiskPolicyContext:
    return HedgeRiskPolicyContext(
        account=RiskAccountState.initial(1.0),
        mark=mark,
        uncertainty_score=1.0,
        funding_rate=float(funding_rate),
        feature_age_steps=int(feature_age_steps),
        projection_fresh=False,
        failed_probe_long=int(failed_probe_long),
        failed_probe_short=int(failed_probe_short),
        downside_semideviation=float(downside_semideviation),
        pending_reward_fraction=float(pending_reward_fraction),
    )


def context_from_central_projection(
    projection: CentralRuntimeProjection,
    *,
    pair: str,
    mark: float,
    profile: RiskLevelProfile,
    uncertainty_score: float = 0.5,
    funding_rate: float = 0.0,
    feature_age_steps: int = 0,
    failed_probe_long: int = 0,
    failed_probe_short: int = 0,
    downside_semideviation: float = 0.0,
    pending_reward_fraction: float = 0.0,
) -> HedgeRiskPolicyContext:
    """Build inference context from the legacy exchange-only central projection."""

    mark_value = float(mark)
    if not math.isfinite(mark_value) or mark_value <= 0:
        raise ValueError("mark must be finite and positive")
    risk = projection.risk
    if risk is None:
        return _empty_context(
            mark=mark_value,
            funding_rate=funding_rate,
            feature_age_steps=feature_age_steps,
            failed_probe_long=failed_probe_long,
            failed_probe_short=failed_probe_short,
            downside_semideviation=downside_semideviation,
            pending_reward_fraction=pending_reward_fraction,
        )

    required_checks = (
        "exchange.rest_calibrated",
        "exchange.user_stream_fresh",
        "exchange.reconciliation_converged",
        "exchange.risk_snapshot_valid",
    )
    checks_fresh = all(bool(projection.checks.get(name, False)) for name in required_checks)
    projection_fresh = bool(
        not projection.stale
        and risk.effective_risk_data_valid
        and checks_fresh
        and projection.reconciliation_status == "HEALTHY"
        and _leverage_compatible(projection.positions, pair=pair, profile=profile)
    )
    account = _account_from_facts(
        positions=projection.positions,
        risk=risk,
        pair=pair,
        mark=mark_value,
        profile=profile,
        step=risk.source_version,
    )
    return HedgeRiskPolicyContext(
        account=account,
        mark=mark_value,
        uncertainty_score=float(uncertainty_score),
        funding_rate=float(funding_rate),
        feature_age_steps=int(feature_age_steps),
        projection_fresh=projection_fresh,
        failed_probe_long=int(failed_probe_long),
        failed_probe_short=int(failed_probe_short),
        downside_semideviation=float(downside_semideviation),
        pending_reward_fraction=float(pending_reward_fraction),
    )


def _runtime_mark(view: HedgeRuntimeView, *, pair: str, fallback: float) -> float:
    symbol = canonicalize_symbol(pair)
    marks: list[float] = []
    for position in view.positions:
        if position.symbol != symbol:
            continue
        reference = float(position.reference_price)
        if math.isfinite(reference) and reference > 0:
            marks.append(reference)
    if marks:
        return sum(marks) / len(marks)
    fallback_value = float(fallback)
    return fallback_value if math.isfinite(fallback_value) and fallback_value > 0 else 1.0


def _runtime_projection_fresh(
    view: HedgeRuntimeView,
    *,
    pair: str,
    profile: RiskLevelProfile,
) -> bool:
    risk = view.risk
    if risk is None or not risk.effective_risk_data_valid:
        return False
    if view.stale or view.halted or not view.ready:
        return False
    checks = dict(view.checks)
    if not checks or not all(bool(value) for value in checks.values()):
        return False
    if not _leverage_compatible(view.positions, pair=pair, profile=profile):
        return False
    if view.source is HedgeProjectionSource.EXCHANGE:
        return view.reconciliation_status == "HEALTHY"
    if view.source is HedgeProjectionSource.PAPER:
        return view.reconciliation_status in {"HEALTHY", "NOT_APPLICABLE"}
    if view.source is HedgeProjectionSource.LIVE:
        return view.reconciliation_status in {"HEALTHY", "NOT_APPLICABLE"}
    return False


def context_from_runtime_view(
    view: HedgeRuntimeView,
    *,
    pair: str,
    profile: RiskLevelProfile,
    fallback_mark: float = 1.0,
    peak_equity: float | None = None,
    uncertainty_score: float = 0.5,
    funding_rate: float = 0.0,
    feature_age_steps: int = 0,
    failed_probe_long: int = 0,
    failed_probe_short: int = 0,
    downside_semideviation: float = 0.0,
    pending_reward_fraction: float = 0.0,
) -> HedgeRiskPolicyContext:
    """Build a policy context from the effective source-separated runtime view.

    PAPER projections are intentionally supported.  Requiring exchange user-stream checks
    for PAPER would force every Docker/native dry-run policy to FLAT even when the paper
    ledger itself is healthy.
    """

    mark = _runtime_mark(view, pair=pair, fallback=fallback_mark)
    risk = view.risk
    if risk is None:
        return _empty_context(
            mark=mark,
            funding_rate=funding_rate,
            feature_age_steps=feature_age_steps,
            failed_probe_long=failed_probe_long,
            failed_probe_short=failed_probe_short,
            downside_semideviation=downside_semideviation,
            pending_reward_fraction=pending_reward_fraction,
        )
    account = _account_from_facts(
        positions=view.positions,
        risk=risk,
        pair=pair,
        mark=mark,
        profile=profile,
        step=view.sequence,
        peak_equity=peak_equity,
    )
    return HedgeRiskPolicyContext(
        account=account,
        mark=mark,
        uncertainty_score=float(uncertainty_score),
        funding_rate=float(funding_rate),
        feature_age_steps=int(feature_age_steps),
        projection_fresh=_runtime_projection_fresh(view, pair=pair, profile=profile),
        failed_probe_long=int(failed_probe_long),
        failed_probe_short=int(failed_probe_short),
        downside_semideviation=float(downside_semideviation),
        pending_reward_fraction=float(pending_reward_fraction),
    )


@dataclass(slots=True)
class _RuntimeHistory:
    source: HedgeProjectionSource | None = None
    sequence: int = -1
    peak_equity: float = 0.0
    last_equity: float | None = None
    downside_returns: deque[float] = field(default_factory=lambda: deque(maxlen=64))


class HedgeRiskRuntimeContextProvider:
    """Stateful callable that binds a risk-level FreqAI model to ``HedgeRuntime``.

    Account history advances once per runtime sequence, not once per DataFrame row.  This
    matters because FreqAI prediction can ask for many historical rows while the account
    projection is a single current snapshot.
    """

    def __init__(
        self,
        runtime: HedgeRuntime,
        *,
        profile: RiskLevelProfile,
        uncertainty_score: float = 0.5,
        downside_window: int = 64,
    ) -> None:
        if not isinstance(runtime, HedgeRuntime):
            raise TypeError("runtime must be a HedgeRuntime")
        if isinstance(downside_window, bool) or int(downside_window) < 2:
            raise ValueError("downside_window must be >= 2")
        self.runtime = runtime
        self.profile = profile
        self.uncertainty_score = float(uncertainty_score)
        if not 0.0 <= self.uncertainty_score <= 1.0:
            raise ValueError("uncertainty_score must be within [0, 1]")
        self.downside_window = int(downside_window)
        self._history: dict[str, _RuntimeHistory] = {}
        self._lock = RLock()

    def reset(self) -> None:
        with self._lock:
            self._history.clear()

    def _advance_history(self, pair: str, view: HedgeRuntimeView) -> _RuntimeHistory:
        key = canonicalize_symbol(pair)
        history = self._history.get(key)
        if history is None:
            history = _RuntimeHistory(downside_returns=deque(maxlen=self.downside_window))
            self._history[key] = history
        risk = view.risk
        if risk is None:
            return history
        equity = float(risk.equity)
        reset = history.source is not None and (
            view.source is not history.source or view.sequence < history.sequence
        )
        if reset:
            history.source = None
            history.sequence = -1
            history.peak_equity = 0.0
            history.last_equity = None
            history.downside_returns.clear()
        if view.sequence != history.sequence or view.source is not history.source:
            if history.last_equity is not None and history.last_equity > 0 and equity > 0:
                value = math.log(equity / history.last_equity)
                history.downside_returns.append(min(0.0, value))
            history.source = view.source
            history.sequence = view.sequence
            history.peak_equity = max(history.peak_equity, equity, float(risk.wallet_balance))
            history.last_equity = equity
        return history

    @staticmethod
    def _downside_semideviation(history: _RuntimeHistory) -> float:
        values = history.downside_returns
        if not values:
            return 0.0
        return math.sqrt(sum(value * value for value in values) / len(values))

    def __call__(self, pair: str, tick: int, index_value: object) -> HedgeRiskPolicyContext:
        del tick, index_value
        with self._lock:
            view = self.runtime.view()
            history = self._advance_history(pair, view)
            context = context_from_runtime_view(
                view,
                pair=pair,
                profile=self.profile,
                peak_equity=history.peak_equity or None,
                uncertainty_score=self.uncertainty_score,
                downside_semideviation=self._downside_semideviation(history),
            )
            if history.peak_equity > context.account.peak_equity:
                context = replace(
                    context,
                    account=replace(context.account, peak_equity=history.peak_equity),
                )
            return context
