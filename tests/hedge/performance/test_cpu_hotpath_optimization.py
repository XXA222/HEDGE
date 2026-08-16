from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

from freqtrade.hedge.planning.context import (
    IntentAction,
    OrderIntent,
    OrderSide,
    PlannerConfig,
    PositionBucket,
    PositionSide,
    StrategyLegState,
    TrailingPhase,
)
from freqtrade.hedge.simulation.cross_wallet import CrossWallet
from freqtrade.hedge.simulation.exchange import BarEvent, SignalEvent
from freqtrade.hedge.simulation.matcher import ConservativeMatcher
from freqtrade.hedge.simulation.replay import EventReplayEngine


NOW = datetime(2026, 8, 13, tzinfo=UTC)


def _intent(*, price: str = "95", side: PositionSide = PositionSide.LONG) -> OrderIntent:
    return OrderIntent.deterministic(
        symbol="BTC/USDT:USDT",
        position_side=side,
        order_side=OrderSide.BUY if side is PositionSide.LONG else OrderSide.SELL,
        action=IntentAction.OPEN,
        bucket=PositionBucket.CORE,
        quantity=Decimal("0.01"),
        price=Decimal(price),
        reduce_only=False,
    )


def _bar(*, low: str = "99", high: str = "101") -> BarEvent:
    return BarEvent(
        timestamp=NOW,
        symbol="BTC/USDT:USDT",
        open=Decimal(100),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(100),
        volume=Decimal(100),
    )


def test_no_touch_active_order_skips_matcher_wallet_clone() -> None:
    wallet = CrossWallet(Decimal(1000))
    wallet.accept_order("far", _intent(price="90"), accepted_at=NOW - timedelta(minutes=1))
    matcher = ConservativeMatcher()
    with patch.object(matcher, "_clone", side_effect=AssertionError("clone should not run")):
        outcome = matcher.match_outcome(_bar(low="99", high="101"), wallet)
    assert outcome.fills == ()
    assert outcome.expired_order_ids == ()
    assert wallet.remaining("far") == Decimal("0.01")


def test_touching_active_order_keeps_full_conservative_matcher_path() -> None:
    wallet = CrossWallet(Decimal(1000))
    wallet.accept_order("touch", _intent(price="99.5"), accepted_at=NOW - timedelta(minutes=1))
    matcher = ConservativeMatcher()
    original = matcher._clone
    calls = 0

    def counted(value: CrossWallet) -> CrossWallet:
        nonlocal calls
        calls += 1
        return original(value)

    with patch.object(matcher, "_clone", side_effect=counted):
        outcome = matcher.match_outcome(_bar(low="99", high="101"), wallet)
    assert calls == 2
    assert outcome.fills


def test_planner_active_order_projection_is_cached_until_order_book_changes() -> None:
    wallet = CrossWallet(Decimal(1000))
    item = _intent(price="90")
    wallet.accept_order("order-1", item, accepted_at=NOW)
    first = wallet.planner_snapshot(Decimal(100), NOW)
    second = wallet.planner_snapshot(Decimal(101), NOW + timedelta(minutes=1))
    assert first.active_orders is second.active_orders
    wallet.cancel_order("order-1")
    third = wallet.planner_snapshot(Decimal(101), NOW + timedelta(minutes=2))
    assert third.active_orders == ()
    assert first.active_orders is not third.active_orders


def test_leg_projection_cache_invalidates_after_position_change() -> None:
    wallet = CrossWallet(Decimal(1000))
    first = wallet.long.immutable()
    second = wallet.long.immutable()
    assert first is second
    item = _intent(price="100")
    wallet.accept_order("entry", item, accepted_at=NOW)
    from freqtrade.hedge.simulation.exchange import FillEvent

    fill = FillEvent(
        event_id="fill-1",
        timestamp=NOW,
        order_id="entry",
        intent_id=item.intent_id,
        symbol=item.symbol,
        position_side=item.position_side,
        quantity=item.quantity,
        price=Decimal(100),
        fee=Decimal(0),
        reduce_only=False,
        bucket=item.bucket,
        action=item.action,
        layer=item.layer,
    )
    wallet.apply_fill(fill)
    third = wallet.long.immutable()
    assert third is not first
    assert third.quantity == Decimal("0.01")


def test_strategy_leg_trusted_evolution_preserves_immutable_semantics() -> None:
    state = StrategyLegState(PositionSide.LONG)
    next_state = state.next_sequence()
    assert next_state is not state
    assert next_state.sequence == 1
    assert state.sequence == 0
    changed = state._evolve_trusted(
        trailing_phase=TrailingPhase.ARMED,
        trailing_extreme=Decimal(100),
        trailing_started_at=NOW,
    )
    assert changed.trailing_phase is TrailingPhase.ARMED
    assert changed.trailing_extreme == Decimal(100)
    assert state.trailing_phase is TrailingPhase.IDLE


def test_compact_replay_caches_effective_planner_config_for_constant_risk_scalars() -> None:
    engine = EventReplayEngine(initial_balance=Decimal(1000))
    events = []
    for index in range(8):
        ts = NOW + timedelta(minutes=index + 1)
        events.extend(
            (
                SignalEvent(
                    timestamp=ts,
                    symbol="BTC/USDT:USDT",
                    long_signal=Decimal("0.5"),
                    short_signal=Decimal("0.2"),
                    confidence=Decimal("0.8"),
                    risk_scale=Decimal("0.9"),
                ),
                BarEvent(
                    timestamp=ts,
                    symbol="BTC/USDT:USDT",
                    open=Decimal(100),
                    high=Decimal(101),
                    low=Decimal(99),
                    close=Decimal(100),
                    volume=Decimal(100),
                ),
            )
        )
    with patch(
        "freqtrade.hedge.simulation.replay.planner_config_for_directive",
        wraps=__import__(
            "freqtrade.hedge.simulation.replay", fromlist=["planner_config_for_directive"]
        ).planner_config_for_directive,
    ) as wrapped:
        engine.replay_ordered_stream(events)
    assert wrapped.call_count == 1


def test_compact_planner_diagnostics_are_optional_but_detailed_default_is_preserved() -> None:
    engine = EventReplayEngine(initial_balance=Decimal(1000), planner_config=PlannerConfig())
    bar = _bar()
    detailed = engine._plan(bar, emit_events=False, collect_diagnostics=True)
    # _plan returns events; diagnostics are intentionally internal to PlanningResult.
    # Verify the fast flag does not alter wallet/order semantics by replaying from a
    # fresh engine and comparing final reports.
    engine_fast = EventReplayEngine(initial_balance=Decimal(1000), planner_config=PlannerConfig())
    engine_fast._plan(bar, emit_events=False, collect_diagnostics=False)
    assert engine.wallet.active_orders == engine_fast.wallet.active_orders
    assert engine.long_state == engine_fast.long_state
    assert engine.short_state == engine_fast.short_state
    assert detailed == ()


def test_no_fill_state_advance_is_a_noop_outside_liquidation() -> None:
    engine = EventReplayEngine(initial_balance=Decimal(1000))
    long_before = engine.long_state
    short_before = engine.short_state
    engine._advance_states_after_fills(_bar(), ())
    assert engine.long_state is long_before
    assert engine.short_state is short_before
