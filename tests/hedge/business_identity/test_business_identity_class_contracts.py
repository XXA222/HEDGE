from __future__ import annotations

from dataclasses import fields

from freqtrade.hedge.integration.paper_events import RecoveredPaperFill
from freqtrade.hedge.planning.context import ActiveOrder, LegPosition, TacticalLot


def _field_names(model: type[object]) -> set[str]:
    return {item.name for item in fields(model)}


def test_r4_2_class_scoped_fields_are_installed_on_the_correct_dataclasses() -> None:
    tactical_fields = _field_names(TacticalLot)
    active_fields = _field_names(ActiveOrder)
    leg_fields = _field_names(LegPosition)
    recovered_fields = _field_names(RecoveredPaperFill)

    assert {"business_identity", "bucket"} <= tactical_fields
    assert "strategy_entry_key" in active_fields
    assert "position_lots" in leg_fields
    assert {"business_identity", "order_role"} <= recovered_fields
