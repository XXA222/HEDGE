from datetime import UTC, datetime
from decimal import Decimal

from freqtrade.hedge.research.offline_rl import OfflineRLTransition, offline_dataset_sha256


def test_offline_transition_binds_pit_mask_and_behavior_probability() -> None:
    row = OfflineRLTransition("t1", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC), "a"*64, 3, "b"*64, Decimal("0.2"), Decimal("1"), "c"*64, False, "RISK_LEVEL_RL")
    assert len(offline_dataset_sha256((row,))) == 64
