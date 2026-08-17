from decimal import Decimal

from freqtrade.hedge.production.release_qualification import QualificationScorecard, ReleaseHardGates, qualify_release


def test_release_requires_every_hard_gate_and_score_dimension() -> None:
    gates = ReleaseHardGates(*([True] * 10))
    score = QualificationScorecard(*([Decimal("0.8")] * 7))
    assert qualify_release(gates, score).qualified


def test_one_failed_gate_blocks_release() -> None:
    gates = ReleaseHardGates(True, True, True, True, False, True, True, True, True, True)
    score = QualificationScorecard(*([Decimal("0.8")] * 7))
    decision = qualify_release(gates, score)
    assert not decision.qualified
    assert "EXCHANGE" in decision.reasons
