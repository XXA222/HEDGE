from __future__ import annotations

from datetime import UTC, datetime

from freqtrade.hedge.production.source_authority import SourceAuthority
from freqtrade.hedge.production.model_governance import ApprovalRecord, ModelIdentity, ModelStatus
from freqtrade.hedge.risk.policy_identity import RiskPolicyIdentity


HASH = "b" * 64
GIT = "c" * 40


def test_clean_hedge_authority_is_promotable_and_stable() -> None:
    authority = SourceAuthority(
        repository="XXA222/HEDGE", branch="master", commit_sha=GIT, tree_sha=GIT,
        manifest_sha256=HASH, source_dirty=False, release_id="r1",
        generated_at=datetime(2026, 8, 17, tzinfo=UTC),
    )
    assert authority.promotable
    assert len(authority.identity_sha256) == 64


def test_dirty_or_foreign_source_never_promotes() -> None:
    authority = SourceAuthority(
        repository="XXA222/HPRL", branch="master", commit_sha=GIT, tree_sha=GIT,
        manifest_sha256=HASH, source_dirty=True, release_id="r1",
        generated_at=datetime(2026, 8, 17, tzinfo=UTC),
    )
    assert not authority.promotable


def test_model_promotion_requires_matching_clean_source_identity() -> None:
    authority = SourceAuthority(
        repository="XXA222/HEDGE", branch="master", commit_sha=GIT, tree_sha=GIT,
        manifest_sha256=HASH, source_dirty=False, release_id="r1",
        generated_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    identity = ModelIdentity(
        "model", "HPRL", HASH, HASH, HASH, HASH, "torch",
        source_authority_sha256=authority.identity_sha256,
    )
    record = ApprovalRecord(
        identity, ModelStatus.APPROVED, datetime(2020, 1, 1, tzinfo=UTC), "ops",
        True, True, True, "safe",
    )
    assert record.promotable_from(authority)


def test_risk_policy_identity_is_component_stable() -> None:
    identity = RiskPolicyIdentity("v1", "hedge", HASH, HASH, HASH, HASH, HASH)
    assert len(identity.identity_sha256) == 64
