from freqtrade.persistence.hedge_migrations import migration_plan_ids


def test_protection_migrations_follow_business_identity_migrations_contiguously() -> None:
    plan = migration_plan_ids()
    expected = (
        "H3-038-verify-business-identity",
        "H3-039-business-protection-schema",
        "H3-040-business-protection-constraints",
        "H3-041-verify-business-protection",
    )
    start = plan.index(expected[0])
    assert tuple(plan[start : start + len(expected)]) == expected
