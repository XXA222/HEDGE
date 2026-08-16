from freqtrade.hedge.production.hprl_algorithm_qualification import (
    AlgorithmQualificationPolicy,
    AlgorithmTrialEvidence,
    qualify_algorithm,
    qualify_candidate_set,
    qualify_real_market_model,
)

SOURCE = "a" * 64
BEHAVIOR = "b" * 64


def _trial(
    algorithm,
    seed,
    window,
    ret,
    *,
    behavior=True,
    parity=True,
    latency=20.0,
    drawdown=0.08,
):
    return AlgorithmTrialEvidence(
        algorithm=algorithm,
        seed=seed,
        window_id=window,
        sample_count=20_000,
        net_return=ret,
        sharpe=1.2,
        sortino=1.5,
        calmar=1.1,
        max_drawdown=drawdown,
        cvar95=0.03,
        turnover=1.0,
        fee_fraction=0.002,
        funding_fraction=0.001,
        slippage_fraction=0.001,
        behavior_passed=behavior,
        behavior_sha256=BEHAVIOR,
        replay_parity_passed=parity,
        inference_p95_ms=latency,
        source_sha256=SOURCE,
    )


def _matrix(algorithm, premium):
    rows = []
    for seed in range(3):
        for window in ("wf1", "wf2", "wf3"):
            rows.append(_trial(algorithm, seed, window, 0.04 + premium + seed * 0.001))
    return rows


def test_candidate_must_beat_paired_deterministic_baseline():
    baseline = _matrix("deterministic", 0.0)
    candidate = _matrix("fast_td3", 0.03)
    report = qualify_algorithm(candidate, baseline)
    assert report.passed
    assert report.paired_trials == 9
    assert report.median_excess_return > 0


def test_behavior_failure_blocks_profitable_algorithm():
    baseline = _matrix("deterministic", 0.0)
    candidate = _matrix("fast_td3", 0.10)
    candidate[0] = _trial("fast_td3", 0, "wf1", 0.20, behavior=False)
    report = qualify_algorithm(candidate, baseline)
    assert not report.passed
    assert "ALGORITHM_POSITION_BEHAVIOR_NOT_UNIVERSALLY_PASSED" in report.reasons


def test_selection_ranks_only_qualified_candidates():
    rows = _matrix("deterministic", 0.0) + _matrix("fast_td3", 0.03) + _matrix("xqc", 0.01)
    report = qualify_candidate_set(rows)
    assert report.passed
    assert report.champion == "fast_td3"


def test_missing_walkforward_coverage_fails():
    baseline = _matrix("deterministic", 0.0)
    candidate = [_trial("fast_td3", seed, "wf1", 0.2) for seed in range(3)]
    report = qualify_algorithm(
        candidate,
        baseline,
        policy=AlgorithmQualificationPolicy(minimum_pair_coverage=1.0),
    )
    assert not report.passed
    assert "ALGORITHM_WALKFORWARD_COVERAGE_INSUFFICIENT" in report.reasons


class _RealMarket:
    passed = True
    production_evidence_eligible = True
    real_trade_write_count = 0
    model_target_feed = True
    evidence_sha256 = "c" * 64


class _Behavior:
    passed = True
    semantic_sha256 = "d" * 64


def test_real_market_model_composite_requires_zero_write_and_true_model_feed():
    selection = qualify_candidate_set(_matrix("deterministic", 0.0) + _matrix("fast_td3", 0.03))
    report = qualify_real_market_model(_RealMarket(), _Behavior(), selection)
    assert report.passed
    assert report.champion == "fast_td3"


def test_real_market_model_composite_rejects_exchange_write():
    class _UnsafeMarket(_RealMarket):
        real_trade_write_count = 1

    selection = qualify_candidate_set(_matrix("deterministic", 0.0) + _matrix("fast_td3", 0.03))
    report = qualify_real_market_model(_UnsafeMarket(), _Behavior(), selection)
    assert not report.passed
    assert "REAL_MARKET_ZERO_WRITE_INVARIANT_FAILED" in report.reasons
