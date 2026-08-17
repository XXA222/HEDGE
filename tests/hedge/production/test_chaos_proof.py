from freqtrade.hedge.production.chaos_proof import ChaosCaseResult, ChaosScenario, qualify_chaos_recovery


def test_all_six_recovery_scenarios_are_required() -> None:
    rows = tuple(ChaosCaseResult(scenario, True, True, True, True) for scenario in ChaosScenario)
    assert qualify_chaos_recovery(rows) == (True, ())


def test_missing_scenario_fails_closed() -> None:
    passed, reasons = qualify_chaos_recovery(())
    assert not passed and len(reasons) == 6
