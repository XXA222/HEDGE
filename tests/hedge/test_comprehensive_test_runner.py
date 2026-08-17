import sys
from datetime import UTC, datetime
from pathlib import Path

from tools.run_hedge_comprehensive_tests import (
    PhaseResult,
    PhaseStatus,
    _json_payload,
    _markdown_report,
    execute_phase,
    run_phases,
    select_phases,
)
from tools.run_hedge_comprehensive_tests import TestPhase as _TestPhase


def _phase(identifier: str, **changes: object) -> _TestPhase:
    values = {
        "identifier": identifier,
        "description": identifier,
        "command": ("python", "-c", "pass"),
    }
    values.update(changes)
    return _TestPhase(**values)  # type: ignore[arg-type]


def _result(phase: _TestPhase, status: PhaseStatus) -> PhaseResult:
    return PhaseResult(
        phase.identifier,
        status,
        0.1,
        0 if status is PhaseStatus.PASS else 1,
        phase.command,
        "phase.log",
        None,
        "" if status is PhaseStatus.PASS else "expected failure",
    )


def test_failed_phase_does_not_stop_later_phases() -> None:
    phases = (_phase("first"), _phase("second"), _phase("third"))
    visited: list[str] = []

    def executor(phase: _TestPhase) -> PhaseResult:
        visited.append(phase.identifier)
        return _result(phase, PhaseStatus.FAIL if phase.identifier == "first" else PhaseStatus.PASS)

    results = run_phases(phases, executor)
    assert visited == ["first", "second", "third"]
    assert [result.status for result in results] == [
        PhaseStatus.FAIL,
        PhaseStatus.PASS,
        PhaseStatus.PASS,
    ]


def test_phase_selection_skips_only_opt_in_or_unavailable_capabilities() -> None:
    phases = (
        _phase("normal"),
        _phase("network", requires_external=True),
        _phase("long", requires_long_run=True),
        _phase("gpu", requires_cuda=True),
    )
    selected, skipped = select_phases(
        phases,
        profile="standard",
        include_external=False,
        include_long=False,
        cuda_available=False,
    )
    assert [phase.identifier for phase in selected] == ["normal"]
    assert {row.identifier for row in skipped} == {"network", "long", "gpu"}
    assert {row.status for row in skipped} == {PhaseStatus.SKIPPED}


def test_summary_records_failures_without_losing_later_results() -> None:
    phases = (_phase("failure"), _phase("later"))
    payload = _json_payload(
        root=Path("D:/Program Files/HEDGE"),
        profile="standard",
        results=(_result(phases[0], PhaseStatus.FAIL), _result(phases[1], PhaseStatus.PASS)),
        started_at=datetime(2026, 8, 17, tzinfo=UTC),
        finished_at=datetime(2026, 8, 17, 0, 1, tzinfo=UTC),
    )
    assert payload["passed"] is False
    assert payload["status_counts"] == {
        "PASS": 1,
        "FAIL": 1,
        "TIMEOUT": 0,
        "ERROR": 0,
        "SKIPPED": 0,
    }
    report = _markdown_report(payload)
    assert "| failure | FAIL |" in report
    assert "| later | PASS |" in report


def test_real_subprocess_failure_is_logged_and_later_phase_runs(tmp_path: Path) -> None:
    phases = (
        _phase("fail", command=(sys.executable, "-c", "raise SystemExit(7)")),
        _phase("later", command=(sys.executable, "-c", "print('later phase ran')")),
    )

    def executor(phase: _TestPhase) -> PhaseResult:
        return execute_phase(
            phase,
            root=Path.cwd(),
            output_dir=tmp_path,
            python=sys.executable,
            timeout_seconds=10,
        )

    results = run_phases(phases, executor)
    assert [result.status for result in results] == [PhaseStatus.FAIL, PhaseStatus.PASS]
    assert "later phase ran" in Path(results[1].log_path).read_text(encoding="utf-8")
