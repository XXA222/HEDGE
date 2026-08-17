#!/usr/bin/env python3
"""Run HEDGE test domains independently and continue after every failed phase.

Every test domain runs in a dedicated subprocess. A failed, timed-out, or collection-broken
phase is recorded but never prevents later domains from running. The runner emits JSON,
Markdown, phase logs and JUnit XML reports for pytest phases.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class PhaseStatus(StrEnum):
    PASS = "PASS"  # noqa: S105 - status label, never a credential
    FAIL = "FAIL"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class TestPhase:
    identifier: str
    description: str
    command: tuple[str, ...]
    pytest_phase: bool = False
    profiles: tuple[str, ...] = ("standard", "full")
    requires_external: bool = False
    requires_long_run: bool = False
    requires_cuda: bool = False

    def __post_init__(self) -> None:
        if not self.identifier.replace("_", "").isalnum():
            raise ValueError("phase identifier must be alphanumeric with underscores")
        if not self.description.strip() or not self.command:
            raise ValueError("phase description and command are required")
        if not set(self.profiles) <= {"smoke", "standard", "full"}:
            raise ValueError("unknown phase profile")


@dataclass(frozen=True, slots=True)
class PhaseResult:
    identifier: str
    status: PhaseStatus
    duration_seconds: float
    return_code: int | None
    command: tuple[str, ...]
    log_path: str
    junit_path: str | None
    detail: str = ""


def _pytest(identifier: str, *paths: str) -> TestPhase:
    return TestPhase(
        identifier=f"pytest_{identifier}",
        description="pytest: " + ", ".join(paths),
        command=("{python}", "-m", "pytest", "-q", *paths),
        pytest_phase=True,
    )


def default_phases() -> tuple[TestPhase, ...]:
    """Return the test matrix without executing it, suitable for audit and extension."""
    phases = (
        TestPhase(
            "static_ruff_correctness",
            "Ruff fatal-correctness analysis for HEDGE source, tests and tools",
            (
                "{python}",
                "-m",
                "ruff",
                "check",
                "freqtrade/hedge",
                "tests/hedge",
                "tools",
                "--select",
                "E9,F63,F7,F82",
            ),
            profiles=("smoke", "standard", "full"),
        ),
        TestPhase(
            "clean_mainline",
            "Clean-mainline source authority and exhaustive manifest verification",
            (
                "{python}",
                "tools/validate_clean_mainline.py",
                "--project-root",
                ".",
                "--workspace-mode",
            ),
            profiles=("smoke", "standard", "full"),
        ),
        _pytest(
            "core_contracts",
            "tests/hedge/test_architecture.py",
            "tests/hedge/test_contracts.py",
            "tests/hedge/test_domain.py",
            "tests/hedge/test_domain_safety.py",
            "tests/hedge/test_numeric_symbols.py",
            "tests/hedge/test_patch_integrity.py",
        ),
        _pytest(
            "risk_wallet_planning",
            "tests/hedge/risk",
            "tests/hedge/wallet",
            "tests/hedge/planning",
        ),
        _pytest(
            "execution_exchange_api",
            "tests/hedge/execution",
            "tests/hedge/exchange",
            "tests/hedge/api",
        ),
        _pytest(
            "persistence_concurrency_migrations",
            "tests/hedge/persistence",
            "tests/hedge/concurrency",
            "tests/hedge/migrations",
        ),
        _pytest("research_optimization", "tests/hedge/research", "tests/hedge/optimization"),
        _pytest(
            "roadmap_research",
            "tests/hedge/test_benchmark_tower.py",
            "tests/hedge/test_evidence_dag.py",
            "tests/hedge/test_offline_rl.py",
            "tests/hedge/test_ope_qualification.py",
            "tests/hedge/test_perpetual_features.py",
            "tests/hedge/test_point_in_time_data_plane.py",
            "tests/hedge/test_position_management_evidence.py",
            "tests/hedge/test_qualification_funnel.py",
            "tests/hedge/test_regime_ml.py",
            "tests/hedge/test_tail_evaluation.py",
        ),
        _pytest("mlrl", "tests/hedge/mlrl"),
        _pytest("hprl", "tests/hedge/hprl"),
        TestPhase(
            "gpu_torch_smoke",
            "PyTorch CUDA availability and finite GPU matrix multiplication",
            (
                "{python}",
                "-c",
                (
                    "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
                    "x=torch.randn((128,128),device='cuda:0'); y=x@x.T; torch.cuda.synchronize(); "
                    "assert bool(torch.isfinite(y).all()); print(torch.cuda.get_device_name(0))"
                ),
            ),
            profiles=("standard", "full"),
            requires_cuda=True,
        ),
        _pytest(
            "production_control_plane",
            "tests/hedge/production",
            "tests/hedge/acceptance",
            "tests/hedge/readiness",
            "tests/hedge/readonly",
            "tests/hedge/control",
        ),
        _pytest(
            "integrated_runtime",
            "tests/hedge/integration",
            "tests/hedge/deployment",
            "tests/hedge/operations",
            "tests/hedge/backtesting",
            "tests/hedge/simulation",
            "tests/hedge/strategies",
        ),
        TestPhase(
            "audit_800",
            "800-round audit remediation validator",
            ("{python}", "tools/validate_hedge_audit_remediation_800.py"),
            profiles=("full",),
        ),
        TestPhase(
            "full_ruff_debt_audit",
            "Full Ruff style and security debt audit across HEDGE source, tests and tools",
            ("{python}", "-m", "ruff", "check", "freqtrade/hedge", "tests/hedge", "tools"),
            profiles=("full",),
        ),
        TestPhase(
            "upstream_freqtrade",
            "Freqtrade upstream command, data, strategy, optimization and persistence regression",
            (
                "{python}",
                "-m",
                "pytest",
                "-q",
                "tests/commands",
                "tests/data",
                "tests/strategy",
                "tests/optimize",
                "tests/persistence",
            ),
            pytest_phase=True,
            profiles=("full",),
        ),
        TestPhase(
            "exchange_online",
            "Exchange online compatibility tests; requires network access",
            ("{python}", "-m", "pytest", "-q", "tests/exchange_online"),
            pytest_phase=True,
            profiles=("full",),
            requires_external=True,
        ),
        TestPhase(
            "long_resource_gate",
            "Long-run resource gate contract; does not launch a two-year replay itself",
            (
                "{python}",
                "-m",
                "pytest",
                "-q",
                "tests/hedge/production/test_long_run_resource_gate.py",
            ),
            pytest_phase=True,
            profiles=("full",),
            requires_long_run=True,
        ),
    )
    return phases


def select_phases(
    phases: Sequence[TestPhase],
    *,
    profile: str,
    include_external: bool,
    include_long: bool,
    cuda_available: bool,
) -> tuple[tuple[TestPhase, ...], tuple[PhaseResult, ...]]:
    selected: list[TestPhase] = []
    skipped: list[PhaseResult] = []
    for phase in phases:
        reason = ""
        if profile not in phase.profiles:
            reason = f"excluded by profile={profile}"
        elif phase.requires_external and not include_external:
            reason = "requires --include-external"
        elif phase.requires_long_run and not include_long:
            reason = "requires --include-long"
        elif phase.requires_cuda and not cuda_available:
            reason = "CUDA unavailable"
        if reason:
            skipped.append(
                PhaseResult(
                    phase.identifier,
                    PhaseStatus.SKIPPED,
                    0.0,
                    None,
                    phase.command,
                    "",
                    None,
                    reason,
                )
            )
        else:
            selected.append(phase)
    return tuple(selected), tuple(skipped)


def _cuda_available(python: str, root: Path) -> bool:
    try:
        completed = subprocess.run(
            [python, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _phase_command(
    phase: TestPhase,
    *,
    python: str,
    base_temp: Path,
    junit_path: Path,
) -> tuple[str, ...]:
    command = tuple(item.replace("{python}", python) for item in phase.command)
    if phase.pytest_phase:
        return (
            *command,
            "--continue-on-collection-errors",
            "--basetemp",
            str(base_temp),
            "--junitxml",
            str(junit_path),
        )
    return command


def execute_phase(
    phase: TestPhase,
    *,
    root: Path,
    output_dir: Path,
    python: str,
    timeout_seconds: int,
) -> PhaseResult:
    log_path = output_dir / "logs" / f"{phase.identifier}.log"
    junit_path = output_dir / "junit" / f"{phase.identifier}.xml" if phase.pytest_phase else None
    base_temp = output_dir / "pytest-tmp" / phase.identifier
    for path in (log_path.parent, base_temp):
        path.mkdir(parents=True, exist_ok=True)
    if junit_path is not None:
        junit_path.parent.mkdir(parents=True, exist_ok=True)
    command = _phase_command(
        phase,
        python=python,
        base_temp=base_temp,
        junit_path=junit_path or output_dir / "unused.xml",
    )
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("COMMAND: " + " ".join(command) + "\n\n")
            completed = subprocess.run(
                command,
                cwd=root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                env={**os.environ, "PYTHONUTF8": "1"},
            )
        status = PhaseStatus.PASS if completed.returncode == 0 else PhaseStatus.FAIL
        detail = "" if status is PhaseStatus.PASS else f"exit code {completed.returncode}"
        return PhaseResult(
            phase.identifier,
            status,
            time.monotonic() - started,
            completed.returncode,
            command,
            str(log_path),
            None if junit_path is None else str(junit_path),
            detail,
        )
    except subprocess.TimeoutExpired:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nTIMEOUT after {timeout_seconds} seconds\n")
        return PhaseResult(
            phase.identifier,
            PhaseStatus.TIMEOUT,
            time.monotonic() - started,
            None,
            command,
            str(log_path),
            None if junit_path is None else str(junit_path),
            f"timed out after {timeout_seconds} seconds",
        )
    except OSError as exc:
        return PhaseResult(
            phase.identifier,
            PhaseStatus.ERROR,
            time.monotonic() - started,
            None,
            command,
            str(log_path),
            None if junit_path is None else str(junit_path),
            f"{type(exc).__name__}: {exc}",
        )


PhaseExecutor = Callable[[TestPhase], PhaseResult]


def run_phases(phases: Sequence[TestPhase], executor: PhaseExecutor) -> tuple[PhaseResult, ...]:
    """Execute all phases in order; deliberately do not short-circuit on any result."""
    results: list[PhaseResult] = []
    for phase in phases:
        results.append(executor(phase))
    return tuple(results)


def _json_payload(
    *,
    root: Path,
    profile: str,
    results: Sequence[PhaseResult],
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, object]:
    status_counts = {
        status.value: sum(row.status is status for row in results)
        for status in PhaseStatus
    }
    return {
        "schema_version": "hedge-comprehensive-test-report-v1",
        "project_root": str(root),
        "profile": profile,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "phase_count": len(results),
        "status_counts": status_counts,
        "passed": status_counts[PhaseStatus.FAIL.value] == 0
        and status_counts[PhaseStatus.TIMEOUT.value] == 0
        and status_counts[PhaseStatus.ERROR.value] == 0,
        "results": [asdict(row) | {"status": row.status.value} for row in results],
    }


def _markdown_report(payload: dict[str, object]) -> str:
    rows = payload["results"]
    if not isinstance(rows, list):
        raise TypeError("report results must be a list")
    lines = [
        "# HEDGE Comprehensive Test Report",
        "",
        f"- Profile: `{payload['profile']}`",
        f"- Started: `{payload['started_at']}`",
        f"- Finished: `{payload['finished_at']}`",
        f"- Overall: `{'PASS' if payload['passed'] else 'FAIL'}`",
        "",
        "| Phase | Status | Seconds | Detail |",
        "| --- | --- | ---: | --- |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("report result must be a mapping")
        lines.append(
            "| {identifier} | {status} | {duration_seconds:.2f} | {detail} |".format(
                identifier=row["identifier"],
                status=row["status"],
                duration_seconds=float(row["duration_seconds"]),
                detail=str(row["detail"]).replace("|", "\\|"),
            )
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used by every phase",
    )
    parser.add_argument("--profile", choices=("smoke", "standard", "full"), default="standard")
    parser.add_argument("--include-external", action="store_true")
    parser.add_argument("--include-long", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list-phases", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.timeout_seconds < 1:
        raise ValueError("timeout-seconds must be positive")
    root = args.project_root.resolve()
    phases = default_phases()
    if args.list_phases:
        for phase in phases:
            print(f"{phase.identifier}: {phase.description}")
        return 0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or root / "artifacts" / "comprehensive-tests" / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    cuda_available = _cuda_available(args.python, root)
    selected, skipped = select_phases(
        phases,
        profile=args.profile,
        include_external=args.include_external,
        include_long=args.include_long,
        cuda_available=cuda_available,
    )
    started_at = datetime.now(UTC)
    print(f"HEDGE comprehensive test run: {len(selected)} selected, {len(skipped)} skipped")

    def executor(phase: TestPhase) -> PhaseResult:
        print(f"[START] {phase.identifier}: {phase.description}", flush=True)
        result = execute_phase(
            phase,
            root=root,
            output_dir=output_dir,
            python=args.python,
            timeout_seconds=args.timeout_seconds,
        )
        print(
            (
                f"[{result.status.value}] {result.identifier} "
                f"{result.duration_seconds:.2f}s {result.detail}"
            ),
            flush=True,
        )
        return result

    results = (*skipped, *run_phases(selected, executor))
    finished_at = datetime.now(UTC)
    payload = _json_payload(
        root=root,
        profile=args.profile,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_markdown_report(payload), encoding="utf-8")
    print(f"Summary: {output_dir / 'summary.md'}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
