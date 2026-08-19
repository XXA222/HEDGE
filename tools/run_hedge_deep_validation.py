#!/usr/bin/env python3
"""High-intensity, failure-isolated validation and evidence packaging for HEDGE.

This runner treats the supplied unified checklist as an inventory, executes independent phases
in subprocesses, records every result, and continues after FAIL/TIMEOUT/ERROR.  It never enables
exchange writes.  External Binance read-only and 24/72-hour soak evidence are opt-in/manual gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class Status(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class Phase:
    identifier: str
    description: str
    command: tuple[str, ...]
    profiles: tuple[str, ...] = ("deep", "maximum")
    timeout_seconds: int = 1800
    requires_cuda: bool = False
    requires_data: bool = False
    manual_only: bool = False


@dataclass(frozen=True, slots=True)
class Result:
    identifier: str
    description: str
    status: Status
    duration_seconds: float
    return_code: int | None
    command: tuple[str, ...]
    log_path: str
    detail: str = ""


ITEM_RE = re.compile(
    r"^#{2,3}\s+(?P<id>[A-Z][A-Z0-9]*-\d+|G\d+)\s*[—-]\s*"
    r"(?P<title>.*?)(?:\s+`(?P<priority>P[012])`)?\s*$"
)


def parse_checklist(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ITEM_RE.match(line.strip())
        if match:
            rows.append(
                {
                    "id": match.group("id"),
                    "title": match.group("title").strip(),
                    "priority": match.group("priority") or "P1",
                }
            )
    return rows


def command(*parts: str) -> tuple[str, ...]:
    return ("{python}", *parts)


def build_phases(args: argparse.Namespace, *, cuda: bool, data_available: bool) -> list[Phase]:
    root = "{root}"
    phases = [
        Phase(
            "source_manifest_strict",
            "Manifest, compile, package hygiene and source authority",
            command("tools/validate_clean_mainline.py", "--project-root", root, "--workspace-mode"),
            profiles=("smoke", "deep", "maximum"),
            timeout_seconds=1800,
        ),
        Phase(
            "source_correctness",
            "Fatal Ruff correctness and training-health contract tests",
            command(
                "-m", "ruff", "check", "freqtrade/hedge", "freqtrade/freqai/RL",
                "tests/hedge", "tools", "--select", "E9,F63,F7,F82",
            ),
            profiles=("smoke", "deep", "maximum"),
            timeout_seconds=900,
        ),
        Phase(
            "training_health_contracts",
            "Gradient, policy-distribution and rolling-collapse telemetry contracts",
            command(
                "-m", "pytest", "-q", "tests/hedge/telemetry/test_training_health.py",
                "tests/hedge/hprl/test_training_health.py",
                "tests/hedge/mlrl/test_risk_level_rl_mainline_integration.py",
            ),
            profiles=("smoke", "deep", "maximum"),
            timeout_seconds=1800,
        ),
        Phase(
            "comprehensive_standard",
            "Independent standard HEDGE domain matrix",
            command(
                "tools/run_hedge_comprehensive_tests.py", "--project-root", root,
                "--profile", "standard", "--timeout-seconds", str(args.phase_timeout),
                "--output-dir", "{output}/nested-comprehensive-standard",
            ),
            profiles=("deep", "maximum"),
            timeout_seconds=args.phase_timeout + 300,
        ),
        Phase(
            "risk_level_training_stress",
            "Synthetic Risk-Level PPO high-timestep training soak",
            command(
                "tools/run_hedge_risklevel_training.py", "--timesteps", str(args.risk_timesteps),
                "--rows", str(args.risk_rows), "--seed", "42", "--output",
                "{output}/risk-level-training-model",
            ),
            profiles=("deep", "maximum"),
            timeout_seconds=args.training_timeout,
        ),
        Phase(
            "hprl_algorithm_stress",
            "All HPRL algorithms with high update and replay-iteration counts",
            command(
                "tools/benchmark_hprl_performance.py", "--device", "{hprl_device}",
                "--algorithms", "xqc,fast_td3,fast_dsac,simba_sac,rebrac_v2",
                "--batch-sizes", "256", "--hidden-dim", "128", "--hidden-depth", "2",
                "--warmup", "128", "--iterations", str(args.hprl_iterations),
                "--replay-capacity", "200000", "--replay-batch", "256",
                "--replay-iterations", str(args.hprl_replay_iterations),
            ),
            profiles=("deep", "maximum"),
            timeout_seconds=args.training_timeout,
            requires_cuda=False,
        ),
        Phase(
            "hprl_cuda_smoke",
            "HPRL CUDA mixed-precision eager-train smoke with CPU replay",
            command(
                "-m", "freqtrade.hedge.hprl", "train-smoke", "--device", "cuda",
                "--algorithm", "xqc", "--mixed-precision", "--expected-updates", "10000",
                "--replay-device", "cpu", "--compile-mode", "off",
            ),
            profiles=("deep", "maximum"),
            timeout_seconds=1800,
            requires_cuda=True,
        ),
        Phase(
            "eth_two_year_data",
            "ETH 1m/15m/1h/8h/1d two-year data integrity and cadence",
            command(
                "tools/validate_eth_two_year_deep.py", "--data-root", "{data_root}",
                "--output", "{output}/eth-two-year-validation.json",
            ),
            profiles=("deep", "maximum"),
            timeout_seconds=1800,
            requires_data=True,
        ),
        Phase(
            "performance_memory_benchmark",
            "High-cycle bounded performance benchmark",
            command(
                "tools/benchmark_hedge_performance.py", "--cycles", str(args.performance_cycles),
                "--retention", str(max(args.performance_cycles * 2, 2000)),
            ),
            profiles=("deep", "maximum"),
            timeout_seconds=args.training_timeout,
        ),
    ]
    if args.profile == "maximum":
        phases.extend(
            [
                Phase(
                    "comprehensive_full",
                    "Full HEDGE matrix including upstream and long-run contract phases",
                    command(
                        "tools/run_hedge_comprehensive_tests.py", "--project-root", root,
                        "--profile", "full", "--include-long", "--timeout-seconds",
                        str(args.phase_timeout), "--output-dir", "{output}/nested-comprehensive-full",
                    ),
                    profiles=("maximum",),
                    timeout_seconds=args.phase_timeout + 300,
                ),
                Phase(
                    "upstream_regression",
                    "Freqtrade upstream command/data/strategy/optimization/persistence regression",
                    command("-m", "pytest", "-q", "tests/commands", "tests/data", "tests/strategy", "tests/optimize", "tests/persistence"),
                    profiles=("maximum",),
                    timeout_seconds=args.phase_timeout,
                ),
                Phase(
                    "full_ruff_audit",
                    "Full style/debt audit retained as evidence, never hidden",
                    command("-m", "ruff", "check", "freqtrade/hedge", "tests/hedge", "tools"),
                    profiles=("maximum",),
                    timeout_seconds=1800,
                ),
            ]
        )
    return phases


def _render(parts: tuple[str, ...], *, root: Path, output: Path, data_root: Path, hprl_device: str, python: str) -> tuple[str, ...]:
    return tuple(
        part.replace("{python}", str(python))
        .replace("{root}", str(root))
        .replace("{output}", str(output))
        .replace("{data_root}", str(data_root))
        .replace("{hprl_device}", hprl_device)
        for part in parts
    )


def execute(phase: Phase, *, root: Path, output: Path, data_root: Path, hprl_device: str, python: str) -> Result:
    log_path = output / "logs" / f"{phase.identifier}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = _render(phase.command, root=root, output=output, data_root=data_root, hprl_device=hprl_device, python=python)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("COMMAND: " + json.dumps(rendered, ensure_ascii=False) + "\n\n")
        try:
            completed = subprocess.run(
                rendered,
                cwd=root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=phase.timeout_seconds,
                check=False,
                env={**os.environ, "PYTHONUTF8": "1", "HEDGE_DEEP_VALIDATION": "1"},
            )
            status = Status.PASS if completed.returncode == 0 else Status.FAIL
            detail = "" if status is Status.PASS else f"exit code {completed.returncode}"
            return Result(phase.identifier, phase.description, status, time.monotonic() - started, completed.returncode, rendered, str(log_path), detail)
        except subprocess.TimeoutExpired:
            handle.write(f"\nTIMEOUT after {phase.timeout_seconds} seconds\n")
            return Result(phase.identifier, phase.description, Status.TIMEOUT, time.monotonic() - started, None, rendered, str(log_path), f"timeout after {phase.timeout_seconds}s")
        except OSError as exc:
            return Result(phase.identifier, phase.description, Status.ERROR, time.monotonic() - started, None, rendered, str(log_path), f"{type(exc).__name__}: {exc}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_reports(output: Path, *, root: Path, checklist: Path, items: list[dict[str, str]], results: list[Result], args: argparse.Namespace) -> None:
    counts = {status.value: sum(row.status is status for row in results) for status in Status}
    payload: dict[str, Any] = {
        "schema": "hedge-deep-validation-report-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "project_root": str(root),
        "commit": subprocess.run([args.python, "-c", "import subprocess; print(subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())"], cwd=root, capture_output=True, text=True, check=False).stdout.strip(),
        "checklist": str(checklist),
        "checklist_item_count": len(items),
        "profile": args.profile,
        "status_counts": counts,
        "passed": counts["FAIL"] == 0 and counts["TIMEOUT"] == 0 and counts["ERROR"] == 0,
        "results": [asdict(row) | {"status": row.status.value} for row in results],
    }
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# HEDGE Deep Validation Report", "", f"- Profile: `{args.profile}`",
        f"- Commit: `{payload['commit']}`", f"- Checklist items: `{len(items)}`",
        f"- Overall: `{'PASS' if payload['passed'] else 'FAIL'}`", "",
        "| Phase | Status | Seconds | Detail |", "|---|---|---:|---|",
    ]
    for row in results:
        detail = row.detail.replace("|", "\\|")
        lines.append(f"| {row.identifier} | {row.status.value} | {row.duration_seconds:.2f} | {detail} |")
    lines.extend(["", "## Interpretation", "", "A phase result is not a claim that every checklist item passed. External Binance write, PostgreSQL production, 24/72-hour soak, and real-market acceptance remain explicitly external/manual unless separately supplied.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    coverage = []
    for item in items:
        prefix = item["id"].split("-")[0]
        phase = "comprehensive_standard"
        if prefix in {"SRC", "CON", "VAL"}:
            phase = "source_manifest_strict"
        elif prefix in {"RLR", "HPRL", "RL21"}:
            phase = "hprl_algorithm_stress"
        elif prefix in {"BT", "SIM", "OPT", "RES"}:
            phase = "comprehensive_standard"
        elif prefix in {"SYS", "E2E", "PROD"}:
            phase = "comprehensive_full" if args.profile == "maximum" else "comprehensive_standard"
        coverage.append(item | {"mapped_phase": phase, "claim": "requires phase evidence and checklist-specific review"})
    (output / "checklist_inventory.json").write_text(json.dumps({"count": len(coverage), "items": coverage}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _package(output: Path, *, root: Path, checklist: Path) -> Path:
    scripts = output / "scripts"
    scripts.mkdir(exist_ok=True)
    this_file = Path(__file__).resolve()
    shutil.copy2(this_file, scripts / this_file.name)
    validator = root / "tools" / "validate_eth_two_year_deep.py"
    if validator.is_file():
        shutil.copy2(validator, scripts / validator.name)
    wrapper = root / "tools" / "Run-HEDGE-DeepValidation.ps1"
    if wrapper.is_file():
        shutil.copy2(wrapper, scripts / wrapper.name)
    if checklist.is_file():
        shutil.copy2(checklist, output / "checklist-source.md")
    (output / "README.md").write_text(
        "# HEDGE deep validation evidence\n\n"
        "This archive contains phase-isolated logs, JSON/Markdown reports, the exact checklist, "
        "and the scripts used to generate the package. A failed phase does not stop later phases.\n\n"
        "The runner does not enable exchange writes. Binance production, PostgreSQL production, "
        "24/72-hour soak, and other external gates remain SKIPPED/manual unless separately supplied.\n",
        encoding="utf-8",
    )
    archive = output.parent / f"HEDGE-deep-validation-{output.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output).as_posix())
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checklist", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "deep", "maximum"), default="deep")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=Path("artifacts/eth-two-year-deep"))
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--phase-timeout", type=int, default=1800)
    parser.add_argument("--training-timeout", type=int, default=7200)
    parser.add_argument("--risk-timesteps", type=int, default=20000)
    parser.add_argument("--risk-rows", type=int, default=6000)
    parser.add_argument("--hprl-iterations", type=int, default=2000)
    parser.add_argument("--hprl-replay-iterations", type=int, default=10000)
    parser.add_argument("--performance-cycles", type=int, default=1500)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if min(args.phase_timeout, args.training_timeout, args.risk_timesteps, args.risk_rows, args.hprl_iterations, args.hprl_replay_iterations, args.performance_cycles) < 1:
        raise SystemExit("all workload and timeout values must be positive")
    root = args.project_root.expanduser().resolve()
    checklist = args.checklist.expanduser().resolve()
    if not root.is_dir() or not checklist.is_file():
        raise SystemExit("project root or checklist does not exist")
    data_root = args.data_root if args.data_root.is_absolute() else root / args.data_root
    data_root = data_root.resolve()
    if args.require_data and not data_root.is_dir():
        raise SystemExit(f"required ETH data root does not exist: {data_root}")
    cuda = subprocess.run([args.python, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"], cwd=root, check=False).returncode == 0
    hprl_device = "cuda" if cuda else "cpu"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or root / "artifacts" / "deep-validation" / stamp).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    items = parse_checklist(checklist)
    (output / "environment.json").write_text(json.dumps({"python": args.python, "cuda": cuda, "hprl_device": hprl_device, "data_root": str(data_root), "data_available": data_root.is_dir(), "checklist_sha256": _sha256(checklist)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    phases = build_phases(args, cuda=cuda, data_available=data_root.is_dir())
    results: list[Result] = []
    print(f"HEDGE deep validation: profile={args.profile}, checklist_items={len(items)}, phases={len(phases)}")
    for phase in phases:
        if args.profile not in phase.profiles:
            results.append(Result(phase.identifier, phase.description, Status.SKIPPED, 0.0, None, phase.command, "", "excluded by profile"))
            continue
        if phase.requires_cuda and not cuda:
            results.append(Result(phase.identifier, phase.description, Status.SKIPPED, 0.0, None, phase.command, "", "CUDA unavailable"))
            continue
        if phase.requires_data and not data_root.is_dir():
            status = Status.FAIL if args.require_data else Status.SKIPPED
            results.append(Result(phase.identifier, phase.description, status, 0.0, None, phase.command, "", "ETH data root unavailable"))
            continue
        print(f"[START] {phase.identifier}: {phase.description}", flush=True)
        result = execute(phase, root=root, output=output, data_root=data_root, hprl_device=hprl_device, python=args.python)
        results.append(result)
        print(f"[{result.status.value}] {phase.identifier} {result.duration_seconds:.2f}s {result.detail}", flush=True)
    _write_reports(output, root=root, checklist=checklist, items=items, results=results, args=args)
    archive = _package(output, root=root, checklist=checklist)
    print(f"Summary: {output / 'summary.md'}")
    print(f"Package: {archive}")
    return 0 if all(row.status not in {Status.FAIL, Status.TIMEOUT, Status.ERROR} for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
