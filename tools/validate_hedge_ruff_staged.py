#!/usr/bin/env python3
"""Run staged Ruff debt gates for the Hedge remediation line.

The gate intentionally does not run ``ruff --fix``.  It measures production, Risk-Level
RL, tests, and tools separately so cleanup can be reviewed in semantic batches.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "freqtrade-hedge-ruff-staged-remediation-v1"

# All staged areas are now clean.  Any future finding is a regression and must fail the
# gate rather than being absorbed by a renewed debt budget.
STAGE_BUDGETS = {
    "freqtrade/hedge": 0,
    "freqtrade/freqai/hedge_rl": 0,
    "tests/hedge": 0,
    "tools": 0,
}

# Checked-in ceilings from the previous remediation cycle.  A future change may
# only lower these values; raising a budget is an explicit policy violation.
PREVIOUS_STAGE_BUDGETS = {
    "freqtrade/hedge": 161,
    "freqtrade/freqai/hedge_rl": 0,
    "tests/hedge": 340,
    "tools": 300,
}

BLOCKING_RULES = "E9,F63,F7,F82"


def _run_ruff(root: Path, paths: list[str], *, select: str | None = None) -> list[dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        *paths,
        "--no-cache",
        "--output-format",
        "json",
    ]
    if select is not None:
        command.extend(["--select", select])
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    raw = completed.stdout.strip()
    if not raw:
        if completed.returncode == 0:
            return []
        raise RuntimeError(
            "ruff produced no JSON output: "
            + ((completed.stderr or "").strip() or f"exit={completed.returncode}")
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid Ruff JSON: {raw[:500]}") from exc
    if not isinstance(payload, list):
        raise TypeError("Ruff JSON output must be a list")
    return payload


def _rule_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        code = str(item.get("code") or "UNKNOWN")
        counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    stages = {
        "freqtrade/hedge": ["freqtrade/hedge"],
        "freqtrade/freqai/hedge_rl": ["freqtrade/freqai/hedge_rl"],
        "tests/hedge": ["tests/hedge"],
        "tools": ["tools"],
    }

    failures: list[str] = []
    reports: dict[str, Any] = {}

    for name, budget in STAGE_BUDGETS.items():
        previous = PREVIOUS_STAGE_BUDGETS[name]
        if budget > previous:
            failures.append(
                f"{name}: budget increased from {previous} to {budget}; budgets must only decrease"
            )

    for name, paths in stages.items():
        full = _run_ruff(root, paths)
        blocking = _run_ruff(root, paths, select=BLOCKING_RULES)
        budget = STAGE_BUDGETS[name]
        reports[name] = {
            "issues": len(full),
            "budget": budget,
            "blocking_issues": len(blocking),
            "rule_counts": _rule_counts(full),
        }
        if blocking:
            failures.append(f"{name}: blocking Ruff correctness issues={len(blocking)}")
        if len(full) > budget:
            failures.append(f"{name}: Ruff debt={len(full)} exceeds budget={budget}")

    # The independent Risk-Level module is intentionally held to the strictest target.
    if reports["freqtrade/freqai/hedge_rl"]["issues"] != 0:
        failures.append("freqtrade/freqai/hedge_rl must remain Ruff-clean")

    report = {
        "schema": SCHEMA,
        "project_root": str(root),
        "stages": reports,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
