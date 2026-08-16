"""Unified entry point for the Hedge validation suites.

The existing validation scripts remain independently executable for CI and
backward compatibility. This module provides one stable dispatcher so local
operators do not need to remember which script owns each gate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUITES = (
    "clean-mainline",
    "clean-mainline-200",
    "ruff",
    "research",
    "mlrl",
    "performance",
    "all",
)


def _commands(root: Path, suite: str) -> list[list[str]]:
    python = sys.executable
    common = ["--project-root", str(root)]
    commands: dict[str, list[list[str]]] = {
        "clean-mainline": [
            [python, "tools/validate_clean_mainline.py", *common, "--workspace-mode"],
        ],
        "clean-mainline-200": [
            [python, "tools/validate_clean_mainline_200.py", *common, "--workspace-mode"],
        ],
        "ruff": [[python, "tools/validate_hedge_ruff_staged.py", *common]],
        "research": [[python, "tools/validate_hedge_research_quality.py", *common]],
        "mlrl": [
            [
                python,
                "tools/validate_hedge_mlrl_code_quality.py",
                "--source",
                str(root),
            ],
            [
                python,
                "tools/run_hedge_mlrl_validation.py",
                "--source",
                str(root),
            ],
        ],
        "performance": [
            [
                python,
                "tools/benchmark_hedge_audit_h01_1500.py",
                "--cycles",
                "1500",
                "--retention",
                "2000",
            ]
        ],
    }
    if suite == "all":
        return [command for name in SUITES[:-1] for command in commands[name]]
    return commands[suite]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    failures = 0
    for command in _commands(root, args.suite):
        print("===", " ".join(command[1:]), "===")
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode:
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
