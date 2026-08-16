#!/usr/bin/env python3
"""Operator CLI for HPRL five-priority evidence closure."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any


# When this tool is executed by absolute path, Python places the tools directory
# at sys.path[0], not the repository root.  Bootstrap the canonical source root
# explicitly so the operator CLI always imports the source tree being validated
# instead of depending on cwd, an editable install, or a stale site-package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write(path: str | None, payload: object) -> None:
    text = json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n"
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")


def _connect_factory(dsn: str):
    try:
        import psycopg  # type: ignore

        return lambda: psycopg.connect(dsn)
    except ImportError:
        try:
            import psycopg2  # type: ignore

            return lambda: psycopg2.connect(dsn)
        except ImportError as exc:
            raise RuntimeError(
                "psycopg/psycopg2 is required for PostgreSQL visibility evidence"
            ) from exc



def isolated_bootstrap(args: argparse.Namespace) -> int:
    from freqtrade.hedge.production.runtime_test_isolation import create_isolated_test_environment

    report = create_isolated_test_environment(
        args.root,
        runtime_python=args.runtime_python,
        environment_root=args.environment_root,
        install_missing=not args.no_install_missing,
        timeout_seconds=args.timeout,
    )
    payload = asdict(report)
    payload["schema"] = "hprl-priority5-isolated-test-env-v1"
    payload["status"] = "PASS" if report.passed else "FAIL"
    _write(args.output, payload)
    return 0 if report.passed else 2


def isolated_pytest(args: argparse.Namespace) -> int:
    from freqtrade.hedge.production.runtime_test_isolation import run_isolated_pytest

    report = run_isolated_pytest(
        args.root,
        environment_python=args.environment_python,
        targets=tuple(args.targets),
        minimum_tests=args.minimum_tests,
        junit_path=args.junit,
        timeout_seconds=args.timeout,
        use_xdist=args.xdist,
    )
    payload = asdict(report)
    payload["schema"] = "hprl-priority5-isolated-pytest-v1"
    payload["status"] = "PASS" if report.passed else "FAIL"
    _write(args.output, payload)
    return 0 if report.passed else 2

def source_probe(args: argparse.Namespace) -> int:
    from freqtrade.hedge.production.runtime_test_isolation import source_tree_sha256

    digest = source_tree_sha256(args.root)
    payload = {
        "schema": "hprl-priority5-source-runtime-v1",
        "status": "PASS",
        "source_root": str(Path(args.root).resolve()),
        "source_sha256": digest,
        "observed_at": datetime.now(UTC).isoformat(),
    }
    _write(args.output, payload)
    return 0


def fault_focused(args: argparse.Namespace) -> int:
    from freqtrade.hedge.production.fault_campaign_qualification import (
        focused_runtime_campaign_report,
    )

    report = focused_runtime_campaign_report()
    payload = asdict(report)
    payload["schema"] = "hprl-priority5-focused-fault-v1"
    payload["status"] = "PASS" if report.passed else "FAIL"
    _write(args.output, payload)
    return 0 if report.passed else 2


def postgres_visibility(args: argparse.Namespace) -> int:
    from freqtrade.hedge.production.postgres_visibility import run_postgres_dual_session_visibility

    dsn = Path(args.dsn_file).read_text(encoding="utf-8").strip()
    if not dsn:
        raise ValueError("PostgreSQL DSN file is empty")
    report = run_postgres_dual_session_visibility(
        _connect_factory(dsn),
        now=datetime.now(UTC),
    )
    payload = asdict(report)
    payload["schema"] = "hprl-priority5-postgres-visibility-v1"
    payload["status"] = "PASS" if report.passed else "FAIL"
    _write(args.output, payload)
    return 0 if report.passed else 2


def algorithm_qualify(args: argparse.Namespace) -> int:
    from freqtrade.hedge.production.hprl_algorithm_qualification import (
        AlgorithmTrialEvidence,
        qualify_candidate_set,
    )

    payload = json.loads(Path(args.trials).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("trials JSON must contain a list")
    rows = tuple(AlgorithmTrialEvidence(**item) for item in payload)
    report = qualify_candidate_set(rows)
    result = asdict(report)
    result["schema"] = "hprl-priority5-algorithm-qualification-v1"
    result["status"] = "PASS" if report.passed else "FAIL"
    _write(args.output, result)
    return 0 if report.passed else 2


def resource_qualify(args: argparse.Namespace) -> int:
    from freqtrade.hedge.production.long_run_resource_gate import (
        ResourceSample,
        evaluate_resource_stability,
    )

    payload = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("samples JSON must contain a list")
    report = evaluate_resource_stability(tuple(ResourceSample(**item) for item in payload))
    result = asdict(report)
    result["schema"] = "hprl-priority5-resource-stability-v1"
    result["status"] = "PASS" if report.passed else "FAIL"
    _write(args.output, result)
    return 0 if report.passed else 2


def closure(args: argparse.Namespace) -> int:
    from freqtrade.hedge.production.priority5_closure import (
        evaluate_priority5_closure,
        load_priority_evidence,
    )

    rows = tuple(load_priority_evidence(path) for path in args.evidence)
    report = evaluate_priority5_closure(
        rows,
        expected_source_sha256=args.source_sha256,
        now=datetime.now(UTC),
    )
    payload = asdict(report)
    payload["schema"] = "hprl-priority5-closure-v1"
    payload["status"] = "PASS" if report.passed else "PENDING_OR_FAIL"
    _write(args.output, payload)
    return 0 if report.passed else 3


def offline_audit(args: argparse.Namespace) -> int:
    from freqtrade.hedge.production.fault_campaign_qualification import (
        focused_runtime_campaign_report,
    )
    from freqtrade.hedge.production.runtime_test_isolation import source_tree_sha256

    source = source_tree_sha256(args.root)
    fault = focused_runtime_campaign_report()
    payload: dict[str, Any] = {
        "schema": "hprl-priority5-offline-foundation-v1",
        "status": "PASS" if fault.passed else "FAIL",
        "source_sha256": source,
        "baseline_commit": "b05a5214ffb64cdf3e5385c8deca5fa6fc0f9917",
        "source_runtime": "MEASURED_LOCALLY",
        "focused_fault_campaign": asdict(fault),
        "postgresql": "PENDING_REAL_MEASURED_EVIDENCE",
        "model_real_market": "PENDING_REAL_MEASURED_EVIDENCE",
        "long_run": "PENDING_TWO_YEAR_MEASURED_EVIDENCE",
        "note": "Offline foundation never promotes PostgreSQL, real-market or two-year evidence.",
        "observed_at": datetime.now(UTC).isoformat(),
    }
    _write(args.output, payload)
    return 0 if fault.passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)


    p = sub.add_parser("isolated-bootstrap")
    p.add_argument("--root", required=True)
    p.add_argument("--runtime-python", required=True)
    p.add_argument("--environment-root", required=True)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--no-install-missing", action="store_true")
    p.add_argument("--output")
    p.set_defaults(func=isolated_bootstrap)

    p = sub.add_parser("isolated-pytest")
    p.add_argument("--root", required=True)
    p.add_argument("--environment-python", required=True)
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--minimum-tests", type=int, required=True)
    p.add_argument("--junit", required=True)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--xdist", action="store_true")
    p.add_argument("--output")
    p.set_defaults(func=isolated_pytest)

    p = sub.add_parser("source-probe")
    p.add_argument("--root", required=True)
    p.add_argument("--output")
    p.set_defaults(func=source_probe)

    p = sub.add_parser("fault-focused")
    p.add_argument("--output")
    p.set_defaults(func=fault_focused)

    p = sub.add_parser("postgres-visibility")
    p.add_argument("--dsn-file", required=True)
    p.add_argument("--output")
    p.set_defaults(func=postgres_visibility)

    p = sub.add_parser("algorithm-qualify")
    p.add_argument("--trials", required=True)
    p.add_argument("--output")
    p.set_defaults(func=algorithm_qualify)

    p = sub.add_parser("resource-qualify")
    p.add_argument("--samples", required=True)
    p.add_argument("--output")
    p.set_defaults(func=resource_qualify)

    p = sub.add_parser("closure")
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--evidence", nargs="+", required=True)
    p.add_argument("--output")
    p.set_defaults(func=closure)

    p = sub.add_parser("offline-audit")
    p.add_argument("--root", required=True)
    p.add_argument("--output")
    p.set_defaults(func=offline_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
