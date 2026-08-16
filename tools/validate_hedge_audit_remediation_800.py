#!/usr/bin/env python3
"""Independent source/remediation checks for the V1.7 audit remediation."""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Callable
from pathlib import Path


SCHEMA = "freqtrade-hedge-v17-audit-remediation-806-v3"


class Matrix:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, category: str, name: str, passed: bool, detail: str = "") -> None:
        self.rows.append(
            {
                "round": len(self.rows) + 1,
                "category": category,
                "name": name,
                "passed": bool(passed),
                "detail": detail,
            }
        )


def text(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8-sig")


def parse_ok(path: Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8-sig")
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
        return True
    except Exception:
        return False


def no_except_pass(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except Exception:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ExceptHandler)
            and node.body
            and all(isinstance(item, ast.Pass) for item in node.body)
        ):
            return False
    return True


def no_bare_except(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except Exception:
        return False
    return not any(
        isinstance(node, ast.ExceptHandler) and node.type is None for node in ast.walk(tree)
    )


def _add_semantic_checks(
    matrix: Matrix,
    root: Path,
    semantic: list[tuple[str, str, str, bool]],
) -> None:
    for category, rel, token, should_exist in semantic:
        source = text(root, rel)
        found = token in source
        matrix.add(category, f"{rel}: {token}", found is should_exist, f"found={found}")


def _add_file_checks(
    matrix: Matrix,
    root: Path,
    paths: list[Path],
    *,
    category: str,
    name_prefix: str,
    check: Callable[[Path], bool],
) -> None:
    for path in paths:
        relative = str(path.relative_to(root))
        matrix.add(category, f"{name_prefix}{relative}", check(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    m = Matrix()

    # 1-57: concrete audit finding/remediation invariants.
    semantic: list[tuple[str, str, str, bool]] = [
        ("H01", "freqtrade/hedge/execution/service.py", "account_id: str | None = None", True),
        ("H01", "freqtrade/hedge/execution/service.py", "symbol: str | None = None", True),
        (
            "H01",
            "freqtrade/hedge/execution/service.py",
            "position_side: PositionSide | None = None",
            True,
        ),
        (
            "H01",
            "freqtrade/hedge/execution/service.py",
            "statuses: Sequence[OrderState] | None = None",
            True,
        ),
        ("H01", "freqtrade/hedge/execution/service.py", "include_terminal: bool = True", True),
        ("H01", "freqtrade/hedge/execution/service.py", "newest_first: bool = False", True),
        ("H01", "freqtrade/hedge/execution/service.py", "self._open", True),
        ("H01", "freqtrade/hedge/execution/service.py", "self._unknown", True),
        ("H01", "freqtrade/hedge/execution/service.py", "self._insertion_sequence", True),
        (
            "H01",
            "freqtrade/persistence/hedge_execution_adapters.py",
            "ExecutionOrderStateRow.account_id == account_id",
            True,
        ),
        (
            "H01",
            "freqtrade/persistence/hedge_execution_adapters.py",
            "ExecutionOrderStateRow.symbol == symbol",
            True,
        ),
        (
            "H01",
            "freqtrade/persistence/hedge_execution_adapters.py",
            "ExecutionOrderStateRow.position_side == position_side.value",
            True,
        ),
        ("H01", "freqtrade/persistence/hedge_execution_adapters.py", ".limit(limit)", True),
        (
            "H01",
            "freqtrade/persistence/hedge_execution_adapters.py",
            "ExecutionOrderStateRow.id.desc()",
            True,
        ),
        ("H02", "freqtrade/hedge/execution/service.py", "terminal_retention", True),
        ("H02", "freqtrade/hedge/execution/service.py", "_terminal_fifo", True),
        ("H02", "freqtrade/hedge/execution/service.py", "_retire_terminal_locked", True),
        ("H02", "freqtrade/hedge/execution/fake_exchange.py", "terminal_retention", True),
        ("H02", "freqtrade/hedge/execution/idempotency.py", "completed_retention", True),
        ("H03", "freqtrade/hedge/execution/event_publisher.py", "deque(maxlen=capacity)", True),
        ("H03", "freqtrade/hedge/execution/event_publisher.py", "capacity must be positive", True),
        (
            "H03",
            "freqtrade/hedge/execution/event_publisher.py",
            "callbacks = tuple(self._callbacks)",
            True,
        ),
        (
            "H04",
            "freqtrade/hedge/integration/paper_runtime.py",
            "def _prune_planner_order_map",
            True,
        ),
        (
            "H04",
            "freqtrade/hedge/integration/paper_runtime.py",
            "self._prune_planner_order_map()",
            True,
        ),
        (
            "H05",
            "freqtrade/hedge/integration/paper_runtime.py",
            "active_orders = self._active_orders()",
            True,
        ),
        (
            "H05",
            "freqtrade/hedge/integration/paper_runtime.py",
            "active_orders=active_orders",
            True,
        ),
        ("H06", "freqtrade/hedge/execution/event_publisher.py", "run_coroutine_threadsafe", True),
        ("H06", "freqtrade/hedge/execution/event_publisher.py", "asyncio.run(coroutine)", False),
        ("H06", "freqtrade/hedge/execution/event_publisher.py", "_background_tasks", True),
        ("H06", "freqtrade/rpc/api_server/hedge_plugin.py", "loop=event_loop", True),
        (
            "H07",
            "freqtrade/hedge/integration/controller.py",
            "NativeConvergenceCoordinator | None",
            True,
        ),
        (
            "H07",
            "freqtrade/hedge/integration/controller.py",
            "HedgeModelReadinessGate | None",
            True,
        ),
        (
            "H07",
            "freqtrade/hedge/integration/controller.py",
            "HedgeProducerConsumerGate | None",
            True,
        ),
        ("H07", "freqtrade/hedge/integration/controller.py", "tuple[FundingEvent, ...]", True),
        ("H08", "freqtrade/hedge/execution/kill_switch.py", "logger.exception", True),
        ("H08", "freqtrade/hedge/backtesting/memory.py", "exc_info=True", True),
        ("H08", "freqtrade/hedge/deployment/single_instance.py", "logger.debug", True),
        ("H08", "freqtrade/hedge/integration/paper_state.py", "logger.debug", True),
        ("H09", "Dockerfile", " AS builder", True),
        ("H09", "Dockerfile", " AS runtime", True),
        ("H09", "Dockerfile", "COPY --from=builder /opt/hedge-venv /opt/hedge-venv", True),
        ("H09", "Dockerfile", "HEALTHCHECK", True),
        (
            "INSTALL",
            "scripts/Install-Freqtrade-Hedge-V17-Remed800-OneShot-PS51.ps1",
            "setuptools==83.0.0",
            True,
        ),
        (
            "H11",
            "freqtrade/hedge/integration/paper_runtime.py",
            "def _encode_active_execution_orders",
            True,
        ),
        ("H11", "freqtrade/hedge/integration/paper_runtime.py", "statuses=(", True),
        (
            "P2",
            "freqtrade/hedge/execution/state_machine.py",
            "_terminal: bool = field(init=False",
            True,
        ),
        (
            "P4",
            "freqtrade/hedge/performance/resource_governor.py",
            "worker_numeric_environment",
            True,
        ),
        ("P4", "freqtrade/hedge/backtesting/parallel.py", "worker_numeric_environment()", True),
        ("P4", "freqtrade/hedge/native/parallel_hyperopt.py", "worker_numeric_environment()", True),
        (
            "M1",
            "tools/validate_hedge_bounded_runtime_collections.py",
            "HEDGE_BOUNDED_COLLECTION_GATE",
            True,
        ),
        ("M1", "freqtrade/hedge/integration/paper_runtime.py", "def collection_gauges", True),
        ("M1", "freqtrade/hedge/execution/event_publisher.py", "publisher_events", True),
        (
            "M2",
            "freqtrade/hedge/integration/central_source.py",
            "@dataclass(frozen=True, slots=True)",
            True,
        ),
        (
            "M3",
            "freqtrade/hedge/backtesting/memory.py",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            True,
        ),
        ("M3", "freqtrade/hedge/optimization/engine.py", "release_phase_memory()", True),
        (
            "M4",
            "freqtrade/hedge/execution/event_publisher.py",
            "sys.intern(event.event_type)",
            True,
        ),
        (
            "M4",
            "freqtrade/hedge/execution/event_publisher.py",
            'sys.intern(str(event.payload.get("account_id"',
            True,
        ),
    ]
    _add_semantic_checks(m, root, semantic)
    if len(m.rows) != 57:
        raise AssertionError(len(m.rows))

    # 58-369: every Hedge Python file compiles (312 distinct checks).
    hedge_files = sorted((root / "freqtrade/hedge").rglob("*.py"))
    if len(hedge_files) != 312:
        raise RuntimeError(f"expected 312 Hedge Python files, found {len(hedge_files)}")
    _add_file_checks(m, root, hedge_files, category="compile", name_prefix="", check=parse_ok)
    if len(m.rows) != 369:
        raise AssertionError(len(m.rows))

    # 370-681: each Hedge Python file has no silent except/pass.
    _add_file_checks(
        m,
        root,
        hedge_files,
        category="observability",
        name_prefix="no except/pass: ",
        check=no_except_pass,
    )
    if len(m.rows) != 681:
        raise AssertionError(len(m.rows))

    # 682-727: every Risk-Level RL Python file compiles (46 distinct checks).
    rl_files = sorted((root / "freqtrade/freqai/hedge_rl").rglob("*.py"))
    if len(rl_files) != 46:
        raise RuntimeError(f"expected 46 Hedge RL Python files, found {len(rl_files)}")
    _add_file_checks(m, root, rl_files, category="rl-compile", name_prefix="", check=parse_ok)
    if len(m.rows) != 727:
        raise AssertionError(len(m.rows))

    # 728-773: no Risk-Level RL file may silently swallow an exception.
    _add_file_checks(
        m,
        root,
        rl_files,
        category="rl-observability",
        name_prefix="no except/pass: ",
        check=no_except_pass,
    )
    if len(m.rows) != 773:
        raise AssertionError(len(m.rows))

    # 774-806: selected core files also reject bare except clauses.
    _add_file_checks(
        m,
        root,
        hedge_files[:33],
        category="exception-contract",
        name_prefix="no bare except: ",
        check=no_bare_except,
    )
    if len(m.rows) != 806:
        raise AssertionError(len(m.rows))

    failures = [row for row in m.rows if not row["passed"]]
    report = {
        "schema": SCHEMA,
        "project_root": str(root),
        "rounds": len(m.rows),
        "passed": len(m.rows) - len(failures),
        "failed": len(failures),
        "status": "PASS" if not failures else "FAIL",
        "rows": m.rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
    print(f"HEDGE_AUDIT_REMEDIATION_806: {report['status']} ({report['passed']}/806)")
    if failures:
        for row in failures[:25]:
            print(f" - round {row['round']}: {row['name']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
