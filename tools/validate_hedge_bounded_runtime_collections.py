#!/usr/bin/env python3
"""Fail closed when critical long-lived Hedge collections lose capacity contracts."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SNIPPETS = {
    "freqtrade/hedge/execution/service.py": (
        "terminal_retention",
        "_terminal_fifo",
        "_open",
        "_unknown",
        "collection_gauges",
    ),
    "freqtrade/hedge/execution/event_publisher.py": (
        "capacity",
        "deque(maxlen=capacity)",
        "collection_gauges",
    ),
    "freqtrade/hedge/execution/fake_exchange.py": (
        "terminal_retention",
        "recent_fill_capacity",
        "call_history_capacity",
        "collection_gauges",
    ),
    "freqtrade/hedge/execution/idempotency.py": (
        "completed_retention",
        "collection_gauges",
    ),
    "freqtrade/hedge/integration/paper_runtime.py": (
        "_prune_planner_order_map",
        "collection_gauges",
    ),
}


def main() -> int:
    failures: list[str] = []
    for relative, snippets in REQUIRED_SNIPPETS.items():
        path = ROOT / relative
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        for snippet in snippets:
            if snippet not in source:
                failures.append(f"{relative}: missing bounded-runtime marker {snippet!r}")
    if failures:
        print("HEDGE_BOUNDED_COLLECTION_GATE: FAIL")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("HEDGE_BOUNDED_COLLECTION_GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
