#!/usr/bin/env python3
"""Compare compact and detailed Hedge backtest result JSON artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from freqtrade.hedge.backtesting.consistency import compare_backtest_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=pathlib.Path, required=True)
    parser.add_argument("--detailed", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    detailed = json.loads(args.detailed.read_text(encoding="utf-8"))
    report = compare_backtest_results(compact, detailed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
