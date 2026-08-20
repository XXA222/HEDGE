# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
_here_text = str(HERE)
if _here_text in sys.path:
    sys.path.remove(_here_text)
sys.path.insert(0, _here_text)

from suite_specs import MODELS, TIMEFRAMES, input_timeframes_for


def utc_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    path.write_text(text, encoding="utf-8")


def python_executable(repo_root: Path) -> Path:
    candidates = [
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / ".venv" / "bin" / "python",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return Path(sys.executable)


def default_datadir(repo_root: Path) -> Path:
    candidates = [
        repo_root / "user_data" / "data" / "binance",
        repo_root / "user_data" / "data",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def git_sha(repo_root: Path) -> str:
    try:
        cp = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return cp.stdout.strip() if cp.returncode == 0 else ""
    except Exception:
        return ""


def run_process(
    command: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
    timeout: int = 0,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND\n")
        log.write(subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        try:
            cp = subprocess.run(
                command,
                cwd=str(command_cwd(command)),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=None if timeout <= 0 else timeout,
            )
            return {
                "returncode": int(cp.returncode),
                "seconds": round(time.time() - started, 3),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            log.write("\nTIMEOUT\n")
            return {
                "returncode": 124,
                "seconds": round(time.time() - started, 3),
                "timed_out": True,
            }
        except Exception as exc:
            log.write("\nMASTER ERROR\n" + traceback.format_exc())
            return {
                "returncode": 125,
                "seconds": round(time.time() - started, 3),
                "timed_out": False,
                "master_error": repr(exc),
            }


def command_cwd(command: list[str]) -> Path:
    # All subprocesses are explicitly constructed with the repo root as argv metadata through
    # HPRL_SUITE_REPO_ROOT. Keeping this helper separate makes tests easy and avoids global cwd.
    raw = os.environ.get("HPRL_SUITE_REPO_ROOT")
    return Path(raw).resolve() if raw else Path.cwd()


def _validation_dates(mode: str, train_start: str, backtest_start: str, end: str):
    if mode == "strict-oos":
        return train_start, backtest_start, backtest_start, end
    if mode == "two-year-split":
        start_dt = datetime.fromisoformat(backtest_start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        split = start_dt + (end_dt - start_dt) * 0.40
        split = split.replace(hour=0, minute=0, second=0, microsecond=0)
        split_text = split.astimezone(UTC).isoformat().replace("+00:00", "Z")
        return backtest_start, split_text, split_text, end
    if mode == "integration-full":
        start_dt = datetime.fromisoformat(backtest_start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        split = start_dt + (end_dt - start_dt) * 0.40
        split = split.replace(hour=0, minute=0, second=0, microsecond=0)
        split_text = split.astimezone(UTC).isoformat().replace("+00:00", "Z")
        return backtest_start, split_text, backtest_start, end
    raise ValueError(mode)


def freqtrade_timerange(start: str, end: str) -> str:
    left = datetime.fromisoformat(start.replace("Z", "+00:00")).strftime("%Y%m%d")
    right = datetime.fromisoformat(end.replace("Z", "+00:00")).strftime("%Y%m%d")
    return f"{left}-{right}"


def parse_result(result_path: Path) -> dict[str, object]:
    if not result_path.is_file():
        return {}
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"result_parse_error": repr(exc)}
    report = payload.get("report") if isinstance(payload, dict) else None
    row: dict[str, object] = {
        "pair": payload.get("pair"),
        "timeframe": payload.get("timeframe"),
        "start": payload.get("start"),
        "end": payload.get("end"),
        "bar_count": payload.get("bar_count"),
        "signal_count": payload.get("signal_count"),
        "funding_count": payload.get("funding_count"),
        "missing_candle_count": payload.get("missing_candle_count"),
        "data_fingerprint": payload.get("data_fingerprint"),
        "result_fingerprint": payload.get("result_fingerprint"),
    }
    if isinstance(report, dict):
        for key, value in report.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"report_{key}"] = value
    return row


def source_snapshot(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "README_CN.md",
        "suite_specs.py",
        "artifact_contract.py",
        "features.py",
        "prepare_models.py",
        "run_suite.py",
        "run_all.ps1",
        "run_all.sh",
        "validate_suite.py",
        "PACKAGE_MANIFEST.json",
        "SOURCE_IMPROVEMENTS.md",
        "Install-HPRL-Freqtrade-MTF-V3.ps1",
    ):
        src = HERE / name
        if src.exists():
            shutil.copy2(src, destination / name)
    shutil.copytree(HERE / "strategies", destination / "strategies", dirs_exist_ok=True)
    shutil.copytree(HERE / "configs", destination / "configs", dirs_exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def artifact_inventory(user_data: Path, output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = user_data / "hprl_freqtrade_models"
    for model in MODELS:
        for timeframe in TIMEFRAMES:
            directory = root / model / timeframe
            record: dict[str, object] = {
                "model": model,
                "timeframe": timeframe,
                "directory": str(directory),
            }
            for filename in (
                "checkpoint.pt",
                "checkpoint.pt.json",
                "scaler.npz",
                "metadata.json",
                "artifact_manifest.json",
            ):
                path = directory / filename
                record[f"{filename}_exists"] = path.is_file()
                if path.is_file():
                    record[f"{filename}_bytes"] = path.stat().st_size
                    record[f"{filename}_sha256"] = sha256_file(path)
            rows.append(record)
    write_json(output, rows)
    return rows


def artifact_data_format(
    user_data: Path,
    model: str,
    timeframe: str,
    requested: str,
) -> str | None:
    if requested != "auto":
        return requested
    metadata = user_data / "hprl_freqtrade_models" / model / timeframe / "metadata.json"
    if not metadata.is_file():
        return None
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        value = str(payload.get("data_format") or "").strip()
        return value or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    p = argparse.ArgumentParser(
        description=(
            "Run 5 native HPRL Strategies x 6 base timeframes with causal "
            "Freqtrade informative MTF inputs."
        )
    )
    p.add_argument("--repo-root", default=".")
    p.add_argument("--datadir")
    p.add_argument("--data-format", default="auto")
    p.add_argument("--train-start", default="2023-08-19T00:00:00Z")
    p.add_argument("--backtest-start", default="2024-08-19T00:00:00Z")
    p.add_argument("--end", default="2026-08-19T00:00:00Z")
    p.add_argument(
        "--validation-mode",
        choices=("strict-oos", "two-year-split", "integration-full"),
        default="strict-oos",
    )
    p.add_argument("--budget", choices=("fast", "balanced", "deep"), default="balanced")
    p.add_argument("--device", default="cpu")
    p.add_argument("--strategy-device", default="cpu")
    p.add_argument("--parallel-envs", type=int, default=16)
    p.add_argument("--task-timeout", type=int, default=0)
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--force-training", action="store_true")
    p.add_argument("--output-dir")
    args = p.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "freqtrade").is_dir():
        raise SystemExit(f"Not a HEDGE/Freqtrade source root: {repo_root}")
    user_data = repo_root / "user_data"
    datadir = Path(args.datadir).resolve() if args.datadir else default_datadir(repo_root)
    py = python_executable(repo_root)
    output_base = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else repo_root / "artifacts" / "hprl_freqtrade_results"
    )
    run_root = output_base / f"HPRL_FREQTRADE_MTF_30_{utc_tag()}"
    (run_root / "logs" / "training").mkdir(parents=True, exist_ok=False)
    (run_root / "logs" / "backtests").mkdir(parents=True)
    (run_root / "backtests").mkdir(parents=True)
    source_snapshot(run_root / "source_snapshot")

    train_start, train_end, bt_start, bt_end = _validation_dates(
        args.validation_mode, args.train_start, args.backtest_start, args.end
    )
    timerange = freqtrade_timerange(bt_start, bt_end)
    sha = git_sha(repo_root)
    manifest = {
        "schema": "hprl-freqtrade-mtf-30-suite-v3",
        "repository": "XXA222/HEDGE",
        "git_sha": sha,
        "validation_mode": args.validation_mode,
        "training_window": [train_start, train_end],
        "formal_backtest_window": [bt_start, bt_end],
        "formal_backtest_timerange": timerange,
        "performance_authority": "python -m freqtrade hedge-backtesting",
        "policy_provider": (
            "native HPRL agent + causal Freqtrade informative MTF features + "
            "VectorizedHedgeEnv + HprlHedgeAdapter inside real IStrategy"
        ),
        "pair": "ETH/USDT:USDT",
        "base_timeframes": list(TIMEFRAMES),
        "mtf_inputs": {tf: list(input_timeframes_for(tf)) for tf in TIMEFRAMES},
        "models": list(MODELS),
        "tasks_expected": len(TIMEFRAMES) * len(MODELS),
        "budget": args.budget,
        "device": args.device,
        "strategy_device": args.strategy_device,
        "parallel_envs": args.parallel_envs,
        "datadir": str(datadir),
        "user_data_dir": str(user_data),
        "started_at": datetime.now(UTC).isoformat(),
    }
    write_json(run_root / "run_manifest.json", manifest)

    failures: list[dict[str, object]] = []
    commands: list[str] = []
    training_rows: list[dict[str, object]] = []
    env = os.environ.copy()
    env["HPRL_SUITE_REPO_ROOT"] = str(repo_root)
    env["HPRL_STRATEGY_DEVICE"] = args.strategy_device
    os.environ["HPRL_SUITE_REPO_ROOT"] = str(repo_root)

    if not args.skip_training:
        for model in MODELS:
            for tf in TIMEFRAMES:
                cmd = [
                    str(py), str(HERE / "prepare_models.py"),
                    "--repo-root", str(repo_root),
                    "--user-data-dir", str(user_data),
                    "--datadir", str(datadir),
                    "--data-format", args.data_format,
                    "--model", model,
                    "--timeframe", tf,
                    "--train-start", train_start,
                    "--train-end", train_end,
                    "--budget", args.budget,
                    "--device", args.device,
                    "--parallel-envs", str(args.parallel_envs),
                ]
                if args.force_training:
                    cmd.append("--force")
                commands.append(subprocess.list2cmdline(cmd))
                status = run_process(
                    cmd,
                    run_root / "logs" / "training" / f"{model}_{tf}.log",
                    env,
                    args.task_timeout,
                )
                row = {
                    "phase": "training",
                    "model": model,
                    "timeframe": tf,
                    "input_timeframes": list(input_timeframes_for(tf)),
                    **status,
                }
                training_rows.append(row)
                if status["returncode"] != 0:
                    failures.append({**row, "log": f"logs/training/{model}_{tf}.log"})

    results: list[dict[str, object]] = []
    for model, spec in MODELS.items():
        config = HERE / "configs" / f"{model}.json"
        for tf in TIMEFRAMES:
            task_name = f"{model}_{tf}"
            result_path = run_root / "backtests" / f"{task_name}.json"
            cmd = [
                str(py), "-m", "freqtrade", "hedge-backtesting",
                "--config", str(config),
                "--strategy", spec.strategy_class,
                "--strategy-path", str(HERE / "strategies"),
                "--userdir", str(user_data),
                "--datadir", str(datadir),
                "--timeframe", tf,
                "--timerange", timerange,
                "--cache", "none",
                "--hedge-export-filename", str(result_path),
            ]
            resolved_data_format = artifact_data_format(
                user_data, model, tf, args.data_format
            )
            if resolved_data_format:
                cmd.extend(["--data-format-ohlcv", resolved_data_format])
            commands.append(subprocess.list2cmdline(cmd))
            status = run_process(
                cmd,
                run_root / "logs" / "backtests" / f"{task_name}.log",
                env,
                args.task_timeout,
            )
            row: dict[str, object] = {
                "model": model,
                "algorithm": spec.algorithm,
                "strategy": spec.strategy_class,
                "timeframe": tf,
                "input_timeframes": "+".join(input_timeframes_for(tf)),
                "status": "PASS" if status["returncode"] == 0 and result_path.is_file() else "FAIL",
                **status,
                "result_file": (
                    str(result_path.relative_to(run_root))
                    if result_path.is_file()
                    else ""
                ),
                "log_file": f"logs/backtests/{task_name}.log",
            }
            row.update(parse_result(result_path))
            if result_path.is_file():
                row["result_sha256"] = sha256_file(result_path)
            results.append(row)
            if row["status"] != "PASS":
                failures.append({"phase": "backtest", **row})

    (run_root / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    write_json(run_root / "training_status.json", training_rows)
    write_json(run_root / "summary.json", results)
    write_csv(run_root / "summary.csv", results)
    write_json(run_root / "failures.json", failures)
    artifact_inventory(user_data, run_root / "model_artifacts.json")

    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["backtests_passed"] = sum(1 for row in results if row["status"] == "PASS")
    manifest["backtests_failed"] = len(results) - manifest["backtests_passed"]
    manifest["failures_total"] = len(failures)
    write_json(run_root / "run_manifest.json", manifest)

    zip_path = output_base / f"{run_root.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(run_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(run_root.parent))
    digest = sha256_file(zip_path)
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(
        f"{digest}  {zip_path.name}\n", encoding="ascii"
    )

    print(json.dumps({
        "status": "PASS" if manifest["backtests_failed"] == 0 else "PARTIAL",
        "backtests_passed": manifest["backtests_passed"],
        "backtests_failed": manifest["backtests_failed"],
        "result_zip": str(zip_path),
        "sha256": digest,
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["backtests_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
