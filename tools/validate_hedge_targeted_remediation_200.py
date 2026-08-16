#!/usr/bin/env python3
"""Deterministic 200-point validation for the focused V1.7 remediation."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Callable
from pathlib import Path


SCHEMA = "freqtrade-hedge-v17-targeted-remediation-200-v1"


class Matrix:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, category: str, name: str, condition: bool, detail: object = "") -> None:
        self.rows.append(
            {
                "round": len(self.rows) + 1,
                "category": category,
                "name": name,
                "passed": bool(condition),
                "detail": detail,
            }
        )


def _text(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def _contains_all(text: str, tokens: tuple[str, ...]) -> bool:
    return all(token in text for token in tokens)


def _python_ok(path: Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8-sig")
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
        return True
    except Exception:
        return False


def _add_token_checks(
    matrix: Matrix,
    category: str,
    text: str,
    tokens: list[tuple[str, str]],
) -> None:
    for name, token in tokens:
        matrix.add(category, name, token in text, token)


def _add_callable_checks(
    matrix: Matrix,
    category: str,
    checks: list[tuple[str, Callable[[], bool]]],
) -> None:
    for name, check in checks:
        matrix.add(category, name, check())


def _add_windows_token_checks(
    matrix: Matrix,
    script: str,
    tokens: list[tuple[str, str]],
) -> None:
    for name, token in tokens:
        condition = token not in script if name == "no pip editable install" else token in script
        matrix.add("windows-runtime", name, condition, token)


def _add_decimal_checks(matrix: Matrix, root: Path, pattern: re.Pattern[str]) -> None:
    for relative in ("freqtrade/hedge", "tests/hedge", "tools"):
        matches = sum(
            len(pattern.findall(path.read_text(encoding="utf-8")))
            for path in (root / relative).rglob("*.py")
        )
        matrix.add(
            "ruff-staged",
            f"no FURB157 integer-string Decimal in {relative}",
            matches == 0,
            matches,
        )


def _add_python_file_checks(
    matrix: Matrix,
    category: str,
    root: Path,
    relatives: list[str],
) -> None:
    for relative in relatives:
        matrix.add(category, f"compile {relative}", _python_ok(root / relative))


def _add_named_python_checks(matrix: Matrix, root: Path, directory: Path, names: list[str]) -> None:
    for name in names:
        matrix.add("ruff-staged", f"hedge_rl syntax {name}", _python_ok(directory / name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    matrix = Matrix()

    # ------------------------------------------------------------------
    # 1-40: Optimization dataset contract and regressions.
    # ------------------------------------------------------------------
    adapter = _text(root, "freqtrade/hedge/optimization/freqtrade_adapter.py")
    opt_tests = _text(root, "tests/hedge/optimization/test_code_quality.py")

    _add_token_checks(
        matrix,
        "optimization-contract",
        adapter,
        [
            ("contract helper exists", "_require_optimization_dataset_contract"),
            ("contract fields constant exists", "_REQUIRED_OPTIMIZATION_DATASET_FIELDS"),
            ("pair field required", '"pair"'),
            ("timeframe field required", '"timeframe"'),
            ("start field required", '"start"'),
            ("end field required", '"end"'),
            ("bar_count field required", '"bar_count"'),
            ("data_fingerprint field required", '"data_fingerprint"'),
            ("missing_candle_count field required", '"missing_candle_count"'),
            ("missing contract TypeError", "is missing required field(s)"),
            ("gap metadata bool rejected", "isinstance(missing_candle_count, bool)"),
            ("gap metadata integer required", "missing_candle_count must be an integer"),
            ("negative gap count rejected", "missing_candle_count cannot be negative"),
            ("bar count bool rejected", "isinstance(dataset.bar_count, bool)"),
            ("bar count integer required", "dataset bar_count must be an integer"),
            ("bar count positive required", "dataset bar_count must be positive"),
            ("fingerprint string required", "data_fingerprint must be a string"),
            ("pair non-empty required", "dataset pair must be a non-empty string"),
            ("timeframe non-empty required", "dataset timeframe must be a non-empty string"),
            ("probe validated before timestamp build", 'source="probe"'),
            ("probe gap fail closed", "requires a gap-free compact dataset"),
            ("trial validated", 'source="trial"'),
            ("trial gap fail closed", "trial backtest returned a dataset with missing candles"),
            ("trial fingerprint compared", "run_dataset.data_fingerprint"),
            ("probe fingerprint alias used", "probe_dataset.data_fingerprint"),
        ],
    )
    _add_token_checks(
        matrix,
        "optimization-tests",
        opt_tests,
        [
            ("fake dataset carries gap count", "missing_candle_count=missing_candle_count"),
            ("fake runner gap argument defaults zero", "missing_candle_count: int = 0"),
            ("probe gap regression exists", "test_round_20_probe_missing_candles_fail_closed"),
            (
                "probe missing contract regression exists",
                "test_round_21_missing_probe_dataset_contract_is_explicit",
            ),
            (
                "trial missing contract regression exists",
                "test_round_22_trial_missing_dataset_contract_becomes_failed_trial",
            ),
            ("trial gap regression exists", "test_round_23_trial_missing_candles_fail_closed"),
            (
                "old four regression 15 retained",
                "test_round_15_full_baseline_fingerprint_drift_is_still_rejected",
            ),
            (
                "old four regression 16 retained",
                "test_round_16_walk_forward_slice_fingerprint_may_differ_from_full_probe",
            ),
            (
                "old four regression 17 retained",
                "test_round_17_explicit_stress_fingerprint_may_differ_from_probe",
            ),
            (
                "old four regression 19 retained",
                "test_round_19_trial_artifacts_are_cleaned_when_trial_raises",
            ),
            ("probe missing count removes field", "del run.dataset.missing_candle_count"),
            ("explicit TypeError expected", "TypeError,"),
            (
                "gap ValueError expected",
                'assertRaisesRegex(ValueError, "gap-free compact dataset")',
            ),
            (
                "trial contract error retained in trial result",
                "missing required field(s): missing_candle_count",
            ),
            ("trial gap error retained in trial result", 'self.assertIn("missing candles"'),
        ],
    )
    if len(matrix.rows) != 40:
        raise AssertionError(len(matrix.rows))

    # ------------------------------------------------------------------
    # 41-80: Full Freqtrade config schema example.
    # ------------------------------------------------------------------
    config_path = root / "config_examples/config_hedge_paper.example.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    api = config.get("api_server", {})
    operations_test = _text(root, "tests/hedge/operations/test_config.py")

    config_checks: list[tuple[str, Callable[[], bool]]] = [
        ("example is JSON object", lambda: isinstance(config, dict)),
        ("dry_run true", lambda: config.get("dry_run") is True),
        ("trading mode futures", lambda: config.get("trading_mode") == "futures"),
        ("margin mode cross", lambda: config.get("margin_mode") == "cross"),
        ("position mode hedge", lambda: config.get("position_mode") == "hedge"),
        ("exchange binance", lambda: config.get("exchange", {}).get("name") == "binance"),
        ("pair whitelist exists", lambda: bool(config.get("exchange", {}).get("pair_whitelist"))),
        ("api object exists", lambda: isinstance(api, dict)),
        ("api disabled", lambda: api.get("enabled") is False),
        ("api loopback", lambda: api.get("listen_ip_address") == "127.0.0.1"),
        (
            "api port valid",
            lambda: isinstance(api.get("listen_port"), int) and 1024 <= api["listen_port"] <= 65535,
        ),
        ("api username string", lambda: isinstance(api.get("username"), str)),
        ("api password string", lambda: isinstance(api.get("password"), str)),
        ("api jwt string", lambda: isinstance(api.get("jwt_secret_key"), str)),
        ("api jwt min length", lambda: len(api.get("jwt_secret_key", "")) >= 32),
        ("api CORS list", lambda: isinstance(api.get("CORS_origins"), list)),
        ("api verbosity valid", lambda: api.get("verbosity") in {"error", "info"}),
        ("hedge object exists", lambda: isinstance(config.get("hedge"), dict)),
        ("hedge readonly", lambda: config["hedge"].get("read_only") is True),
        ("live trading disabled", lambda: config["hedge"].get("live_trading_enabled") is False),
        ("paper mode", lambda: config["hedge"].get("operation_mode") == "paper"),
        (
            "paper initial balance",
            lambda: config["hedge"].get("paper", {}).get("initial_balance") == "1000",
        ),
        ("paper leverage", lambda: config["hedge"].get("paper", {}).get("leverage") == "3"),
        (
            "closed candle required",
            lambda: config["hedge"].get("paper", {}).get("require_closed_candle") is True,
        ),
        (
            "max missing candles zero",
            lambda: config["hedge"].get("paper", {}).get("max_missing_candles") == 0,
        ),
    ]
    _add_callable_checks(matrix, "paper-config", config_checks)

    _add_token_checks(
        matrix,
        "paper-config-test",
        operations_test,
        [
            ("setup_utils_configuration imported", "setup_utils_configuration"),
            ("RunMode imported", "RunMode"),
            (
                "full pipeline test exists",
                "test_paper_example_passes_full_freqtrade_configuration_pipeline",
            ),
            ("hedge-db command supplied", '"command": "hedge-db"'),
            ("config file list supplied", '"config": [str(path)]'),
            ("UTIL_NO_EXCHANGE used", "RunMode.UTIL_NO_EXCHANGE"),
            ("dry run asserted", 'config["dry_run"] is True'),
            ("api disabled asserted", 'config["api_server"]["enabled"] is False'),
            ("loopback asserted", 'config["api_server"]["listen_ip_address"] == "127.0.0.1"'),
            ("hedge readonly asserted", 'config["hedge"]["read_only"] is True'),
        ],
    )
    # schema-required fields are checked individually again as a contract set.
    _add_token_checks(
        matrix,
        "paper-config-schema",
        api,
        [
            (f"api required field {key}", key)
            for key in ("enabled", "listen_ip_address", "listen_port", "username", "password")
        ],
    )
    if len(matrix.rows) != 80:
        raise AssertionError(len(matrix.rows))

    # ------------------------------------------------------------------
    # 81-110: Windows local venv alignment.
    # ------------------------------------------------------------------
    win_script = _text(root, "scripts/Align-Freqtrade-Hedge-V17-Windows-Runtime-PS51.ps1")
    dual_script = _text(root, "scripts/Test-Freqtrade-Hedge-RiskLevelRL-DualRuntime.ps1")
    req_rl = _text(root, "requirements-freqai-rl.txt")
    win_tokens = [
        ("project root parameter", "[string]$ProjectRoot"),
        ("optional proxy parameter", "[string]$Proxy"),
        ("strict mode", "Set-StrictMode -Version Latest"),
        ("stop on errors", '$ErrorActionPreference = "Stop"'),
        ("project local venv", ".venv\\Scripts\\python.exe"),
        ("requirements authority", "requirements-freqai-rl.txt"),
        ("dynamic tqdm regex", "(?m)^tqdm=="),
        ("metadata version query", "importlib.metadata"),
        ("before version captured", "$Before"),
        ("target version printed", "$RequiredTqdm"),
        ("conditional install only", "if ($Before -ne $RequiredTqdm)"),
        ("no pip editable install", "pip install -e"),
        ("disable pip version check", "--disable-pip-version-check"),
        ("no pip cache", "--no-cache-dir"),
        ("only-if-needed", '"only-if-needed"'),
        ("proxy is optional", "IsNullOrWhiteSpace($Proxy)"),
        ("exact tqdm install", '"tqdm==" + $RequiredTqdm'),
        ("after version captured", "$After"),
        ("post alignment equality", "$After -ne $RequiredTqdm"),
        ("pip check", "-m pip check"),
        ("PASS marker", "WINDOWS_V17_RUNTIME_ALIGNMENT: PASS"),
    ]
    _add_windows_token_checks(matrix, win_script, win_tokens)
    matrix.add("windows-runtime", "requirements pin is 4.69.0", "tqdm==4.69.0" in req_rl)
    matrix.add(
        "windows-runtime",
        "dual runtime exact tqdm gate",
        "m.version('tqdm') == '4.69.0'" in dual_script,
    )
    matrix.add("windows-runtime", "alignment script ASCII", win_script.isascii())
    matrix.add(
        "windows-runtime", "alignment does not touch Docker", "docker " not in win_script.lower()
    )
    matrix.add("windows-runtime", "alignment does not mutate source", "Copy-Item" not in win_script)
    matrix.add("windows-runtime", "alignment does not install torch", "torch==" not in win_script)
    matrix.add(
        "windows-runtime",
        "alignment resolves pin before install",
        win_script.index("$RequiredTqdm") < win_script.index('"tqdm==" + $RequiredTqdm'),
    )
    matrix.add(
        "windows-runtime",
        "alignment verifies Python exists",
        "Project-local Python is missing" in win_script,
    )
    matrix.add(
        "windows-runtime",
        "alignment verifies requirements exists",
        "requirements-freqai-rl.txt is missing" in win_script,
    )
    if len(matrix.rows) != 110:
        raise AssertionError(len(matrix.rows))

    # ------------------------------------------------------------------
    # 111-150: Docker isolated pytest harness.
    # ------------------------------------------------------------------
    docker_script = _text(root, "scripts/Test-Freqtrade-Hedge-Docker-IsolatedPytest-PS51.ps1")
    docker_tokens = [
        ("existing container parameter", "freqtrade-hedge-clean-v121-dryrun"),
        ("proxy parameter", "host.docker.internal:7897"),
        ("target parameter", '$PytestTarget = "tests/hedge"'),
        ("keep temp switch", "$KeepTemporaryVenv"),
        ("runtime python fixed", "/opt/hedge-venv/bin/python"),
        ("runtime site fixed", "/opt/hedge-venv/lib/python3.12/site-packages"),
        ("test venv under tmp", "/tmp/freqtrade-hedge-pytest-venv"),  # noqa: S108 - static token
        ("test python under temp", '$TestPython = $TestVenv + "/bin/python"'),
        ("pth bridge", "freqtrade-hedge-runtime.pth"),
        ("pytest pin", "pytest==9.1.1"),
        ("pytest asyncio pin", "pytest-asyncio==1.4.0"),
        ("pytest cov pin", "pytest-cov==7.1.0"),
        ("pytest mock pin", "pytest-mock==3.15.1"),
        ("pytest random pin", "pytest-random-order==1.2.0"),
        ("pytest timeout pin", "pytest-timeout==2.4.0"),
        ("pytest xdist pin", "pytest-xdist==3.8.0"),
        ("container running gate", "must already be running in maintenance/test mode"),
        ("runtime freeze before", "$Before"),
        ("runtime freeze after", "$After"),
        ("runtime immutability check", "$Before -ne $After"),
        ("contamination error", "runtime venv contamination detected"),
        ("temporary venv created", "$RuntimePython -m venv $TestVenv"),
        ("runtime site added to pth", '$RuntimeSite + "`n/opt/freqtrade-hedge`n"'),
        ("test pip used", '$TestPython, "-m", "pip", "install"'),
        ("pytest executed with test python", "$TestPython -m pytest"),
        ("no bytecode", "PYTHONDONTWRITEBYTECODE=1"),
        ("hash seed fixed", "PYTHONHASHSEED=0"),
        ("cacheprovider disabled", "no:cacheprovider"),
        ("addopts cleared", '"addopts="'),
        ("temp venv cleanup", "rm -rf $TestVenv"),
        ("PASS marker", "DOCKER_ISOLATED_PYTEST: PASS"),
    ]
    _add_token_checks(matrix, "docker-pytest", docker_script, docker_tokens)
    matrix.add("docker-pytest", "no docker build", "docker build" not in docker_script.lower())
    matrix.add("docker-pytest", "no docker run", "docker run" not in docker_script.lower())
    matrix.add(
        "docker-pytest",
        "no runtime pip install",
        '$RuntimePython, "-m", "pip", "install"' not in docker_script,
    )
    matrix.add("docker-pytest", "script ASCII", docker_script.isascii())
    static_test = _text(root, "tests/hedge/deployment/test_targeted_runtime_remediation.py")
    matrix.add(
        "docker-pytest", "static harness tests exist", "test_docker_pytest_harness" in static_test
    )
    matrix.add("docker-pytest", "static venv test exists", "isolated" in static_test.lower())
    matrix.add(
        "docker-pytest",
        "runtime path immutability tested",
        "runtime venv contamination detected" in static_test,
    )
    matrix.add(
        "docker-pytest",
        "all test plugin pins statically asserted",
        "pytest-xdist==3.8.0" in static_test,
    )
    matrix.add(
        "docker-pytest", "temporary venv optional retention", "KeepTemporaryVenv" in docker_script
    )
    if len(matrix.rows) != 150:
        raise AssertionError(len(matrix.rows))

    # ------------------------------------------------------------------
    # 151-200: staged Ruff cleanup and syntax safety.
    # ------------------------------------------------------------------
    decimal_integer = re.compile(r'Decimal\("-?(?:0|[1-9][0-9]*)"\)')
    _add_decimal_checks(matrix, root, decimal_integer)

    hedge_rl = root / "freqtrade/freqai/hedge_rl"
    critical_rl_files = [
        "risk_bridge.py",
        "risk_environment.py",
        "risk_gym.py",
        "risk_levels.py",
        "risk_memory.py",
        "risk_portfolio.py",
        "risk_reward.py",
        "risk_runtime.py",
    ]
    _add_named_python_checks(matrix, root, hedge_rl, critical_rl_files)

    risk_levels = _text(root, "freqtrade/freqai/hedge_rl/risk_levels.py")
    matrix.add(
        "ruff-staged",
        "bool risk level raises TypeError",
        'raise TypeError("boolean is not a risk level")' in risk_levels,
    )
    matrix.add(
        "ruff-staged",
        "bool joint id raises TypeError",
        'raise TypeError("joint risk action id' in risk_levels,
    )
    risk_tests = _text(root, "tests/hedge/mlrl/test_risk_action_reward_contract.py")
    matrix.add(
        "ruff-staged",
        "bool action TypeError test",
        "test_action_parser_rejects_bool_levels" in risk_tests
        and "pytest.raises(TypeError)" in risk_tests,
    )
    matrix.add(
        "ruff-staged",
        "bool joint TypeError test",
        "test_joint_id_rejects_bool_input_as_type_error" in risk_tests,
    )

    ruff_tool = _text(root, "tools/validate_hedge_ruff_staged.py")
    _add_token_checks(
        matrix,
        "ruff-staged",
        ruff_tool,
        [
            "STAGE_BUDGETS",
            '"freqtrade/hedge": 0',
            '"freqtrade/freqai/hedge_rl": 0',
            '"tests/hedge": 0',
            '"tools": 0',
            'BLOCKING_RULES = "E9,F63,F7,F82"',
            "--output-format",
            '"json"',
            "blocking Ruff correctness issues",
            "must remain Ruff-clean",
        ],
    )

    # Compile a broad set of changed/critical files as independent rounds.
    compile_targets = [
        "freqtrade/hedge/optimization/freqtrade_adapter.py",
        "tests/hedge/optimization/test_code_quality.py",
        "tests/hedge/operations/test_config.py",
        "tests/hedge/deployment/test_targeted_runtime_remediation.py",
        "tests/hedge/mlrl/test_risk_action_reward_contract.py",
        "tools/validate_hedge_ruff_staged.py",
        "tools/validate_hedge_targeted_remediation_200.py",
        "freqtrade/freqai/hedge_rl/risk_bridge.py",
        "freqtrade/freqai/hedge_rl/risk_environment.py",
        "freqtrade/freqai/hedge_rl/risk_gym.py",
        "freqtrade/freqai/hedge_rl/risk_levels.py",
        "freqtrade/freqai/hedge_rl/risk_memory.py",
        "freqtrade/freqai/hedge_rl/risk_portfolio.py",
        "freqtrade/freqai/hedge_rl/risk_reward.py",
        "freqtrade/freqai/hedge_rl/risk_runtime.py",
        "freqtrade/hedge/backtesting/dataset.py",
        "freqtrade/configuration/config_setup.py",
        "freqtrade/config_schema/config_schema.py",
        "freqtrade/commands/hedge_db_commands.py",
        "freqtrade/hedge/backtesting/memory.py",
        "freqtrade/hedge/research/validation_matrix.py",
    ]
    _add_python_file_checks(matrix, "syntax-safety", root, compile_targets)

    # Final policy checks complete the 200-point matrix.
    matrix.add(
        "ruff-staged",
        "no blind Ruff fix in Windows alignment",
        "ruff --fix" not in win_script.lower(),
    )
    matrix.add(
        "ruff-staged",
        "no blind Ruff fix in Docker harness",
        "ruff --fix" not in docker_script.lower(),
    )
    matrix.add(
        "ruff-staged",
        "staged Ruff validator itself compiles",
        _python_ok(root / "tools/validate_hedge_ruff_staged.py"),
    )
    matrix.add("ruff-staged", "remediation validator compiles", _python_ok(Path(__file__)))

    if len(matrix.rows) != 200:
        raise AssertionError(len(matrix.rows))
    failed = [row for row in matrix.rows if not row["passed"]]
    report = {
        "schema": SCHEMA,
        "rounds": 200,
        "passed": 200 - len(failed),
        "failed": len(failed),
        "failures": failed,
        "status": "PASS" if not failed else "FAIL",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
