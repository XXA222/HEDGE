from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from suite_specs import TIMEFRAMES, input_timeframes_for


HERE = Path(__file__).resolve().parent
EXPECTED = {
    "fast_td3": "HPRLFastTD3ETHStrategy",
    "fast_dsac": "HPRLFastDSACETHStrategy",
    "simba_sac": "HPRLSimbaSACETHStrategy",
    "xqc": "HPRLXQCETHStrategy",
    "rebrac_v2": "HPRLReBRACv2ETHStrategy",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:  # noqa: C901
    errors: list[str] = []
    parsed = 0
    for path in sorted(HERE.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")
            parsed += 1
        except Exception as exc:
            errors.append(f"python:{path.relative_to(HERE)}:{type(exc).__name__}:{exc}")

    base = (HERE / "strategies" / "hprl_mtf_v3_base.py").read_text(encoding="utf-8")
    required_tokens = (
        "informative_pairs",
        "get_pair_dataframe",
        "inference_arrays_mtf",
        "input_timeframes_for",
        "create_agent",
        "VectorizedHedgeEnv",
        "load_checkpoint",
        "HprlHedgeAdapter",
        "PlannedExecutionIntent",
        "executed_level_index",
        "artifact_manifest.json",
        "runtime_contract_sha256",
        "formal_start_position",
        "hedge_long_score",
        "hedge_target_net",
        "hedge_target_net_ratio",
        "hedge_long_exposure_scale",
        "hedge_allow_new_risk",
    )
    for token in required_tokens:
        if token not in base:
            errors.append(f"strategy-base-missing:{token}")
    if "HPRL_ETH_MULTI_TF" in base or "runner.py" in base:
        errors.append("formal Strategy must not import or invoke the legacy runner")

    feature_source = (HERE / "features.py").read_text(encoding="utf-8")
    for token in (
        "align_multi_timeframe_features",
        "source_close_ns",
        "decision_close_ns",
        "searchsorted",
        "age_frac",
        "stale informative data",
        "training_arrays_mtf",
        "inference_arrays_mtf",
    ):
        if token not in feature_source:
            errors.append(f"feature-contract-missing:{token}")

    run_source = (HERE / "run_suite.py").read_text(encoding="utf-8")
    if '"hedge-backtesting"' not in run_source:
        errors.append("formal orchestrator does not invoke hedge-backtesting")
    if "evaluate_agent" in run_source:
        errors.append("formal orchestrator must not use custom HPRL evaluate_agent")
    if "--data-format-ohlcv" not in run_source:
        errors.append("formal orchestrator does not preserve prepared OHLCV data format")

    artifact_source = (HERE / "artifact_contract.py").read_text(encoding="utf-8")
    for token in (
        "SOURCE_CONTRACT_SCHEMA",
        "RUNTIME_CONTRACT_SCHEMA",
        "input_timeframes",
        "alignment_contract",
        "freqtrade/data/dataprovider.py",
        "freqtrade/strategy/interface.py",
    ):
        if token not in artifact_source:
            errors.append(f"artifact-contract-missing:{token}")

    for base_tf in TIMEFRAMES:
        inputs = input_timeframes_for(base_tf)
        if inputs[0] != base_tf:
            errors.append(f"mtf-order:{base_tf}:{inputs}")
        base_index = TIMEFRAMES.index(base_tf)
        if inputs != TIMEFRAMES[base_index:]:
            errors.append(f"mtf-surface:{base_tf}:{inputs}")

    for model, strategy in EXPECTED.items():
        config_path = HERE / "configs" / f"{model}.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"json:{config_path.name}:{exc}")
            continue
        if config.get("strategy") != strategy:
            errors.append(f"strategy-config:{model}:{config.get('strategy')} != {strategy}")
        if config.get("trading_mode") != "futures" or config.get("margin_mode") != "cross":
            errors.append(f"futures-config:{model}")
        if config.get("position_mode") != "hedge" or config.get("hedge_mode_enabled") is not True:
            errors.append(f"hedge-mode-config:{model}")
        if config.get("exchange", {}).get("pair_whitelist") != ["ETH/USDT:USDT"]:
            errors.append(f"pair-config:{model}")
        planner = config.get("hedge", {}).get("planner", {})
        expected_planner = {
            "core_wallet_exposure_long": "0.40",
            "core_wallet_exposure_short": "0.40",
            "tactical_wallet_exposure_long": "0",
            "tactical_wallet_exposure_short": "0",
            "max_wallet_exposure_long": "0.40",
            "max_wallet_exposure_short": "0.40",
            "max_gross_wallet_exposure": "0.70",
        }
        for key, value in expected_planner.items():
            if planner.get(key) != value:
                errors.append(f"planner-profile:{model}:{key}:{planner.get(key)} != {value}")

    matrix = [(model, tf) for model in EXPECTED for tf in TIMEFRAMES]
    if len(matrix) != 30:
        errors.append(f"matrix-size:{len(matrix)}")

    for path in sorted(HERE.glob("*.ps1")):
        raw = path.read_bytes()
        try:
            raw.decode("ascii")
        except UnicodeDecodeError:
            errors.append(f"powershell-not-ascii:{path.name}")

    manifest_files = []
    for path in sorted(HERE.rglob("*")):
        if (
            path.is_file()
            and path.name != "PACKAGE_MANIFEST.json"
            and "__pycache__" not in path.parts
        ):
            manifest_files.append({
                "path": path.relative_to(HERE).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha(path),
            })

    payload = {
        "schema": "hprl-freqtrade-mtf-source-validation-v3",
        "status": (
            "SOURCE_STATIC_CHECK_OK_TESTS_DEFERRED"
            if not errors
            else "SOURCE_STATIC_CHECK_FAILED"
        ),
        "python_files_parsed": parsed,
        "strategies": len(EXPECTED),
        "base_timeframes": list(TIMEFRAMES),
        "mtf_inputs": {tf: list(input_timeframes_for(tf)) for tf in TIMEFRAMES},
        "formal_backtest_tasks_available_for_later": len(matrix),
        "training_executed": False,
        "backtests_executed": False,
        "performance_test_executed": False,
        "errors": errors,
        "files": manifest_files,
    }
    (HERE / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    visible = {key: value for key, value in payload.items() if key != "files"}
    print(json.dumps(visible, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
