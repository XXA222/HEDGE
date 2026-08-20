from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


MODEL_METADATA_SCHEMA = "hprl-freqtrade-model-v5-mtf"
ARTIFACT_MANIFEST_SCHEMA = "hprl-freqtrade-artifact-manifest-v3-mtf"
RUNTIME_CONTRACT_SCHEMA = "hprl-freqtrade-runtime-contract-v3-mtf"
SOURCE_CONTRACT_SCHEMA = "hprl-freqtrade-source-contract-v2-mtf"
REQUIRED_ARTIFACT_FILES = ("checkpoint.pt", "checkpoint.pt.json", "scaler.npz", "metadata.json")

_SEMANTIC_SUITE_FILES = (
    "artifact_contract.py",
    "features.py",
    "prepare_models.py",
    "suite_specs.py",
    "strategies/hprl_mtf_v3_base.py",
    "strategies/hprl_fast_td3_eth.py",
    "strategies/hprl_fast_dsac_eth.py",
    "strategies/hprl_simba_sac_eth.py",
    "strategies/hprl_xqc_eth.py",
    "strategies/hprl_rebrac_v2_eth.py",
)

# These repository surfaces define the data/Strategy/HEDGE lifecycle consumed by this suite.
_REPOSITORY_INTEGRATION_FILES = (
    "freqtrade/data/dataprovider.py",
    "freqtrade/strategy/interface.py",
    "freqtrade/resolvers/strategy_resolver.py",
    "freqtrade/data/history/history_utils.py",
    "freqtrade/hedge/strategies/contract.py",
    "freqtrade/hedge/production/hprl_hedge_adapter.py",
)


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        _jsonable(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    view = memoryview(array).cast("B")
    chunk = 8 * 1024 * 1024
    for start in range(0, len(view), chunk):
        digest.update(view[start : start + chunk])
    return digest.hexdigest()


def source_contract_payload(*, suite_root: str | Path, repo_root: str | Path) -> dict[str, object]:
    """Hash all source surfaces that define MTF training and Strategy inference semantics."""
    suite = Path(suite_root).resolve()
    repo = Path(repo_root).resolve()
    suite_files: dict[str, str] = {}
    for relative in _SEMANTIC_SUITE_FILES:
        path = suite / relative
        if not path.is_file():
            raise FileNotFoundError(f"HPRL semantic suite source is missing: {path}")
        suite_files[relative] = sha256_file(path)

    hprl_root = repo / "freqtrade" / "hedge" / "hprl"
    if not hprl_root.is_dir():
        raise FileNotFoundError(f"HPRL source tree is missing: {hprl_root}")
    repository_files: dict[str, str] = {}
    for path in sorted(hprl_root.rglob("*.py")):
        relative = path.relative_to(repo).as_posix()
        repository_files[relative] = sha256_file(path)
    for relative in _REPOSITORY_INTEGRATION_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"HPRL integration source is missing: {path}")
        repository_files[relative] = sha256_file(path)
    if not repository_files:
        raise RuntimeError("HPRL semantic repository source set is empty")

    return {
        "schema": SOURCE_CONTRACT_SCHEMA,
        "suite_files": suite_files,
        "repository_files": repository_files,
    }


def runtime_contract_payload(
    *,
    model_key: str,
    algorithm: str,
    strategy_class: str,
    base_timeframe: str,
    input_timeframes: tuple[str, ...],
    feature_version: str,
    feature_names: tuple[str, ...],
    alignment_contract: Mapping[str, object],
    action_config: Mapping[str, object],
    cost_config: Mapping[str, object],
    reward_config: Mapping[str, object],
    model_spec: object,
    source_contract: Mapping[str, object],
) -> dict[str, object]:
    if source_contract.get("schema") != SOURCE_CONTRACT_SCHEMA:
        raise ValueError("invalid HPRL source contract schema")
    if not input_timeframes or input_timeframes[0] != base_timeframe:
        raise ValueError("runtime contract input_timeframes must start with base_timeframe")
    return {
        "schema": RUNTIME_CONTRACT_SCHEMA,
        "model": model_key,
        "algorithm": algorithm,
        "strategy_class": strategy_class,
        "timeframe": base_timeframe,
        "base_timeframe": base_timeframe,
        "input_timeframes": list(input_timeframes),
        "feature_version": feature_version,
        "feature_names": list(feature_names),
        "alignment_contract": _jsonable(alignment_contract),
        "action_config": _jsonable(action_config),
        "cost_config": _jsonable(cost_config),
        "reward_config": _jsonable(reward_config),
        "model_spec": _jsonable(model_spec),
        "source_contract": _jsonable(source_contract),
    }


def atomic_write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    text = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


def atomic_save_npz(path: str | Path, **arrays: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


def verify_manifest_files(root: str | Path, manifest: Mapping[str, object]) -> None:
    base = Path(root)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("HPRL artifact manifest has no file hash table")
    if set(files) != set(REQUIRED_ARTIFACT_FILES):
        raise RuntimeError(
            "HPRL artifact manifest must hash exactly the committed artifact members: "
            f"expected={sorted(REQUIRED_ARTIFACT_FILES)}, actual={sorted(map(str, files))}"
        )
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise TypeError("HPRL artifact manifest file hashes are malformed")
        path = base / name
        if not path.is_file():
            raise FileNotFoundError(f"HPRL artifact manifest file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"HPRL artifact hash mismatch for {name}: expected={expected}, actual={actual}"
            )
