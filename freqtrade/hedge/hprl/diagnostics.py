"""Numerical, dataset and artifact diagnostics for HPRL research workflows."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class FiniteFieldDiagnostics:
    field: str
    shape: tuple[int, ...]
    total: int
    finite: int
    nan: int
    posinf: int
    neginf: int
    finite_min: float | None
    finite_max: float | None
    first_bad_index: tuple[int, ...] | None

    @property
    def ok(self) -> bool:
        return self.finite == self.total


class NonFiniteTransitionError(ValueError):
    """Fail-closed replay error that preserves the field and execution context."""

    def __init__(
        self,
        diagnostics: Mapping[str, FiniteFieldDiagnostics],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.diagnostics = dict(diagnostics)
        self.context = dict(context or {})
        failed = [name for name, value in self.diagnostics.items() if not value.ok]
        message = "non-finite replay transition"
        if failed:
            message += f" fields={','.join(failed)}"
        if self.context:
            compact = ",".join(f"{key}={value}" for key, value in sorted(self.context.items()) if isinstance(value, (str, int, float, bool)))
            if compact:
                message += f" context={compact}"
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": str(self),
            "context": dict(self.context),
            "fields": {name: asdict(value) for name, value in self.diagnostics.items()},
        }


def tensor_finite_diagnostics(name: str, value: object) -> FiniteFieldDiagnostics:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - HPRL runtime always has torch.
        raise RuntimeError("tensor diagnostics require torch") from exc
    tensor = torch.as_tensor(value)
    finite_mask = torch.isfinite(tensor)
    nan_mask = torch.isnan(tensor)
    posinf_mask = torch.isposinf(tensor)
    neginf_mask = torch.isneginf(tensor)
    bad = (~finite_mask).nonzero(as_tuple=False)
    finite_values = tensor[finite_mask]
    finite_min = float(finite_values.min().item()) if finite_values.numel() else None
    finite_max = float(finite_values.max().item()) if finite_values.numel() else None
    first_bad = tuple(int(index) for index in bad[0].tolist()) if bad.numel() else None
    return FiniteFieldDiagnostics(
        field=name,
        shape=tuple(int(size) for size in tensor.shape),
        total=int(tensor.numel()),
        finite=int(finite_mask.sum().item()),
        nan=int(nan_mask.sum().item()),
        posinf=int(posinf_mask.sum().item()),
        neginf=int(neginf_mask.sum().item()),
        finite_min=finite_min,
        finite_max=finite_max,
        first_bad_index=first_bad,
    )


def validate_replay_transition_finite(
    *,
    obs: object,
    action: object,
    reward: object,
    next_obs: object,
    done: object,
    context: Mapping[str, Any] | None = None,
) -> None:
    values = {
        "obs": obs,
        "action": action,
        "reward": reward,
        "next_obs": next_obs,
        "done": done,
    }
    diagnostics = {name: tensor_finite_diagnostics(name, value) for name, value in values.items()}
    if any(not value.ok for value in diagnostics.values()):
        raise NonFiniteTransitionError(diagnostics, context=context)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    files = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "schema": "hprl-artifact-manifest-v1",
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def write_artifact_manifest(root: str | Path, *, filename: str = "ARTIFACT-MANIFEST.json") -> Path:
    base = Path(root).resolve()
    target = base / filename
    payload = artifact_manifest(base)
    # Do not recursively include an old manifest as authoritative evidence.
    payload["files"] = [item for item in payload["files"] if item["path"] != filename]
    payload["file_count"] = len(payload["files"])
    payload["total_bytes"] = sum(item["bytes"] for item in payload["files"])
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def finite_mapping(values: Mapping[str, object]) -> bool:
    for value in values.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True
