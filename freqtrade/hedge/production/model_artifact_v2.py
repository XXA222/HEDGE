"""Promotion-complete model artifact identity."""

from dataclasses import dataclass
from hashlib import sha256
import json


def _sha(value: str) -> str:
    value = value.lower().strip()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("artifact identity fields must be sha256")
    return value


@dataclass(frozen=True, slots=True)
class ModelArtifactV2:
    model_id: str
    algorithm: str
    model_sha256: str
    source_authority_sha256: str
    dataset_sha256: str
    feature_set_sha256: str
    risk_policy_sha256: str
    simulator_calibration_sha256: str
    benchmark_protocol_sha256: str
    evidence_dag_sha256: str

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.algorithm.strip():
            raise ValueError("model identity is required")
        for name in tuple(self.__dataclass_fields__)[2:]:
            object.__setattr__(self, name, _sha(getattr(self, name)))

    @property
    def fingerprint(self) -> str:
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ModelArtifactRegistryV2:
    def __init__(self) -> None:
        self._artifacts: dict[str, ModelArtifactV2] = {}

    def register(self, artifact: ModelArtifactV2) -> str:
        fingerprint = artifact.fingerprint
        existing = self._artifacts.get(artifact.model_id)
        if existing is not None and existing.fingerprint != fingerprint:
            raise ValueError("model_id already maps to a different immutable artifact")
        self._artifacts[artifact.model_id] = artifact
        return fingerprint

    def get(self, model_id: str) -> ModelArtifactV2 | None:
        return self._artifacts.get(model_id)
