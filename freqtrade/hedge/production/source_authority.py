"""Immutable source identity used to gate promotable HEDGE artifacts.

The manifest proves file content while Git proves the reviewed source revision.  Both
are retained here so a model or release cannot be promoted from an untracked local
overlay by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess


_HEX = frozenset("0123456789abcdef")


def _sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str")
    result = value.strip().lower()
    if len(result) != 64 or any(char not in _HEX for char in result):
        raise ValueError(f"{name} must be sha256")
    return result


def _git_sha(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str or None")
    result = value.strip().lower()
    if len(result) != 40 or any(char not in _HEX for char in result):
        raise ValueError(f"{name} must be a full git SHA")
    return result


@dataclass(frozen=True, slots=True)
class MigrationProvenance:
    source: str
    source_revision: str
    migration_id: str

    def __post_init__(self) -> None:
        for name in ("source", "source_revision", "migration_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be str")
            if not value.strip():
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value.strip())


@dataclass(frozen=True, slots=True)
class SourceAuthority:
    """The sole auditable identity for current HEDGE source artifacts."""

    repository: str
    branch: str
    commit_sha: str | None
    tree_sha: str | None
    manifest_sha256: str
    source_dirty: bool
    release_id: str
    generated_at: datetime
    migration_provenance: tuple[MigrationProvenance, ...] = ()

    def __post_init__(self) -> None:
        for name in ("repository", "branch", "release_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be str")
            if not value.strip():
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "commit_sha", _git_sha(self.commit_sha, name="commit_sha"))
        object.__setattr__(self, "tree_sha", _git_sha(self.tree_sha, name="tree_sha"))
        object.__setattr__(self, "manifest_sha256", _sha256(self.manifest_sha256, name="manifest_sha256"))
        if not isinstance(self.source_dirty, bool):
            raise TypeError("source_dirty must be bool")
        if not isinstance(self.generated_at, datetime):
            raise TypeError("generated_at must be datetime")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        object.__setattr__(self, "generated_at", self.generated_at.astimezone(UTC))
        provenance = tuple(self.migration_provenance)
        if not all(isinstance(item, MigrationProvenance) for item in provenance):
            raise TypeError("migration_provenance must contain MigrationProvenance")
        object.__setattr__(self, "migration_provenance", provenance)

    @property
    def identity_sha256(self) -> str:
        payload = {
            "repository": self.repository,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "tree_sha": self.tree_sha,
            "manifest_sha256": self.manifest_sha256,
            "source_dirty": self.source_dirty,
            "release_id": self.release_id,
            "migration_provenance": [
                {"source": item.source, "source_revision": item.source_revision, "migration_id": item.migration_id}
                for item in self.migration_provenance
            ],
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def promotable(self) -> bool:
        return (
            self.repository == "XXA222/HEDGE"
            and self.commit_sha is not None
            and self.tree_sha is not None
            and not self.source_dirty
        )


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-c", f"safe.directory={root}", "-C", str(root), *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def discover_source_authority(
    root: str | Path,
    *,
    repository: str = "XXA222/HEDGE",
    release_id: str = "HEDGE",
    migration_provenance: tuple[MigrationProvenance, ...] = (),
    now: datetime | None = None,
) -> SourceAuthority:
    """Discover Git and manifest facts locally; missing Git is never promotable."""

    root_path = Path(root).resolve()
    manifest = root_path / "CLEAN-MAINLINE-MANIFEST.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    branch = _git(root_path, "branch", "--show-current") or "DETACHED_OR_UNKNOWN"
    commit_sha = _git(root_path, "rev-parse", "HEAD")
    tree_sha = _git(root_path, "rev-parse", "HEAD^{tree}")
    status = _git(root_path, "status", "--porcelain=v1")
    return SourceAuthority(
        repository=repository,
        branch=branch,
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        manifest_sha256=sha256(manifest.read_bytes()).hexdigest(),
        source_dirty=bool(status) if status is not None else True,
        release_id=release_id,
        generated_at=now or datetime.now(UTC),
        migration_provenance=migration_provenance,
    )
