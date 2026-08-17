"""Local-only Binance credential loading with redaction-safe contracts.

Credential files are never copied into project configuration, artifacts, or reports.
Callers obtain a mapping intended only for a child-process environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class CredentialFileError(ValueError):
    """The local credential source is unavailable or lacks a usable key pair."""


@dataclass(frozen=True, slots=True)
class BinanceCredentials:
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)

    def __post_init__(self) -> None:
        for name in ("api_key", "api_secret"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
                raise CredentialFileError(f"{name} must be a nonempty single-line value")
            if len(value) > 1024:
                raise CredentialFileError(f"{name} is unexpectedly long")

    def environment(self) -> dict[str, str]:
        return {
            "FREQTRADE__EXCHANGE__KEY": self.api_key,
            "FREQTRADE__EXCHANGE__SECRET": self.api_secret,
        }


_KEY_ALIASES = ("key", "api_key", "apikey", "binance_api_key")
_SECRET_ALIASES = ("secret", "api_secret", "apisecret", "binance_api_secret")


def _mapping_credentials(values: dict[object, object]) -> BinanceCredentials | None:
    normalized = {str(name).strip().lower(): str(value).strip() for name, value in values.items()}
    key_values = tuple(normalized[name] for name in _KEY_ALIASES if name in normalized)
    secret_values = tuple(normalized[name] for name in _SECRET_ALIASES if name in normalized)
    if len(set(key_values)) > 1 or len(set(secret_values)) > 1:
        raise CredentialFileError("credential file has conflicting key or secret labels")
    key = key_values[0] if key_values else None
    secret = secret_values[0] if secret_values else None
    return None if key is None or secret is None else BinanceCredentials(key, secret)


def _line_credentials(raw: str) -> BinanceCredentials | None:
    lines = tuple(
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(lines) == 2:
        labelled = _labelled_credentials(lines)
        return labelled if labelled is not None else BinanceCredentials(lines[0], lines[1])
    return _labelled_credentials(lines)


def _labelled_credentials(lines: tuple[str, ...]) -> BinanceCredentials | None:
    values: dict[str, str] = {}
    for line in lines:
        delimiter = "=" if "=" in line else ":" if ":" in line else ""
        if not delimiter:
            return None
        name, value = line.split(delimiter, maxsplit=1)
        values[name.strip()] = value.strip()
    return _mapping_credentials(values)


def load_binance_credentials(path: Path) -> BinanceCredentials:
    """Load a JSON, labelled-line, or two-line key/secret credential file.

    Error messages deliberately contain field names and paths, never credential values.
    """
    if not isinstance(path, Path):
        raise TypeError("credential path must be pathlib.Path")
    source = path.expanduser()
    if not source.is_file():
        raise CredentialFileError(f"credential file is unavailable: {source}")
    try:
        raw = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise CredentialFileError(f"credential file cannot be read: {source}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        credentials = _line_credentials(raw)
    else:
        credentials = _mapping_credentials(payload) if isinstance(payload, dict) else None
    if credentials is None:
        raise CredentialFileError("credential file must contain Binance API key and secret")
    return credentials
