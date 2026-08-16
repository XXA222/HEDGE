"""Real PostgreSQL dual-session visibility and composite closure evidence.

The existing production PostgreSQL implementation owns ledger, backup/restore and
failover semantics.  This module adds an independent DB-API two-session transaction probe
and then composes all measured PostgreSQL evidence without weakening those authorities.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import re
from typing import Any
from uuid import uuid4

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _safe_identifier(value: str, *, field: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} is not a safe SQL identifier")
    return value


def _cursor_scalar(connection: Any, sql: str, params: tuple[object, ...] = ()) -> object:
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return None if row is None else row[0]
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            close()


def _cursor_execute(connection: Any, sql: str, params: tuple[object, ...] = ()) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params)
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            close()


def _close(connection: Any | None) -> None:
    if connection is None:
        return
    close = getattr(connection, "close", None)
    if callable(close):
        close()


def _rollback(connection: Any | None) -> None:
    if connection is None:
        return
    rollback = getattr(connection, "rollback", None)
    if callable(rollback):
        try:
            rollback()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class PostgresVisibilityReport:
    writer_backend_pid: int
    observer_backend_pid: int
    distinct_sessions: bool
    writer_isolation: str
    observer_isolation: str
    first_uncommitted_hidden: bool
    first_commit_visible: bool
    reverse_uncommitted_hidden: bool
    reverse_commit_visible: bool
    cleanup_ok: bool
    evidence_sha256: str
    observed_at: datetime
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.distinct_sessions
            and self.first_uncommitted_hidden
            and self.first_commit_visible
            and self.reverse_uncommitted_hidden
            and self.reverse_commit_visible
            and self.cleanup_ok
            and not self.reasons
        )


def run_postgres_dual_session_visibility(
    connection_factory: Callable[[], Any],
    *,
    now: datetime,
    schema: str = "freqtrade_hedge_probe",
    table_prefix: str = "hprl_visibility",
) -> PostgresVisibilityReport:
    observed = _aware(now)
    schema = _safe_identifier(schema, field="schema")
    table_prefix = _safe_identifier(table_prefix, field="table_prefix")
    table = _safe_identifier(f"{table_prefix}_{uuid4().hex[:12]}", field="table")
    qualified = f'"{schema}"."{table}"'
    writer = observer = None
    writer_pid = observer_pid = 0
    writer_isolation = observer_isolation = ""
    first_hidden = first_visible = reverse_hidden = reverse_visible = cleanup = False
    reasons: list[str] = []
    token = uuid4().hex
    updated = token + "-observer"
    try:
        writer = connection_factory()
        observer = connection_factory()
        writer_pid = int(_cursor_scalar(writer, "SELECT pg_backend_pid()") or 0)
        observer_pid = int(_cursor_scalar(observer, "SELECT pg_backend_pid()") or 0)
        writer_isolation = str(_cursor_scalar(writer, "SHOW transaction_isolation") or "")
        observer_isolation = str(_cursor_scalar(observer, "SHOW transaction_isolation") or "")
        if writer_pid <= 0 or observer_pid <= 0 or writer_pid == observer_pid:
            reasons.append("POSTGRES_VISIBILITY_REQUIRES_DISTINCT_SESSIONS")

        _cursor_execute(writer, f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        _cursor_execute(
            writer,
            f"CREATE TABLE {qualified} (probe_id text PRIMARY KEY, payload text NOT NULL)",
        )
        writer.commit()
        _rollback(observer)

        _cursor_execute(
            writer,
            f"INSERT INTO {qualified} (probe_id,payload) VALUES (%s,%s)",
            (token, token),
        )
        before = int(
            _cursor_scalar(
                observer,
                f"SELECT count(*) FROM {qualified} WHERE probe_id=%s",
                (token,),
            )
            or 0
        )
        first_hidden = before == 0
        writer.commit()
        after = str(
            _cursor_scalar(
                observer,
                f"SELECT payload FROM {qualified} WHERE probe_id=%s",
                (token,),
            )
            or ""
        )
        first_visible = after == token

        _cursor_execute(
            observer,
            f"UPDATE {qualified} SET payload=%s WHERE probe_id=%s",
            (updated, token),
        )
        writer_view_before = str(
            _cursor_scalar(
                writer,
                f"SELECT payload FROM {qualified} WHERE probe_id=%s",
                (token,),
            )
            or ""
        )
        reverse_hidden = writer_view_before == token
        observer.commit()
        writer_view_after = str(
            _cursor_scalar(
                writer,
                f"SELECT payload FROM {qualified} WHERE probe_id=%s",
                (token,),
            )
            or ""
        )
        reverse_visible = writer_view_after == updated

        _cursor_execute(writer, f"DROP TABLE {qualified}")
        writer.commit()
        cleanup = True
    except Exception as exc:
        reasons.append(f"POSTGRES_VISIBILITY_PROBE:{type(exc).__name__}:{str(exc)[:500]}")
        _rollback(writer)
        _rollback(observer)
        if writer is not None:
            try:
                _cursor_execute(writer, f"DROP TABLE IF EXISTS {qualified}")
                writer.commit()
                cleanup = True
            except Exception:
                _rollback(writer)
    finally:
        _close(observer)
        _close(writer)

    checks = (
        (
            writer_pid > 0 and observer_pid > 0 and writer_pid != observer_pid,
            "POSTGRES_SESSIONS_NOT_DISTINCT",
        ),
        (first_hidden, "POSTGRES_UNCOMMITTED_INSERT_VISIBLE"),
        (first_visible, "POSTGRES_COMMITTED_INSERT_NOT_VISIBLE"),
        (reverse_hidden, "POSTGRES_UNCOMMITTED_UPDATE_VISIBLE"),
        (reverse_visible, "POSTGRES_COMMITTED_UPDATE_NOT_VISIBLE"),
        (cleanup, "POSTGRES_VISIBILITY_CLEANUP_FAILED"),
    )
    for ok, reason in checks:
        if not ok and reason not in reasons:
            reasons.append(reason)
    payload = {
        "writer_backend_pid": writer_pid,
        "observer_backend_pid": observer_pid,
        "writer_isolation": writer_isolation,
        "observer_isolation": observer_isolation,
        "first_hidden": first_hidden,
        "first_visible": first_visible,
        "reverse_hidden": reverse_hidden,
        "reverse_visible": reverse_visible,
        "cleanup": cleanup,
        "observed_at": observed.isoformat(),
    }
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return PostgresVisibilityReport(
        writer_backend_pid=writer_pid,
        observer_backend_pid=observer_pid,
        distinct_sessions=writer_pid > 0 and observer_pid > 0 and writer_pid != observer_pid,
        writer_isolation=writer_isolation,
        observer_isolation=observer_isolation,
        first_uncommitted_hidden=first_hidden,
        first_commit_visible=first_visible,
        reverse_uncommitted_hidden=reverse_hidden,
        reverse_commit_visible=reverse_visible,
        cleanup_ok=cleanup,
        evidence_sha256=digest,
        observed_at=observed,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _passed(value: object | None) -> bool:
    return bool(value is not None and getattr(value, "passed", False))


def _digest(value: object | None) -> str:
    if value is None:
        return ""
    for name in ("evidence_sha256", "archive_sha256", "aggregate_sha256", "semantic_sha256"):
        raw = str(getattr(value, name, "") or "").lower()
        if len(raw) == 64 and all(ch in "0123456789abcdef" for ch in raw):
            return raw
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else repr(value)
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class PostgresCompositePolicy:
    require_backup: bool = True
    require_restore: bool = True
    require_failover: bool = True
    require_visibility: bool = True


@dataclass(frozen=True, slots=True)
class PostgresCompositeReport:
    passed: bool
    promotion_eligible: bool
    core_passed: bool
    visibility_passed: bool
    backup_passed: bool
    restore_passed: bool
    failover_passed: bool
    component_sha256: tuple[tuple[str, str], ...]
    evidence_sha256: str
    reasons: tuple[str, ...]


def evaluate_postgres_composite(
    *,
    core: object,
    visibility: PostgresVisibilityReport | None,
    backup: object | None = None,
    restore: object | None = None,
    failover: object | None = None,
    policy: PostgresCompositePolicy | None = None,
) -> PostgresCompositeReport:
    p = policy or PostgresCompositePolicy()
    checks = {
        "core": _passed(core),
        "visibility": _passed(visibility),
        "backup": _passed(backup),
        "restore": _passed(restore),
        "failover": _passed(failover),
    }
    reasons: list[str] = []
    if not checks["core"]:
        reasons.append("POSTGRES_CORE_EVIDENCE_FAILED")
    if p.require_visibility and not checks["visibility"]:
        reasons.append("POSTGRES_VISIBILITY_EVIDENCE_MISSING_OR_FAILED")
    if p.require_backup and not checks["backup"]:
        reasons.append("POSTGRES_BACKUP_EVIDENCE_MISSING_OR_FAILED")
    if p.require_restore and not checks["restore"]:
        reasons.append("POSTGRES_RESTORE_EVIDENCE_MISSING_OR_FAILED")
    if p.require_failover and not checks["failover"]:
        reasons.append("POSTGRES_FAILOVER_EVIDENCE_MISSING_OR_FAILED")
    components = tuple(
        (name, _digest(value))
        for name, value in (
            ("core", core),
            ("visibility", visibility),
            ("backup", backup),
            ("restore", restore),
            ("failover", failover),
        )
        if value is not None
    )
    digest = sha256(json.dumps(components, separators=(",", ":")).encode()).hexdigest()
    promotion_eligible = all(checks.values())
    return PostgresCompositeReport(
        passed=not reasons,
        promotion_eligible=promotion_eligible,
        core_passed=checks["core"],
        visibility_passed=checks["visibility"],
        backup_passed=checks["backup"],
        restore_passed=checks["restore"],
        failover_passed=checks["failover"],
        component_sha256=components,
        evidence_sha256=digest,
        reasons=tuple(reasons),
    )
