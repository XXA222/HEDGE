"""Strict external exchange-risk evidence contract for HPRL release qualification.

The tensor research environment does not pretend its synthetic bankruptcy threshold is a Binance
maintenance-margin liquidation engine.  Release qualification may consume independently produced,
hashable evidence from HEDGE's exchange/risk acceptance tooling instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .diagnostics import sha256_file


@dataclass(frozen=True, slots=True)
class ExchangeRiskEvidence:
    schema: str
    symbol: str
    margin_mode: str
    position_mode: str
    liquidation_model: str
    maintenance_margin_source: str
    accepted: bool
    observed_at: str
    checks: Mapping[str, bool]
    source_sha256: str
    source_path: str

    def __post_init__(self) -> None:
        if self.schema != "hedge-exchange-risk-evidence-v1":
            raise ValueError("unsupported exchange-risk evidence schema")
        if not self.symbol.strip():
            raise ValueError("exchange-risk evidence symbol cannot be empty")
        if self.margin_mode.lower() != "cross":
            raise ValueError("HPRL ETH release requires cross-margin evidence")
        if self.position_mode.lower() not in {"hedge", "dual_side", "dual-side"}:
            raise ValueError("HPRL ETH release requires hedge/dual-side position mode evidence")
        if not self.liquidation_model.strip() or not self.maintenance_margin_source.strip():
            raise ValueError("exchange-risk evidence must identify liquidation and maintenance-margin sources")
        if not isinstance(self.accepted, bool):
            raise TypeError("exchange-risk evidence accepted must be boolean")
        if not self.checks or any(not isinstance(value, bool) for value in self.checks.values()):
            raise ValueError("exchange-risk evidence checks must be a non-empty bool mapping")
        try:
            parsed = datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("exchange-risk observed_at must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("exchange-risk observed_at must be timezone-aware")
        if len(self.source_sha256) != 64:
            raise ValueError("exchange-risk evidence source hash must be SHA256")

    @property
    def verified(self) -> bool:
        return self.accepted and all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "symbol": self.symbol,
            "margin_mode": self.margin_mode,
            "position_mode": self.position_mode,
            "liquidation_model": self.liquidation_model,
            "maintenance_margin_source": self.maintenance_margin_source,
            "accepted": self.accepted,
            "observed_at": self.observed_at,
            "checks": dict(self.checks),
            "source_sha256": self.source_sha256,
            "source_path": self.source_path,
            "verified": self.verified,
        }


def load_exchange_risk_evidence(path: str | Path, *, expected_symbol: str) -> ExchangeRiskEvidence:
    target = Path(path).expanduser().resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("exchange-risk evidence root must be an object")
    evidence = ExchangeRiskEvidence(
        schema=str(payload.get("schema", "")),
        symbol=str(payload.get("symbol", "")),
        margin_mode=str(payload.get("margin_mode", "")),
        position_mode=str(payload.get("position_mode", "")),
        liquidation_model=str(payload.get("liquidation_model", "")),
        maintenance_margin_source=str(payload.get("maintenance_margin_source", "")),
        accepted=payload.get("accepted"),
        observed_at=str(payload.get("observed_at", "")),
        checks=dict(payload.get("checks", {})),
        source_sha256=sha256_file(target),
        source_path=str(target),
    )
    if evidence.symbol != expected_symbol:
        raise ValueError(
            f"exchange-risk evidence symbol mismatch: expected {expected_symbol!r}, got {evidence.symbol!r}"
        )
    if not evidence.verified:
        raise ValueError("exchange-risk evidence is not fully accepted")
    return evidence
