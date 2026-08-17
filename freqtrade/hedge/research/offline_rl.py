"""Unified point-in-time offline-RL transition contract for all Hedge policies."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import json

from freqtrade.hedge.contracts import finite_decimal


def _hash(value: str) -> str:
    value = value.lower().strip()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("hash must be sha256")
    return value


@dataclass(frozen=True, slots=True)
class OfflineRLTransition:
    transition_id: str
    event_time: datetime
    available_at: datetime
    observation_sha256: str
    action_id: int
    action_mask_sha256: str
    behavior_probability: Decimal
    reward: Decimal
    next_observation_sha256: str
    terminated: bool
    policy_family: str

    def __post_init__(self) -> None:
        if not self.transition_id.strip() or not self.policy_family.strip():
            raise ValueError("transition_id and policy_family are required")
        for name in ("event_time", "available_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(UTC))
        if self.available_at < self.event_time:
            raise ValueError("available_at cannot precede event_time")
        for name in ("observation_sha256", "action_mask_sha256", "next_observation_sha256"):
            object.__setattr__(self, name, _hash(getattr(self, name)))
        if isinstance(self.action_id, bool) or not isinstance(self.action_id, int) or self.action_id < 0:
            raise ValueError("action_id must be nonnegative int")
        probability = finite_decimal(self.behavior_probability, field_name="behavior_probability")
        if not Decimal(0) < probability <= Decimal(1):
            raise ValueError("behavior_probability must be within (0,1]")
        object.__setattr__(self, "behavior_probability", probability)
        object.__setattr__(self, "reward", finite_decimal(self.reward, field_name="reward"))
        if not isinstance(self.terminated, bool):
            raise TypeError("terminated must be bool")


def offline_dataset_sha256(rows: tuple[OfflineRLTransition, ...]) -> str:
    if not rows or len({row.transition_id for row in rows}) != len(rows):
        raise ValueError("unique nonempty transitions required")
    payload = [{name: str(getattr(row, name)) for name in row.__dataclass_fields__} for row in rows]
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
