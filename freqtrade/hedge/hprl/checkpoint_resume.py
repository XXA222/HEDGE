"""Explicit HPRL checkpoint resume modes.

The ETH research workflow intentionally uses weight-only warm starts for final retraining.
It never labels a partially restored trainer as an exact resume.
"""
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from .device import require_torch, torch_device


torch = require_torch()


class ResumeMode(StrEnum):
    WARM_START_WEIGHTS = "warm_start_weights"
    WARM_START_WEIGHTS_OPTIMIZER = "warm_start_weights_optimizer"
    EXACT_RESUME = "exact_resume"


def load_warm_start_weights(path: str | Path, agent: object) -> dict[str, Any]:
    """Restore network weights/targets only; optimizer, RNG, replay and trainer state stay fresh."""
    target = torch_device(str(getattr(agent, "device", "cpu")))
    state = torch.load(Path(path), map_location=target, weights_only=True)
    saved_class = state.get("agent_class")
    current_class = f"{type(agent).__module__}.{type(agent).__qualname__}"
    if saved_class is not None and saved_class != current_class:
        raise ValueError(
            f"checkpoint agent class mismatch: saved={saved_class!r}, current={current_class!r}"
        )
    schema = state.get("schema", 1)
    if schema == 1:
        getattr(agent, "actor").load_state_dict(state["actor"])
        getattr(agent, "critic").load_state_dict(state["critic"])
        if hasattr(agent, "actor_target"):
            agent.actor_target.load_state_dict(agent.actor.state_dict())
        if hasattr(agent, "critic_target"):
            agent.critic_target.load_state_dict(agent.critic.state_dict())
        return dict(state.get("metadata", {}))
    agent_state = state.get("agent_state")
    if not isinstance(agent_state, dict):
        raise ValueError("checkpoint agent_state is required")
    modules = dict(agent_state.get("modules", {}))
    for name in ("actor", "critic", "actor_target", "critic_target"):
        if name in modules and hasattr(agent, name):
            getattr(agent, name).load_state_dict(modules[name])
    if "actor_target" not in modules and hasattr(agent, "actor_target"):
        agent.actor_target.load_state_dict(agent.actor.state_dict())
    if "critic_target" not in modules and hasattr(agent, "critic_target"):
        agent.critic_target.load_state_dict(agent.critic.state_dict())
    return dict(state.get("metadata", {}))


def require_supported_resume_mode(mode: str) -> ResumeMode:
    try:
        resolved = ResumeMode(str(mode).strip().lower())
    except ValueError as exc:
        raise ValueError(f"unknown HPRL resume mode: {mode!r}") from exc
    if resolved is ResumeMode.WARM_START_WEIGHTS_OPTIMIZER:
        raise NotImplementedError(
            "warm_start_weights_optimizer is intentionally fail-closed: carrying optimizer state "
            "into a fresh replay/reward-normalizer distribution is a different experiment and "
            "needs an explicit workflow before it can be enabled"
        )
    if resolved is ResumeMode.EXACT_RESUME:
        raise NotImplementedError(
            "exact_resume is intentionally fail-closed until replay, reward-normalizer, "
            "health-history, environment state and dataset cursor are checkpointed together"
        )
    return resolved
