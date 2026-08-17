"""Shared neural-network and reinforcement-learning health telemetry.

The helpers in this module are deliberately model-agnostic.  They observe gradients,
parameter movement and policy statistics without changing activations, losses or optimizer
behaviour.  Expensive parameter snapshots are intended to be taken at a configurable interval.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import torch
from torch import nn


NamedParameters = Iterable[tuple[str, nn.Parameter]]
ParameterSnapshot = dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class GradientTelemetry:
    global_norm: float
    weight_norm: float
    near_zero_ratio: float
    per_layer_norm: Mapping[str, float]
    parameter_count: int


def _layer_name(parameter_name: str) -> str:
    """Collapse weight/bias parameters into a stable layer identifier."""

    head, separator, tail = parameter_name.rpartition(".")
    if separator and tail in {"weight", "bias"}:
        return head or parameter_name
    return parameter_name


@torch.no_grad()
def snapshot_parameters(parameters: NamedParameters) -> ParameterSnapshot:
    """Clone trainable parameters for an occasional post-step update measurement."""

    return {
        name: parameter.detach().clone()
        for name, parameter in parameters
        if parameter.requires_grad
    }


@torch.no_grad()
def measure_gradients(
    parameters: NamedParameters,
    *,
    near_zero_threshold: float = 1e-12,
) -> GradientTelemetry:
    """Measure global/per-layer norms and the fraction of near-zero gradient elements."""

    if not math.isfinite(near_zero_threshold) or near_zero_threshold < 0.0:
        raise ValueError("near_zero_threshold must be finite and non-negative")
    gradient_squared = 0.0
    weight_squared = 0.0
    near_zero = 0
    elements = 0
    layer_squared: dict[str, float] = {}
    for name, parameter in parameters:
        if not parameter.requires_grad:
            continue
        weight = parameter.detach().float()
        weight_squared += float(torch.sum(weight.square()).item())
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient = gradient.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite gradient in parameter {name}")
        squared = float(torch.sum(gradient.square()).item())
        gradient_squared += squared
        layer = _layer_name(name)
        layer_squared[layer] = layer_squared.get(layer, 0.0) + squared
        near_zero += int(torch.count_nonzero(gradient.abs() <= near_zero_threshold).item())
        elements += gradient.numel()
    return GradientTelemetry(
        global_norm=math.sqrt(max(gradient_squared, 0.0)),
        weight_norm=math.sqrt(max(weight_squared, 0.0)),
        near_zero_ratio=float(near_zero / elements) if elements else 1.0,
        per_layer_norm={name: math.sqrt(max(value, 0.0)) for name, value in layer_squared.items()},
        parameter_count=elements,
    )


@torch.no_grad()
def measure_parameter_update(
    parameters: NamedParameters,
    before: Mapping[str, torch.Tensor],
    *,
    epsilon: float = 1e-12,
) -> tuple[float, dict[str, float]]:
    """Return actual ``||delta parameter|| / ||parameter||`` globally and per layer."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    delta_squared = 0.0
    weight_squared = 0.0
    layer_delta: dict[str, float] = {}
    layer_weight: dict[str, float] = {}
    for name, parameter in parameters:
        old = before.get(name)
        if old is None:
            continue
        current = parameter.detach()
        if current.shape != old.shape:
            raise ValueError(f"parameter shape changed while measuring update: {name}")
        delta_value = float(torch.sum((current.float() - old.float()).square()).item())
        weight_value = float(torch.sum(old.float().square()).item())
        delta_squared += delta_value
        weight_squared += weight_value
        layer = _layer_name(name)
        layer_delta[layer] = layer_delta.get(layer, 0.0) + delta_value
        layer_weight[layer] = layer_weight.get(layer, 0.0) + weight_value
    global_ratio = math.sqrt(max(delta_squared, 0.0)) / max(
        math.sqrt(max(weight_squared, 0.0)), epsilon
    )
    per_layer = {
        name: math.sqrt(max(value, 0.0))
        / max(math.sqrt(max(layer_weight.get(name, 0.0), 0.0)), epsilon)
        for name, value in layer_delta.items()
    }
    return global_ratio, per_layer


@dataclass(frozen=True, slots=True)
class CollapseThresholds:
    window: int = 8
    patience: int = 3
    gradient_norm_min: float = 1e-8
    update_ratio_min: float = 1e-10
    advantage_std_min: float = 1e-7
    entropy_min: float = 1e-3
    action_saturation_max: float = 0.98

    def __post_init__(self) -> None:
        if self.window < 2 or self.patience < 1:
            raise ValueError("collapse window must be >= 2 and patience must be positive")
        numeric = (
            self.gradient_norm_min,
            self.update_ratio_min,
            self.advantage_std_min,
            self.entropy_min,
            self.action_saturation_max,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
            raise ValueError("collapse thresholds must be finite and non-negative")
        if self.action_saturation_max > 1.0:
            raise ValueError("action_saturation_max cannot exceed 1")


@dataclass(slots=True)
class RollingCollapseDetector:
    """Detect sustained learning collapse while resisting one-batch noise."""

    thresholds: CollapseThresholds = field(default_factory=CollapseThresholds)
    _history: dict[str, deque[float]] = field(default_factory=dict, init=False, repr=False)
    _collapse_streak: int = field(default=0, init=False, repr=False)

    def _append(self, name: str, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"training health metric {name} must be finite")
        history = self._history.setdefault(name, deque(maxlen=self.thresholds.window))
        history.append(float(value))

    def _mean(self, name: str) -> float | None:
        values = self._history.get(name)
        if not values:
            return None
        return sum(values) / len(values)

    def update(self, metrics: Mapping[str, float]) -> dict[str, float]:
        aliases = {
            "global_grad_norm": ("global_grad_norm",),
            "policy_grad_norm": ("policy_grad_norm", "actor_grad_norm"),
            "value_grad_norm": ("value_grad_norm", "critic_grad_norm"),
            "parameter_update_ratio": ("parameter_update_ratio",),
            "policy_update_ratio": ("policy_update_ratio", "actor_update_ratio"),
            "value_update_ratio": ("value_update_ratio", "critic_update_ratio"),
            "policy_entropy": ("policy_entropy", "tier_entropy_mean"),
            "action_saturation": ("action_saturation",),
            "advantage_std": ("advantage_std",),
        }
        for canonical, candidates in aliases.items():
            present = [float(metrics[name]) for name in candidates if name in metrics]
            if present:
                value = (
                    max(present)
                    if canonical.endswith("grad_norm")
                    else sum(present) / len(present)
                )
                self._append(canonical, value)

        def ready_for(name: str) -> bool:
            return len(self._history.get(name, ())) >= self.thresholds.window

        ready = any(ready_for(name) for name in self._history)
        global_grad_collapsed = ready_for("global_grad_norm") and (
            self._mean("global_grad_norm") or 0.0
        ) <= self.thresholds.gradient_norm_min
        policy_grad_collapsed = ready_for("policy_grad_norm") and (
            self._mean("policy_grad_norm") or 0.0
        ) <= self.thresholds.gradient_norm_min
        value_grad_collapsed = ready_for("value_grad_norm") and (
            self._mean("value_grad_norm") or 0.0
        ) <= self.thresholds.gradient_norm_min
        grad_collapsed = (
            global_grad_collapsed or policy_grad_collapsed or value_grad_collapsed
        )
        global_update_collapsed = ready_for("parameter_update_ratio") and (
            self._mean("parameter_update_ratio") or 0.0
        ) <= self.thresholds.update_ratio_min
        policy_update_collapsed = ready_for("policy_update_ratio") and (
            self._mean("policy_update_ratio") or 0.0
        ) <= self.thresholds.update_ratio_min
        value_update_collapsed = ready_for("value_update_ratio") and (
            self._mean("value_update_ratio") or 0.0
        ) <= self.thresholds.update_ratio_min
        update_collapsed = (
            global_update_collapsed or policy_update_collapsed or value_update_collapsed
        )
        advantage_mean = self._mean("advantage_std")
        advantage_collapsed = (
            ready_for("advantage_std")
            and advantage_mean is not None
            and advantage_mean <= self.thresholds.advantage_std_min
        )
        entropy_mean = self._mean("policy_entropy")
        saturation_mean = self._mean("action_saturation")
        policy_collapsed = (
            ready_for("policy_entropy")
            and ready_for("action_saturation")
            and entropy_mean is not None
            and saturation_mean is not None
            and entropy_mean <= self.thresholds.entropy_min
            and saturation_mean >= self.thresholds.action_saturation_max
        )
        evidence = grad_collapsed or (
            update_collapsed and (advantage_collapsed or policy_collapsed)
        )
        if ready and evidence:
            self._collapse_streak += 1
        else:
            self._collapse_streak = 0
        collapsed = self._collapse_streak >= self.thresholds.patience
        return {
            "training_health_ready": float(ready),
            "gradient_collapse": float(ready and grad_collapsed),
            "update_collapse": float(ready and update_collapsed),
            "policy_collapse": float(ready and policy_collapsed),
            "policy_gradient_collapse": float(policy_grad_collapsed),
            "value_gradient_collapse": float(value_grad_collapsed),
            "policy_update_collapse": float(policy_update_collapsed),
            "value_update_collapse": float(value_update_collapsed),
            "advantage_collapse": float(ready and advantage_collapsed),
            "training_health_collapse_streak": float(self._collapse_streak),
            "training_health_collapsed": float(collapsed),
        }
