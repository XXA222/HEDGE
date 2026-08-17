"""Stable-Baselines3 callback for gradient and policy-distribution health."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from freqtrade.hedge.telemetry.training_health import (
    CollapseThresholds,
    RollingCollapseDetector,
    measure_gradients,
    measure_parameter_update,
    snapshot_parameters,
)


logger = logging.getLogger(__name__)


class SB3TrainingHealthCallback(BaseCallback):
    """Observe the update that occurs between rollout-end and the next rollout-start."""

    def __init__(
        self,
        *,
        log_prefix: str = "train/health",
        thresholds: CollapseThresholds | None = None,
        near_zero_threshold: float = 1e-12,
    ) -> None:
        super().__init__(verbose=0)
        self.log_prefix = log_prefix.rstrip("/")
        self.near_zero_threshold = float(near_zero_threshold)
        self.detector = RollingCollapseDetector(thresholds or CollapseThresholds())
        self.latest_metrics: dict[str, float] = {}
        self._before: dict[str, Any] | None = None
        self._rollout_metrics: dict[str, float] = {}

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        log_prefix: str = "train/health",
    ) -> SB3TrainingHealthCallback:
        values = config or {}
        threshold_names = {
            "window",
            "patience",
            "gradient_norm_min",
            "update_ratio_min",
            "advantage_std_min",
            "entropy_min",
            "action_saturation_max",
        }
        thresholds = CollapseThresholds(
            **{name: values[name] for name in threshold_names if name in values}
        )
        return cls(
            log_prefix=log_prefix,
            thresholds=thresholds,
            near_zero_threshold=float(values.get("near_zero_threshold", 1e-12)),
        )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        named = tuple(self.model.policy.named_parameters())
        self._before = snapshot_parameters(named)
        self._rollout_metrics = self._measure_rollout()

    def _on_rollout_start(self) -> None:
        self._emit_completed_update()

    def _on_training_end(self) -> None:
        self._emit_completed_update()

    def _measure_rollout(self) -> dict[str, float]:
        buffer = getattr(self.model, "rollout_buffer", None)
        if buffer is None:
            return {}
        metrics: dict[str, float] = {}
        log_probs = np.asarray(getattr(buffer, "log_probs", ()), dtype=np.float64).reshape(-1)
        log_probs = log_probs[np.isfinite(log_probs)]
        if log_probs.size:
            metrics["policy_entropy"] = float(max(0.0, -log_probs.mean()))
        advantages = np.asarray(getattr(buffer, "advantages", ()), dtype=np.float64).reshape(-1)
        advantages = advantages[np.isfinite(advantages)]
        if advantages.size:
            metrics["advantage_std"] = float(advantages.std())
        actions = np.asarray(getattr(buffer, "actions", ()))
        if actions.size:
            metrics["action_saturation"] = self._action_saturation(actions)
        return metrics

    def _action_saturation(self, actions: np.ndarray) -> float:
        space = getattr(self.model, "action_space", None)
        flattened = np.asarray(actions)
        nvec = getattr(space, "nvec", None)
        if nvec is not None:
            values = flattened.reshape(-1, len(nvec))
            upper = np.asarray(nvec).reshape(1, -1) - 1
            return float(np.mean((values <= 0) | (values >= upper)))
        if hasattr(space, "n"):
            values = flattened.reshape(-1).astype(np.int64, copy=False)
            if values.size == 0:
                return 0.0
            counts = np.bincount(values, minlength=int(space.n))
            return float(counts.max() / values.size)
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        if low is not None and high is not None:
            values = flattened.reshape(-1, int(np.asarray(low).size))
            low_array = np.asarray(low).reshape(1, -1)
            high_array = np.asarray(high).reshape(1, -1)
            span = np.maximum(high_array - low_array, 1e-12)
            normalized = (values - low_array) / span
            return float(np.mean((normalized <= 0.02) | (normalized >= 0.98)))
        return 0.0

    @staticmethod
    def _role_parameters(named_parameters, role: str):
        if role == "policy":
            tokens = ("policy_net", "action_net", "actor", ".pi_")
        else:
            tokens = ("value_net", "critic", ".vf_")
        return tuple(
            (name, value)
            for name, value in named_parameters
            if any(token in name for token in tokens)
        )

    def _emit_completed_update(self) -> None:
        if self._before is None:
            return
        named = tuple(self.model.policy.named_parameters())
        gradients = measure_gradients(named, near_zero_threshold=self.near_zero_threshold)
        update_ratio, per_layer_updates = measure_parameter_update(named, self._before)
        metrics = dict(self._rollout_metrics)
        metrics.update(
            {
                "global_grad_norm": gradients.global_norm,
                "near_zero_ratio": gradients.near_zero_ratio,
                "parameter_update_ratio": update_ratio,
            }
        )
        for role in ("policy", "value"):
            selected = self._role_parameters(named, role)
            if selected:
                role_gradients = measure_gradients(
                    selected, near_zero_threshold=self.near_zero_threshold
                )
                role_update_ratio, _ = measure_parameter_update(selected, self._before)
                metrics[f"{role}_grad_norm"] = role_gradients.global_norm
                metrics[f"{role}_near_zero_ratio"] = role_gradients.near_zero_ratio
                metrics[f"{role}_update_ratio"] = role_update_ratio
        metrics.update(self.detector.update(metrics))
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("non-finite SB3 training health metric")
        if metrics["training_health_collapsed"]:
            logger.warning(
                "RL training health collapse detected at timestep=%s metrics=%s",
                self.num_timesteps,
                metrics,
            )
        for name, value in metrics.items():
            self.logger.record(f"{self.log_prefix}/{name}", value)
        for name, value in gradients.per_layer_norm.items():
            self.logger.record(f"{self.log_prefix}/layers/{name}/grad_norm", value)
        for name, value in per_layer_updates.items():
            self.logger.record(f"{self.log_prefix}/layers/{name}/update_ratio", value)
        self.latest_metrics = metrics
        self._before = None
        self._rollout_metrics = {}
