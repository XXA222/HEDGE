"""Adaptive CPU runtime policy for the independent Hedge risk-level learner.

The risk-level policy keeps a single chronological environment by default because
its account state is causal.  CPU parallelism is therefore applied to numerical
policy work at coarse boundaries while independent Hyperopt/research tasks remain
handled by the project-level adaptive process scheduler.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from freqtrade.hedge.performance.resource_governor import AdaptiveResourceGovernor


@dataclass(frozen=True, slots=True)
class RiskRLAdaptiveCpuConfig:
    enabled: bool = True
    max_torch_threads: int = 16
    min_torch_threads: int = 1
    refresh_train_steps: int = 2048

    def __post_init__(self) -> None:
        if self.min_torch_threads < 1:
            raise ValueError("min_torch_threads must be positive")
        if self.max_torch_threads < self.min_torch_threads:
            raise ValueError("max_torch_threads must be >= min_torch_threads")
        if self.refresh_train_steps < 1:
            raise ValueError("refresh_train_steps must be positive")

    @classmethod
    def from_freqtrade_config(cls, config: Mapping[str, Any]) -> RiskRLAdaptiveCpuConfig:
        freqai = config.get("freqai", {}) if isinstance(config, Mapping) else {}
        hedge = freqai.get("hedge_rl_config", {}) if isinstance(freqai, Mapping) else {}
        raw = hedge.get("adaptive_cpu", {}) if isinstance(hedge, Mapping) else {}
        if not isinstance(raw, Mapping):
            raw = {}
        valid = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in raw.items() if key in valid})


class RiskRLAdaptiveCpuController:
    """Translate host-aware resource snapshots into a safe PyTorch thread budget."""

    def __init__(
        self,
        config: RiskRLAdaptiveCpuConfig,
        governor: AdaptiveResourceGovernor | None = None,
    ) -> None:
        self.config = config
        self.governor = governor or AdaptiveResourceGovernor()
        self.last_threads = 1
        self.last_source = "uninitialized"
        self.last_system_cpu_percent = 0.0

    @classmethod
    def from_freqtrade_config(cls, config: Mapping[str, Any]) -> RiskRLAdaptiveCpuController:
        return cls(RiskRLAdaptiveCpuConfig.from_freqtrade_config(config))

    def recommended_threads(self) -> int:
        if not self.config.enabled:
            return self.config.min_torch_threads
        snapshot = self.governor.snapshot(sample_seconds=0.0)
        suggested = self.governor.numeric_threads(
            concurrent_python_workers=1,
            snapshot=snapshot,
        )
        threads = max(
            self.config.min_torch_threads,
            min(
                self.config.max_torch_threads,
                snapshot.physical_cpus,
                suggested,
            ),
        )
        self.last_threads = int(threads)
        self.last_source = snapshot.source
        self.last_system_cpu_percent = float(snapshot.system_cpu_percent)
        return self.last_threads

    def apply_torch(self, torch_module: Any) -> int:
        threads = self.recommended_threads()
        try:
            current = int(torch_module.get_num_threads())
        except (AttributeError, TypeError, ValueError):
            current = -1
        if current != threads:
            torch_module.set_num_threads(threads)
        return threads

    def telemetry(self) -> dict[str, int | float | str | bool]:
        return {
            "enabled": self.config.enabled,
            "torch_threads": self.last_threads,
            "resource_source": self.last_source,
            "system_cpu_percent": self.last_system_cpu_percent,
            "max_torch_threads": self.config.max_torch_threads,
            "refresh_train_steps": self.config.refresh_train_steps,
        }
