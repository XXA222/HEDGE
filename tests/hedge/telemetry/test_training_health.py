from __future__ import annotations

import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from freqtrade.freqai.RL.training_health import SB3TrainingHealthCallback
from freqtrade.freqai.torch.PyTorchDataConvertor import DefaultPyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchModelTrainer
from freqtrade.hedge.telemetry.training_health import (
    CollapseThresholds,
    RollingCollapseDetector,
    measure_gradients,
    measure_parameter_update,
    snapshot_parameters,
)


def test_gradient_and_actual_parameter_update_telemetry() -> None:
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = snapshot_parameters(tuple(model.named_parameters()))

    prediction = model(torch.ones((8, 3)))
    loss = prediction.square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    gradients = measure_gradients(tuple(model.named_parameters()))
    assert math.isfinite(gradients.global_norm) and gradients.global_norm > 0.0
    assert gradients.weight_norm > 0.0
    assert 0.0 <= gradients.near_zero_ratio <= 1.0
    assert gradients.parameter_count == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert set(gradients.per_layer_norm) == {"0", "2"}

    optimizer.step()
    update_ratio, per_layer = measure_parameter_update(
        tuple(model.named_parameters()), before
    )
    assert math.isfinite(update_ratio) and update_ratio > 0.0
    assert set(per_layer) == {"0", "2"}
    assert all(value > 0.0 for value in per_layer.values())


def test_non_finite_gradient_fails_closed() -> None:
    parameter = nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([float("nan"), 0.0])
    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        measure_gradients((("weight", parameter),))


def test_rolling_detector_requires_sustained_collapse() -> None:
    detector = RollingCollapseDetector(
        CollapseThresholds(window=2, patience=2)
    )
    collapsed = {
        "actor_grad_norm": 0.0,
        "critic_grad_norm": 0.0,
        "actor_update_ratio": 0.0,
        "critic_update_ratio": 0.0,
        "policy_entropy": 0.0,
        "action_saturation": 1.0,
        "advantage_std": 0.0,
    }

    assert detector.update(collapsed)["training_health_ready"] == 0.0
    second = detector.update(collapsed)
    assert second["training_health_ready"] == 1.0
    assert second["training_health_collapsed"] == 0.0
    third = detector.update(collapsed)
    assert third["training_health_collapsed"] == 1.0
    assert third["gradient_collapse"] == 1.0
    assert third["policy_collapse"] == 1.0


def test_rolling_detector_recovers_after_healthy_update() -> None:
    detector = RollingCollapseDetector(CollapseThresholds(window=2, patience=1))
    collapsed = {
        "global_grad_norm": 0.0,
        "parameter_update_ratio": 0.0,
        "advantage_std": 0.0,
    }
    detector.update(collapsed)
    assert detector.update(collapsed)["training_health_collapsed"] == 1.0

    healthy = {
        "global_grad_norm": 1.0,
        "parameter_update_ratio": 1e-3,
        "advantage_std": 0.5,
    }
    detector.update(healthy)
    recovered = detector.update(healthy)
    assert recovered["training_health_collapsed"] == 0.0
    assert recovered["training_health_collapse_streak"] == 0.0


def test_detector_identifies_policy_branch_collapse_while_value_branch_learns() -> None:
    detector = RollingCollapseDetector(CollapseThresholds(window=2, patience=1))
    metrics = {
        "actor_grad_norm": 0.0,
        "critic_grad_norm": 1.0,
        "actor_update_ratio": 0.0,
        "critic_update_ratio": 1e-3,
        "policy_entropy": 0.5,
        "action_saturation": 0.1,
        "advantage_std": 0.25,
    }
    detector.update(metrics)
    result = detector.update(metrics)
    assert result["policy_gradient_collapse"] == 1.0
    assert result["value_gradient_collapse"] == 0.0
    assert result["policy_update_collapse"] == 1.0
    assert result["value_update_collapse"] == 0.0
    assert result["training_health_collapsed"] == 1.0


def test_sb3_callback_emits_risk_policy_and_value_health() -> None:
    class Policy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_net = nn.Linear(3, 4)
            self.value_net = nn.Linear(3, 1)

    class MetricLogger:
        def __init__(self) -> None:
            self.values: dict[str, float] = {}

        def record(self, name: str, value: float) -> None:
            self.values[name] = float(value)

    policy = Policy()
    metric_logger = MetricLogger()
    fake_model = SimpleNamespace(
        policy=policy,
        action_space=gym.spaces.MultiDiscrete([5, 5]),
        rollout_buffer=SimpleNamespace(
            log_probs=np.array([-0.5, -0.8, -0.2]),
            advantages=np.array([1.0, -1.0, 0.25]),
            actions=np.array([[0, 4], [2, 3], [1, 2]]),
        ),
        logger=metric_logger,
    )
    callback = SB3TrainingHealthCallback(log_prefix="train/risk_level_health")
    callback.model = fake_model
    callback._on_rollout_end()

    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    features = torch.ones((4, 3))
    loss = policy.policy_net(features).square().mean() + policy.value_net(features).square().mean()
    loss.backward()
    optimizer.step()
    callback._on_training_end()

    assert callback.latest_metrics["global_grad_norm"] > 0.0
    assert callback.latest_metrics["policy_grad_norm"] > 0.0
    assert callback.latest_metrics["value_grad_norm"] > 0.0
    assert callback.latest_metrics["policy_update_ratio"] > 0.0
    assert callback.latest_metrics["value_update_ratio"] > 0.0
    assert 0.0 <= callback.latest_metrics["policy_near_zero_ratio"] <= 1.0
    assert 0.0 <= callback.latest_metrics["value_near_zero_ratio"] <= 1.0
    assert callback.latest_metrics["parameter_update_ratio"] > 0.0
    assert callback.latest_metrics["policy_entropy"] > 0.0
    assert callback.latest_metrics["advantage_std"] > 0.0
    assert callback.latest_metrics["action_saturation"] == pytest.approx(2 / 6)
    assert "train/risk_level_health/global_grad_norm" in metric_logger.values


def test_pytorch_trainer_records_global_and_per_layer_health() -> None:
    class ScalarLogger:
        def __init__(self) -> None:
            self.names: set[str] = set()

        def log_scalar(self, name: str, value: float, step: int) -> None:
            assert math.isfinite(float(value))
            assert step >= 0
            self.names.add(name)

    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 1))
    logger = ScalarLogger()
    trainer = PyTorchModelTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
        criterion=nn.MSELoss(),
        device="cpu",
        data_convertor=DefaultPyTorchDataConvertor(target_tensor_type=torch.float),
        tb_logger=logger,
        n_epochs=1,
        batch_size=4,
        training_health={"interval": 1, "window": 2},
    )
    features = pd.DataFrame(np.arange(24, dtype=np.float32).reshape(8, 3) / 24.0)
    labels = pd.DataFrame(np.linspace(0.0, 1.0, 8, dtype=np.float32))
    trainer.fit({"train_features": features, "train_labels": labels}, ["train"])

    assert trainer.training_health_metrics["global_grad_norm"] > 0.0
    assert trainer.training_health_metrics["parameter_update_ratio"] > 0.0
    assert "training_health/global_grad_norm" in logger.names
    assert "training_health/near_zero_ratio" in logger.names
    assert any(name.endswith("/grad_norm") for name in logger.names)
    assert any(name.endswith("/update_ratio") for name in logger.names)
