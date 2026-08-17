from __future__ import annotations

import math

import pytest
import torch

from freqtrade.hedge.hprl.config import HPRLTrainingConfig
from freqtrade.hedge.hprl.registry import create_agent
from freqtrade.hedge.hprl.replay import ReplayBatch


@pytest.mark.parametrize("algorithm", ["xqc", "fast_td3", "fast_dsac", "simba_sac", "rebrac_v2"])
def test_all_hprl_algorithms_emit_complete_training_health(algorithm: str) -> None:
    config = HPRLTrainingConfig(
        algorithm=algorithm,
        device="cpu",
        replay_device="cpu",
        batch_size=8,
        replay_capacity=32,
        warmup_steps=0,
        hidden_dim=16,
        hidden_depth=1,
        metrics_interval=1,
        compile_mode="off",
    )
    torch.manual_seed(19)
    agent = create_agent(algorithm, 4, 2, config, device="cpu")
    batch = ReplayBatch(
        obs=torch.randn(8, 4),
        action=torch.rand(8, 2),
        reward=torch.randn(8, 1) * 0.1,
        next_obs=torch.randn(8, 4),
        done=torch.zeros(8, 1),
    )

    metrics = agent.update(batch, collect_metrics=True).values
    required = {
        "actor_grad_norm",
        "critic_grad_norm",
        "actor_update_ratio",
        "critic_update_ratio",
        "policy_entropy",
        "action_saturation",
        "advantage_std",
    }
    assert required <= metrics.keys()
    assert all(math.isfinite(metrics[name]) for name in required)
    assert metrics["critic_update_ratio"] > 0.0
    assert 0.0 <= metrics["policy_entropy"] <= 1.0
    assert 0.0 <= metrics["action_saturation"] <= 1.0
    assert metrics["advantage_std"] >= 0.0
