"""Shared helpers for HPRL off-policy algorithms."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping

from ..device import require_torch


torch = require_torch()


@dataclass(frozen=True, slots=True)
class UpdateMetrics:
    values: Mapping[str, float]


def make_metrics(collect: bool, values: Mapping[str, object]) -> UpdateMetrics:
    """Materialize scalar metrics only when requested.

    CUDA tensors are stacked before the host copy so one sampled metrics event causes one device
    synchronization instead of one synchronization per metric.
    """
    if not collect:
        return UpdateMetrics({})
    names = tuple(values)
    device = next(
        (value.device for value in values.values() if torch.is_tensor(value)),
        torch.device("cpu"),
    )
    tensors = [
        (
            value.detach().float().reshape(())
            if torch.is_tensor(value)
            else torch.tensor(float(value))
        ).to(device=device)
        for value in values.values()
    ]
    if not tensors:
        return UpdateMetrics({})
    materialized = torch.stack(tensors).cpu().tolist()
    return UpdateMetrics(dict(zip(names, (float(value) for value in materialized), strict=True)))



class PolyakUpdatePlan:
    """Pre-bound Polyak update plan that removes per-update module introspection.

    Parameter and buffer identities are stable for the lifetime of an HPRL agent.  Binding
    them once avoids tuple construction, named-buffer dictionaries, key comparisons and
    repeated dtype branching on every optimizer update.
    """

    __slots__ = ("target_params", "source_params", "floating_target",
                 "floating_source", "copy_target", "copy_source", "foreach")

    def __init__(self, target, source, *, foreach: bool = True) -> None:
        self.target_params = tuple(target.parameters())
        self.source_params = tuple(source.parameters())
        if len(self.target_params) != len(self.source_params):
            raise ValueError("soft-update module parameters do not match")
        target_buffers = dict(target.named_buffers())
        source_buffers = dict(source.named_buffers())
        if target_buffers.keys() != source_buffers.keys():
            raise ValueError("soft-update module buffers do not match")
        floating_target = []
        floating_source = []
        copy_target = []
        copy_source = []
        for name, target_buffer in target_buffers.items():
            source_buffer = source_buffers[name]
            if target_buffer.dtype.is_floating_point:
                floating_target.append(target_buffer)
                floating_source.append(source_buffer)
            else:
                copy_target.append(target_buffer)
                copy_source.append(source_buffer)
        self.floating_target = tuple(floating_target)
        self.floating_source = tuple(floating_source)
        self.copy_target = tuple(copy_target)
        self.copy_source = tuple(copy_source)
        self.foreach = bool(foreach)

    def step(self, tau: float) -> None:
        if not 0.0 < float(tau) <= 1.0:
            raise ValueError("soft-update tau must be in (0, 1]")
        with torch.no_grad():
            if self.target_params:
                if self.foreach:
                    try:
                        torch._foreach_lerp_(self.target_params, self.source_params, tau)
                    except (RuntimeError, TypeError):
                        for target_param, source_param in zip(
                            self.target_params, self.source_params, strict=True
                        ):
                            target_param.lerp_(source_param, tau)
                else:
                    for target_param, source_param in zip(
                        self.target_params, self.source_params, strict=True
                    ):
                        target_param.lerp_(source_param, tau)
            if self.floating_target:
                if self.foreach:
                    try:
                        torch._foreach_lerp_(self.floating_target, self.floating_source, tau)
                    except (RuntimeError, TypeError):
                        for target_buffer, source_buffer in zip(
                            self.floating_target, self.floating_source, strict=True
                        ):
                            target_buffer.lerp_(source_buffer, tau)
                else:
                    for target_buffer, source_buffer in zip(
                        self.floating_target, self.floating_source, strict=True
                    ):
                        target_buffer.lerp_(source_buffer, tau)
            if self.copy_target:
                if self.foreach:
                    try:
                        torch._foreach_copy_(self.copy_target, self.copy_source)
                    except (RuntimeError, TypeError):
                        for target_buffer, source_buffer in zip(
                            self.copy_target, self.copy_source, strict=True
                        ):
                            target_buffer.copy_(source_buffer)
                else:
                    for target_buffer, source_buffer in zip(
                        self.copy_target, self.copy_source, strict=True
                    ):
                        target_buffer.copy_(source_buffer)




class OptimizerStepPlan:
    """Pre-bound backward/clip/optimizer execution plan for one parameter group."""

    __slots__ = (
        "precision",
        "optimizer",
        "parameters",
        "max_norm",
        "last_update_ratio",
    )

    def __init__(self, precision, optimizer, parameters, max_norm: float) -> None:
        self.precision = precision
        self.optimizer = optimizer
        self.parameters = parameters if isinstance(parameters, tuple) else tuple(parameters)
        self.max_norm = float(max_norm)
        if self.max_norm <= 0.0:
            raise ValueError("optimizer step max_norm must be positive")
        device = self.parameters[0].device if self.parameters else torch.device("cpu")
        self.last_update_ratio = torch.zeros((), device=device)

    def backward_and_clip(self, loss):
        return self.precision.backward_and_clip(
            loss, self.optimizer, self.parameters, self.max_norm
        )

    def optimizer_step(self) -> None:
        self.precision.optimizer_step(self.optimizer)

    def step(self, loss, *, measure_update: bool = False):
        before = snapshot_parameter_values(self.parameters) if measure_update else ()
        norm = self.backward_and_clip(loss)
        self.optimizer_step()
        if measure_update:
            self.last_update_ratio = parameter_update_ratio(self.parameters, before)
        return norm


@torch.no_grad()
def snapshot_parameter_values(parameters):
    """Device-resident snapshots used only on sampled telemetry updates."""

    return tuple(parameter.detach().clone() for parameter in parameters)


@torch.no_grad()
def parameter_update_ratio(parameters, before):
    """Actual global parameter delta divided by the pre-update weight norm."""

    if len(parameters) != len(before):
        raise ValueError("parameter update snapshot does not match parameter group")
    if not parameters:
        return torch.zeros(())
    delta_squared = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    weight_squared = torch.zeros_like(delta_squared)
    for parameter, old in zip(parameters, before, strict=True):
        delta_squared.add_((parameter.detach().float() - old.float()).square().sum())
        weight_squared.add_(old.float().square().sum())
    return delta_squared.sqrt() / weight_squared.sqrt().clamp_min(1e-12)


def _critic_minimum(critic, obs, action):
    first, second = critic(obs, action)
    if isinstance(first, tuple):
        first = first[0]
    if isinstance(second, tuple):
        second = second[0]
    return torch.minimum(first, second)


@torch.no_grad()
def off_policy_health_metrics(agent, batch) -> dict[str, object]:
    """Current-policy collapse signals and one-step TD-advantage dispersion.

    Off-policy agents do not retain PPO-style advantages.  Their corresponding learning signal is
    the one-step TD advantage ``target - Q(s, a)``.  Entropy is measured empirically from current
    policy actions and normalized to [0, 1] per action dimension.
    """

    critic_training = agent.critic.training
    target_training = agent.critic_target.training
    agent.critic.eval()
    agent.critic_target.eval()
    try:
        # Deterministic actions make telemetry observational: sampling here must not advance the
        # RNG or alter the subsequent training trajectory merely because metrics are enabled.
        policy_action = agent.act(batch.obs, deterministic=True).float()
        action_2d = policy_action.reshape(policy_action.shape[0], -1).clamp(0.0, 1.0)
        flat_saturation = (action_2d <= 0.02).float().mean()
        heavy_saturation = (action_2d >= 0.98).float().mean()
        saturation = flat_saturation + heavy_saturation
        policy_action_mean = action_2d.mean()
        policy_action_std = action_2d.std(unbiased=False)
        bins = 16
        encoded = torch.clamp((action_2d * bins).long(), 0, bins - 1)
        entropies = []
        for dimension in range(encoded.shape[1]):
            counts = torch.bincount(encoded[:, dimension], minlength=bins).float()
            probabilities = counts / counts.sum().clamp_min(1.0)
            nonzero = probabilities > 0
            entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
            entropies.append(entropy / math.log(float(bins)))
        policy_entropy = (
            torch.stack(entropies).mean()
            if entropies
            else torch.zeros((), device=policy_action.device)
        )

        current_q = _critic_minimum(agent.critic, batch.obs, batch.action)
        next_action = agent.act(batch.next_obs, deterministic=True).float()
        next_q = _critic_minimum(agent.critic_target, batch.next_obs, next_action)
        target = batch.reward + agent.config.gamma * (1.0 - batch.done) * next_q
        advantage_std = (target - current_q).float().std(unbiased=False)
    finally:
        agent.critic.train(critic_training)
        agent.critic_target.train(target_training)
    return {
        "policy_entropy": policy_entropy,
        "action_saturation": saturation,
        "policy_action_mean": policy_action_mean,
        "policy_action_std": policy_action_std,
        "flat_saturation": flat_saturation,
        "heavy_saturation": heavy_saturation,
        "advantage_std": advantage_std,
    }


class FrozenModulePlan:
    """Reusable freeze context with a pre-bound parameter tuple."""

    __slots__ = ("module", "params", "eval_mode")

    def __init__(self, module, *, eval_mode: bool = False) -> None:
        self.module = module
        self.params = tuple(module.parameters())
        self.eval_mode = bool(eval_mode)

    @contextmanager
    def frozen(self):
        requires_grad = tuple(param.requires_grad for param in self.params)
        was_training = self.module.training
        try:
            for param in self.params:
                param.requires_grad_(False)
            if self.eval_mode:
                self.module.eval()
            yield self.module
        finally:
            for param, enabled in zip(self.params, requires_grad, strict=True):
                param.requires_grad_(enabled)
            self.module.train(was_training)

def soft_update(target, source, tau: float, *, foreach: bool = True) -> None:
    """Polyak-update parameters and stateful buffers such as BatchNorm statistics."""
    if not 0.0 < float(tau) <= 1.0:
        raise ValueError("soft-update tau must be in (0, 1]")
    with torch.no_grad():
        target_params = tuple(target.parameters())
        source_params = tuple(source.parameters())
        if len(target_params) != len(source_params):
            raise ValueError("soft-update module parameters do not match")
        if target_params:
            if foreach:
                try:
                    torch._foreach_lerp_(target_params, source_params, tau)
                except (RuntimeError, TypeError):
                    for target_param, source_param in zip(
                        target_params, source_params, strict=True
                    ):
                        target_param.lerp_(source_param, tau)
            else:
                for target_param, source_param in zip(target_params, source_params, strict=True):
                    target_param.lerp_(source_param, tau)
        target_buffers = dict(target.named_buffers())
        source_buffers = dict(source.named_buffers())
        if target_buffers.keys() != source_buffers.keys():
            raise ValueError("soft-update module buffers do not match")
        floating_target = []
        floating_source = []
        for name, target_buffer in target_buffers.items():
            source_buffer = source_buffers[name]
            if target_buffer.dtype.is_floating_point:
                floating_target.append(target_buffer)
                floating_source.append(source_buffer)
            else:
                target_buffer.copy_(source_buffer)
        if floating_target:
            if foreach:
                try:
                    torch._foreach_lerp_(floating_target, floating_source, tau)
                except (RuntimeError, TypeError):
                    for target_buffer, source_buffer in zip(
                        floating_target, floating_source, strict=True
                    ):
                        target_buffer.lerp_(source_buffer, tau)
            else:
                for target_buffer, source_buffer in zip(
                    floating_target, floating_source, strict=True
                ):
                    target_buffer.lerp_(source_buffer, tau)


def hard_update(target, source) -> None:
    target.load_state_dict(source.state_dict())


def min_q(critic, obs, action):
    q1, q2 = critic(obs, action)
    return torch.minimum(q1, q2)


@contextmanager
def frozen_module(module, *, eval_mode: bool = False):
    """Freeze module parameters while retaining gradients with respect to its inputs."""
    params = tuple(module.parameters())
    requires_grad = tuple(param.requires_grad for param in params)
    was_training = module.training
    try:
        for param in params:
            param.requires_grad_(False)
        if eval_mode:
            module.eval()
        yield module
    finally:
        for param, enabled in zip(params, requires_grad, strict=True):
            param.requires_grad_(enabled)
        module.train(was_training)
