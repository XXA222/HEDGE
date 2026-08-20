from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from freqtrade.enums.hedge import PositionMode
from freqtrade.hedge.config import HedgeRuntimeConfig
from freqtrade.hedge.hprl.config import HPRLTrainingConfig
from freqtrade.hedge.hprl.trainer import OfflineTrainer, OnlineTrainer
from freqtrade.hedge.memory_lifecycle import HedgeMemoryPolicy, clear_dataprovider_caches
from freqtrade.hedge.risk.portfolio_v2 import AssetRiskExposure, aggregate_portfolio_risk
from freqtrade.hedge.runtime import HedgeProjectionSource, HedgeRuntime


def _runtime() -> HedgeRuntime:
    return HedgeRuntime(
        HedgeRuntimeConfig(
            position_mode=PositionMode.HEDGE,
            enabled=True,
            managed_pair="ETH/USDT:USDT",
            account_id="main",
            exchange_adapter="binance",
        )
    )


def _exchange_checks(persistence: bool | None) -> dict[str, bool]:
    checks = {
        "readonly_service_bound": True,
        "rest_calibrated": True,
        "user_stream_fresh": True,
        "reconciliation_converged": True,
        "risk_snapshot_valid": True,
    }
    if persistence is not None:
        checks["common.persistence_healthy"] = persistence
    return checks


def _publish(runtime: HedgeRuntime, checks: dict[str, bool]) -> None:
    now = datetime.now(UTC)
    runtime.publish(
        positions=(),
        risk=None,
        reconciliation_status="HEALTHY",
        reconciliation_at=now,
        stream_state="CONNECTED",
        stream_last_event_at=now,
        stream_reconnect_count=0,
        checks=checks,
    )


def test_exchange_missing_persistence_health_fails_closed() -> None:
    runtime = _runtime()
    with pytest.raises(ValueError, match=r"common\.persistence_healthy"):
        _publish(runtime, _exchange_checks(None))
    assert runtime.view().ready is False
    assert runtime.view().halted is True


def test_exchange_explicit_persistence_false_is_not_ready() -> None:
    runtime = _runtime()
    _publish(runtime, _exchange_checks(False))
    assert runtime.view().ready is False


def test_live_projection_is_reserved_and_rejected() -> None:
    runtime = _runtime()
    with pytest.raises(ValueError, match="LIVE projection is reserved"):
        runtime.publish(
            source=HedgeProjectionSource.LIVE,
            positions=(),
            risk=None,
            reconciliation_status="UNKNOWN",
            reconciliation_at=None,
            stream_state="UNKNOWN",
            stream_last_event_at=None,
            stream_reconnect_count=0,
            checks={},
        )


def _exposures() -> tuple[AssetRiskExposure, ...]:
    return (
        AssetRiskExposure("BTC", Decimal(10), Decimal(10), Decimal(2)),
        AssetRiskExposure("ETH", Decimal(5), Decimal(5), Decimal(1)),
    )


def test_portfolio_fingerprint_includes_correlations() -> None:
    low = aggregate_portfolio_risk(_exposures(), {("BTC", "ETH"): Decimal("0.1")})
    high = aggregate_portfolio_risk(_exposures(), {("BTC", "ETH"): Decimal("0.9")})
    assert low.correlated_variance != high.correlated_variance
    assert low.fingerprint != high.fingerprint


def test_portfolio_fingerprint_is_order_independent() -> None:
    correlations = {("ETH", "BTC"): Decimal("0.5")}
    forward = aggregate_portfolio_risk(_exposures(), correlations)
    reverse = aggregate_portfolio_risk(tuple(reversed(_exposures())), correlations)
    assert forward == reverse


def test_dataprovider_include_backtesting_capability_is_used() -> None:
    class DataProvider:
        def __init__(self) -> None:
            self.include_backtesting = None

        def clear_cache(self, *, include_backtesting: bool = False) -> None:
            self.include_backtesting = include_backtesting

    provider = DataProvider()
    clear_dataprovider_caches(provider, HedgeMemoryPolicy(clear_backtesting_cache=True))
    assert provider.include_backtesting is True


def test_dataprovider_internal_typeerror_is_not_swallowed() -> None:
    class BrokenDataProvider:
        def clear_cache(self, *, include_backtesting: bool = False) -> None:
            del include_backtesting
            raise TypeError("internal cache failure")

    with pytest.raises(TypeError, match="internal cache failure"):
        clear_dataprovider_caches(BrokenDataProvider())


class _FakeDataset:
    observation_dim = 2
    action_dim = 2
    action_unit = "policy_code"

    def __len__(self) -> int:
        return 8

    def tensors(self, device: str, *, chunk_rows: int):
        del chunk_rows
        import torch

        target = torch.device(device)
        return {
            "obs": torch.zeros((8, 2), device=target),
            "action": torch.zeros((8, 2), device=target),
            "reward": torch.zeros((8, 1), device=target),
            "next_obs": torch.zeros((8, 2), device=target),
            "done": torch.zeros((8, 1), device=target),
        }


class _FakeAgent:
    def __init__(self) -> None:
        import torch

        self.device = torch.device("cpu")
        self.calls = 0

    def update(self, batch, *, collect_metrics: bool):
        del batch, collect_metrics
        self.calls += 1
        return SimpleNamespace(values={"critic_loss": 1.0})


class _CollapsedDetector:
    def update(self, metrics):
        del metrics
        return {"training_health_collapsed": 1.0, "policy_collapse": 1.0}


@pytest.mark.parametrize(
    ("fail_mode", "expected_updates", "expected_stopped"),
    (("stop", 1, True), ("warn", 5, False)),
)
def test_offline_health_fail_mode_contract(
    fail_mode: str,
    expected_updates: int,
    expected_stopped: bool,
) -> None:
    config = HPRLTrainingConfig(
        algorithm="rebrac_v2",
        device="cpu",
        batch_size=2,
        metrics_interval=1,
        health_fail_mode=fail_mode,
    )
    agent = _FakeAgent()
    trainer = OfflineTrainer(_FakeDataset(), agent, config, device="cpu")
    trainer.training_health_detector = _CollapsedDetector()
    summary = trainer.run(5)
    assert summary.updates == expected_updates
    assert summary.early_stopped is expected_stopped
    assert agent.calls == expected_updates
    if expected_stopped:
        assert summary.stop_reason == "policy_degeneracy"


def test_pinned_fallback_requests_nonblocking_transfer() -> None:
    import torch

    class FakeBatch:
        def __init__(self) -> None:
            self.obs = SimpleNamespace(device=torch.device("cpu"))
            self.non_blocking = None

        def to(self, device, *, non_blocking: bool):
            del device
            self.non_blocking = non_blocking
            return self

    class FakeBuffer:
        pin_memory = True

        def sample(self, batch_size: int):
            del batch_size
            return FakeBatch()

    trainer = object.__new__(OnlineTrainer)
    trainer.replay_prefetcher = None
    trainer.config = SimpleNamespace(
        batch_size=1,
        replay_reuse_sample_buffers=False,
    )
    trainer.buffer = FakeBuffer()
    trainer.agent = SimpleNamespace(device=torch.device("cuda"))
    batch = trainer._training_batch()
    assert batch.non_blocking is True
