# ruff: noqa: E402, I001
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
from pandas import DataFrame, Timestamp

from freqtrade.exchange import timeframe_to_seconds
from freqtrade.strategy import IStrategy

SUITE_ROOT = Path(__file__).resolve().parent.parent
_suite_root_text = str(SUITE_ROOT)
if _suite_root_text in sys.path:
    sys.path.remove(_suite_root_text)
sys.path.insert(0, _suite_root_text)

from artifact_contract import (
    ARTIFACT_MANIFEST_SCHEMA,
    MODEL_METADATA_SCHEMA,
    canonical_sha256,
    runtime_contract_payload,
    source_contract_payload,
    verify_manifest_files,
)
from features import FEATURE_VERSION, apply_scaler, inference_arrays_mtf
from suite_specs import (
    ACTION_KWARGS,
    COST_KWARGS,
    MODELS,
    MTF_ALIGNMENT_CONTRACT,
    PERIODS_PER_YEAR,
    SOURCE_WARMUP_CANDLES,
    informative_timeframes_for,
    input_timeframes_for,
    reward_kwargs,
)


class HPRLNativeHedgeStrategyBase(IStrategy):
    """Native HPRL policy provider for the canonical HEDGE/Freqtrade Strategy contract.

    The class intentionally does not implement TD3/SAC/XQC/ReBRAC and does not own
    authoritative fills or portfolio PnL.  It loads one repository-native HPRL checkpoint,
    consumes the base timeframe plus Freqtrade informative higher-timeframe OHLCV using strict
    closed-candle alignment, advances the native
    policy state in causal order, validates the emitted tier target with
    ``HprlHedgeAdapter``, and writes only the canonical ``hedge_*`` directive columns.

    The native HPRL environment used here is a policy-state shadow because the current
    historical Freqtrade lifecycle analyzes the Strategy dataframe before HEDGE event replay.
    HEDGE remains the only authority for fills, fees, funding, wallet state and final PnL.
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1m"
    startup_candle_count = SOURCE_WARMUP_CANDLES
    process_only_new_candles = True
    minimal_roi = {"0": 10.0}
    stoploss = -0.99
    trailing_stop = False
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    MODEL_KEY = ""

    @classmethod
    def version(cls) -> str:
        return f"hprl-freqtrade-mtf-v3:{cls.MODEL_KEY}"


    def informative_pairs(self):
        """Register every higher timeframe with Freqtrade/DataProvider.

        Freqtrade requires informative timeframes to be equal to or higher than the Strategy
        timeframe.  The ordered suite contract therefore consumes the base timeframe plus all
        configured higher timeframes, preserving the original 5 x 6 comparison matrix.
        """
        return [("ETH/USDT:USDT", tf) for tf in informative_timeframes_for(self.timeframe)]

    def _artifact_dir(self) -> Path:
        user_data = Path(str(self.config.get("user_data_dir", "user_data"))).resolve()
        return user_data / "hprl_freqtrade_models" / self.MODEL_KEY / self.timeframe

    def _native_configs(self, api, device: str):
        spec = MODELS[self.MODEL_KEY]
        action = api["HPRLActionConfig"](**ACTION_KWARGS)
        costs = api["HPRLCostConfig"](**COST_KWARGS)
        reward = api["HPRLRewardConfig"](**reward_kwargs(spec))
        env = api["HPRLEnvironmentConfig"](
            initial_equity=1000.0,
            parallel_envs=1,
            annualization_periods=PERIODS_PER_YEAR[self.timeframe],
            cvar_alpha=0.05,
            terminate_equity_ratio=0.20,
            runtime_checks=False,
            info_mode="full",
            action=action,
            costs=costs,
            reward=reward,
        )
        train = api["HPRLTrainingConfig"](
            algorithm=spec.algorithm,
            seed=42,
            device=device,
            replay_device="same",
            batch_size=spec.batch_size,
            replay_capacity=spec.replay_capacity,
            warmup_steps=spec.warmup_transitions,
            gradient_steps=1,
            gamma=spec.gamma,
            tau=spec.tau,
            learning_rate=spec.learning_rate,
            hidden_dim=spec.hidden_dim,
            hidden_depth=spec.hidden_depth,
            gradient_clip_norm=spec.gradient_clip_norm,
            mixed_precision=False,
            compile_mode="off",
            expected_updates=0,
            metrics_interval=100,
            tier_entropy_target_fraction=spec.tier_entropy_target_fraction,
            runtime_checks=False,
        )
        memory = api["HPRLMemoryConfig"](
            dataset_mode="auto",
            dataset_window_steps=16_384,
            dataset_gpu_fraction=0.20,
            replay_gpu_fraction=0.30,
            release_offline_source_after_tensorize=True,
        )
        return action, env, train, memory

    @staticmethod
    def _native_api():
        from freqtrade.hedge.hprl.action_space import configure_agent_action_levels
        from freqtrade.hedge.hprl.checkpoint import load_checkpoint
        from freqtrade.hedge.hprl.config import (
            HPRLActionConfig,
            HPRLCostConfig,
            HPRLEnvironmentConfig,
            HPRLMemoryConfig,
            HPRLRewardConfig,
            HPRLTrainingConfig,
        )
        from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
        from freqtrade.hedge.hprl.data import TensorMarketDataset
        from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
        from freqtrade.hedge.hprl.registry import create_agent
        from freqtrade.hedge.production.hprl_hedge_adapter import (
            HprlHedgeAdapter,
            HprlHedgeAdapterPolicy,
            HprlTargetUnit,
        )

        return {
            "configure_agent_action_levels": configure_agent_action_levels,
            "load_checkpoint": load_checkpoint,
            "HPRLActionConfig": HPRLActionConfig,
            "HPRLCostConfig": HPRLCostConfig,
            "HPRLEnvironmentConfig": HPRLEnvironmentConfig,
            "HPRLMemoryConfig": HPRLMemoryConfig,
            "HPRLRewardConfig": HPRLRewardConfig,
            "HPRLTrainingConfig": HPRLTrainingConfig,
            "PlannedExecutionIntent": PlannedExecutionIntent,
            "TensorMarketDataset": TensorMarketDataset,
            "VectorizedHedgeEnv": VectorizedHedgeEnv,
            "create_agent": create_agent,
            "HprlHedgeAdapter": HprlHedgeAdapter,
            "HprlHedgeAdapterPolicy": HprlHedgeAdapterPolicy,
            "HprlTargetUnit": HprlTargetUnit,
        }

    def _runtime_contract(self, feature_names: tuple[str, ...]) -> dict[str, object]:
        import freqtrade

        spec = MODELS[self.MODEL_KEY]
        repo_root = Path(freqtrade.__file__).resolve().parent.parent
        source_contract = source_contract_payload(suite_root=SUITE_ROOT, repo_root=repo_root)
        inputs = input_timeframes_for(self.timeframe)
        return runtime_contract_payload(
            model_key=spec.key,
            algorithm=spec.algorithm,
            strategy_class=spec.strategy_class,
            base_timeframe=self.timeframe,
            input_timeframes=inputs,
            feature_version=FEATURE_VERSION,
            feature_names=feature_names,
            alignment_contract=MTF_ALIGNMENT_CONTRACT,
            action_config=ACTION_KWARGS,
            cost_config=COST_KWARGS,
            reward_config=reward_kwargs(spec),
            model_spec=asdict(spec),
            source_contract=source_contract,
        )

    @staticmethod
    def _utc_close_timestamp(value: object, candle_seconds: int):
        timestamp = Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.to_pydatetime() + timedelta(seconds=candle_seconds)

    @staticmethod
    def _exact_notional_from_tier(action_cfg, decoded, side_index: int) -> float:
        # Never derive production ratios from float32 target_margin.  The codec already
        # exposes the executed integer tier index, so map that index back to the configured
        # decimal level and only then apply leverage.  This prevents a legal 12% -> 20%
        # transition from becoming 0.08000000... and being rejected by Decimal risk checks.
        index = int(decoded.executed_level_index[0, 0, side_index].detach().cpu().item())
        levels = tuple(action_cfg.position_levels)
        if not 0 <= index < len(levels):
            raise RuntimeError(f"native HPRL codec returned invalid tier index: {index}")
        margin = Decimal(str(levels[index]))
        leverage = Decimal(str(action_cfg.leverage))
        return float(margin * leverage)

    def _load_artifact_contract(
        self,
        *,
        artifact: Path,
        feature_names: tuple[str, ...],
    ) -> tuple[dict[str, object], dict[str, object], str, str]:
        manifest_path = artifact / "artifact_manifest.json"
        metadata_path = artifact / "metadata.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != ARTIFACT_MANIFEST_SCHEMA:
            raise RuntimeError("HPRL artifact manifest schema is not supported")
        if metadata.get("schema") != MODEL_METADATA_SCHEMA:
            raise RuntimeError("HPRL model metadata schema is not supported")

        spec = MODELS[self.MODEL_KEY]
        identity = (
            metadata.get("model"),
            metadata.get("algorithm"),
            metadata.get("strategy_class"),
            metadata.get("timeframe"),
            tuple(metadata.get("input_timeframes", ())),
            metadata.get("pair"),
        )
        expected_identity = (
            self.MODEL_KEY,
            spec.algorithm,
            spec.strategy_class,
            self.timeframe,
            input_timeframes_for(self.timeframe),
            "ETH/USDT:USDT",
        )
        if identity != expected_identity:
            raise RuntimeError(
                f"HPRL artifact identity mismatch: expected={expected_identity}, actual={identity}"
            )
        if metadata.get("feature_version") != FEATURE_VERSION:
            raise RuntimeError(
                "HPRL feature contract mismatch: "
                f"{metadata.get('feature_version')} != {FEATURE_VERSION}"
            )
        if tuple(metadata.get("feature_names", ())) != feature_names:
            raise RuntimeError("HPRL metadata feature names do not match Strategy features")

        runtime_contract_sha = canonical_sha256(self._runtime_contract(feature_names))
        artifact_contract_sha = str(metadata.get("artifact_contract_sha256") or "")
        if metadata.get("runtime_contract_sha256") != runtime_contract_sha:
            raise RuntimeError("HPRL runtime contract changed; checkpoint retraining is required")
        if manifest.get("runtime_contract_sha256") != runtime_contract_sha:
            raise RuntimeError(
                "HPRL artifact manifest runtime contract does not match current source"
            )
        if (
            not artifact_contract_sha
            or manifest.get("artifact_contract_sha256") != artifact_contract_sha
        ):
            raise RuntimeError("HPRL artifact contract hash is missing or inconsistent")

        verify_manifest_files(artifact, manifest)
        return metadata, manifest, runtime_contract_sha, artifact_contract_sha

    def _generate_hprl_columns(  # noqa: C901
        self, dataframe: DataFrame, pair: str, frames: dict[str, DataFrame]
    ) -> DataFrame:
        if self.MODEL_KEY not in MODELS:
            raise RuntimeError(f"Unknown HPRL Strategy model key: {self.MODEL_KEY!r}")
        if pair != "ETH/USDT:USDT":
            raise RuntimeError(f"HPRL ETH Strategy received unsupported pair: {pair!r}")
        if int(self.startup_candle_count) < SOURCE_WARMUP_CANDLES:
            raise RuntimeError(
                "HPRL MTF startup_candle_count is below the source warmup contract: "
                f"{self.startup_candle_count} < {SOURCE_WARMUP_CANDLES}"
            )

        artifact = self._artifact_dir()
        checkpoint = artifact / "checkpoint.pt"
        checkpoint_sidecar = artifact / "checkpoint.pt.json"
        scaler_path = artifact / "scaler.npz"
        metadata_path = artifact / "metadata.json"
        manifest_path = artifact / "artifact_manifest.json"
        required = (checkpoint, checkpoint_sidecar, scaler_path, metadata_path, manifest_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Native HPRL model artifact set is incomplete; prepare/retrain the model first: "
                + ", ".join(missing)
            )

        candle_seconds = timeframe_to_seconds(self.timeframe)
        input_timeframes = input_timeframes_for(self.timeframe)
        (
            x_raw,
            y,
            available,
            timestamps,
            feature_names,
            valid_positions,
            alignment_diagnostics,
        ) = inference_arrays_mtf(
            frames,
            base_timeframe=self.timeframe,
            input_timeframes=input_timeframes,
        )
        if len(x_raw) < 2:
            raise RuntimeError(
                "Not enough causal HPRL feature rows in Freqtrade analyzed dataframe"
            )

        with np.load(scaler_path, allow_pickle=False) as scale:
            mean = scale["mean"].astype(np.float32, copy=True)
            std = scale["std"].astype(np.float32, copy=True)
            saved_names = tuple(str(value) for value in scale["feature_names"].tolist())
            scaler_runtime_sha = str(scale["runtime_contract_sha256"].item())
            scaler_artifact_sha = str(scale["artifact_contract_sha256"].item())
        if saved_names != feature_names:
            raise RuntimeError("HPRL scaler and Strategy feature names are not identical")

        metadata, _manifest, runtime_contract_sha, artifact_contract_sha = (
            self._load_artifact_contract(artifact=artifact, feature_names=feature_names)
        )
        if (
            scaler_runtime_sha != runtime_contract_sha
            or scaler_artifact_sha != artifact_contract_sha
        ):
            raise RuntimeError(
                "HPRL scaler contract hashes do not match the committed artifact set"
            )
        x = apply_scaler(x_raw, mean, std)

        api = self._native_api()
        import torch

        zeros = np.zeros(len(x), dtype=np.float32)
        dataset = api["TensorMarketDataset"](
            features=torch.as_tensor(x[:, None, :], dtype=torch.float32),
            forward_returns=torch.as_tensor(y[:, None], dtype=torch.float32),
            funding_rates=torch.as_tensor(zeros[:, None], dtype=torch.float32),
            available_notional=torch.as_tensor(available[:, None], dtype=torch.float32),
            symbols=("ETHUSDT",),
        ).validate()

        device = os.environ.get("HPRL_STRATEGY_DEVICE", "cpu")
        action_cfg, env_cfg, train_cfg, memory_cfg = self._native_configs(api, device)
        env = api["VectorizedHedgeEnv"](
            dataset, env_cfg, device=train_cfg.device, memory_config=memory_cfg
        )
        agent = api["create_agent"](
            MODELS[self.MODEL_KEY].algorithm,
            env.observation_dim,
            env.action_dim,
            train_cfg,
            device=None,
        )
        api["configure_agent_action_levels"](agent, action_cfg.level_count)
        loaded_meta = api["load_checkpoint"](
            checkpoint, agent, map_location="agent", restore_rng=False
        )
        checkpoint_identity = (
            loaded_meta.get("schema"),
            loaded_meta.get("model"),
            loaded_meta.get("algorithm"),
            loaded_meta.get("strategy_class"),
            loaded_meta.get("timeframe"),
            tuple(loaded_meta.get("input_timeframes", ())),
            loaded_meta.get("feature_version"),
            loaded_meta.get("runtime_contract_sha256"),
            loaded_meta.get("artifact_contract_sha256"),
            loaded_meta.get("observation_dim"),
            loaded_meta.get("action_dim"),
        )
        expected_checkpoint_identity = (
            MODEL_METADATA_SCHEMA,
            self.MODEL_KEY,
            MODELS[self.MODEL_KEY].algorithm,
            MODELS[self.MODEL_KEY].strategy_class,
            self.timeframe,
            input_timeframes,
            FEATURE_VERSION,
            runtime_contract_sha,
            artifact_contract_sha,
            env.observation_dim,
            env.action_dim,
        )
        if checkpoint_identity != expected_checkpoint_identity:
            env.close(aggressive=False)
            raise RuntimeError(
                "HPRL checkpoint embedded metadata does not match current Strategy contract: "
                f"expected={expected_checkpoint_identity}, actual={checkpoint_identity}"
            )
        if (
            metadata.get("observation_dim") != env.observation_dim
            or metadata.get("action_dim") != env.action_dim
        ):
            env.close(aggressive=False)
            raise RuntimeError(
                "HPRL metadata observation/action dimensions do not match native env"
            )

        adapter_policy = api["HprlHedgeAdapterPolicy"].from_hprl_action_config(
            action_cfg,
            target_unit=api["HprlTargetUnit"].NOTIONAL_EQUITY_RATIO,
        )
        adapter = api["HprlHedgeAdapter"](adapter_policy)

        row_count = len(dataframe)
        long_score = np.zeros(row_count, dtype=np.float32)
        short_score = np.zeros(row_count, dtype=np.float32)
        target_net = np.full(row_count, np.nan, dtype=np.float64)
        target_net_ratio = np.full(row_count, np.nan, dtype=np.float64)
        confidence = np.ones(row_count, dtype=np.float32)
        risk_scale = np.ones(row_count, dtype=np.float32)
        long_scale = np.zeros(row_count, dtype=np.float32)
        short_scale = np.zeros(row_count, dtype=np.float32)
        allow_new_risk = np.zeros(row_count, dtype=bool)
        regime = np.full(row_count, "HPRL", dtype=object)
        reason = np.full(row_count, "HPRL_STARTUP", dtype=object)
        model_version = np.full(row_count, self.version(), dtype=object)

        # Freqtrade analyzes startup candles, then HEDGE trims exactly
        # ``startup_candle_count`` rows before replay.  Reset the shadow account at the same
        # boundary so the first formal HEDGE bar and the policy both start from flat equity.
        formal_start_position = int(self.startup_candle_count)
        start_decision = int(np.searchsorted(valid_positions, formal_start_position, side="left"))
        if (
            start_decision >= len(valid_positions)
            or int(valid_positions[start_decision]) != formal_start_position
        ):
            env.close(aggressive=False)
            raise RuntimeError(
                "HPRL startup_candle_count does not cover the complete feature warmup at the "
                f"formal replay boundary: startup={formal_start_position}, "
                f"first_valid={int(valid_positions[0])}"
            )
        if start_decision >= env.market.time_steps - 1:
            env.close(aggressive=False)
            raise RuntimeError("HPRL formal replay has fewer than two realizable policy timesteps")

        obs, _ = env.reset(start_index=start_decision)
        previous_projection = None
        model_id = (
            f"hprl/{MODELS[self.MODEL_KEY].algorithm}/{self.timeframe}/"
            + "+".join(input_timeframes)
        )
        shadow_autoresets = 0
        try:
            for decision_index in range(start_decision, env.market.time_steps):
                # Decide first.  No realized forward return for this row has been consumed yet.
                action = agent.act(obs, deterministic=True)
                shaped = action
                if tuple(action.shape) == (env.envs, env.action_dim):
                    shaped = action.reshape(env.envs, env.symbols, 2)
                if tuple(shaped.shape) != (env.envs, env.symbols, 2):
                    raise RuntimeError(
                        f"native HPRL agent returned invalid action shape: {tuple(action.shape)}"
                    )
                if env.tiered_codec is None:
                    raise RuntimeError("Formal HPRL Strategy requires tiered action mode")
                decoded = env.tiered_codec.decode(shaped, env.margin_position)
                long_notional = self._exact_notional_from_tier(action_cfg, decoded, 0)
                short_notional = self._exact_notional_from_tier(action_cfg, decoded, 1)

                observed_at = self._utc_close_timestamp(
                    timestamps[decision_index], candle_seconds
                )
                intent = api["PlannedExecutionIntent"](
                    symbol=pair,
                    target_long_exposure=long_notional,
                    target_short_exposure=short_notional,
                    confidence=1.0,
                    model_id=model_id,
                    metadata={
                        "source": "freqtrade-strategy-native-hprl-mtf",
                        "base_timeframe": self.timeframe,
                        "input_timeframes": list(input_timeframes),
                    },
                )
                projection = adapter.adapt(
                    intent,
                    sequence=decision_index - start_decision,
                    observed_at=observed_at,
                    now=observed_at,
                    previous=previous_projection,
                )
                if not projection.accepted:
                    raise RuntimeError(
                        "native HPRL tier codec produced a target rejected by the production "
                        f"adapter at row={decision_index}: {projection.reasons}"
                    )
                signal = adapter.signal_snapshot_kwargs(
                    projection,
                    timeframe=self.timeframe,
                    candle_close_time=observed_at,
                    feature_timestamp=observed_at,
                )
                row_position = int(valid_positions[decision_index])
                long_score[row_position] = float(signal["long_score"])
                short_score[row_position] = float(signal["short_score"])
                confidence[row_position] = float(signal["confidence"])
                risk_scale[row_position] = float(signal["risk_scale"])
                long_scale[row_position] = float(signal["long_exposure_scale"])
                short_scale[row_position] = float(signal["short_exposure_scale"])
                allow_new_risk[row_position] = bool(signal["allow_new_risk"])
                regime[row_position] = str(signal["regime"])
                reason[row_position] = str(signal["reason"])
                model_version[row_position] = str(signal["model_version"])
                previous_projection = projection

                if decision_index >= env.market.time_steps - 1:
                    break

                # Realize only after the target for this row is frozen.  A shadow-account
                # termination must not terminate the formal HEDGE Strategy.  The native env
                # autoresets terminated rows internally and returns the reset observation.
                step = env.step(action)
                if bool(step.terminated[0].item()):
                    shadow_autoresets += 1
                if bool(step.info.get("time_done", False)):
                    break
                obs = step.observation
        finally:
            env.close(aggressive=False)

        # Assign columns in bulk.  Avoid a million DataFrame.at writes on 1m history and keep
        # the exact canonical HEDGE surface explicit, including both nullable target fields.
        dataframe["hedge_long_score"] = long_score
        dataframe["hedge_short_score"] = short_score
        dataframe["hedge_target_net"] = target_net
        dataframe["hedge_target_net_ratio"] = target_net_ratio
        dataframe["hedge_confidence"] = confidence
        dataframe["hedge_risk_scale"] = risk_scale
        dataframe["hedge_long_exposure_scale"] = long_scale
        dataframe["hedge_short_exposure_scale"] = short_scale
        dataframe["hedge_allow_new_risk"] = allow_new_risk
        dataframe["hedge_regime"] = regime
        dataframe["hedge_reason"] = reason
        dataframe["hedge_model_version"] = model_version
        # Keep the autoreset counter internal: extra full-history diagnostic columns would
        # unnecessarily increase the 1m dataframe footprint.  The important semantic rule is
        # that a shadow autoreset never terminates formal Strategy signal generation.
        _ = (shadow_autoresets, alignment_diagnostics)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = str(metadata.get("pair") or "ETH/USDT:USDT")
        if pair != "ETH/USDT:USDT":
            raise RuntimeError(f"HPRL ETH Strategy received unsupported pair: {pair!r}")
        if getattr(self, "dp", None) is None:
            raise RuntimeError("Freqtrade DataProvider is required for HPRL MTF Strategy")

        frames: dict[str, DataFrame] = {self.timeframe: dataframe}
        for timeframe in informative_timeframes_for(self.timeframe):
            informative = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
            if informative is None or informative.empty:
                raise RuntimeError(
                    f"Freqtrade DataProvider returned no informative OHLCV for {pair} {timeframe}"
                )
            frames[timeframe] = informative
        return self._generate_hprl_columns(dataframe, pair, frames)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        del metadata
        allowed = dataframe["hedge_allow_new_risk"].astype(bool)
        dataframe["enter_long"] = (
            (dataframe["hedge_long_exposure_scale"] > 0) & allowed
        ).astype(int)
        dataframe["enter_short"] = (
            (dataframe["hedge_short_exposure_scale"] > 0) & allowed
        ).astype(int)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        del metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
