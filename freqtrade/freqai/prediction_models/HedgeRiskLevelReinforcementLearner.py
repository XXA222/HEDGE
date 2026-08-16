"""Memory-efficient FreqAI learner for independent LONG/SHORT target risk levels."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th
from pandas import DataFrame
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.hedge_rl.risk_bridge import HedgeRiskLevelPolicyBridge, HedgeRiskPolicyContext
from freqtrade.freqai.hedge_rl.risk_environment import HedgeRiskEnvConfig, HedgeRiskLevelEnv
from freqtrade.freqai.hedge_rl.risk_levels import RiskLevelProfile
from freqtrade.freqai.hedge_rl.risk_memory import (
    HedgeRLMemoryConfig,
    compact_feature_matrix,
    compact_training_dataframe,
    release_rl_phase_memory,
)
from freqtrade.freqai.hedge_rl.risk_projection_adapter import HedgeRiskRuntimeContextProvider
from freqtrade.freqai.hedge_rl.risk_runtime import RiskRLAdaptiveCpuController
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner


logger = logging.getLogger(__name__)
HedgeRiskContextProvider = Callable[[str, int, object], HedgeRiskPolicyContext]


class _AdaptiveRiskCpuCallback(BaseCallback):
    """Refresh the host-aware PyTorch thread budget at coarse training intervals."""

    def __init__(self, controller: RiskRLAdaptiveCpuController) -> None:
        super().__init__(verbose=0)
        self.controller = controller
        self.interval = controller.config.refresh_train_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.interval == 0:
            self.controller.apply_torch(th)
        return True


class HedgeRiskLevelReinforcementLearner(ReinforcementLearner):
    """Official FreqAI RL lifecycle with MultiDiscrete([5, 5]) Hedge targets.

    This remains independent from HPRL.  V2 adds a memory-specific lifecycle:

    * no unused deep copy of ``train_features`` into ``df_raw``;
    * transformed market features are downcast to float32 by default;
    * train/eval environments compact DataFrames into narrow arrays and do not retain config;
    * train/eval environments are closed and detached after ``learn()``;
    * inference converts to at most one compact numeric matrix and emits int8 level columns.
    """

    MyRLEnv = HedgeRiskLevelEnv  # type: ignore[assignment]
    _SUPPORTED_MODEL_TYPES = {"PPO", "A2C", "TRPO", "RecurrentPPO", "MaskablePPO"}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.model_type not in self._SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"{self.model_type} does not support this MultiDiscrete Hedge action contract; "
                f"choose one of {sorted(self._SUPPORTED_MODEL_TYPES)}"
            )
        self.hedge_memory_config = HedgeRLMemoryConfig.from_freqtrade_config(self.config)
        self.hedge_cpu_controller = RiskRLAdaptiveCpuController.from_freqtrade_config(self.config)
        self.hedge_cpu_controller.apply_torch(th)
        self._hedge_train_rows = 0
        self._hedge_train_bounds: tuple[str, str] | None = None
        self._hedge_oos_rows = 0
        self._hedge_oos_bounds: tuple[str, str] | None = None
        self._hedge_risk_context_provider: HedgeRiskContextProvider | None = None

    def set_hedge_context_provider(self, provider: HedgeRiskContextProvider | None) -> None:
        self._hedge_risk_context_provider = provider

    def set_hedge_runtime(self, runtime: Any) -> None:
        """Bind the live source-separated Hedge runtime to risk-level inference."""

        profile = RiskLevelProfile.from_freqtrade_config(self.config)
        self._hedge_risk_context_provider = HedgeRiskRuntimeContextProvider(
            runtime,
            profile=profile,
        )
        logger.info("Hedge risk-level runtime context provider bound.")

    @staticmethod
    def _optimizer_state_to_cpu(optimizer: Any) -> None:
        state = optimizer.state if hasattr(optimizer, "state") else None
        if not isinstance(state, dict):
            return
        for values in state.values():
            if not isinstance(values, dict):
                continue
            for key, value in tuple(values.items()):
                if th.is_tensor(value):
                    values[key] = value.to(device="cpu")

    def _ensure_model_cpu(self, model: Any) -> Any:
        """Move continual-learning policy and optimizer state to CPU before learn()."""

        policy = model.policy if hasattr(model, "policy") else None
        if policy is not None and hasattr(policy, "to"):
            policy.to(th.device("cpu"))
        optimizer = (
            policy.optimizer if policy is not None and hasattr(policy, "optimizer") else None
        )
        if optimizer is not None:
            self._optimizer_state_to_cpu(optimizer)
        if hasattr(model, "device"):
            try:
                model.device = th.device("cpu")
            except (AttributeError, TypeError):
                pass
        return model

    def _policy_context(
        self,
        pair: str,
        tick: int,
        index_value: object,
        *,
        fallback_mark: float,
    ) -> HedgeRiskPolicyContext:
        provider = self._hedge_risk_context_provider
        if provider is None:
            context = HedgeRiskPolicyContext.flat(1.0, mark=max(float(fallback_mark), 1e-12))
            return HedgeRiskPolicyContext(
                account=context.account,
                mark=context.mark,
                uncertainty_score=1.0,
                projection_fresh=False,
            )
        context = provider(pair, tick, index_value)
        if not isinstance(context, HedgeRiskPolicyContext):
            raise TypeError("Hedge risk context provider must return HedgeRiskPolicyContext")
        return context

    def train(self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """Freqtrade RL training path without the base class' unused ``df_raw`` deepcopy."""

        logger.info(
            "--------------------Starting memory-efficient Hedge RL training %s "
            "--------------------",
            pair,
        )
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )
        dd: dict[str, Any] = dk.make_train_test_datasets(features_filtered, labels_filtered)
        # Our environment never consumes df_raw.  The upstream RL base deep-copies the entire
        # training DataFrame here; keep an empty frame instead of a second full dataset.
        self.df_raw = DataFrame()
        del features_filtered, labels_filtered, unfiltered_df
        dk.fit_labels()

        prices_train, prices_test = self.build_ohlc_price_dataframes(dk.data_dictionary, pair, dk)
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        (dd["train_features"], dd["train_labels"], dd["train_weights"]) = (
            dk.feature_pipeline.fit_transform(
                dd["train_features"], dd["train_labels"], dd["train_weights"]
            )
        )
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            (dd["test_features"], dd["test_labels"], dd["test_weights"]) = (
                dk.feature_pipeline.transform(
                    dd["test_features"], dd["test_labels"], dd["test_weights"]
                )
            )

        feature_dtype = self.hedge_memory_config.feature_dtype
        dd["train_features"] = compact_training_dataframe(dd["train_features"], dtype=feature_dtype)
        train_index = dd["train_features"].index
        if len(prices_train) != len(dd["train_features"]) or not prices_train.index.equals(
            train_index
        ):
            raise ValueError("Hedge RL train features/prices lost chronological index alignment")
        if not train_index.is_monotonic_increasing or not train_index.is_unique:
            raise ValueError("Hedge RL training index must be strictly chronological and unique")
        if isinstance(dd.get("test_features"), pd.DataFrame) and not dd["test_features"].empty:
            dd["test_features"] = compact_training_dataframe(
                dd["test_features"], dtype=feature_dtype
            )
            test_index = dd["test_features"].index
            if len(prices_test) != len(dd["test_features"]) or not prices_test.index.equals(
                test_index
            ):
                raise ValueError("Hedge RL OOS features/prices lost chronological index alignment")
            if not test_index.is_monotonic_increasing or not test_index.is_unique:
                raise ValueError("Hedge RL OOS index must be strictly chronological and unique")
            try:
                split_is_causal = bool(train_index[-1] < test_index[0])
            except TypeError as exc:
                raise ValueError(
                    "Hedge RL train/OOS indexes are not chronologically comparable"
                ) from exc
            if not split_is_causal:
                raise ValueError(
                    "Hedge RL requires train_end < oos_start; shuffled splits are invalid"
                )
            self._hedge_oos_rows = len(dd["test_features"])
            self._hedge_oos_bounds = (
                str(test_index[0]),
                str(test_index[-1]),
            )
        else:
            self._hedge_oos_rows = 0
            self._hedge_oos_bounds = None
        self._hedge_train_rows = len(dd["train_features"])
        self._hedge_train_bounds = (
            str(train_index[0]),
            str(train_index[-1]),
        )
        logger.info(
            "Training Hedge risk-level model on %s features, %s points, dtype=%s",
            len(dd["train_features"].columns),
            self._hedge_train_rows,
            dd["train_features"].dtypes.iloc[0] if len(dd["train_features"].columns) else "n/a",
        )

        self.set_train_and_eval_environments(dd, prices_train, prices_test, dk)
        # Environments have detached the narrow arrays they need. Drop the temporary price
        # DataFrame references before the long policy-optimization phase.
        del prices_train, prices_test
        release_rl_phase_memory(trim_allocator=False)
        model = self.fit(dd, dk)
        logger.info("--------------------done Hedge RL training %s--------------------", pair)
        return model

    def _cpu_training_parameters(self) -> dict[str, Any]:
        """Return SB3 parameters pinned to the project's CPU-only training baseline."""

        parameters = dict(self.freqai_info.get("model_training_parameters", {}))
        parameters["device"] = "cpu"
        return parameters

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """Train while releasing environment-sized state immediately after optimization."""

        train_rows = self._hedge_train_rows or len(data_dictionary["train_features"])
        active_threads = self.hedge_cpu_controller.apply_torch(th)
        logger.info(
            "Hedge risk-level adaptive CPU threads=%s telemetry=%s",
            active_threads,
            self.hedge_cpu_controller.telemetry(),
        )
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * train_rows
        policy_kwargs = {"activation_fn": th.nn.ReLU, "net_arch": self.net_arch}
        tb_path = (
            Path(dk.full_path / "tensorboard" / dk.pair.split("/")[0])
            if self.activate_tensorboard
            else None
        )
        if dk.pair not in self.dd.model_dictionary or not self.continual_learning:
            model = self.MODELCLASS(
                self.policy_type,
                self.train_env,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tb_path,
                **self._cpu_training_parameters(),
            )
        else:
            logger.info("Continual Hedge RL training - reattaching fresh compact environment.")
            model = self.dd.model_dictionary[dk.pair]
            self._ensure_model_cpu(model)
            model.set_env(self.train_env)

        self._ensure_model_cpu(model)
        adaptive_cpu_callback = _AdaptiveRiskCpuCallback(self.hedge_cpu_controller)
        callbacks: list[Any] = [
            adaptive_cpu_callback,
            self.eval_callback,
            self.tensorboard_callback,
        ]
        progressbar_callback: ProgressBarCallback | None = None
        if self.rl_config.get("progress_bar", False):
            progressbar_callback = ProgressBarCallback()
            callbacks.insert(0, progressbar_callback)
        candidate = model
        try:
            model.learn(total_timesteps=int(total_timesteps), callback=callbacks)
            best_model_path = Path(dk.data_path / "best_model.zip")
            if best_model_path.is_file():
                logger.info("Callback found a best Hedge risk-level model.")
                candidate = self.MODELCLASS.load(dk.data_path / "best_model", device="cpu")
            else:
                logger.info("Couldn't find best model, using final Hedge risk-level model instead.")

            from freqtrade.freqai.hedge_rl.risk_learning_audit import (
                RiskLearningAuditThresholds,
                audit_enabled,
                audit_required,
                run_risk_level_learning_audit_on_env,
                write_risk_learning_audit,
            )

            if audit_enabled(self.config):
                required = audit_required(self.config)
                try:
                    if self.eval_env is None or self._hedge_oos_rows <= 0:
                        raise RuntimeError(
                            "Risk-Level post-fit audit requires a non-empty chronological OOS split"
                        )
                    bounds = self._hedge_oos_bounds or ("", "")
                    train_bounds = self._hedge_train_bounds or ("", "")
                    audit = run_risk_level_learning_audit_on_env(
                        model=candidate,
                        env=self.eval_env,
                        thresholds=RiskLearningAuditThresholds.from_freqtrade_config(self.config),
                        metadata={
                            "pair": dk.pair,
                            "model_type": self.model_type,
                            "training_rows": train_rows,
                            "train_start": train_bounds[0],
                            "train_end": train_bounds[1],
                            "oos_rows": self._hedge_oos_rows,
                            "oos_start": bounds[0],
                            "oos_end": bounds[1],
                            "automatic_post_fit_audit": True,
                            "continual_learning": bool(self.continual_learning),
                        },
                    )
                    audit_path = Path(dk.data_path / "risk-level-learning-audit.json")
                    write_risk_learning_audit(audit, audit_path)
                    logger.info(
                        "Risk-Level OOS sizing audit status=%s path=%s",
                        "PASS" if audit.passed else "FAIL",
                        audit_path,
                    )
                    if required and not audit.passed:
                        raise RuntimeError(
                            "Risk-Level post-fit learning audit failed the required "
                            "dynamic-sizing gate"
                        )
                except Exception:
                    if required:
                        raise
                    logger.exception("Non-blocking Risk-Level post-fit learning audit failed")
        finally:
            if progressbar_callback:
                progressbar_callback.on_training_end()
            if self.hedge_memory_config.release_training_envs_after_fit:
                self._release_training_environments(model)
            callbacks.clear()
            adaptive_cpu_callback = None  # type: ignore[assignment]
            progressbar_callback = None
            if self.hedge_memory_config.release_phase_memory_after_fit:
                release_rl_phase_memory()

        return candidate

    def _release_training_environments(self, model: Any) -> None:
        """Close feature-owning environments and break the model->env retention chain."""

        for name in ("train_env", "eval_env"):
            env = getattr(self, name, None)
            if env is not None:
                try:
                    env.close()
                except (AttributeError, RuntimeError):
                    pass
            setattr(self, name, None)
        # SB3 predict() does not need an environment. The next continual-learning cycle calls
        # model.set_env() with a newly prepared environment before model.learn().
        if model.env is not None:
            try:
                model.env = None
            except AttributeError:
                pass
        for name in ("_last_obs", "_last_original_obs", "_last_episode_starts"):
            if hasattr(model, name):
                try:
                    setattr(model, name, None)
                except AttributeError:
                    pass
        self.eval_callback = None
        self.tensorboard_callback = None  # type: ignore[assignment]  # rebuilt by set_train_and_eval_environments()

    def rl_model_predict(
        self,
        dataframe: pd.DataFrame,
        dk: FreqaiDataKitchen,
        model,
    ) -> pd.DataFrame:
        if len(dk.label_list) != 2:
            raise ValueError(
                "HedgeRiskLevelReinforcementLearner requires exactly two labels: "
                "LONG level then SHORT level"
            )
        features = compact_feature_matrix(
            dataframe,
            dtype=self.hedge_memory_config.numpy_feature_dtype,
            readonly=True,
        )
        profile = RiskLevelProfile.from_freqtrade_config(self.config)
        env_config = HedgeRiskEnvConfig.from_freqtrade_config(self.config)
        bridge = HedgeRiskLevelPolicyBridge(
            feature_names=tuple(str(column) for column in dataframe.columns),
            window_size=self.CONV_WIDTH,
            profile=profile,
            feature_clip=env_config.feature_clip,
            max_episode_steps=env_config.max_episode_steps,
            max_feature_age_steps=int(
                self.config.get("freqai", {})
                .get("hedge_rl_config", {})
                .get("max_feature_age_steps", 1)
            ),
        )
        # Levels are only 0..4. int8 cuts output memory to 1/8 of V1 int64 predictions.
        output_values = np.zeros((len(dataframe), 2), dtype=np.int8)
        for tick in range(self.CONV_WIDTH - 1, len(dataframe)):
            context = self._policy_context(
                dk.pair,
                tick,
                dataframe.index[tick],
                fallback_mark=1.0,
            )
            observation = bridge.observation(features, tick=tick, context=context)
            action = bridge.predict_action(model, observation, context=context)
            output_values[tick, 0] = int(action.long_level)
            output_values[tick, 1] = int(action.short_level)
        output = pd.DataFrame(output_values, index=dataframe.index, columns=dk.label_list)
        del features, output_values
        if self.hedge_memory_config.release_phase_memory_after_predict:
            release_rl_phase_memory(trim_allocator=False)
        return output
