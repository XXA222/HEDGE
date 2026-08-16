"""Modular JSON-schema extension for the Hedge runtime.

This module is intentionally isolated from Freqtrade's upstream schema so future
upgrades only require a stable one-line hook in ``config_schema.py``.
"""

from __future__ import annotations

from typing import Any


HEDGE_ROLE_SCHEMA: dict[str, Any] = {
    "description": "Per-user Hedge control-plane role mapping.",
    "type": "object",
    "additionalProperties": {
        "type": "string",
        "enum": ["VIEWER", "OPERATOR", "RISK_MANAGER", "ADMIN"],
    },
    "default": {},
}


HEDGE_RL_ACTION_SPACE_SCHEMA: dict[str, Any] = {
    "description": "Dual-leg Hedge RL target risk-level action space.",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "enabled": {"type": "boolean", "default": True},
        "position_levels": {
            "type": "array",
            "items": {"type": "number", "minimum": 0.0, "maximum": 0.50},
            "minItems": 5,
            "maxItems": 5,
            "default": [0.0, 0.05, 0.12, 0.25, 0.40],
        },
        "long_leverage": {"type": "number", "exclusiveMinimum": 0.0},
        "short_leverage": {"type": "number", "exclusiveMinimum": 0.0},
        "max_combined_margin_fraction": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0},
        "minimum_reserve_margin_fraction": {
            "type": "number",
            "minimum": 0.0,
            "exclusiveMaximum": 1.0,
        },
        "hard_max_margin_fraction_per_leg": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": 0.50,
        },
        "rebalance_deadband_fraction": {"type": "number", "minimum": 0.0, "maximum": 0.05},
    },
}


HEDGE_RL_REWARD_SCHEMA: dict[str, Any] = {
    "description": "Account-equity-first reward and risk-shaping parameters for Hedge RL.",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "equity_log_return_scale": {"type": "number", "minimum": 0.0},
        "drawdown_weight": {"type": "number", "minimum": 0.0},
        "downside_exposure_weight": {"type": "number", "minimum": 0.0},
        "downside_ewma_weight": {"type": "number", "minimum": 0.0},
        "downside_ewma_alpha": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0},
        "uncertainty_exposure_weight": {"type": "number", "minimum": 0.0},
        "leverage_exposure_weight": {"type": "number", "minimum": 0.0},
        "preferred_reserve_margin_fraction": {
            "type": "number",
            "minimum": 0.0,
            "exclusiveMaximum": 1.0,
        },
        "reserve_pressure_weight": {"type": "number", "minimum": 0.0},
        "minimum_liquidation_buffer_fraction": {
            "type": "number",
            "minimum": 0.0,
            "exclusiveMaximum": 1.0,
        },
        "liquidation_buffer_weight": {"type": "number", "minimum": 0.0},
        "wrong_level_loss_weight": {"type": "number", "minimum": 0.0},
        "position_success_bonus_weight": {"type": "number", "minimum": 0.0},
        "loss_level_multipliers": {
            "type": "array",
            "items": {"type": "number", "minimum": 0.0},
            "minItems": 5,
            "maxItems": 5,
        },
        "win_level_multipliers": {
            "type": "array",
            "items": {"type": "number", "minimum": 0.0},
            "minItems": 5,
            "maxItems": 5,
        },
        "adverse_scale_in_weight": {"type": "number", "minimum": 0.0},
        "upward_jump_weight": {"type": "number", "minimum": 0.0},
        "level_churn_weight": {"type": "number", "minimum": 0.0},
        "turnover_shaping_weight": {"type": "number", "minimum": 0.0},
        "repeated_probe_weight": {"type": "number", "minimum": 0.0},
        "risk_reduction_bonus_weight": {"type": "number", "minimum": 0.0},
        "profit_lock_bonus_weight": {"type": "number", "minimum": 0.0},
        "hedge_efficiency_weight": {"type": "number", "minimum": 0.0},
        "hedge_waste_weight": {"type": "number", "minimum": 0.0},
        "delayed_scale_bonus_weight": {"type": "number", "minimum": 0.0},
        "delayed_probe_bonus_weight": {"type": "number", "minimum": 0.0},
        "scale_confirmation_steps": {"type": "integer", "minimum": 1},
        "probe_confirmation_steps": {"type": "integer", "minimum": 1},
        "probe_drawdown_limit": {"type": "number", "minimum": 0.0},
        "max_positive_shaping": {"type": "number", "minimum": 0.0},
        "reward_clip": {"type": "number", "exclusiveMinimum": 0.0},
        "soft_clip": {"type": "boolean"},
    },
}


HEDGE_RL_LEARNING_AUDIT_SCHEMA: dict[str, Any] = {
    "description": "Post-fit OOS and counterfactual evidence for Risk-Level RL sizing.",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "enabled": {"type": "boolean", "default": False},
        "fail_training_on_gate": {"type": "boolean", "default": False},
        "drawdown_weight": {"type": "number", "minimum": 0.0},
        "min_sizing_edge": {"type": "number", "minimum": 0.0},
        "max_active_joint_action_share": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": 1.0,
        },
        "min_distinct_nonzero_levels": {"type": "integer", "minimum": 1, "maximum": 4},
        "min_active_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "min_nonzero_level_entropy": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "min_magnitude_change_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "shuffle_trials": {"type": "integer", "minimum": 1},
        "shuffle_quantile": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "permutation_trials": {"type": "integer", "minimum": 1, "maximum": 23},
        "permutation_quantile": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "max_permutation_exceedance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "segment_count": {"type": "integer", "minimum": 1},
        "min_segment_steps": {"type": "integer", "minimum": 2},
        "min_segments": {"type": "integer", "minimum": 1},
        "min_segment_pass_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


HEDGE_RL_RUNTIME_SCHEMA: dict[str, Any] = {
    "description": "Shared runtime/memory/adaptive-CPU settings for Hedge RL learners.",
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "observation_window": {"type": "integer", "minimum": 2},
        "max_episode_steps": {"type": "integer", "minimum": 1},
        "starting_balance": {"type": "number", "exclusiveMinimum": 0.0},
        "fee_rate": {"type": "number", "minimum": 0.0},
        "slippage_bps": {"type": "number", "minimum": 0.0, "exclusiveMaximum": 10000.0},
        "funding_rate_per_step": {"type": "number"},
        "drawdown_stop": {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
        "maintenance_margin_ratio": {"type": "number", "minimum": 0.0, "exclusiveMaximum": 1.0},
        "feature_clip": {"type": "number", "exclusiveMinimum": 0.0},
        "default_uncertainty_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "random_start": {"type": "boolean"},
        "seed": {"type": "integer"},
        "max_feature_age_steps": {"type": "integer", "minimum": 0},
        "memory": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "feature_dtype": {"type": "string", "enum": ["float32", "float64"]},
                "reward_breakdown_interval": {"type": "integer", "minimum": 0},
                "gc_collect_every_episodes": {"type": "integer", "minimum": 0},
                "release_training_envs_after_fit": {"type": "boolean"},
                "release_phase_memory_after_fit": {"type": "boolean"},
                "release_phase_memory_after_predict": {"type": "boolean"},
                "max_pending_reward_outcomes": {"type": "integer", "minimum": 8},
            },
        },
        "adaptive_cpu": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "enabled": {"type": "boolean"},
                "max_torch_threads": {"type": "integer", "minimum": 1},
                "min_torch_threads": {"type": "integer", "minimum": 1},
                "refresh_train_steps": {"type": "integer", "minimum": 1},
            },
        },
    },
}

HEDGE_RL_RUNTIME_SCHEMA["properties"]["learning_audit"] = HEDGE_RL_LEARNING_AUDIT_SCHEMA


def extend_config_schema(conf_schema: dict[str, Any]) -> None:
    """Idempotently attach Hedge fields to an upstream Freqtrade schema."""

    properties = conf_schema.setdefault("properties", {})

    # Extend FreqAI RL through the existing Hedge schema boundary rather than editing
    # upstream rl_config definitions.  This makes action/reward mistakes fail at config
    # validation instead of several minutes into training.
    definitions = conf_schema.get("definitions", {})
    freqai_definition = definitions.get("freqai") if isinstance(definitions, dict) else None
    if isinstance(freqai_definition, dict):
        freqai_properties = freqai_definition.setdefault("properties", {})
        rl_config = freqai_properties.get("rl_config")
        if isinstance(rl_config, dict):
            rl_properties = rl_config.setdefault("properties", {})
            rl_properties.setdefault("hedge_action_space", HEDGE_RL_ACTION_SPACE_SCHEMA)
            rl_properties.setdefault("hedge_reward", HEDGE_RL_REWARD_SCHEMA)
        freqai_properties.setdefault("hedge_rl_config", HEDGE_RL_RUNTIME_SCHEMA)

    # Remove an empty sibling accidentally created by an older extension attempt.
    freqai_ref = properties.get("freqai")
    if isinstance(freqai_ref, dict) and freqai_ref.get("properties") == {}:
        freqai_ref.pop("properties", None)

    api_server = properties.get("api_server")
    if isinstance(api_server, dict):
        api_properties = api_server.setdefault("properties", {})
        api_properties.setdefault("hedge_roles", HEDGE_ROLE_SCHEMA)

    # HEDGE_T_P1_SCHEMA_BEGIN
    conf_schema.setdefault("properties", {}).update(
        {
            "position_mode": {
                "description": "Account position mode used by the hedge extension.",
                "type": "string",
                "enum": ["oneway", "hedge"],
                "default": "oneway",
            },
            "hedge_mode_enabled": {
                "description": "Feature gate for the custom hedge execution path.",
                "type": "boolean",
                "default": False,
            },
            "managed_pair": {
                "description": "Single futures pair managed by the hedge MVP.",
                "type": ["string", "null"],
                "default": None,
            },
            "hedge": {
                "description": "Strict safety and adapter settings for the hedge extension.",
                "type": "object",
                "default": {},
                "properties": {
                    "exchange_adapter": {
                        "type": ["string", "null"],
                        "enum": ["binance", "gate", None],
                        "default": None,
                    },
                    "read_only": {"type": "boolean", "default": True},
                    "live_trading_enabled": {"type": "boolean", "default": False},
                    "account_id": {"type": "string", "minLength": 1, "default": "hedge-main"},
                    "operation_mode": {
                        "type": "string",
                        "enum": ["paper", "readonly", "shadow"],
                        "default": "paper",
                    },
                    "target_leverage": {"type": ["string", "number", "integer"]},
                    "max_clock_skew_ms": {"type": "integer", "minimum": 0},
                    "user_stream_max_age_ms": {"type": "integer", "minimum": 1},
                    "rest_reconcile_interval_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "fast_calibration_interval_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "full_reconcile_interval_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "full_calibration_interval_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "calibration_error_retry_interval_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "calibration_stale_after_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "fill_lookback_hours": {"type": ["number", "integer"], "exclusiveMinimum": 0},
                    "history_overlap_seconds": {"type": ["number", "integer"], "minimum": 0},
                    "listen_key_ttl_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "listen_key_renew_interval_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "listen_key_error_retry_interval_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "futures_base_url": {"type": "string", "minLength": 1},
                    "spot_base_url": {"type": "string", "minLength": 1},
                    "websocket_base_url": {"type": "string", "minLength": 1},
                    "rest_proxy_url": {"type": ["string", "null"]},
                    "websocket_proxy_url": {"type": ["string", "null"]},
                    "trust_env_proxy": {"type": "boolean", "default": False},
                    "request_timeout_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "recv_window_ms": {"type": "integer", "minimum": 1},
                    "max_collection_span_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "min_reconnect_delay_seconds": {"type": ["number", "integer"], "minimum": 0},
                    "max_reconnect_delay_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "reconnect_reset_after_seconds": {
                        "type": ["number", "integer"],
                        "exclusiveMinimum": 0,
                    },
                    "drift_verification_attempts": {"type": "integer", "minimum": 1},
                    "max_history_backfill_days": {
                        "type": ["number", "integer", "null"],
                        "minimum": 0,
                    },
                    "quantity_tolerance": {"type": ["string", "number", "integer"]},
                    "financial_tolerance": {"type": ["string", "number", "integer"]},
                    "system_client_order_prefixes": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "uniqueItems": True,
                    },
                    "max_gross_notional": {"type": ["string", "number", "integer"]},
                    "max_gross_exposure_ratio": {"type": ["string", "number", "integer"]},
                    "max_margin_utilization": {"type": ["string", "number", "integer"]},
                    "min_liquidation_buffer_ratio": {"type": ["string", "number", "integer"]},
                    "max_single_order_notional": {"type": ["string", "number", "integer"]},
                    "main_loop": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "mode": {
                                "type": "string",
                                "enum": ["HEDGE_SIMULATED", "HEDGE_PRODUCTION_LOCKED"],
                            },
                            "allowed_symbols": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "pattern": "^[A-Za-z0-9/_:-]+$",
                                    "minLength": 6,
                                },
                                "uniqueItems": True,
                                "minItems": 1,
                            },
                            "state_backend": {"type": "string", "enum": ["sql", "json", "memory"]},
                            "state_path": {"type": ["string", "null"]},
                            "max_submissions_per_cycle": {"type": "integer", "minimum": 1},
                            "max_cancellations_per_cycle": {"type": "integer", "minimum": 1},
                            "block_new_risk_on_external_side": {"type": "boolean"},
                            "require_stream_fresh": {"type": "boolean"},
                            "require_rest_fresh": {"type": "boolean"},
                            "require_reconciliation_consistent": {"type": "boolean"},
                            "recover_on_start": {"type": "boolean"},
                        },
                    },
                    "control_plane": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "enabled": {"type": "boolean", "default": False},
                            "confirmation_ttl_seconds": {
                                "type": "integer",
                                "minimum": 10,
                                "maximum": 3600,
                                "default": 120,
                            },
                            "max_pending_confirmations": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100000,
                                "default": 10000,
                            },
                        },
                    },
                    "paper": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "initial_balance": {"type": ["string", "number", "integer"]},
                            "leverage": {"type": ["string", "number", "integer"]},
                            "auto_fill": {"type": "boolean"},
                            "fill_model": {"type": "string", "enum": ["conservative", "instant"]},
                            "long_signal": {"type": ["string", "number", "integer"]},
                            "short_signal": {"type": ["string", "number", "integer"]},
                            "tick_size": {"type": ["string", "number", "integer"]},
                            "qty_step": {"type": ["string", "number", "integer"]},
                            "min_qty": {"type": ["string", "number", "integer"]},
                            "min_notional": {"type": ["string", "number", "integer"]},
                            "ephemeral": {"type": "boolean"},
                            "state_backend": {"type": "string", "enum": ["sql", "json"]},
                            "state_path": {"type": "string", "minLength": 1},
                            "ohlcv_source": {
                                "type": "string",
                                "enum": ["dataprovider", "ticker_compat"],
                            },
                            "require_closed_candle": {"type": "boolean"},
                            "candle_max_age_seconds": {"type": "integer", "minimum": 0},
                            "max_catchup_candles": {"type": "integer", "minimum": 1},
                            "max_missing_candles": {"type": "integer", "minimum": 0},
                            "reject_revised_candle": {"type": "boolean"},
                            "funding_source": {"type": "string", "enum": ["exchange", "none"]},
                            "funding_max_age_seconds": {"type": "integer", "minimum": 1},
                            "funding_poll_interval_seconds": {"type": "integer", "minimum": 0},
                            "account_events_enabled": {"type": "boolean"},
                            "idempotency_lease_seconds": {"type": "integer", "minimum": 1},
                            "fee_rate": {"type": ["string", "number", "integer"]},
                            "maker_fee_rate": {"type": ["string", "number", "integer"]},
                            "taker_fee_rate": {"type": ["string", "number", "integer"]},
                            "volume_participation": {"type": ["string", "number", "integer"]},
                            "market_slippage_bps": {"type": ["string", "number", "integer"]},
                            "max_entry_layers_per_bar": {"type": "integer", "minimum": 0},
                            "max_reduce_layers_per_bar": {"type": "integer", "minimum": 0},
                            "max_fill_ratio_per_order": {"type": ["string", "number", "integer"]},
                            "max_fills_per_bar": {"type": "integer", "minimum": 0},
                            "bar_volume": {"type": ["string", "number", "integer", "null"]},
                        },
                    },
                    "planner": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "long_enabled": {"type": "boolean"},
                            "short_enabled": {"type": "boolean"},
                            "max_grid_layers": {"type": "integer", "minimum": 0},
                            "take_profit_layers": {"type": "integer", "minimum": 0},
                            "cooldown_seconds": {"type": "integer", "minimum": 0},
                            "replace_price_tolerance_ticks": {"type": "integer", "minimum": 0},
                            "replace_qty_tolerance_steps": {"type": "integer", "minimum": 0},
                            "replace_min_age_seconds": {"type": "integer", "minimum": 0},
                            "trailing_timeout_seconds": {"type": "integer", "minimum": 0},
                            "max_pending_entries": {"type": "integer", "minimum": 0},
                            "unstuck_max_holding_seconds": {"type": "integer", "minimum": 0},
                            "unstuck_min_cooldown_seconds": {"type": "integer", "minimum": 0},
                            "core_wallet_exposure_long": {"type": ["string", "number", "integer"]},
                            "core_wallet_exposure_short": {"type": ["string", "number", "integer"]},
                            "tactical_wallet_exposure_long": {
                                "type": ["string", "number", "integer"]
                            },
                            "tactical_wallet_exposure_short": {
                                "type": ["string", "number", "integer"]
                            },
                            "max_wallet_exposure_long": {"type": ["string", "number", "integer"]},
                            "max_wallet_exposure_short": {"type": ["string", "number", "integer"]},
                            "max_gross_wallet_exposure": {"type": ["string", "number", "integer"]},
                            "initial_entry_fraction": {"type": ["string", "number", "integer"]},
                            "grid_spacing": {"type": ["string", "number", "integer"]},
                            "grid_spacing_growth": {"type": ["string", "number", "integer"]},
                            "grid_qty_growth": {"type": ["string", "number", "integer"]},
                            "qty_scale": {"type": ["string", "number", "integer"]},
                            "trailing_rebound": {"type": ["string", "number", "integer"]},
                            "take_profit_spacing": {"type": ["string", "number", "integer"]},
                            "tactical_reduce_fraction": {"type": ["string", "number", "integer"]},
                            "core_min_fraction": {"type": ["string", "number", "integer"]},
                            "unstuck_trigger_gross_exposure": {
                                "type": ["string", "number", "integer"]
                            },
                            "unstuck_reduce_fraction": {"type": ["string", "number", "integer"]},
                            "maintenance_margin_rate": {"type": ["string", "number", "integer"]},
                            "target_net_wallet_exposure": {"type": ["string", "number", "integer"]},
                            "net_repair_threshold": {"type": ["string", "number", "integer"]},
                            "trailing_trigger_distance": {"type": ["string", "number", "integer"]},
                            "grid_initial_distance": {"type": ["string", "number", "integer"]},
                            "max_single_order_notional": {"type": ["string", "number", "integer"]},
                            "unstuck_daily_loss_budget": {"type": ["string", "number", "integer"]},
                            "unstuck_weekly_loss_budget": {"type": ["string", "number", "integer"]},
                            "unstuck_min_risk_improvement": {
                                "type": ["string", "number", "integer"]
                            },
                            "liquidation_fee_rate": {"type": ["string", "number", "integer"]},
                            "liquidation_buffer_warning_ratio": {
                                "type": ["string", "number", "integer"]
                            },
                        },
                    },
                    "simulation": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "maker_fee": {"type": ["string", "number", "integer"]},
                            "taker_fee": {"type": ["string", "number", "integer"]},
                            "partial_fill_ratio": {"type": ["string", "number", "integer"]},
                            "max_fills_per_bar": {"type": "integer", "minimum": 0},
                            "volume_participation": {"type": ["string", "number", "integer"]},
                            "market_slippage_bps": {"type": ["string", "number", "integer"]},
                            "bar_volume": {"type": ["string", "number", "integer", "null"]},
                        },
                    },
                },
                "additionalProperties": False,
            },
        }
    )
    hedge_schema = conf_schema.get("properties", {}).get("hedge")
    if isinstance(hedge_schema, dict):
        hedge_properties = hedge_schema.setdefault("properties", {})
        hedge_properties.setdefault(
            "operations",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "enabled": {"type": "boolean", "default": False},
                    "state_path": {"type": "string", "minLength": 1},
                    "market_max_age_seconds": {"type": "integer", "minimum": 1},
                    "market_max_gap_candles": {"type": "integer", "minimum": 0},
                    "mark_index_max_divergence_bps": {"type": ["string", "number", "integer"]},
                    "warmup_candles": {"type": "integer", "minimum": 1},
                    "informative_warmup": {
                        "type": "object",
                        "additionalProperties": {"type": "integer", "minimum": 1},
                    },
                    "max_gross_ratio": {"type": ["string", "number", "integer"]},
                    "max_margin_ratio": {"type": ["string", "number", "integer"]},
                    "max_net_ratio": {"type": ["string", "number", "integer"]},
                    "drawdown_warning": {"type": ["string", "number", "integer"]},
                    "drawdown_pause": {"type": ["string", "number", "integer"]},
                    "drawdown_kill": {"type": ["string", "number", "integer"]},
                    "drawdown_recovery": {"type": ["string", "number", "integer"]},
                },
            },
        )
        hedge_properties.setdefault(
            "native_convergence",
            {
                "type": "object",
                "additionalProperties": False,
                "default": {},
                "properties": {
                    "enabled": {"type": "boolean", "default": True},
                    "callback_mode": {
                        "type": "string",
                        "enum": ["native_only", "legacy_view", "both_conservative"],
                        "default": "native_only",
                    },
                    "fail_closed_confirmations": {"type": "boolean", "default": True},
                    "exits": {"type": "object"},
                    "rpc_projection": {"type": "boolean", "default": True},
                    "notifications": {"type": "boolean", "default": True},
                    "capital_policy": {"type": "object"},
                },
            },
        )
        hedge_properties.setdefault(
            "universe",
            {
                "type": "object",
                "additionalProperties": False,
                "default": {},
                "properties": {
                    "enabled": {"type": "boolean", "default": False},
                    "pairs": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 3},
                        "uniqueItems": True,
                    },
                    "weights": {"type": "object"},
                    "blacklist_policy": {
                        "type": "string",
                        "enum": ["reduce_only", "drain", "immediate_exit"],
                        "default": "drain",
                    },
                    "drain_removed_pairs": {"type": "boolean", "default": True},
                    "max_pairs": {"type": "integer", "minimum": 1, "default": 20},
                    "minimum_pair_weight": {"type": ["string", "number"], "default": "0"},
                },
            },
        )
        hedge_properties.setdefault(
            "freqai_native",
            {
                "type": "object",
                "additionalProperties": False,
                "default": {},
                "properties": {
                    "enabled": {"type": "boolean", "default": False},
                    "required": {"type": "boolean", "default": False},
                    "maximum_signal_age_seconds": {"type": "integer", "minimum": 1},
                    "expected_feature_schema": {"type": ["string", "null"]},
                    "model_manifest_path": {"type": ["string", "null"]},
                    "fail_closed_on_expiry": {"type": "boolean", "default": True},
                    "compatible_pairs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                },
            },
        )
        hedge_properties.setdefault(
            "producer_consumer",
            {
                "type": "object",
                "additionalProperties": False,
                "default": {},
                "properties": {
                    "enabled": {"type": "boolean", "default": False},
                    "required": {"type": "boolean", "default": False},
                    "maximum_age_seconds": {"type": "integer", "minimum": 1},
                    "conflict_tolerance": {"type": ["string", "number"]},
                    "fail_on_conflict": {"type": "boolean", "default": True},
                    "producers": {"type": "object"},
                },
            },
        )
        hedge_properties.setdefault(
            "hyperopt_native",
            {
                "type": "object",
                "additionalProperties": False,
                "default": {},
                "properties": {
                    "enabled": {"type": "boolean", "default": False},
                    "epochs": {"type": "integer", "minimum": 1},
                    "jobs": {"type": "integer"},
                    "seed": {"type": "integer"},
                    "loss_weights": {"type": "object"},
                    "result_path": {"type": ["string", "null"]},
                },
            },
        )
        hedge_properties.setdefault(
            "rl_native",
            {
                "type": "object",
                "additionalProperties": False,
                "default": {},
                "properties": {
                    "enabled": {"type": "boolean", "default": False},
                    "max_step_ratio": {"type": ["string", "number"]},
                    "min_liquidation_buffer": {"type": ["string", "number"]},
                    "max_gross_exposure": {"type": ["string", "number"]},
                    "require_confidence": {"type": ["string", "number"]},
                    "reward_weights": {"type": "object"},
                    "vector_envs": {"type": "integer", "minimum": 1},
                    "device": {"type": "string", "enum": ["auto", "cpu", "cuda"]},
                },
            },
        )
        hedge_properties.setdefault(
            "dashboard",
            {
                "type": "object",
                "additionalProperties": False,
                "default": {},
                "properties": {
                    "enabled": {"type": "boolean", "default": False},
                    "local_only": {"type": "boolean", "default": True},
                    "refresh_seconds": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 300,
                        "default": 5,
                    },
                    "telemetry_capacity": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 100000,
                        "default": 2000,
                    },
                    "telemetry_backend": {
                        "type": "string",
                        "enum": ["memory", "jsonl"],
                        "default": "memory",
                    },
                    "telemetry_path": {"type": "string", "minLength": 1},
                    "control_state_path": {"type": "string", "minLength": 1},
                },
            },
        )
    if isinstance(api_server, dict):
        api_properties = api_server.setdefault("properties", {})
        api_properties.setdefault(
            "hedge_ws_accounts",
            {
                "type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "string"}},
                "default": {},
            },
        )
        api_properties.setdefault(
            "hedge_ws_allow_sensitive_admin", {"type": "boolean", "default": False}
        )

    # HEDGE_T_P1_SCHEMA_END
