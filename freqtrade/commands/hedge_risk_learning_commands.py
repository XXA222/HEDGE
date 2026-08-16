"""CLI commands for Risk-Level RL sizing attribution and walk-forward evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise OperationalException(
        f"Unsupported audit table format {path.suffix!r}; use CSV or Parquet."
    )


def _is_default_range_index(frame: pd.DataFrame) -> bool:
    index = frame.index
    return isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1


def _align_frames(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    index_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Require evidence stronger than same-length positional CSV rows."""
    selected = str(index_column or "").strip()
    if selected:
        if selected not in features.columns or selected not in prices.columns:
            raise OperationalException(
                f"--index-column {selected!r} must exist in both features and prices."
            )
        left = pd.to_datetime(features.pop(selected), utc=True, errors="raise")
        right = pd.to_datetime(prices.pop(selected), utc=True, errors="raise")
        features.index = left
        prices.index = right
        mode = f"EXPLICIT_COLUMN:{selected}"
    elif features.index.equals(prices.index) and not _is_default_range_index(features):
        mode = "PRESERVED_INDEX"
    else:
        common = next(
            (
                candidate
                for candidate in ("date", "timestamp", "datetime")
                if candidate in features.columns and candidate in prices.columns
            ),
            None,
        )
        if common is None:
            raise OperationalException(
                "Risk-Level OOS audit refuses positional-only alignment. Use Parquet with "
                "the original index, or provide --index-column present in both files."
            )
        left = pd.to_datetime(features.pop(common), utc=True, errors="raise")
        right = pd.to_datetime(prices.pop(common), utc=True, errors="raise")
        features.index = left
        prices.index = right
        mode = f"AUTO_COLUMN:{common}"
    if not features.index.equals(prices.index):
        raise OperationalException("OOS feature/price indexes are not identical after alignment.")
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise OperationalException("OOS audit index must be strictly chronological and unique.")
    return features, prices, mode


def _training_boundary_metadata(
    args: dict[str, Any],
    *,
    oos_start: object,
) -> tuple[str, str]:
    train_start = str(args.get("hedge_risk_audit_train_start") or "").strip()
    train_end = str(args.get("hedge_risk_audit_train_end") or "").strip()
    if bool(train_start) != bool(train_end):
        raise OperationalException("--train-start and --train-end must be supplied together.")
    if not train_start:
        return "", ""
    start = pd.to_datetime(train_start, utc=True, errors="raise")
    end = pd.to_datetime(train_end, utc=True, errors="raise")
    oos = pd.to_datetime(oos_start, utc=True, errors="raise")
    if end <= start:
        raise OperationalException("Risk-Level training window end must be after start.")
    if end >= oos:
        raise OperationalException("Risk-Level audit requires train_end < oos_start.")
    return str(start), str(end)


def _load_model(model_type: str, path: Path):
    from stable_baselines3.common.base_class import BaseAlgorithm

    normalized = model_type.strip()
    model_class: type[BaseAlgorithm]
    if normalized == "PPO":
        from stable_baselines3 import PPO

        model_class = PPO
    elif normalized == "A2C":
        from stable_baselines3 import A2C

        model_class = A2C
    elif normalized == "TRPO":
        from sb3_contrib import TRPO

        model_class = TRPO
    elif normalized == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO

        model_class = RecurrentPPO
    elif normalized == "MaskablePPO":
        from sb3_contrib import MaskablePPO

        model_class = MaskablePPO
    else:  # pragma: no cover - argparse restricts this
        raise OperationalException(f"Unsupported Risk-Level RL model type: {normalized}")
    return model_class.load(path, device="cpu")


def start_hedge_risk_level_audit(args: dict[str, Any]) -> int:
    """Run causal OOS counterfactual attribution for one trained Risk-Level model."""
    from freqtrade.configuration.config_setup import setup_utils_configuration
    from freqtrade.freqai.hedge_rl.risk_learning_audit import (
        RiskLearningAuditThresholds,
        run_risk_level_learning_audit,
        write_risk_learning_audit,
    )

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    features_path = Path(str(args["hedge_risk_audit_features"])).expanduser().resolve()
    prices_path = Path(str(args["hedge_risk_audit_prices"])).expanduser().resolve()
    model_path = Path(str(args["hedge_risk_audit_model"])).expanduser().resolve()
    output_path = Path(str(args["hedge_risk_audit_output"])).expanduser().resolve()

    for label, path in (
        ("features", features_path),
        ("prices", prices_path),
        ("model", model_path),
    ):
        if not path.is_file():
            raise OperationalException(f"Risk-Level audit {label} file not found: {path}")

    features = _read_frame(features_path)
    prices = _read_frame(prices_path)
    features, prices, alignment_mode = _align_frames(
        features,
        prices,
        index_column=args.get("hedge_risk_audit_index_column"),
    )
    model_type = str(args.get("hedge_risk_audit_model_type", "PPO"))
    model = _load_model(model_type, model_path)
    thresholds = RiskLearningAuditThresholds(
        drawdown_weight=float(args.get("hedge_risk_audit_drawdown_weight", 1.0)),
        min_sizing_edge=float(args.get("hedge_risk_audit_min_sizing_edge", 0.0005)),
        max_active_joint_action_share=float(
            args.get("hedge_risk_audit_max_active_action_share", 0.90)
        ),
        min_distinct_nonzero_levels=int(args.get("hedge_risk_audit_min_distinct_levels", 3)),
        min_active_fraction=float(args.get("hedge_risk_audit_min_active_fraction", 0.02)),
        min_nonzero_level_entropy=float(args.get("hedge_risk_audit_min_nonzero_entropy", 0.20)),
        min_magnitude_change_fraction=float(
            args.get("hedge_risk_audit_min_magnitude_change_fraction", 0.005)
        ),
        shuffle_trials=int(args.get("hedge_risk_audit_shuffle_trials", 8)),
        shuffle_quantile=float(args.get("hedge_risk_audit_shuffle_quantile", 0.75)),
        permutation_trials=int(args.get("hedge_risk_audit_permutation_trials", 23)),
        permutation_quantile=float(args.get("hedge_risk_audit_permutation_quantile", 0.75)),
        max_permutation_exceedance=float(
            args.get("hedge_risk_audit_max_permutation_exceedance", 0.25)
        ),
        segment_count=int(args.get("hedge_risk_audit_segment_count", 4)),
        min_segment_steps=int(args.get("hedge_risk_audit_min_segment_steps", 128)),
        min_segments=int(args.get("hedge_risk_audit_min_segments", 2)),
        min_segment_pass_ratio=float(args.get("hedge_risk_audit_min_segment_pass_ratio", 0.50)),
    )
    train_start, train_end = _training_boundary_metadata(
        args,
        oos_start=features.index[0],
    )
    report = run_risk_level_learning_audit(
        model=model,
        features=features,
        prices=prices,
        config=config,
        thresholds=thresholds,
        shuffle_seed=int(args.get("hedge_risk_audit_shuffle_seed", 20260815)),
        metadata={
            "model_type": model_type,
            "model_path": str(model_path),
            "train_start": train_start,
            "train_end": train_end,
            "oos_start": str(features.index[0]),
            "oos_end": str(features.index[-1]),
            "alignment": alignment_mode,
        },
    )
    write_risk_learning_audit(report, output_path)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.passed else 3


def start_hedge_risk_walkforward_audit(args: dict[str, Any]) -> int:
    """Aggregate independent sequential model/OOS audits into a walk-forward gate."""
    from freqtrade.freqai.hedge_rl.risk_walkforward_audit import (
        RiskWalkForwardThresholds,
        aggregate_risk_walk_forward,
        load_risk_learning_audit,
        write_risk_walk_forward_report,
    )

    raw_paths = args.get("hedge_risk_wf_audits") or []
    paths = [Path(str(item)).expanduser().resolve() for item in raw_paths]
    audit_directory = args.get("hedge_risk_wf_audit_directory")
    if audit_directory not in (None, ""):
        directory = Path(str(audit_directory)).expanduser().resolve()
        if not directory.is_dir():
            raise OperationalException(f"Walk-forward audit directory not found: {directory}")
        paths = sorted(directory.rglob("risk-level-learning-audit.json"))
        if not paths:
            raise OperationalException(
                f"No risk-level-learning-audit.json files found below {directory}"
            )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise OperationalException("Walk-forward audit files not found: " + ", ".join(missing))
    thresholds = RiskWalkForwardThresholds(
        min_folds=int(args.get("hedge_risk_wf_min_folds", 3)),
        min_fold_pass_ratio=float(args.get("hedge_risk_wf_min_pass_ratio", 0.67)),
        min_positive_fixed_edge_ratio=float(
            args.get("hedge_risk_wf_min_positive_fixed_ratio", 0.67)
        ),
        min_positive_permutation_edge_ratio=float(
            args.get("hedge_risk_wf_min_positive_permutation_ratio", 0.67)
        ),
        min_distinct_model_ratio=float(args.get("hedge_risk_wf_min_distinct_model_ratio", 1.0)),
        min_median_fixed_edge=float(args.get("hedge_risk_wf_min_median_fixed_edge", 0.0)),
        min_median_permutation_edge=float(
            args.get("hedge_risk_wf_min_median_permutation_edge", 0.0)
        ),
    )
    report = aggregate_risk_walk_forward(
        [load_risk_learning_audit(path) for path in paths],
        thresholds=thresholds,
    )
    output = Path(str(args["hedge_risk_wf_output"])).expanduser().resolve()
    write_risk_walk_forward_report(report, output)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.passed else 4
