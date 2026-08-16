"""Hedge CLI registration isolated from Freqtrade's upstream parser layout."""

from __future__ import annotations

from argparse import ArgumentParser, _SubParsersAction
from typing import Any


def _start_hedge_db(args: dict[str, Any]) -> Any:
    """Load the Hedge database command only when it is executed."""
    from freqtrade.commands.hedge_db_commands import start_hedge_db

    return start_hedge_db(args)


def _start_hedge_backtesting(args: dict[str, Any]) -> Any:
    """Load the Hedge backtesting runtime only when it is executed."""
    from freqtrade.commands.hedge_runtime_commands import start_hedge_backtesting

    return start_hedge_backtesting(args)


def _start_hedge_paper(args: dict[str, Any]) -> Any:
    """Load the durable Hedge Paper runtime only when it is executed."""
    from freqtrade.commands.hedge_runtime_commands import start_hedge_paper

    return start_hedge_paper(args)


def _start_hedge_readonly_check(args: dict[str, Any]) -> Any:
    """Load the Binance readonly preflight only when it is executed."""
    from freqtrade.commands.hedge_readonly_commands import start_hedge_readonly_check

    return start_hedge_readonly_check(args)


def _start_hedge_native_audit(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_native_audit

    return start_hedge_native_audit(args)


def _start_hedge_model_check(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_model_check

    return start_hedge_model_check(args)


def _start_hedge_contracts(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_contracts

    return start_hedge_contracts(args)


def _start_hedge_result_analysis(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_result_analysis

    return start_hedge_result_analysis(args)


def _start_hedge_lookahead_file_analysis(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_lookahead_file_analysis

    return start_hedge_lookahead_file_analysis(args)


def _start_hedge_recursive_file_analysis(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_recursive_file_analysis

    return start_hedge_recursive_file_analysis(args)


def _start_hedge_native_hyperopt(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_native_hyperopt

    return start_hedge_native_hyperopt(args)


def _start_hedge_research_optimize(args: dict[str, Any]) -> Any:
    """Load the resumable research optimizer without replacing native hyperopt."""
    from freqtrade.commands.hedge_runtime_commands import start_hedge_research_optimize

    return start_hedge_research_optimize(args)


def _start_hedge_research_capabilities(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_research_commands import start_hedge_research_capabilities

    return start_hedge_research_capabilities(args)


def _start_hedge_research_validate(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_research_commands import start_hedge_research_validate

    return start_hedge_research_validate(args)


def _start_hedge_runtime_acceptance(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_acceptance_commands import start_hedge_runtime_acceptance

    return start_hedge_runtime_acceptance(args)


def _start_hedge_risk_level_audit(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_risk_learning_commands import start_hedge_risk_level_audit

    return start_hedge_risk_level_audit(args)


def _start_hedge_risk_walkforward_audit(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_risk_learning_commands import start_hedge_risk_walkforward_audit

    return start_hedge_risk_walkforward_audit(args)


def _register_runtime_acceptance_command(
    subparsers: _SubParsersAction,
    common_parser: ArgumentParser,
    choices: dict[str, Any],
) -> None:
    if "hedge-runtime-acceptance" in choices:
        return
    command = subparsers.add_parser(
        "hedge-runtime-acceptance",
        help="Run the 20-round state-integrity Runtime Acceptance.",
        parents=[common_parser],
    )
    command.set_defaults(func=_start_hedge_runtime_acceptance)
    command.add_argument(
        "--mode",
        dest="hedge_acceptance_mode",
        choices=("deterministic", "live-readonly"),
        default="deterministic",
    )
    command.add_argument("--project-root", dest="project_root")
    command.add_argument("--output-directory", dest="hedge_acceptance_output_directory")
    command.add_argument("--acceptance-db", dest="hedge_acceptance_database")
    command.add_argument(
        "--observe-seconds", dest="hedge_acceptance_observe_seconds", type=float, default=60.0
    )
    command.add_argument(
        "--target-soak-stage",
        dest="hedge_acceptance_target_soak_stage",
        choices=("smoke", "1h", "6h", "24h", "72h"),
        default="smoke",
    )


def register_hedge_subcommands(  # noqa: C901
    manager: Any,
    subparsers: _SubParsersAction,
    common_parser: ArgumentParser,
    strategy_parser: ArgumentParser,
    *,
    trade_options: list[str],
    backtest_options: list[str],
) -> None:
    """Idempotently register Hedge-only commands without importing runtimes."""

    choices = subparsers.choices
    if "hedge-paper" not in choices:
        hedge_paper_cmd = subparsers.add_parser(
            "hedge-paper",
            help="Run SQL-durable Hedge Paper with real DataProvider OHLCV.",
            parents=[common_parser, strategy_parser],
        )
        hedge_paper_cmd.set_defaults(func=_start_hedge_paper)
        manager._build_args(optionlist=trade_options, parser=hedge_paper_cmd)

    if "hedge-backtesting" not in choices:
        hedge_backtesting_cmd = subparsers.add_parser(
            "hedge-backtesting",
            help="Backtest dual-leg Hedge planning with next-bar execution.",
            parents=[common_parser, strategy_parser],
        )
        hedge_backtesting_cmd.set_defaults(func=_start_hedge_backtesting)
        manager._build_args(optionlist=backtest_options, parser=hedge_backtesting_cmd)
        hedge_backtesting_cmd.add_argument(
            "--hedge-export-filename",
            dest="hedge_export_filename",
            help="Write the Hedge result JSON to this path.",
        )
        hedge_backtesting_cmd.add_argument(
            "--hedge-export-events",
            dest="hedge_export_events",
            action="store_true",
            default=False,
            help="Include the full event ledger in the result JSON.",
        )

    if "hedge-db" not in choices:
        hedge_db = subparsers.add_parser(
            "hedge-db",
            help="Plan, migrate, or verify the explicitly gated Hedge schema.",
            parents=[common_parser],
        )
        hedge_db.set_defaults(func=_start_hedge_db)
        hedge_db.add_argument(
            "--action",
            dest="hedge_db_action",
            choices=("status", "plan", "migrate", "verify"),
            default="status",
        )
        hedge_db.add_argument("--db-url", dest="db_url")
        hedge_db.add_argument(
            "--backup-directory",
            dest="hedge_backup_directory",
        )

    if "hedge-readonly-check" not in choices:
        readonly_check = subparsers.add_parser(
            "hedge-readonly-check",
            help="Run a fail-closed Binance REST-only account preflight.",
            parents=[common_parser],
        )
        readonly_check.set_defaults(func=_start_hedge_readonly_check)
        readonly_check.add_argument(
            "--output",
            dest="hedge_readonly_output",
            help="Write the sanitized preflight JSON report to this path.",
        )
        readonly_check.add_argument(
            "--include-history",
            dest="hedge_readonly_include_history",
            action="store_true",
            default=False,
            help="Also collect order, fill and income history during preflight.",
        )
    if "hedge-native-audit" not in choices:
        command = subparsers.add_parser(
            "hedge-native-audit",
            help="Run fail-closed Hedge native convergence source checks.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_native_audit)
        command.add_argument("--project-root", dest="project_root")
        command.add_argument("--output", dest="hedge_native_output")

    if "hedge-model-check" not in choices:
        command = subparsers.add_parser(
            "hedge-model-check",
            help="Validate a Hedge FreqAI model manifest and expiry.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_model_check)
        command.add_argument("--manifest", dest="hedge_model_manifest", required=True)
        command.add_argument("--output", dest="hedge_native_output")

    if "hedge-native-contracts" not in choices:
        command = subparsers.add_parser(
            "hedge-native-contracts",
            help="Print Hedge Hyperopt and dual-leg RL contracts.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_contracts)
        command.add_argument("--output", dest="hedge_native_output")
    if "hedge-result-analysis" not in choices:
        command = subparsers.add_parser(
            "hedge-result-analysis",
            help="Rank Hedge v4 result artifacts.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_result_analysis)
        command.add_argument("--result", dest="hedge_result_files", action="append", required=True)
        command.add_argument("--metric", dest="hedge_result_metric", default="total_return")
        command.add_argument("--ascending", dest="hedge_result_ascending", action="store_true")
        command.add_argument("--output", dest="hedge_native_output")

    if "hedge-lookahead-analysis" not in choices:
        command = subparsers.add_parser(
            "hedge-lookahead-analysis",
            help="Compare full and truncated Hedge result prefixes.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_lookahead_file_analysis)
        command.add_argument("--baseline", dest="hedge_baseline_result", required=True)
        command.add_argument(
            "--candidate",
            dest="hedge_candidate_results",
            action="append",
            required=True,
            help="CUTOFF=PATH",
        )
        command.add_argument("--field", dest="hedge_analysis_fields", action="append")
        command.add_argument("--tolerance", dest="hedge_analysis_tolerance", default="0")
        command.add_argument("--output", dest="hedge_native_output")

    if "hedge-recursive-analysis" not in choices:
        command = subparsers.add_parser(
            "hedge-recursive-analysis",
            help="Compare Hedge terminal outputs across startup windows.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_recursive_file_analysis)
        command.add_argument(
            "--result",
            dest="hedge_recursive_results",
            action="append",
            required=True,
            help="STARTUP_CANDLES=PATH",
        )
        command.add_argument("--compare-tail", dest="hedge_compare_tail", type=int, default=1)
        command.add_argument("--tolerance", dest="hedge_analysis_tolerance", default="0")
        command.add_argument("--output", dest="hedge_native_output")

    if "hedge-research-optimize" not in choices:
        command = subparsers.add_parser(
            "hedge-research-optimize",
            help="Run resumable research-grade Hedge parameter optimization.",
            parents=[common_parser, strategy_parser],
        )
        command.set_defaults(func=_start_hedge_research_optimize)
        manager._build_args(optionlist=backtest_options, parser=command)
        command.add_argument("--hedge-study-name", dest="hedge_study_name")
        command.add_argument("--hedge-trials", dest="hedge_trials", type=int)
        command.add_argument(
            "--hedge-workers",
            dest="hedge_workers",
            type=int,
            help="Hedge worker policy: 0=adaptive, -1=resource max, 1=serial, N=upper bound.",
        )
        command.add_argument(
            "--hedge-sampler",
            dest="hedge_sampler",
            choices=("grid", "random"),
        )
        command.add_argument(
            "--hedge-optimization-output",
            dest="hedge_optimization_output",
        )

    if "hedge-hyperopt" not in choices:
        command = subparsers.add_parser(
            "hedge-hyperopt",
            help="Run Hedge-native parameter search.",
            parents=[common_parser, strategy_parser],
        )
        command.set_defaults(func=_start_hedge_native_hyperopt)
        manager._build_args(optionlist=backtest_options, parser=command)
        command.add_argument("--hedge-epochs", dest="hedge_epochs", type=int, default=10)
        command.add_argument(
            "--hedge-random-state", dest="hedge_random_state", type=int, default=42
        )
        command.add_argument(
            "--hedge-workers",
            dest="hedge_workers",
            type=int,
            default=0,
            help="Hedge worker policy: 0=adaptive, -1=resource max, 1=serial, N=upper bound.",
        )
        command.add_argument("--hedge-hyperopt-directory", dest="hedge_hyperopt_directory")
        command.add_argument("--output", dest="hedge_native_output")
    if "hedge-research-capabilities" not in choices:
        command = subparsers.add_parser(
            "hedge-research-capabilities",
            help="Show the 200-round Hedge research capability surface.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_research_capabilities)
        command.add_argument("--output", dest="hedge_research_output")

    if "hedge-research-validate" not in choices:
        command = subparsers.add_parser(
            "hedge-research-validate",
            help="Run the fail-fast 200-round Hedge research validation suite.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_research_validate)
        command.add_argument("--output", dest="hedge_research_output")

    if "hedge-risk-level-audit" not in choices:
        command = subparsers.add_parser(
            "hedge-risk-level-audit",
            help="Prove or reject learned Risk-Level RL dynamic position sizing on OOS data.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_risk_level_audit)
        command.add_argument("--features", dest="hedge_risk_audit_features", required=True)
        command.add_argument("--prices", dest="hedge_risk_audit_prices", required=True)
        command.add_argument("--model", dest="hedge_risk_audit_model", required=True)
        command.add_argument(
            "--model-type",
            dest="hedge_risk_audit_model_type",
            choices=("PPO", "A2C", "TRPO", "RecurrentPPO", "MaskablePPO"),
            default="PPO",
        )
        command.add_argument("--index-column", dest="hedge_risk_audit_index_column")
        command.add_argument("--train-start", dest="hedge_risk_audit_train_start")
        command.add_argument("--train-end", dest="hedge_risk_audit_train_end")
        command.add_argument("--output", dest="hedge_risk_audit_output", required=True)
        command.add_argument(
            "--drawdown-weight",
            dest="hedge_risk_audit_drawdown_weight",
            type=float,
            default=1.0,
        )
        command.add_argument(
            "--min-sizing-edge",
            dest="hedge_risk_audit_min_sizing_edge",
            type=float,
            default=0.0005,
        )
        command.add_argument(
            "--max-active-action-share",
            dest="hedge_risk_audit_max_active_action_share",
            type=float,
            default=0.90,
        )
        command.add_argument(
            "--min-distinct-levels",
            dest="hedge_risk_audit_min_distinct_levels",
            type=int,
            default=3,
        )
        command.add_argument(
            "--min-active-fraction",
            dest="hedge_risk_audit_min_active_fraction",
            type=float,
            default=0.02,
        )
        command.add_argument(
            "--min-nonzero-entropy",
            dest="hedge_risk_audit_min_nonzero_entropy",
            type=float,
            default=0.20,
        )
        command.add_argument(
            "--min-magnitude-change-fraction",
            dest="hedge_risk_audit_min_magnitude_change_fraction",
            type=float,
            default=0.005,
        )
        command.add_argument(
            "--shuffle-trials",
            dest="hedge_risk_audit_shuffle_trials",
            type=int,
            default=8,
        )
        command.add_argument(
            "--shuffle-quantile",
            dest="hedge_risk_audit_shuffle_quantile",
            type=float,
            default=0.75,
        )
        command.add_argument(
            "--permutation-trials",
            dest="hedge_risk_audit_permutation_trials",
            type=int,
            default=23,
        )
        command.add_argument(
            "--permutation-quantile",
            dest="hedge_risk_audit_permutation_quantile",
            type=float,
            default=0.75,
        )
        command.add_argument(
            "--max-permutation-exceedance",
            dest="hedge_risk_audit_max_permutation_exceedance",
            type=float,
            default=0.25,
        )
        command.add_argument(
            "--segment-count",
            dest="hedge_risk_audit_segment_count",
            type=int,
            default=4,
        )
        command.add_argument(
            "--min-segment-steps",
            dest="hedge_risk_audit_min_segment_steps",
            type=int,
            default=128,
        )
        command.add_argument(
            "--min-segments",
            dest="hedge_risk_audit_min_segments",
            type=int,
            default=2,
        )
        command.add_argument(
            "--min-segment-pass-ratio",
            dest="hedge_risk_audit_min_segment_pass_ratio",
            type=float,
            default=0.50,
        )
        command.add_argument(
            "--shuffle-seed",
            dest="hedge_risk_audit_shuffle_seed",
            type=int,
            default=20260815,
        )

    if "hedge-risk-walkforward-audit" not in choices:
        command = subparsers.add_parser(
            "hedge-risk-walkforward-audit",
            help="Aggregate sequential independently-trained Risk-Level OOS audits.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_risk_walkforward_audit)
        sources = command.add_mutually_exclusive_group(required=True)
        sources.add_argument("--audit", dest="hedge_risk_wf_audits", action="append")
        sources.add_argument("--audit-directory", dest="hedge_risk_wf_audit_directory")
        command.add_argument("--output", dest="hedge_risk_wf_output", required=True)
        command.add_argument("--min-folds", dest="hedge_risk_wf_min_folds", type=int, default=3)
        command.add_argument(
            "--min-pass-ratio",
            dest="hedge_risk_wf_min_pass_ratio",
            type=float,
            default=0.67,
        )
        command.add_argument(
            "--min-positive-fixed-ratio",
            dest="hedge_risk_wf_min_positive_fixed_ratio",
            type=float,
            default=0.67,
        )
        command.add_argument(
            "--min-positive-permutation-ratio",
            dest="hedge_risk_wf_min_positive_permutation_ratio",
            type=float,
            default=0.67,
        )
        command.add_argument(
            "--min-distinct-model-ratio",
            dest="hedge_risk_wf_min_distinct_model_ratio",
            type=float,
            default=1.0,
        )
        command.add_argument(
            "--min-median-fixed-edge",
            dest="hedge_risk_wf_min_median_fixed_edge",
            type=float,
            default=0.0,
        )
        command.add_argument(
            "--min-median-permutation-edge",
            dest="hedge_risk_wf_min_median_permutation_edge",
            type=float,
            default=0.0,
        )

    _register_runtime_acceptance_command(subparsers, common_parser, choices)
