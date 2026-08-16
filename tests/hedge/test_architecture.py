from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _python_files(relative: str) -> tuple[Path, ...]:
    return tuple((PROJECT_ROOT / relative).rglob("*.py"))


def test_zero_has_one_canonical_definition() -> None:
    definitions = []
    for path in _python_files("freqtrade/hedge"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id == "ZERO"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "Decimal"
            ):
                definitions.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert definitions == ["freqtrade/hedge/numeric.py"]


def test_contracts_public_import_does_not_load_execution() -> None:
    source = (PROJECT_ROOT / "freqtrade/hedge/contracts/__init__.py").read_text(encoding="utf-8")
    assert "freqtrade.hedge.execution" not in source


def test_contracts_adapters_do_not_import_execution() -> None:
    source = (PROJECT_ROOT / "freqtrade/hedge/contracts/adapters.py").read_text(encoding="utf-8")
    assert "freqtrade.hedge.execution" not in source


def test_simulation_and_strategies_are_not_mutually_dependent() -> None:
    simulation_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in _python_files("freqtrade/hedge/simulation")
    )
    strategy_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in _python_files("freqtrade/hedge/strategies")
    )
    assert "freqtrade.hedge.integration" not in simulation_sources
    assert "freqtrade.hedge.simulation" not in strategy_sources


def test_execution_does_not_import_integration() -> None:
    execution_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in _python_files("freqtrade/hedge/execution")
    )
    assert "freqtrade.hedge.integration" not in execution_sources


def test_boundary_errors_share_reason_code_hierarchy() -> None:
    from freqtrade.hedge.contracts.errors import HedgeContractError, ReasonCode
    from freqtrade.hedge.errors import (
        HedgeContractViolation,
        HedgeDefinitiveError,
        HedgeError,
        HedgeRetryableError,
    )
    from freqtrade.hedge.exchange.rate_limit import (
        BinanceDataError,
        BinancePermissionError,
        BinanceRateLimitError,
        BinanceRetryableError,
    )

    contract_error = HedgeContractError(ReasonCode.UNKNOWN_ORDER)
    assert isinstance(contract_error, HedgeContractViolation)
    assert isinstance(contract_error, HedgeError)
    assert contract_error.reason_code == ReasonCode.UNKNOWN_ORDER
    assert issubclass(BinanceDataError, HedgeDefinitiveError)
    assert issubclass(BinancePermissionError, HedgeDefinitiveError)
    assert issubclass(BinanceRateLimitError, HedgeRetryableError)
    assert issubclass(BinanceRetryableError, HedgeRetryableError)
    assert isinstance(BinanceDataError("malformed"), HedgeError)
    assert isinstance(BinanceRetryableError("temporary", retryable=True), HedgeError)


def test_canonical_hedge_error_module_has_no_parallel_exception_root() -> None:
    import inspect

    import freqtrade.hedge.errors as errors

    for name, candidate in vars(errors).items():
        if name.startswith("_") or not inspect.isclass(candidate):
            continue
        if candidate.__module__ != errors.__name__:
            continue
        assert issubclass(candidate, errors.HedgeError), name
