from __future__ import annotations

import ast
from functools import cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEDGE_ROOT = PROJECT_ROOT / "freqtrade/hedge"


def _python_files(relative: str) -> tuple[Path, ...]:
    return tuple((PROJECT_ROOT / relative).rglob("*.py"))


def _module_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


@cache
def _direct_hedge_imports(path: Path) -> set[str]:
    """Return statically-resolved imports that stay inside ``freqtrade.hedge``."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current = _module_name(path)
    package = current if path.name == "__init__.py" else current.rpartition(".")[0]
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            parts = package.split(".")
            base = ".".join(parts[: len(parts) - node.level + 1])
            module = f"{base}.{node.module}" if node.module else base
        else:
            module = node.module or ""
        if module:
            imports.add(module)
        if node.module is None:
            imports.update(f"{module}.{alias.name}" for alias in node.names)
    return {module for module in imports if module.startswith("freqtrade.hedge")}


@cache
def _hedge_import_graph() -> dict[str, set[str]]:
    return {
        _module_name(path): _direct_hedge_imports(path)
        for path in HEDGE_ROOT.rglob("*.py")
    }


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
    graph = _hedge_import_graph()
    pending = ["freqtrade.hedge.contracts"]
    visited = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        pending.extend(graph.get(module, ()))
    assert not any(module.startswith("freqtrade.hedge.execution") for module in visited)


def test_contracts_adapters_do_not_import_execution() -> None:
    imports = _direct_hedge_imports(
        PROJECT_ROOT / "freqtrade/hedge/contracts/adapters.py"
    )
    assert not any(module.startswith("freqtrade.hedge.execution") for module in imports)


def test_simulation_and_strategies_are_not_mutually_dependent() -> None:
    graph = _hedge_import_graph()
    simulation_imports = {
        imported
        for module, imports in graph.items()
        if module.startswith("freqtrade.hedge.simulation")
        for imported in imports
    }
    strategy_imports = {
        imported
        for module, imports in graph.items()
        if module.startswith("freqtrade.hedge.strategies")
        for imported in imports
    }
    assert not any(
        module.startswith("freqtrade.hedge.integration") for module in simulation_imports
    )
    assert not any(
        module.startswith("freqtrade.hedge.simulation") for module in strategy_imports
    )


def test_execution_does_not_import_integration() -> None:
    graph = _hedge_import_graph()
    execution_imports = {
        imported
        for module, imports in graph.items()
        if module.startswith("freqtrade.hedge.execution")
        for imported in imports
    }
    assert not any(
        module.startswith("freqtrade.hedge.integration") for module in execution_imports
    )


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


def test_domain_error_types_preserve_the_canonical_hierarchy() -> None:
    """Keep reportable domain failures catchable as ``HedgeError``.

    HPRL retains its own public error family because it is an independently usable
    training submodule.  The main Hedge runtime must not grow another root.
    """

    from freqtrade.hedge.control.service import (
        ControlConfirmationError,
        ControlPermissionError,
    )
    from freqtrade.hedge.deployment.config import DeploymentConfigError
    from freqtrade.hedge.deployment.readiness import DeploymentReadinessError
    from freqtrade.hedge.deployment.single_instance import InstanceLockError
    from freqtrade.hedge.errors import HedgeError
    from freqtrade.hedge.execution.binance_usdm_adapter import BinanceExecutionApiError
    from freqtrade.hedge.execution.kill_switch import ExecutionHaltedError
    from freqtrade.hedge.execution.production_gate import ExecutionWriteLockedError
    from freqtrade.hedge.execution.service import (
        DefinitiveExchangeOperationError,
        ExecutionBlockedError,
    )
    from freqtrade.hedge.integration.central_source import IntegrationSafetyError
    from freqtrade.hedge.operations.state import StateCorruptionError
    from freqtrade.hedge.production.closed_loop import ClosedLoopJournalConcurrencyError
    from freqtrade.hedge.production.evidence import EvidenceConcurrencyError

    domain_errors = (
        ControlConfirmationError,
        ControlPermissionError,
        DeploymentConfigError,
        DeploymentReadinessError,
        InstanceLockError,
        BinanceExecutionApiError,
        ExecutionHaltedError,
        ExecutionWriteLockedError,
        DefinitiveExchangeOperationError,
        ExecutionBlockedError,
        IntegrationSafetyError,
        StateCorruptionError,
        ClosedLoopJournalConcurrencyError,
        EvidenceConcurrencyError,
    )
    assert all(issubclass(error, HedgeError) for error in domain_errors)
    assert issubclass(ControlPermissionError, PermissionError)
    assert issubclass(ExecutionWriteLockedError, PermissionError)
