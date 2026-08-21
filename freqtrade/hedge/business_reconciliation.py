"""Business-lot reconciliation for HEDGE managed positions and orders.

Exchange positions can be side-level aggregates while HEDGE owns the durable
business-lot ledger.  This module compares those authorities without attempting
historical attribution or automatic repair.  Any ambiguous managed identity is
reported and can be used by operations/readiness to fail closed for new risk.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from uuid import UUID

from freqtrade.hedge.contracts.business_identity import (
    BusinessIdentity,
    canonical_business_side,
    canonical_business_symbol,
)


ZERO = Decimal(0)
ONE = Decimal(1)


def _decimal(value: object, *, name: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return result


def _text(value: object, *, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    result = str(value).strip()
    if not result:
        if optional:
            return None
        raise ValueError(f"{name} is required")
    return result


@dataclass(frozen=True, slots=True)
class BusinessLotBalance:
    identity: BusinessIdentity
    open_quantity: Decimal
    bucket: str = "TACTICAL"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, BusinessIdentity):
            raise TypeError("identity must be BusinessIdentity")
        quantity = _decimal(self.open_quantity, name="open_quantity")
        if quantity < ZERO:
            raise ValueError("open_quantity must be nonnegative")
        bucket = str(getattr(self.bucket, "value", self.bucket)).strip().upper()
        if bucket not in {"CORE", "TACTICAL"}:
            raise ValueError("bucket must be CORE or TACTICAL")
        object.__setattr__(self, "open_quantity", quantity)
        object.__setattr__(self, "bucket", bucket)

    @property
    def business_lot_id(self) -> UUID:
        return self.identity.business_lot_id


@dataclass(frozen=True, slots=True)
class BusinessReconciliationIssue:
    code: str
    detail: str
    position_side: str | None = None
    local_amount: Decimal | None = None
    remote_amount: Decimal | None = None
    business_lot_id: str | None = None
    client_order_id: str | None = None

    def __post_init__(self) -> None:
        code = str(self.code).strip().upper()
        detail = str(self.detail).strip()
        if not code or not detail:
            raise ValueError("reconciliation issue code/detail are required")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "detail", detail)
        if self.position_side is not None:
            object.__setattr__(
                self,
                "position_side",
                canonical_business_side(self.position_side),
            )
        if self.local_amount is not None:
            object.__setattr__(
                self,
                "local_amount",
                _decimal(self.local_amount, name="local_amount"),
            )
        if self.remote_amount is not None:
            object.__setattr__(
                self,
                "remote_amount",
                _decimal(self.remote_amount, name="remote_amount"),
            )


@dataclass(frozen=True, slots=True)
class BusinessReconciliationResult:
    consistent: bool
    lot_sum_consistent: bool
    managed_order_identity_consistent: bool
    issues: tuple[BusinessReconciliationIssue, ...]
    long_lot_quantity: Decimal
    short_lot_quantity: Decimal
    remote_long_quantity: Decimal
    remote_short_quantity: Decimal
    managed_order_count: int
    managed_identity_count: int
    display_ids: tuple[str, ...]
    amount_tolerance: Decimal

    def __post_init__(self) -> None:
        for field_name in (
            "long_lot_quantity",
            "short_lot_quantity",
            "remote_long_quantity",
            "remote_short_quantity",
            "amount_tolerance",
        ):
            value = _decimal(getattr(self, field_name), name=field_name)
            if value < ZERO:
                raise ValueError(f"{field_name} must be nonnegative")
            object.__setattr__(self, field_name, value)
        if self.managed_order_count < 0 or self.managed_identity_count < 0:
            raise ValueError("managed order counters must be nonnegative")
        if self.managed_identity_count > self.managed_order_count:
            raise ValueError("managed identity count exceeds managed order count")
        if len(self.display_ids) != len(set(self.display_ids)):
            raise ValueError("business display ids must be unique")
        expected = not self.issues
        if self.consistent != expected:
            raise ValueError("consistent must agree with reconciliation issues")

    @property
    def managed_order_identity_coverage(self) -> Decimal:
        if self.managed_order_count == 0:
            return ONE
        return Decimal(self.managed_identity_count) / Decimal(self.managed_order_count)

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(item.code for item in self.issues)

    def operation_details(self, *, limit: int = 100) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        details: list[str] = []
        for issue in self.issues[:limit]:
            suffix = ""
            if issue.client_order_id:
                suffix = f":order={issue.client_order_id}"
            elif issue.business_lot_id:
                suffix = f":lot={issue.business_lot_id}"
            details.append(f"{issue.code}{suffix}")
        return tuple(details)


def _coerce_lot(value: object) -> BusinessLotBalance:
    if isinstance(value, BusinessLotBalance):
        return value
    if isinstance(value, tuple) and len(value) == 3:
        identity, bucket, quantity = value
        return BusinessLotBalance(
            identity=identity,  # type: ignore[arg-type]
            bucket=str(getattr(bucket, "value", bucket)),
            open_quantity=_decimal(quantity, name="open_quantity"),
        )
    identity = getattr(value, "business_identity", None)
    if not isinstance(identity, BusinessIdentity):
        raise TypeError("business lot is missing BusinessIdentity")
    quantity = getattr(value, "quantity", None)
    bucket = getattr(value, "bucket", "TACTICAL")
    return BusinessLotBalance(
        identity=identity,
        bucket=str(getattr(bucket, "value", bucket)),
        open_quantity=_decimal(quantity, name="open_quantity"),
    )


def _order_identity(order: object) -> tuple[object, BusinessIdentity | None]:
    intent = getattr(order, "intent", order)
    identity = getattr(intent, "business_identity", None)
    if identity is not None and not isinstance(identity, BusinessIdentity):
        raise TypeError("managed order business_identity is invalid")
    return intent, identity


def _order_client_id(order: object) -> str | None:
    raw = getattr(order, "client_order_id", None)
    if raw is None:
        raw = getattr(order, "order_id", None)
    return _text(raw, name="client_order_id", optional=True)


def _order_reduces_risk(intent: object) -> bool:
    value = getattr(intent, "reduces_risk", None)
    if isinstance(value, bool):
        return value
    action = str(getattr(getattr(intent, "action", None), "value", "")).upper()
    return action in {"REDUCE", "CLOSE", "UNSTUCK"} or bool(getattr(intent, "reduce_only", False))


def reconcile_business_state(  # noqa: C901 - explicit reconciliation boundary
    *,
    open_lots: Iterable[object],
    managed_orders: Iterable[object],
    remote_long_quantity: Decimal | str | int | float,
    remote_short_quantity: Decimal | str | int | float,
    amount_tolerance: Decimal | str | int | float = Decimal("0.00000001"),
    account_id: str | None = None,
    symbol: str | None = None,
) -> BusinessReconciliationResult:
    """Compare exact HEDGE business lots and managed-order identity coverage.

    ``remote_*_quantity`` is the side-level authority from the exchange/fake
    exchange.  No lot attribution is inferred from that aggregate.  HEDGE lot
    identity must already exist locally; ambiguity is reported rather than
    repaired here.
    """

    tolerance = _decimal(amount_tolerance, name="amount_tolerance")
    if tolerance < ZERO:
        raise ValueError("amount_tolerance must be nonnegative")
    remote_long = _decimal(remote_long_quantity, name="remote_long_quantity")
    remote_short = _decimal(remote_short_quantity, name="remote_short_quantity")
    if remote_long < ZERO or remote_short < ZERO:
        raise ValueError("remote quantities must be nonnegative")

    normalized_account = None if account_id is None else str(account_id).strip()
    if normalized_account == "":
        raise ValueError("account_id cannot be empty")
    normalized_symbol = None if symbol is None else canonical_business_symbol(symbol)

    issues: list[BusinessReconciliationIssue] = []
    lots: list[BusinessLotBalance] = []
    lot_ids: set[UUID] = set()
    display_ids: set[str] = set()
    long_lot = ZERO
    short_lot = ZERO

    for raw in open_lots:
        try:
            lot = _coerce_lot(raw)
        except (TypeError, ValueError) as exc:
            issues.append(
                BusinessReconciliationIssue(
                    "BUSINESS_LOT_INVALID",
                    f"Business lot cannot be reconciled: {exc}",
                )
            )
            continue
        identity = lot.identity
        if identity.business_lot_id in lot_ids:
            issues.append(
                BusinessReconciliationIssue(
                    "DUPLICATE_BUSINESS_LOT",
                    "Business lot appears more than once in the open-lot projection.",
                    position_side=identity.position_side,
                    business_lot_id=str(identity.business_lot_id),
                )
            )
            continue
        lot_ids.add(identity.business_lot_id)
        display_ids.add(identity.display_id)
        if normalized_account is not None and identity.account_id != normalized_account:
            issues.append(
                BusinessReconciliationIssue(
                    "BUSINESS_LOT_SCOPE_MISMATCH",
                    "Business lot account does not match the reconciliation scope.",
                    position_side=identity.position_side,
                    business_lot_id=str(identity.business_lot_id),
                )
            )
        if normalized_symbol is not None and identity.symbol != normalized_symbol:
            issues.append(
                BusinessReconciliationIssue(
                    "BUSINESS_LOT_SCOPE_MISMATCH",
                    "Business lot symbol does not match the reconciliation scope.",
                    position_side=identity.position_side,
                    business_lot_id=str(identity.business_lot_id),
                )
            )
        lots.append(lot)
        if identity.position_side == "LONG":
            long_lot += lot.open_quantity
        else:
            short_lot += lot.open_quantity

    lot_sum_issue_count_before = len(issues)
    if abs(long_lot - remote_long) > tolerance:
        issues.append(
            BusinessReconciliationIssue(
                "LOT_SUM_LONG_MISMATCH",
                "Sum of open LONG business lots differs from the side-level position.",
                position_side="LONG",
                local_amount=long_lot,
                remote_amount=remote_long,
            )
        )
    if abs(short_lot - remote_short) > tolerance:
        issues.append(
            BusinessReconciliationIssue(
                "LOT_SUM_SHORT_MISMATCH",
                "Sum of open SHORT business lots differs from the side-level position.",
                position_side="SHORT",
                local_amount=short_lot,
                remote_amount=remote_short,
            )
        )
    lot_sum_consistent = len(issues) == lot_sum_issue_count_before

    managed_count = 0
    managed_identity_count = 0
    managed_issue_count_before = len(issues)
    for order in managed_orders:
        managed_count += 1
        client_order_id = _order_client_id(order)
        try:
            intent, identity = _order_identity(order)
        except TypeError as exc:
            issues.append(
                BusinessReconciliationIssue(
                    "MANAGED_ORDER_IDENTITY_INVALID",
                    str(exc),
                    client_order_id=client_order_id,
                )
            )
            continue
        if identity is None:
            issues.append(
                BusinessReconciliationIssue(
                    "MANAGED_ORDER_IDENTITY_MISSING",
                    "Managed active order has no complete BusinessIdentity.",
                    client_order_id=client_order_id,
                )
            )
            continue
        managed_identity_count += 1
        display_ids.add(identity.display_id)
        try:
            identity.assert_matches(
                account_id=str(intent.account_id),
                symbol=str(intent.symbol),
                position_side=intent.position_side,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            issues.append(
                BusinessReconciliationIssue(
                    "MANAGED_ORDER_IDENTITY_SCOPE_MISMATCH",
                    f"Managed order identity scope is invalid: {exc}",
                    position_side=identity.position_side,
                    business_lot_id=str(identity.business_lot_id),
                    client_order_id=client_order_id,
                )
            )
            continue
        if normalized_account is not None and identity.account_id != normalized_account:
            issues.append(
                BusinessReconciliationIssue(
                    "MANAGED_ORDER_IDENTITY_SCOPE_MISMATCH",
                    "Managed order account does not match reconciliation scope.",
                    position_side=identity.position_side,
                    business_lot_id=str(identity.business_lot_id),
                    client_order_id=client_order_id,
                )
            )
        if normalized_symbol is not None and identity.symbol != normalized_symbol:
            issues.append(
                BusinessReconciliationIssue(
                    "MANAGED_ORDER_IDENTITY_SCOPE_MISMATCH",
                    "Managed order symbol does not match reconciliation scope.",
                    position_side=identity.position_side,
                    business_lot_id=str(identity.business_lot_id),
                    client_order_id=client_order_id,
                )
            )
        if _order_reduces_risk(intent) and identity.business_lot_id not in lot_ids:
            issues.append(
                BusinessReconciliationIssue(
                    "TARGET_BUSINESS_LOT_NOT_OPEN",
                    "Reduce-only managed order targets a business lot that is not open.",
                    position_side=identity.position_side,
                    business_lot_id=str(identity.business_lot_id),
                    client_order_id=client_order_id,
                )
            )

    managed_consistent = len(issues) == managed_issue_count_before
    return BusinessReconciliationResult(
        consistent=not issues,
        lot_sum_consistent=lot_sum_consistent,
        managed_order_identity_consistent=managed_consistent,
        issues=tuple(issues),
        long_lot_quantity=long_lot,
        short_lot_quantity=short_lot,
        remote_long_quantity=remote_long,
        remote_short_quantity=remote_short,
        managed_order_count=managed_count,
        managed_identity_count=managed_identity_count,
        display_ids=tuple(sorted(display_ids)),
        amount_tolerance=tolerance,
    )


def business_reconciliation_log_payload(
    result: BusinessReconciliationResult,
) -> dict[str, object]:
    if not isinstance(result, BusinessReconciliationResult):
        raise TypeError("result must be BusinessReconciliationResult")
    return {
        "business_reconciliation_consistent": result.consistent,
        "business_lot_sum_consistent": result.lot_sum_consistent,
        "managed_order_identity_consistent": result.managed_order_identity_consistent,
        "managed_order_count": result.managed_order_count,
        "managed_identity_count": result.managed_identity_count,
        "managed_identity_coverage": str(result.managed_order_identity_coverage),
        "business_long_lot_quantity": str(result.long_lot_quantity),
        "business_short_lot_quantity": str(result.short_lot_quantity),
        "remote_long_quantity": str(result.remote_long_quantity),
        "remote_short_quantity": str(result.remote_short_quantity),
        "business_reconciliation_issue_codes": result.issue_codes,
        "business_trade_display_ids": result.display_ids,
    }
