"""Read-only reconciliation for business-lot protection coverage and state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

from .contracts import (
    ONE,
    ZERO,
    BusinessLotProtectionSnapshot,
    ProtectionGroup,
    ProtectionGroupStatus,
    ProtectionKind,
    ProtectionLegStatus,
)


@dataclass(frozen=True, slots=True)
class ProtectionReconciliationIssue:
    code: str
    detail: str
    business_lot_id: str | None = None
    protection_group_id: str | None = None

    def __post_init__(self) -> None:
        code = str(self.code).strip().upper()
        detail = str(self.detail).strip()
        if not code or not detail:
            raise ValueError("protection reconciliation issue requires code/detail")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "detail", detail)


@dataclass(frozen=True, slots=True)
class ProtectionReconciliationResult:
    consistent: bool
    issues: tuple[ProtectionReconciliationIssue, ...]
    open_lot_count: int
    protected_lot_count: int
    stop_covered_lot_count: int
    active_group_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "open_lot_count",
            "protected_lot_count",
            "stop_covered_lot_count",
            "active_group_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative")
        if self.protected_lot_count > self.open_lot_count:
            raise ValueError("protected_lot_count exceeds open_lot_count")
        if self.stop_covered_lot_count > self.protected_lot_count:
            raise ValueError("stop_covered_lot_count exceeds protected_lot_count")
        if self.consistent != (not self.issues):
            raise ValueError("consistent must agree with issues")

    @property
    def protection_coverage(self) -> Decimal:
        if self.open_lot_count == 0:
            return ONE
        return Decimal(self.protected_lot_count) / Decimal(self.open_lot_count)

    @property
    def stop_coverage(self) -> Decimal:
        if self.open_lot_count == 0:
            return ONE
        return Decimal(self.stop_covered_lot_count) / Decimal(self.open_lot_count)

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


def reconcile_protection_state(  # noqa: C901 - complete protection integrity pass
    *,
    open_lots: Iterable[BusinessLotProtectionSnapshot],
    groups: Iterable[ProtectionGroup],
    require_protection_for_all: bool = True,
    require_stop_for_all: bool = True,
) -> ProtectionReconciliationResult:
    if not isinstance(require_protection_for_all, bool):
        raise TypeError("require_protection_for_all must be a boolean")
    if not isinstance(require_stop_for_all, bool):
        raise TypeError("require_stop_for_all must be a boolean")

    issues: list[ProtectionReconciliationIssue] = []
    lots: dict[UUID, BusinessLotProtectionSnapshot] = {}
    for lot in open_lots:
        if not isinstance(lot, BusinessLotProtectionSnapshot):
            raise TypeError("open_lots must contain BusinessLotProtectionSnapshot")
        lot_id = lot.identity.business_lot_id
        if lot.open_quantity <= ZERO:
            continue
        if lot_id in lots:
            issues.append(
                ProtectionReconciliationIssue(
                    "DUPLICATE_OPEN_BUSINESS_LOT",
                    "Open business lot appears more than once in protection reconciliation.",
                    business_lot_id=str(lot_id),
                )
            )
        lots[lot_id] = lot

    active_by_lot: dict[UUID, ProtectionGroup] = {}
    stop_covered: set[UUID] = set()
    active_count = 0
    for group in groups:
        if not isinstance(group, ProtectionGroup):
            raise TypeError("groups must contain ProtectionGroup")
        if group.status.terminal:
            continue
        active_count += 1
        lot_id = group.business_lot_id
        if lot_id not in lots:
            issues.append(
                ProtectionReconciliationIssue(
                    "PROTECTION_TARGET_LOT_NOT_OPEN",
                    "Active protection group targets a business lot that is not open.",
                    business_lot_id=str(lot_id),
                    protection_group_id=str(group.protection_group_id),
                )
            )
            continue
        if lot_id in active_by_lot:
            issues.append(
                ProtectionReconciliationIssue(
                    "MULTIPLE_ACTIVE_PROTECTION_GROUPS",
                    "One business lot has multiple active protection groups.",
                    business_lot_id=str(lot_id),
                    protection_group_id=str(group.protection_group_id),
                )
            )
            continue
        active_by_lot[lot_id] = group
        if group.business_identity != lots[lot_id].identity:
            issues.append(
                ProtectionReconciliationIssue(
                    "PROTECTION_IDENTITY_MISMATCH",
                    "Protection group identity differs from the target business lot.",
                    business_lot_id=str(lot_id),
                    protection_group_id=str(group.protection_group_id),
                )
            )
        active_execution = sum(
            leg.status
            in {
                ProtectionLegStatus.TRIGGERED,
                ProtectionLegStatus.SUBMITTED,
                ProtectionLegStatus.PARTIAL,
            }
            for leg in group.legs
        )
        if group.status is ProtectionGroupStatus.ACTIVE and active_execution != 0:
            issues.append(
                ProtectionReconciliationIssue(
                    "PROTECTION_GROUP_STATE_INVALID",
                    "ACTIVE group contains an in-flight protection execution.",
                    business_lot_id=str(lot_id),
                    protection_group_id=str(group.protection_group_id),
                )
            )
        if group.status is ProtectionGroupStatus.EXECUTING and active_execution != 1:
            issues.append(
                ProtectionReconciliationIssue(
                    "PROTECTION_GROUP_STATE_INVALID",
                    "EXECUTING group must contain exactly one in-flight execution.",
                    business_lot_id=str(lot_id),
                    protection_group_id=str(group.protection_group_id),
                )
            )
        has_stop = any(
            leg.kind in {ProtectionKind.STOP_LOSS, ProtectionKind.TRAILING_STOP}
            and leg.status not in {ProtectionLegStatus.CANCELED, ProtectionLegStatus.FAILED}
            for leg in group.legs
        )
        if has_stop:
            stop_covered.add(lot_id)
        elif group.require_stop or require_stop_for_all:
            issues.append(
                ProtectionReconciliationIssue(
                    "BUSINESS_LOT_STOP_COVERAGE_MISSING",
                    "Open business lot has no active stop-loss or trailing-stop protection.",
                    business_lot_id=str(lot_id),
                    protection_group_id=str(group.protection_group_id),
                )
            )

    if require_protection_for_all:
        for lot_id in sorted(set(lots) - set(active_by_lot), key=str):
            issues.append(
                ProtectionReconciliationIssue(
                    "BUSINESS_LOT_PROTECTION_MISSING",
                    "Open business lot has no active protection group.",
                    business_lot_id=str(lot_id),
                )
            )

    return ProtectionReconciliationResult(
        consistent=not issues,
        issues=tuple(issues),
        open_lot_count=len(lots),
        protected_lot_count=len(active_by_lot),
        stop_covered_lot_count=len(stop_covered),
        active_group_count=active_count,
    )
