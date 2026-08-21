"""SQL source adapter for Business Identity reconciliation."""

from __future__ import annotations

from decimal import Decimal

from freqtrade.hedge.business_reconciliation import (
    BusinessLotBalance,
    BusinessReconciliationResult,
    reconcile_business_state,
)
from freqtrade.persistence.hedge_business_identity import SqlBusinessIdentityAllocator
from freqtrade.persistence.hedge_execution_adapters import SqlExecutionStore


class SqlBusinessReconciliationSource:
    """Load exact business lots/orders from HEDGE SQL and compare side totals.

    The exchange/fake-exchange side quantities are supplied by the caller.  This
    adapter never invents lot attribution from those aggregate quantities.
    """

    def __init__(self, session_factory: object) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory
        self._allocator = SqlBusinessIdentityAllocator(session_factory)
        self._orders = SqlExecutionStore(session_factory)

    def reconcile(
        self,
        *,
        account_id: str,
        symbol: str,
        remote_long_quantity: Decimal | str | int | float,
        remote_short_quantity: Decimal | str | int | float,
        amount_tolerance: Decimal | str | int | float = Decimal("0.00000001"),
    ) -> BusinessReconciliationResult:
        balances: list[BusinessLotBalance] = []
        for side in ("LONG", "SHORT"):
            balances.extend(
                BusinessLotBalance(identity=identity, bucket=bucket, open_quantity=quantity)
                for identity, bucket, quantity in self._allocator.list_open_lots(
                    account_id=account_id,
                    symbol=symbol,
                    position_side=side,
                )
            )
        managed_orders = self._orders.list_orders(
            account_id=account_id,
            symbol=symbol,
            include_terminal=False,
        )
        return reconcile_business_state(
            open_lots=balances,
            managed_orders=managed_orders,
            remote_long_quantity=remote_long_quantity,
            remote_short_quantity=remote_short_quantity,
            amount_tolerance=amount_tolerance,
            account_id=account_id,
            symbol=symbol,
        )
