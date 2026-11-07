"""Interfaccia comune dei broker e broker simulato locale.

Ogni broker reale (Alpaca, Interactive Brokers, Directa…) dovrà implementare
``Broker``; il resto del sistema parla solo con questa interfaccia, così
cambiare broker significa cambiare un adattatore.

``SimulatedBroker`` vive in memoria: esegue gli ordini ai prezzi forniti,
applica una commissione percentuale, rifiuta acquisti senza cash sufficiente
e vendite allo scoperto, ed è idempotente sul ``client_order_id``.
Non usa rete e non invia nulla a nessuno.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Protocol

from project_alpha.execution.orders import Order, Side

QUANTITY_TOLERANCE = 1e-9


class OrderStatus(StrEnum):
    FILLED = "FILLED"
    # Accettato dal broker ma non ancora eseguito (es. in coda per l'apertura).
    SUBMITTED = "SUBMITTED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    positions: Mapping[str, float]
    equity: float


@dataclass(frozen=True)
class OrderReport:
    client_order_id: str
    symbol: str
    side: Side
    quantity: float
    status: OrderStatus
    fill_price: float | None
    commission: float
    reason: str | None = None


class Broker(Protocol):
    def account(self, prices: Mapping[str, float]) -> AccountSnapshot: ...

    def submit(self, order: Order, *, fill_price: float) -> OrderReport: ...

    def order_report(self, client_order_id: str) -> OrderReport | None: ...


@dataclass
class SimulatedBroker:
    """Broker in memoria per test e prove del motore (nessuna rete)."""

    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    commission_rate: float = 0.001
    _reports: dict[str, OrderReport] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isfinite(self.cash) or self.cash < 0:
            raise ValueError("Il cash iniziale deve essere finito e non negativo.")

        if not 0 <= self.commission_rate < 0.1:
            raise ValueError("commission_rate non plausibile.")

        self.positions = {symbol.upper(): float(qty) for symbol, qty in self.positions.items()}

    def account(self, prices: Mapping[str, float]) -> AccountSnapshot:
        value = 0.0

        for symbol, quantity in self.positions.items():
            price = prices.get(symbol)

            if price is None:
                raise ValueError(f"Prezzo mancante per valutare la posizione {symbol}.")

            value += quantity * price

        return AccountSnapshot(
            cash=self.cash,
            positions=dict(self.positions),
            equity=self.cash + value,
        )

    def order_report(self, client_order_id: str) -> OrderReport | None:
        return self._reports.get(client_order_id)

    def submit(self, order: Order, *, fill_price: float) -> OrderReport:
        # Idempotenza: un id già visto restituisce lo stesso esito, senza rieseguire.
        previous = self._reports.get(order.client_order_id)
        if previous is not None:
            return previous

        def finish(status: OrderStatus, price: float | None, fee: float, reason=None):
            report = OrderReport(
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                status=status,
                fill_price=price,
                commission=fee,
                reason=reason,
            )
            self._reports[order.client_order_id] = report
            return report

        if order.quantity <= 0 or not isfinite(fill_price) or fill_price <= 0:
            return finish(OrderStatus.REJECTED, None, 0.0, "Quantità o prezzo non validi.")

        notional = order.quantity * fill_price
        fee = notional * self.commission_rate
        held = self.positions.get(order.symbol, 0.0)

        if order.side is Side.BUY:
            if notional + fee > self.cash + QUANTITY_TOLERANCE:
                return finish(OrderStatus.REJECTED, None, 0.0, "Cash insufficiente.")
            self.cash -= notional + fee
            self.positions[order.symbol] = held + order.quantity
        else:
            if order.quantity > held + QUANTITY_TOLERANCE:
                return finish(OrderStatus.REJECTED, None, 0.0, "Vendita allo scoperto non ammessa.")
            self.cash += notional - fee
            remaining = held - order.quantity
            if abs(remaining) <= QUANTITY_TOLERANCE:
                self.positions.pop(order.symbol, None)
            else:
                self.positions[order.symbol] = remaining

        return finish(OrderStatus.FILLED, fill_price, fee)
