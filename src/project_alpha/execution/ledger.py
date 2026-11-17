"""Conto paper interno (registro locale), per strategie in prova in parallelo.

Serve quando una seconda strategia non può condividere il conto Alpaca con
la prima: le posizioni si mescolerebbero. Il registro tiene cash, posizioni
e ordini in un file JSON e simula l'esecuzione in modo prudente:

- un ordine inviato oggi si esegue al **prezzo di apertura** della prima
  seduta successiva presente nei dati (come gli ordini serali su Alpaca);
- le vendite si regolano prima degli acquisti;
- un acquisto senza cash sufficiente viene ridotto al cash disponibile;
- commissione proporzionale (default 5 punti base) per coprire lo spread.

Nessuna rete e nessun broker: è un conto finto per definizione.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from project_alpha.execution.broker import AccountSnapshot, OrderReport, OrderStatus
from project_alpha.execution.orders import Order, Side
from project_alpha.io_utils import write_text_atomic

MARKET_TZ = "America/New_York"
TOLERANCE = 1e-9


def _session(moment: datetime | pd.Timestamp) -> str:
    return pd.Timestamp(moment).tz_convert(MARKET_TZ).date().isoformat()


@dataclass
class LedgerBroker:
    path: Path
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    pending: list[dict[str, Any]] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    commission_rate: float = 0.0005
    now: datetime = field(default_factory=lambda: datetime.now(UTC))

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        initial_cash: float,
        commission_rate: float = 0.0005,
        now: datetime | None = None,
    ) -> LedgerBroker:
        file = Path(path)
        clock = now or datetime.now(UTC)
        if not file.exists():
            return cls(file, float(initial_cash), commission_rate=commission_rate, now=clock)
        data = json.loads(file.read_text(encoding="utf-8"))
        return cls(
            file,
            float(data["cash"]),
            {k: float(v) for k, v in data.get("positions", {}).items()},
            list(data.get("pending", [])),
            list(data.get("fills", [])),
            commission_rate=commission_rate,
            now=clock,
        )

    def save(self) -> None:
        payload = {
            "cash": self.cash,
            "positions": self.positions,
            "pending": self.pending,
            "fills": self.fills[-500:],
        }
        write_text_atomic(self.path, json.dumps(payload, indent=1))

    # ------------------------------------------------------------------
    # Interfaccia Broker
    # ------------------------------------------------------------------

    def account(self, prices: Mapping[str, float]) -> AccountSnapshot:
        value = 0.0
        for symbol, quantity in self.positions.items():
            price = prices.get(symbol)
            if price is None:
                raise ValueError(f"Prezzo mancante per valutare la posizione {symbol}.")
            value += quantity * price
        return AccountSnapshot(
            cash=self.cash, positions=dict(self.positions), equity=self.cash + value
        )

    def open_orders(self) -> list[dict[str, Any]]:
        return list(self.pending)

    def order_report(self, client_order_id: str) -> OrderReport | None:
        return None

    def submit(self, order: Order, *, fill_price: float) -> OrderReport:
        if any(p["client_order_id"] == order.client_order_id for p in self.pending):
            status = OrderStatus.SUBMITTED
        elif order.quantity <= 0:
            return OrderReport(
                order.client_order_id,
                order.symbol,
                order.side,
                order.quantity,
                OrderStatus.REJECTED,
                None,
                0.0,
                "Quantità non valida.",
            )
        else:
            self.pending.append(
                {
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "side": str(order.side),
                    "quantity": order.quantity,
                    "session": _session(self.now),
                }
            )
            status = OrderStatus.SUBMITTED
        return OrderReport(
            order.client_order_id, order.symbol, order.side, order.quantity, status, None, 0.0
        )

    # ------------------------------------------------------------------
    # Esecuzione degli ordini in coda
    # ------------------------------------------------------------------

    def settle(self, bars: pd.DataFrame) -> int:
        """Esegue gli ordini in coda all'apertura della seduta successiva.

        Restituisce il numero di ordini eseguiti.
        """
        if not self.pending:
            return 0

        data = bars.copy()
        data["symbol"] = data["symbol"].astype(str).str.upper()
        data["session"] = (
            pd.to_datetime(data["timestamp"], utc=True).dt.tz_convert(MARKET_TZ).dt.date.astype(str)
        )
        opens = data.set_index(["symbol", "session"])["open"].sort_index()

        executed = 0
        remaining: list[dict[str, Any]] = []
        ordered = sorted(self.pending, key=lambda p: p["side"] != str(Side.SELL))

        for item in ordered:
            symbol = item["symbol"]
            try:
                series = opens.loc[symbol]
            except KeyError:
                remaining.append(item)
                continue
            later = series[series.index > item["session"]]
            if later.empty:
                remaining.append(item)
                continue

            session, price = later.index[0], float(later.iloc[0])
            quantity = float(item["quantity"])
            held = self.positions.get(symbol, 0.0)

            if item["side"] == str(Side.SELL):
                quantity = min(quantity, held)
                fee = quantity * price * self.commission_rate
                self.cash += quantity * price - fee
                left = held - quantity
                if left <= TOLERANCE:
                    self.positions.pop(symbol, None)
                else:
                    self.positions[symbol] = left
            else:
                affordable = self.cash / (price * (1 + self.commission_rate))
                quantity = max(0.0, min(quantity, affordable))
                fee = quantity * price * self.commission_rate
                self.cash -= quantity * price + fee
                if quantity > TOLERANCE:
                    self.positions[symbol] = held + quantity

            self.fills.append(
                {
                    "client_order_id": item["client_order_id"],
                    "symbol": symbol,
                    "side": item["side"],
                    "quantity": round(quantity, 6),
                    "price": price,
                    "fee": round(fee, 4),
                    "session": session,
                }
            )
            executed += 1

        self.pending = remaining
        return executed
