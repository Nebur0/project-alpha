"""Pianificazione degli ordini a partire dai pesi target."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import floor, isfinite


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Order:
    client_order_id: str
    symbol: str
    side: Side
    quantity: float
    reference_price: float

    @property
    def notional(self) -> float:
        return self.quantity * self.reference_price


@dataclass(frozen=True)
class OrderPlanSettings:
    # Ordini sotto questo controvalore non valgono il costo: si saltano.
    min_order_notional: float = 25.0
    # Azioni frazionarie (Alpaca sì, molti broker europei no).
    allow_fractional: bool = False
    fractional_decimals: int = 4
    # Quota di equity lasciata in cash per costi e arrotondamenti.
    cash_buffer: float = 0.005

    def __post_init__(self) -> None:
        if self.min_order_notional < 0 or not isfinite(self.min_order_notional):
            raise ValueError("min_order_notional deve essere finito e non negativo.")

        if not 0.0 <= self.cash_buffer < 1.0:
            raise ValueError("cash_buffer deve essere compreso in [0, 1).")

        if not 0 <= self.fractional_decimals <= 9:
            raise ValueError("fractional_decimals deve essere tra 0 e 9.")


def client_order_id(
    strategy: str,
    as_of: str,
    symbol: str,
    side: Side,
    quantity: float,
) -> str:
    """Identificativo deterministico: lo stesso ordine ripetuto ha lo stesso id.

    Così un ribilanciamento rilanciato dopo un errore non duplica gli ordini
    (il broker rifiuta o ignora un id già visto).
    """
    payload = f"{strategy}|{as_of}|{symbol}|{side}|{quantity:.9f}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"pa-{digest}"


def _round_quantity(quantity: float, settings: OrderPlanSettings) -> float:
    if settings.allow_fractional:
        factor = 10**settings.fractional_decimals
        return floor(quantity * factor) / factor

    return float(floor(quantity))


def plan_orders(
    target_weights: Mapping[str, float],
    positions: Mapping[str, float],
    prices: Mapping[str, float],
    equity: float,
    *,
    strategy: str,
    as_of: str,
    settings: OrderPlanSettings | None = None,
) -> list[Order]:
    """Ordini per passare dalle posizioni attuali (quantità) ai pesi target.

    - le vendite vengono prima degli acquisti (liberano cash);
    - le quantità sono arrotondate per difetto (mai comprare più del target);
    - le vendite verso zero chiudono l'intera posizione;
    - i delta sotto ``min_order_notional`` vengono ignorati.
    """
    effective = settings or OrderPlanSettings()

    if not isfinite(equity) or equity <= 0:
        raise ValueError("equity deve essere positiva.")

    symbols = sorted(
        {symbol.upper() for symbol in target_weights} | {symbol.upper() for symbol in positions}
    )
    targets = {symbol.upper(): float(weight) for symbol, weight in target_weights.items()}
    held = {symbol.upper(): float(quantity) for symbol, quantity in positions.items()}
    price_map = {symbol.upper(): float(price) for symbol, price in prices.items()}
    investable = equity * (1.0 - effective.cash_buffer)

    sells: list[Order] = []
    buys: list[Order] = []

    for symbol in symbols:
        target_weight = targets.get(symbol, 0.0)

        if target_weight < 0:
            raise ValueError(f"Peso negativo per {symbol}: il motore è long-only.")

        price = price_map.get(symbol)
        current = held.get(symbol, 0.0)

        if price is None or not isfinite(price) or price <= 0:
            if target_weight == 0.0 and current == 0.0:
                continue
            raise ValueError(f"Prezzo mancante o non valido per {symbol}.")

        target_quantity = _round_quantity(target_weight * investable / price, effective)
        delta = target_quantity - current

        if target_weight == 0.0 and current > 0:
            delta = -current  # chiusura completa, anche di frazioni residue

        if abs(delta) * price < effective.min_order_notional and not (
            target_weight == 0.0 and current > 0
        ):
            continue

        if delta == 0:
            continue

        side = Side.BUY if delta > 0 else Side.SELL
        quantity = abs(delta)
        order = Order(
            client_order_id=client_order_id(strategy, as_of, symbol, side, quantity),
            symbol=symbol,
            side=side,
            quantity=quantity,
            reference_price=price,
        )
        (buys if side is Side.BUY else sells).append(order)

    return sells + buys
