"""Adattatore Alpaca **solo paper trading** (soldi finti).

Implementa ``Broker`` verso l'API di trading paper di Alpaca. Barriere:

- l'indirizzo è fisso su ``paper-api.alpaca.markets``: il conto reale non è
  raggiungibile da questo modulo, nemmeno per errore di configurazione;
- le chiavi devono essere chiavi paper (prefisso ``PK``); le chiavi live
  (``AK``) vengono rifiutate prima di qualunque chiamata;
- solo ordini a mercato, ``time_in_force="day"``: inviati dopo la chiusura
  restano in coda e vengono eseguiti all'apertura successiva;
- ``client_order_id`` deterministico: un ordine già inviato non viene
  duplicato (Alpaca rifiuta l'id ripetuto e qui si recupera l'originale);
- un invio con esito incerto (errore di rete) non viene ripetuto alla cieca:
  si cerca l'ordine per id, altrimenti lo si segnala come rifiutato e il
  motore si ferma.

Le chiavi arrivano da ``Settings`` (variabili d'ambiente / ``.env``) e non
compaiono mai in log o messaggi d'errore.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from math import isfinite
from typing import Any

import requests

from project_alpha.config import Settings
from project_alpha.data.alpaca import _get_with_retries
from project_alpha.execution.broker import AccountSnapshot, OrderReport, OrderStatus
from project_alpha.execution.orders import Order, Side

PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
PAPER_KEY_PREFIX = "PK"

_FILLED = {"filled"}
_REJECTED = {"rejected", "canceled", "expired", "suspended", "stopped", "done_for_day"}


class AlpacaPaperError(RuntimeError):
    """Errore dell'API di trading paper."""


def _status(raw: str | None) -> OrderStatus:
    value = (raw or "").lower()

    if value in _FILLED:
        return OrderStatus.FILLED

    if value in _REJECTED:
        return OrderStatus.REJECTED

    return OrderStatus.SUBMITTED


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if isfinite(number) else None


class AlpacaPaperBroker:
    """Broker Alpaca paper. Nessun parametro permette di puntare al conto reale."""

    def __init__(
        self,
        settings: Settings,
        *,
        session: requests.Session | None = None,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.alpaca_api_key.startswith(PAPER_KEY_PREFIX):
            raise AlpacaPaperError(
                "La chiave Alpaca non è una chiave paper (deve iniziare con 'PK'). "
                "Questo modulo lavora solo sul conto paper: usare le chiavi paper."
            )

        self._headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
            "accept": "application/json",
        }
        self._session = session if session is not None else requests.Session()
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._sleep = sleep
        self._reports: dict[str, OrderReport] = {}

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, str | int] | None = None) -> Any:
        try:
            response = _get_with_retries(
                self._session,
                f"{PAPER_TRADING_URL}{path}",
                headers=self._headers,
                params=params or {},
                max_retries=self._max_retries,
                backoff_seconds=self._backoff,
                sleep=self._sleep,
            )
        except RuntimeError as exc:
            raise AlpacaPaperError(str(exc)) from exc

        return response.json()

    # ------------------------------------------------------------------
    # Conto
    # ------------------------------------------------------------------

    def _account_payload(self) -> dict[str, Any]:
        payload = self._get("/v2/account")

        if payload.get("trading_blocked") or payload.get("account_blocked"):
            raise AlpacaPaperError("Conto paper bloccato da Alpaca: nessuna operazione.")

        return payload

    def _positions_payload(self) -> list[dict[str, Any]]:
        return list(self._get("/v2/positions") or [])

    def market_prices(self) -> dict[str, float]:
        """Prezzi correnti delle posizioni detenute (per valutare simboli fuori universo)."""
        prices: dict[str, float] = {}

        for position in self._positions_payload():
            price = _float(position.get("current_price"))
            if price is not None and price > 0:
                prices[str(position["symbol"]).upper()] = price

        return prices

    def account(self, prices: Mapping[str, float]) -> AccountSnapshot:
        """Cash e posizioni dal broker, valutati ai prezzi forniti."""
        payload = self._account_payload()
        cash = _float(payload.get("cash"))

        if cash is None:
            raise AlpacaPaperError("Cash del conto non leggibile.")

        positions: dict[str, float] = {}
        value = 0.0

        for position in self._positions_payload():
            symbol = str(position["symbol"]).upper()
            quantity = _float(position.get("qty"))

            if quantity is None:
                raise AlpacaPaperError(f"Quantità non leggibile per {symbol}.")

            if quantity < 0:
                raise AlpacaPaperError(
                    f"Posizione short inattesa su {symbol}: il motore è long-only."
                )

            price = prices.get(symbol)
            if price is None:
                raise ValueError(f"Prezzo mancante per valutare la posizione {symbol}.")

            positions[symbol] = quantity
            value += quantity * price

        return AccountSnapshot(cash=cash, positions=positions, equity=cash + value)

    def open_orders(self) -> list[dict[str, Any]]:
        return list(self._get("/v2/orders", {"status": "open", "limit": 500}) or [])

    # ------------------------------------------------------------------
    # Ordini
    # ------------------------------------------------------------------

    def _report_from_payload(self, order: Order, payload: Mapping[str, Any]) -> OrderReport:
        status = _status(payload.get("status"))
        fill_price = _float(payload.get("filled_avg_price"))
        reason = (
            None if status is not OrderStatus.REJECTED else f"Stato Alpaca: {payload.get('status')}"
        )

        return OrderReport(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            status=status,
            fill_price=fill_price,
            commission=0.0,
            reason=reason,
        )

    def _find_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        try:
            return self._get(
                "/v2/orders:by_client_order_id",
                {"client_order_id": client_order_id},
            )
        except AlpacaPaperError:
            return None

    def order_report(self, client_order_id: str) -> OrderReport | None:
        return self._reports.get(client_order_id)

    def submit(self, order: Order, *, fill_price: float) -> OrderReport:
        """Invia un ordine a mercato (``fill_price`` è solo il prezzo di riferimento)."""
        previous = self._reports.get(order.client_order_id)
        if previous is not None:
            return previous

        def remember(report: OrderReport) -> OrderReport:
            self._reports[order.client_order_id] = report
            return report

        if order.quantity <= 0:
            return remember(
                OrderReport(
                    order.client_order_id,
                    order.symbol,
                    order.side,
                    order.quantity,
                    OrderStatus.REJECTED,
                    None,
                    0.0,
                    "Quantità non valida.",
                )
            )

        body = {
            "symbol": order.symbol,
            "qty": f"{order.quantity:.9f}".rstrip("0").rstrip("."),
            "side": "buy" if order.side is Side.BUY else "sell",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": order.client_order_id,
        }

        try:
            response = self._session.post(
                f"{PAPER_TRADING_URL}/v2/orders",
                headers=self._headers,
                json=body,
                timeout=45,
            )
        except (requests.ConnectionError, requests.Timeout):
            # Esito incerto: mai reinviare alla cieca, cercare l'ordine per id.
            existing = self._find_by_client_id(order.client_order_id)
            if existing is not None:
                return remember(self._report_from_payload(order, existing))
            return remember(
                OrderReport(
                    order.client_order_id,
                    order.symbol,
                    order.side,
                    order.quantity,
                    OrderStatus.REJECTED,
                    None,
                    0.0,
                    "Esito dell'invio incerto (errore di rete): verificare il conto.",
                )
            )

        if response.status_code == 422 and "client_order_id" in response.text:
            existing = self._find_by_client_id(order.client_order_id)
            if existing is not None:
                return remember(self._report_from_payload(order, existing))

        if response.status_code >= 400:
            return remember(
                OrderReport(
                    order.client_order_id,
                    order.symbol,
                    order.side,
                    order.quantity,
                    OrderStatus.REJECTED,
                    None,
                    0.0,
                    f"Alpaca HTTP {response.status_code}: {response.text[:300]}",
                )
            )

        return remember(self._report_from_payload(order, response.json()))
