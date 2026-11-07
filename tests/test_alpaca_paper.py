"""Test dell'adattatore Alpaca paper con una sessione HTTP finta (nessuna rete)."""

import pytest
import requests

from project_alpha.config import Settings
from project_alpha.execution.alpaca_paper import (
    PAPER_TRADING_URL,
    AlpacaPaperBroker,
    AlpacaPaperError,
)
from project_alpha.execution.broker import OrderStatus
from project_alpha.execution.orders import Order, Side

PAPER = Settings(alpaca_api_key="PKTEST", alpaca_api_secret="secret")


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or ("" if payload is None else str(payload))
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, routes=None, post=None):
        self.routes = routes or {}
        self.post_response = post
        self.calls = []
        self.posted = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        assert url.startswith(PAPER_TRADING_URL)
        path = url[len(PAPER_TRADING_URL) :]
        response = self.routes[path]
        return response(params) if callable(response) else response

    def post(self, url, headers=None, json=None, timeout=None):
        assert url == f"{PAPER_TRADING_URL}/v2/orders"
        self.posted.append(json)
        if isinstance(self.post_response, Exception):
            raise self.post_response
        return self.post_response


ACCOUNT = FakeResponse(payload={"cash": "4000", "equity": "10000", "status": "ACTIVE"})
POSITIONS = FakeResponse(
    payload=[{"symbol": "SPY", "qty": "10", "current_price": "600"}],
)


def _broker(session):
    return AlpacaPaperBroker(PAPER, session=session, sleep=lambda _: None)


def _order(oid="pa-1"):
    return Order(oid, "SPY", Side.BUY, 1.5, 600.0)


def test_live_keys_are_refused_before_any_call() -> None:
    session = FakeSession()
    live = Settings(alpaca_api_key="AKLIVE", alpaca_api_secret="secret")

    with pytest.raises(AlpacaPaperError, match="paper"):
        AlpacaPaperBroker(live, session=session)

    assert session.calls == []


def test_secret_never_in_repr() -> None:
    assert "secret" not in repr(PAPER)
    assert "PKTEST" not in repr(PAPER)


def test_account_values_positions_at_given_prices() -> None:
    broker = _broker(FakeSession({"/v2/account": ACCOUNT, "/v2/positions": POSITIONS}))

    snapshot = broker.account({"SPY": 610.0})

    assert snapshot.cash == 4000.0
    assert snapshot.positions == {"SPY": 10.0}
    assert snapshot.equity == pytest.approx(4000 + 6100)
    assert broker.market_prices() == {"SPY": 600.0}


def test_blocked_account_and_short_positions_stop_everything() -> None:
    blocked = FakeResponse(payload={"cash": "1", "trading_blocked": True})
    with pytest.raises(AlpacaPaperError, match="bloccato"):
        _broker(FakeSession({"/v2/account": blocked, "/v2/positions": POSITIONS})).account({})

    short = FakeResponse(payload=[{"symbol": "SPY", "qty": "-1"}])
    with pytest.raises(AlpacaPaperError, match="short"):
        _broker(FakeSession({"/v2/account": ACCOUNT, "/v2/positions": short})).account({"SPY": 1.0})


def test_submit_sends_market_day_order_and_maps_status() -> None:
    session = FakeSession(post=FakeResponse(payload={"status": "accepted"}))
    broker = _broker(session)

    report = broker.submit(_order(), fill_price=600.0)

    assert report.status is OrderStatus.SUBMITTED
    assert session.posted == [
        {
            "symbol": "SPY",
            "qty": "1.5",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": "pa-1",
        }
    ]
    # Stesso id: nessun secondo invio.
    assert broker.submit(_order(), fill_price=600.0) == report
    assert len(session.posted) == 1


def test_duplicate_client_order_id_recovers_existing_order() -> None:
    session = FakeSession(
        routes={
            "/v2/orders:by_client_order_id": FakeResponse(
                payload={"status": "filled", "filled_avg_price": "601.5"}
            )
        },
        post=FakeResponse(422, text='{"message":"client_order_id must be unique"}'),
    )

    report = _broker(session).submit(_order(), fill_price=600.0)

    assert report.status is OrderStatus.FILLED
    assert report.fill_price == 601.5


def test_network_error_is_not_retried_blindly() -> None:
    missing = FakeResponse(404, text="not found")
    session = FakeSession(
        routes={"/v2/orders:by_client_order_id": missing},
        post=requests.ConnectionError("rete giù"),
    )

    report = _broker(session).submit(_order(), fill_price=600.0)

    assert report.status is OrderStatus.REJECTED
    assert "incerto" in report.reason
    assert len(session.posted) == 1


def test_http_rejection_is_reported() -> None:
    session = FakeSession(post=FakeResponse(403, text="insufficient buying power"))

    report = _broker(session).submit(_order(), fill_price=600.0)

    assert report.status is OrderStatus.REJECTED
    assert "403" in report.reason


def test_open_orders_query() -> None:
    session = FakeSession({"/v2/orders": FakeResponse(payload=[{"id": "1"}])})

    assert _broker(session).open_orders() == [{"id": "1"}]
    assert session.calls[-1][1]["status"] == "open"
