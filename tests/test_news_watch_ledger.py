"""Test di notizie, sentinella oraria e conto interno (nessuna rete)."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_alpha.config import Settings
from project_alpha.execution.broker import SimulatedBroker
from project_alpha.execution.ledger import LedgerBroker
from project_alpha.execution.orders import Order, Side
from project_alpha.news import NewsStore, fetch_news, update_news
from project_alpha.paper import PaperState, load_paper_config, run_paper
from project_alpha.watch import watch_paper

REPO = Path(__file__).resolve().parents[1]
SETTINGS = Settings(alpaca_api_key="PKTEST", alpaca_api_secret="secret")


def _bars(symbols, days=120, seed=0, end="2026-11-09"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=days, tz="UTC") + pd.Timedelta(hours=5)
    rows = []
    for index, symbol in enumerate(symbols):
        price = 100.0 + 10 * index
        for timestamp in dates:
            opening = price * (1 + rng.normal(0, 0.003))
            price = opening * (1 + rng.normal(0, 0.008))
            rows.append({"symbol": symbol, "timestamp": timestamp, "open": opening, "close": price})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Notizie
# ----------------------------------------------------------------------


class FakeResponse:
    status_code = 200
    headers: dict = {}

    def __init__(self, payload):
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        return FakeResponse(self.pages.pop(0))


def _item(i, minutes_ago=10, symbols=("SPY",)):
    created = (datetime(2026, 11, 17, 12, tzinfo=UTC) - timedelta(minutes=minutes_ago)).isoformat()
    return {
        "id": i,
        "headline": f"Notizia {i}",
        "summary": "Riassunto",
        "source": "benzinga",
        "author": "",
        "url": f"https://example.com/{i}",
        "symbols": list(symbols),
        "created_at": created,
        "updated_at": created,
    }


def test_fetch_news_paginates_and_filters_symbols() -> None:
    session = FakeSession(
        [
            {"news": [_item(1), _item(2)], "next_page_token": "x"},
            {"news": [_item(3)], "next_page_token": None},
        ]
    )

    items = fetch_news(
        SETTINGS, symbols=["spy", "GLD"], start=datetime(2026, 11, 16, tzinfo=UTC), session=session
    )

    assert [i["id"] for i in items] == [1, 2, 3]
    assert session.calls[0][0].endswith("/v1beta1/news")
    assert session.calls[0][1]["symbols"] == "GLD,SPY"
    assert session.calls[1][1]["page_token"] == "x"


def test_store_deduplicates_tags_feeds_and_drops_old(tmp_path) -> None:
    store = NewsStore(tmp_path)
    now = datetime(2026, 11, 17, 12, tzinfo=UTC)

    assert (
        store.merge(
            [_item(1), _item(2), _item(9, minutes_ago=60 * 24 * 40)], feed="portfolio", now=now
        )
        == 2
    )
    assert store.merge([_item(2), _item(3)], feed="market", now=now) == 1

    items = {i["id"]: i for i in store.load()}
    assert set(items) == {1, 2, 3}
    assert items[2]["feeds"] == ["portfolio", "market"]


def test_update_news_writes_both_feeds(tmp_path) -> None:
    session = FakeSession(
        [
            {"news": [_item(1)], "next_page_token": None},
            {"news": [_item(2, symbols=("AAPL",))], "next_page_token": None},
        ]
    )

    added = update_news(
        SETTINGS, tmp_path, ["SPY"], now=datetime(2026, 11, 17, 12, tzinfo=UTC), session=session
    )

    assert added == {"portfolio": 1, "market": 1}
    assert len(json.loads((tmp_path / "data" / "news" / "news.json").read_text())) == 2


# ----------------------------------------------------------------------
# Conto interno
# ----------------------------------------------------------------------


def test_ledger_fills_at_next_open_sells_first_and_caps_buys(tmp_path) -> None:
    bars = _bars(["SPY", "TLT"])
    last_two = sorted(bars["timestamp"].unique())[-2:]
    order_day = (pd.Timestamp(last_two[0]) + pd.Timedelta(hours=17)).to_pydatetime()
    ledger = LedgerBroker.load(tmp_path / "l.json", initial_cash=1000.0, now=order_day)
    ledger.positions = {"TLT": 2.0}

    ledger.submit(Order("b", "SPY", Side.BUY, 50.0, 100.0), fill_price=100.0)  # più del cash
    ledger.submit(Order("s", "TLT", Side.SELL, 2.0, 110.0), fill_price=110.0)
    assert len(ledger.open_orders()) == 2

    assert ledger.settle(bars.loc[bars["timestamp"] <= last_two[0]]) == 0  # nessuna seduta dopo
    assert ledger.settle(bars) == 2

    spy_open = bars.loc[(bars.symbol == "SPY") & (bars.timestamp == last_two[1]), "open"].item()
    tlt_open = bars.loc[(bars.symbol == "TLT") & (bars.timestamp == last_two[1]), "open"].item()
    assert ledger.fills[0]["symbol"] == "TLT"  # vendita prima
    assert "TLT" not in ledger.positions
    cash_after_sell = 1000.0 + 2 * tlt_open * (1 - 0.0005)
    assert ledger.positions["SPY"] == pytest.approx(cash_after_sell / (spy_open * 1.0005))
    assert ledger.cash == pytest.approx(0.0, abs=1e-6)

    ledger.save()
    again = LedgerBroker.load(tmp_path / "l.json", initial_cash=0.0)
    assert again.positions == ledger.positions and again.pending == []


def test_permanent_portfolio_runs_on_the_ledger(tmp_path) -> None:
    config = load_paper_config(REPO / "configs" / "paper_permanente_v1.toml")
    config = replace(
        config,
        log=tmp_path / "log.csv",
        state=tmp_path / "state.json",
        ledger=tmp_path / "ledger.json",
    )
    assert config.risk.max_position_weight == 0.25
    bars = _bars(config.symbols)
    days = sorted(bars["timestamp"].unique())

    for day in days[-5:]:
        frame = bars.loc[bars["timestamp"] <= day]
        now = (pd.Timestamp(day) + pd.Timedelta(hours=17)).to_pydatetime()
        ledger = LedgerBroker.load(config.ledger, initial_cash=config.budget, now=now)
        ledger.settle(frame)
        outcome = run_paper(config, ledger, frame=frame, submit=True, now=now)
        ledger.save()

    ledger = LedgerBroker.load(config.ledger, initial_cash=0.0)
    assert set(ledger.positions) == {"SPY", "TLT", "GLD"}
    invested = sum(
        q * bars.loc[bars.symbol == s, "close"].iloc[-1] for s, q in ledger.positions.items()
    )
    assert invested / (invested + ledger.cash) == pytest.approx(0.75, abs=0.03)
    assert outcome.status in {"WITHIN_BANDS", "SUBMITTED"}


# ----------------------------------------------------------------------
# Sentinella
# ----------------------------------------------------------------------


class MarkedBroker(SimulatedBroker):
    """Broker simulato con prezzi correnti (come Alpaca durante la seduta)."""

    prices: dict = {}

    def market_prices(self):
        return dict(self.prices)


@pytest.fixture()
def started(tmp_path):
    config = load_paper_config(REPO / "configs" / "paper_core_v1.toml")
    config = replace(config, log=tmp_path / "log.csv", state=tmp_path / "state.json")
    state = PaperState(
        peak_equity=10_000.0,
        last_equity=10_000.0,
        account_baseline=100_000.0,
        capital_baseline=10_000.0,
    )
    state.save(config.state)
    return config, tmp_path


DURING = datetime(2026, 11, 17, 17, 0, tzinfo=UTC)  # 12:00 a New York, martedì


def _broker(account_equity):
    broker = MarkedBroker(cash=account_equity - 7_000.0, positions={"SPY": 10.0})
    broker.prices = {"SPY": 700.0}
    return broker


def test_watch_is_silent_when_market_is_closed_or_all_is_fine(started) -> None:
    config, root = started
    night = datetime(2026, 11, 17, 23, 0, tzinfo=UTC)

    assert watch_paper(config, _broker(100_000.0), root, now=night).status == "MARKET_CLOSED"
    fine = watch_paper(config, _broker(99_900.0), root, now=DURING, post=lambda *a, **k: None)
    assert fine.status == "CHECKED" and fine.level is None and not fine.notified


def test_watch_alerts_once_per_level_per_day(started) -> None:
    config, root = started
    sent = []

    def post(url, data, headers, timeout):
        sent.append(headers.get("Urgency"))

        class R:
            status_code = 201

        return R()

    # Iscrive un dispositivo finto.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    from project_alpha.push import SubscriptionStore, b64url_encode

    key = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    )
    SubscriptionStore(root / "data" / "push").add(
        {
            "endpoint": "https://web.push.apple.com/x",
            "keys": {"p256dh": b64url_encode(key), "auth": b64url_encode(b"0123456789abcdef")},
        }
    )

    drop = watch_paper(config, _broker(99_600.0), root, now=DURING, post=post)  # -4% in giornata
    again = watch_paper(config, _broker(99_600.0), root, now=DURING + timedelta(hours=1), post=post)
    stop = watch_paper(config, _broker(98_700.0), root, now=DURING + timedelta(hours=2), post=post)

    assert drop.level == "warning" and drop.notified
    assert not again.notified
    assert stop.level == "critical" and stop.notified and "23:00" in stop.message
    assert sent == ["normal", "high"]
