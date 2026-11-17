"""Test del motore dell'app (lavoro serale, idempotenza, recuperi, pianificatore)."""

import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_alpha import jobs
from project_alpha.execution.broker import SimulatedBroker
from project_alpha.jobs import JobContext, Scheduler, daily_due, daily_run, last_complete_session

REPO = Path(__file__).resolve().parents[1]
CORE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "IEF", "LQD", "TLT", "GLD", "VNQ"]


def _bars(end: str, days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(end=end, periods=days, tz="UTC") + pd.Timedelta(hours=5)
    rows = []
    for index, symbol in enumerate(CORE):
        price = 100.0 + 5 * index
        for ts in dates:
            opening = price * (1 + rng.normal(0, 0.002))
            price = opening * (1 + rng.normal(0, 0.007))
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": opening,
                    "high": max(opening, price) * 1.002,
                    "low": min(opening, price) * 0.998,
                    "close": price,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


class Clock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "configs").mkdir()
    for name in ("paper_core_v1.toml", "paper_permanente_v1.toml"):
        shutil.copy(REPO / "configs" / name, tmp_path / "configs" / name)
    return tmp_path


def _context(repo, moment, *, end="2026-11-17", bars=None, broker=None):
    frame = bars if bars is not None else _bars(end)
    calls = {"fetch": 0, "news": 0}
    shared = broker or SimulatedBroker(cash=10_000.0, commission_rate=0.0)

    def fetch(symbols, output):
        calls["fetch"] += 1
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return frame

    def news(root, symbols):
        calls["news"] += 1
        return {"portfolio": 0, "market": 0}

    ctx = JobContext(
        repo,
        fetch=fetch,
        alpaca_broker=lambda: shared,
        news=news,
        post=lambda *a, **k: None,
        clock=Clock(moment),
    )
    return ctx, calls, shared


# 2026-11-17 è martedì. 23:00 in Italia = 17:00 a New York.
EVENING = datetime(2026, 11, 17, 22, 5, tzinfo=UTC)
DURING_SESSION = datetime(2026, 11, 18, 16, 0, tzinfo=UTC)  # mercoledì 11:00 NY


def test_last_complete_session() -> None:
    assert last_complete_session(EVENING) == date(2026, 11, 17)
    assert last_complete_session(datetime(2026, 11, 17, 20, 0, tzinfo=UTC)) == date(2026, 11, 16)
    assert last_complete_session(datetime(2026, 11, 22, 12, 0, tzinfo=UTC)) == date(
        2026, 11, 20
    )  # domenica
    assert last_complete_session(datetime(2026, 11, 23, 9, 0, tzinfo=UTC)) == date(
        2026, 11, 20
    )  # lunedì mattina


def test_daily_run_trades_both_strategies_once_per_session(repo) -> None:
    ctx, calls, broker = _context(repo, EVENING)

    first = daily_run(ctx)
    second = daily_run(ctx)

    assert first["status"] == "OK"
    assert set(first["strategies"]) == {"core_bilanciato_v1", "permanente_v1"}
    assert first["strategies"]["core_bilanciato_v1"]["status"] == "EXECUTED"
    assert (
        first["strategies"]["permanente_v1"]["status"] == "SUBMITTED"
    )  # conto interno: domani all'apertura
    assert broker.positions  # il core ha comprato
    assert second["status"] == "ALREADY_DONE"
    assert calls["fetch"] == 1 and calls["news"] == 1
    assert (repo / "reports" / "paper" / "daily_report.md").exists()
    assert (repo / "reports" / "paper_permanente" / "daily_report.md").exists()


def test_daily_run_waits_while_market_is_open(repo) -> None:
    ctx, calls, _ = _context(repo, DURING_SESSION)

    assert daily_run(ctx)["status"] == "WAIT_FOR_CLOSE"
    assert calls["fetch"] == 0
    assert not daily_due(ctx)


def test_data_errors_stop_trading_and_are_retried_later(repo) -> None:
    bad = _bars("2026-11-17")
    bad.loc[bad.index[5], "close"] = -1.0
    ctx, calls, broker = _context(repo, EVENING, bars=bad)

    result = daily_run(ctx)

    assert result["status"] == "DATA_ERROR"
    assert broker.positions == {}
    assert not daily_due(ctx)  # non riprova subito
    ctx.clock.moment = EVENING + timedelta(minutes=jobs.RETRY_MINUTES + 1)
    assert daily_due(ctx)


def test_fetch_exception_is_caught_and_marked(repo) -> None:
    ctx, _, _ = _context(repo, EVENING)

    def broken(symbols, output):
        raise RuntimeError("Alpaca non raggiungibile")

    ctx.fetch = broken
    result = daily_run(ctx)

    assert result["status"] == "EXCEPTION" and "Alpaca" in result["error"]
    assert jobs.read_marker(ctx)["ok"] is False


def test_dry_run_does_not_mark_the_session(repo) -> None:
    ctx, _, broker = _context(repo, EVENING)

    assert daily_run(ctx, submit=False)["mode"] == "DRY_RUN"
    assert broker.positions == {}
    assert jobs.read_marker(ctx) == {}
    assert daily_due(ctx)


def test_lock_prevents_concurrent_runs(repo) -> None:
    ctx, calls, _ = _context(repo, EVENING)
    lock = repo / "reports" / "paper" / "daily.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("123")

    assert daily_run(ctx)["status"] == "LOCKED"
    assert calls["fetch"] == 0


def test_scheduler_catches_up_after_downtime_and_runs_hourly(repo) -> None:
    # Il PC era spento martedì sera: si riaccende mercoledì alle 8 italiane.
    morning = datetime(2026, 11, 18, 7, 0, tzinfo=UTC)
    ctx, calls, _ = _context(repo, morning)
    scheduler = Scheduler(ctx)

    first = scheduler.step()
    again = scheduler.step()

    assert "daily:OK" in first and "hourly" in first
    assert again == []  # stessa ora, seduta già fatta
    assert jobs.read_marker(ctx)["session"] == "2026-11-17"
