"""Test del ciclo di paper trading con broker simulato (nessuna rete)."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_alpha.execution.broker import SimulatedBroker
from project_alpha.paper import (
    PaperState,
    core_targets,
    load_paper_config,
    resume_paper,
    run_paper,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "paper_core_v1.toml"
NOW = datetime(2026, 11, 9, 22, 30, tzinfo=UTC)


def _bars(symbols, days=120, seed=0, end="2026-11-09"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=days, tz="UTC") + pd.Timedelta(hours=5)
    rows = []
    for index, symbol in enumerate(symbols):
        price = 100.0 + 10 * index
        for timestamp in dates:
            price *= 1 + rng.normal(0, 0.008)
            rows.append({"symbol": symbol, "timestamp": timestamp, "close": price})
    return pd.DataFrame(rows)


@pytest.fixture()
def config(tmp_path):
    loaded = load_paper_config(CONFIG)
    return replace(loaded, log=tmp_path / "paper_log.csv", state=tmp_path / "state.json")


class QueueBroker(SimulatedBroker):
    """Broker simulato con ordini aperti finti, per il controllo di sicurezza."""

    pending: list = []

    def open_orders(self):
        return list(self.pending)


def test_config_matches_ips(config) -> None:
    assert sum(config.weights.values()) == pytest.approx(1.0)
    assert config.budget == 10_000
    assert config.stop_drawdown == 0.12
    assert config.risk.max_drawdown == 0.10
    assert config.risk.allowed_symbols == frozenset(config.weights)
    groups = {}
    for symbol, weight in config.weights.items():
        groups[config.groups[symbol]] = groups.get(config.groups[symbol], 0) + weight
    assert groups == pytest.approx({"azioni": 0.60, "obbligazioni": 0.30, "reali": 0.10})


def test_config_rejects_unknown_keys(tmp_path) -> None:
    bad = tmp_path / "configs" / "bad.toml"
    bad.parent.mkdir()
    bad.write_text(CONFIG.read_text(encoding="utf-8") + "\nbroker_url = 'live'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sconosciute"):
        load_paper_config(bad)


def test_core_targets_scale_with_volatility(config) -> None:
    calm = core_targets(_bars(config.symbols), config)

    assert 0 < calm.exposure <= 1.0
    assert calm.weights["SPY"] == pytest.approx(0.20 * calm.exposure)

    rng = np.random.default_rng(1)
    volatile = _bars(config.symbols)
    volatile["close"] = volatile.groupby("symbol")["close"].transform(
        lambda s: 100 * np.cumprod(1 + rng.normal(0, 0.04, len(s)))
    )
    assert core_targets(volatile, config).exposure < calm.exposure


def test_core_targets_reject_misaligned_data(config) -> None:
    bars = _bars(config.symbols)
    last = bars["timestamp"].max()
    bars = bars.loc[~((bars["symbol"] == "GLD") & (bars["timestamp"] == last))]

    with pytest.raises(ValueError, match="non allineati"):
        core_targets(bars, config)


def test_dry_run_plans_without_trading_or_state(config) -> None:
    broker = QueueBroker(cash=10_000.0)

    outcome = run_paper(config, broker, frame=_bars(config.symbols), now=NOW)

    assert outcome.status == "DRY_RUN"
    assert outcome.rebalance.orders
    assert broker.positions == {}
    assert not config.state.exists()
    assert pd.read_csv(config.log)["mode"].tolist() == ["DRY_RUN"]


def test_submit_builds_portfolio_then_stays_within_bands(config) -> None:
    broker = QueueBroker(cash=10_000.0, commission_rate=0.0)
    bars = _bars(config.symbols)

    first = run_paper(config, broker, frame=bars, submit=True, now=NOW)
    second = run_paper(config, broker, frame=bars, submit=True, now=NOW)

    assert first.status == "EXECUTED"
    assert set(broker.positions) == set(config.weights)
    assert second.status == "WITHIN_BANDS"
    state = PaperState.load(config.state)
    assert state.peak_equity == pytest.approx(first.equity)
    assert len(pd.read_csv(config.log)) == 2


def test_oversized_account_is_refused_at_first_start(config) -> None:
    broker = QueueBroker(cash=100_000.0, commission_rate=0.0)
    bars = _bars(config.symbols)

    dry = run_paper(config, broker, frame=bars, now=NOW)
    assert "Reimpostare il conto" in dry.note

    with pytest.raises(ValueError, match="Reimpostare il conto"):
        run_paper(config, broker, frame=bars, submit=True, now=NOW)
    assert broker.positions == {}


def test_budget_caps_capital_after_growth(config) -> None:
    broker = QueueBroker(cash=10_000.0, commission_rate=0.0)
    bars = _bars(config.symbols)
    run_paper(config, broker, frame=bars, submit=True, now=NOW)

    broker.cash += 5_000.0  # il conto cresce oltre il budget
    outcome = run_paper(config, broker, frame=bars, submit=True, now=NOW)

    invested = outcome.equity - broker.cash
    assert invested <= config.budget * 1.01


def test_open_orders_block_new_orders(config) -> None:
    broker = QueueBroker(cash=10_000.0)
    broker.pending = [{"id": "queued"}]

    outcome = run_paper(config, broker, frame=_bars(config.symbols), submit=True, now=NOW)

    assert outcome.status == "PENDING_ORDERS"
    assert broker.positions == {}


def test_drawdown_stop_liquidates_and_halts_until_resume(config) -> None:
    bars = _bars(config.symbols)
    broker = QueueBroker(cash=10_000.0, commission_rate=0.0)
    run_paper(config, broker, frame=bars, submit=True, now=NOW)

    # Crollo del 20% su tutti i titoli: drawdown oltre lo stop del 12%.
    crash = bars.copy()
    last = crash["timestamp"].max()
    crash.loc[crash["timestamp"] == last, "close"] *= 0.8
    stopped = run_paper(config, broker, frame=crash, submit=True, now=NOW)

    assert stopped.halted
    assert stopped.status == "EXECUTED"
    assert broker.positions == {}
    assert PaperState.load(config.state).halted

    still = run_paper(config, broker, frame=crash, submit=True, now=NOW)
    assert still.status == "HALTED"

    resume_paper(config, now=NOW)
    resumed = run_paper(config, broker, frame=crash, submit=True, now=NOW)
    assert not resumed.halted
    assert resumed.status == "EXECUTED"
    assert broker.positions


def test_stale_data_blocks_buying(config) -> None:
    broker = QueueBroker(cash=10_000.0)
    old = _bars(config.symbols, end="2026-10-20")

    outcome = run_paper(config, broker, frame=old, submit=True, now=NOW)

    assert outcome.status == "REJECTED_BY_RISK"
    assert "STALE_DATA" in outcome.rebalance.risk.violations
    assert broker.positions == {}
