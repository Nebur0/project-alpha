"""Test dell'assistente di monitoraggio (sola lettura)."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_alpha.execution.broker import SimulatedBroker
from project_alpha.monitor import check_paper, render_report, tracking_vs_model
from project_alpha.paper import PaperState, load_paper_config, run_paper

REPO = Path(__file__).resolve().parents[1]


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
    loaded = load_paper_config(REPO / "configs" / "paper_core_v1.toml")
    return replace(loaded, log=tmp_path / "paper_log.csv", state=tmp_path / "state.json")


def _run_days(config, broker, bars, days, *, submit=True):
    """Esegue il ciclo serale sugli ultimi ``days`` giorni dei dati."""
    dates = sorted(bars["timestamp"].unique())[-days:]
    for date in dates:
        frame = bars.loc[bars["timestamp"] <= date]
        now = (pd.Timestamp(date) + pd.Timedelta(hours=17)).to_pydatetime()
        run_paper(config, broker, frame=frame, submit=submit, now=now)
    return (pd.Timestamp(dates[-1]) + pd.Timedelta(hours=18)).to_pydatetime()


def _check(config, bars, now):
    log = pd.read_csv(config.log) if config.log.exists() else None
    return check_paper(config, log, PaperState.load(config.state), bars=bars, now=now)


def test_not_started(config) -> None:
    result = _check(config, None, datetime(2026, 11, 10, tzinfo=UTC))

    assert [a.code for a in result.alerts] == ["NOT_STARTED"]
    assert "Tutto regolare" in render_report(config, result)


def test_dry_run_only_is_reported(config) -> None:
    bars = _bars(config.symbols)
    now = _run_days(config, SimulatedBroker(cash=10_000.0), bars, 1, submit=False)

    result = _check(config, bars, now)

    assert "DRY_RUN_ONLY" in [a.code for a in result.alerts]
    assert result.worst == "info"


def test_regular_runs_have_no_warnings_and_track_the_model(config) -> None:
    bars = _bars(config.symbols)
    broker = SimulatedBroker(cash=10_000.0, commission_rate=0.0)
    now = _run_days(config, broker, bars, 15)

    result = _check(config, bars, now)

    assert result.worst in (None, "info")
    assert result.tracking is not None and result.tracking.intervals >= 10
    # Il broker simulato esegue alla chiusura: conto e modello coincidono quasi.
    assert abs(result.tracking.difference) < 0.005
    report = render_report(config, result)
    assert "Conto contro modello" in report and "Tutto regolare" in report


def test_missing_runs_are_flagged(config) -> None:
    bars = _bars(config.symbols)
    now = _run_days(config, SimulatedBroker(cash=10_000.0), bars, 2)

    late = _check(config, bars, now + timedelta(days=5))
    very_late = _check(config, bars, now + timedelta(days=9))

    assert ("warning", "NO_RECENT_RUN") in [(a.severity, a.code) for a in late.alerts]
    assert ("critical", "NO_RECENT_RUN") in [(a.severity, a.code) for a in very_late.alerts]


def test_halt_and_drawdown_are_critical(config) -> None:
    bars = _bars(config.symbols)
    broker = SimulatedBroker(cash=10_000.0, commission_rate=0.0)
    _run_days(config, broker, bars, 1)
    crash = bars.copy()
    crash.loc[crash["timestamp"] == crash["timestamp"].max(), "close"] *= 0.8
    now = datetime(2026, 11, 10, 9, tzinfo=UTC)
    run_paper(config, broker, frame=crash, submit=True, now=now)

    result = _check(config, crash, now)

    assert result.worst == "critical"
    assert "HALTED" in [a.code for a in result.alerts]
    assert "Serve un tuo intervento" in render_report(config, result)


def test_tracking_detects_an_account_that_diverges(config) -> None:
    bars = _bars(config.symbols)
    broker = SimulatedBroker(cash=10_000.0, commission_rate=0.0)
    _run_days(config, broker, bars, 10)
    log = pd.read_csv(config.log)
    log.loc[log.index[-1], "equity"] = float(log["equity"].iloc[-2]) * 0.95  # -5% inspiegato

    tracking = tracking_vs_model(log, bars)

    assert tracking.difference < -0.03


def test_risk_rejection_and_rejected_orders(config) -> None:
    bars = _bars(config.symbols, end="2026-10-20")  # dati vecchi
    run_paper(
        config,
        SimulatedBroker(cash=10_000.0),
        frame=bars,
        submit=True,
        now=datetime(2026, 11, 9, 22, tzinfo=UTC),
    )

    result = _check(config, bars, datetime(2026, 11, 9, 23, tzinfo=UTC))
    codes = [a.code for a in result.alerts]

    assert "RISK_REJECTED" in codes and "STALE_DATA" in codes
