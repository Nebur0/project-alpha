"""Test della prova in parallelo del portafoglio di crescita."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_alpha.execution.broker import SimulatedBroker
from project_alpha.jobs import JobContext, daily_run, paper_configs
from project_alpha.paper import load_paper_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "paper_crescita_v1.toml"
SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "IEF", "LQD", "TLT", "GLD", "VNQ"]
# 2026-11-17 è martedì: 23:05 in Italia, seduta di New York chiusa.
EVENING = datetime(2026, 11, 17, 22, 5, tzinfo=UTC)


def _bars(end: str = "2026-11-17", days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(end=end, periods=days, tz="UTC") + pd.Timedelta(hours=5)
    rows = []
    for index, symbol in enumerate(SYMBOLS):
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


def context(root: Path, moment: datetime) -> JobContext:
    frame = _bars()
    broker = SimulatedBroker(cash=10_000.0, commission_rate=0.0)

    def fetch(symbols, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return frame

    return JobContext(
        root,
        fetch=fetch,
        alpaca_broker=lambda: broker,
        news=lambda root, symbols: {"portfolio": 0, "market": 0},
        post=lambda *a, **k: None,
        clock=lambda: moment,
    )


def test_config_is_valid_and_fully_invested() -> None:
    config = load_paper_config(CONFIG)

    assert config.name == "crescita_v1"
    assert config.broker == "ledger"
    assert config.vol_target is None
    assert sum(config.weights.values()) == pytest.approx(1.0, abs=1e-9)
    equity = sum(w for s, w in config.weights.items() if config.groups[s] == "azioni")
    assert equity == pytest.approx(0.80, abs=1e-9)
    assert config.risk.max_drawdown < config.stop_drawdown < config.ips_max_drawdown
    assert max(config.weights.values()) <= config.risk.max_position_weight
    assert equity <= config.risk.max_group_weight


def test_engine_runs_three_strategies_with_core_first(tmp_path) -> None:
    (tmp_path / "configs").mkdir()
    for name in ("paper_core_v1.toml", "paper_permanente_v1.toml", "paper_crescita_v1.toml"):
        shutil.copy(REPO / "configs" / name, tmp_path / "configs" / name)

    names = [c.name for c in paper_configs(tmp_path)]
    assert names[0] == "core_bilanciato_v1"
    assert set(names) == {"core_bilanciato_v1", "permanente_v1", "crescita_v1"}

    ctx = context(tmp_path, EVENING)
    result = daily_run(ctx)

    assert result["status"] == "OK"
    assert result["strategies"]["crescita_v1"]["status"] == "SUBMITTED"
    assert (tmp_path / "data" / "paper_crescita" / "ledger.json").exists()
    assert (tmp_path / "reports" / "paper_crescita" / "daily_report.md").exists()
