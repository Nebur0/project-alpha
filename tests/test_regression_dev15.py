"""Regressione: DEV15 V1 con le impostazioni di default riproduce il report.

Fissa i numeri di riferimento (10.000 -> 12.287,34, Sharpe 0,979,
esposizione 20,82%) con una tolleranza sui decimali: differenze di
arrotondamento dell'ambiente (ultime cifre dei float) non lo fanno fallire,
un cambiamento reale del comportamento di default sì.

Gira solo se i dati di sviluppo sono presenti (non sono versionati) e sono
gli stessi con cui è stato prodotto il report.
"""

import json
from pathlib import Path

import pytest

from project_alpha.portfolio_walk_forward import run_walk_forward_portfolio_backtest
from project_alpha.research import (
    _load_frame,
    file_sha256,
    load_research_config,
    run_walk_forward_batch,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "dev15_v1.toml"
DATA = REPO / "data" / "raw" / "bars_dev15.csv"
MANIFEST = REPO / "reports" / "research" / "dev15_v1" / "manifest.json"


def _reference_data_hash() -> str | None:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))["data"]["sha256"]
    except (OSError, KeyError, ValueError):
        return None


@pytest.mark.skipif(not DATA.exists(), reason="dati DEV15 non presenti (non versionati)")
def test_dev15_v1_default_reproduces_report() -> None:
    reference_hash = _reference_data_hash()

    if reference_hash is not None and file_sha256(DATA) != reference_hash:
        pytest.skip("bars_dev15.csv diverso da quello del report di riferimento")

    config = load_research_config(CONFIG, base_dir=REPO)
    assert config.execution.is_default

    frame = _load_frame(config)
    batch = run_walk_forward_batch(config, frame)
    metrics = run_walk_forward_portfolio_backtest(
        frame,
        batch,
        initial_capital=config.walk_forward.initial_capital,
        cost_per_side=config.walk_forward.cost_per_side,
    ).portfolio.metrics

    assert metrics["initial_capital"] == 10_000.0
    assert metrics["final_equity"] == pytest.approx(12_287.34, abs=0.01)
    assert metrics["annualized_sharpe"] == pytest.approx(0.979, abs=5e-4)
    assert metrics["mean_gross_exposure"] == pytest.approx(0.2082, abs=5e-5)
    assert metrics["symbol_count"] == 15
