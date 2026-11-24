"""Test del filtro di tendenza facoltativo e del satellite bitcoin."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_alpha.paper import TrendFilterConfig, core_targets, load_paper_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "paper_bitcoin_trend_v1.toml"


def _frame(prices) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-11-20", periods=len(prices), tz="UTC") + pd.Timedelta(hours=5)
    return pd.DataFrame(
        {
            "symbol": "IBIT",
            "timestamp": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 1_000_000,
        }
    )


def test_bitcoin_config_is_valid() -> None:
    config = load_paper_config(CONFIG)

    assert config.name == "bitcoin_trend_v1"
    assert config.broker == "ledger"
    assert config.trend_filter == TrendFilterConfig(sma_days=200)
    assert config.weights == {"IBIT": 1.0}
    assert config.risk.max_drawdown < config.stop_drawdown < config.ips_max_drawdown


def test_invested_when_above_the_average() -> None:
    config = load_paper_config(CONFIG)
    targets = core_targets(_frame(np.linspace(30, 60, 260)), config)

    assert targets.weights["IBIT"] == pytest.approx(1.0)


def test_cash_when_below_the_average() -> None:
    config = load_paper_config(CONFIG)
    prices = np.concatenate([np.linspace(30, 60, 230), np.linspace(60, 40, 30)])
    targets = core_targets(_frame(prices), config)

    assert targets.weights["IBIT"] == 0.0


def test_cash_without_enough_history() -> None:
    config = load_paper_config(CONFIG)
    targets = core_targets(_frame(np.linspace(30, 60, 150)), config)

    assert targets.weights["IBIT"] == 0.0


def test_without_filter_weights_are_unchanged() -> None:
    config = replace(load_paper_config(CONFIG), trend_filter=None)
    prices = np.concatenate([np.linspace(30, 60, 230), np.linspace(60, 40, 30)])
    targets = core_targets(_frame(prices), config)

    assert targets.weights["IBIT"] == pytest.approx(1.0)


def test_invalid_sma_days() -> None:
    with pytest.raises(ValueError):
        TrendFilterConfig(sma_days=1)
