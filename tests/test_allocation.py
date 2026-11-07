"""Test del portafoglio di policy e del volatility targeting."""

from math import sqrt

import numpy as np
import pandas as pd
import pytest

from project_alpha.allocation import (
    VolTargetConfig,
    run_policy_portfolio,
    volatility_target_exposure,
)

INDEX = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")


def _returns(seed: int, vol: float = 0.01, mean: float = 0.0004) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, vol, INDEX.size), index=INDEX)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_volatility": 0.0},
        {"lookback_days": 1},
        {"max_exposure": 1.5},
        {"min_exposure": 0.6, "max_exposure": 0.5},
        {"warmup_exposure": 1.0, "max_exposure": 0.5},
        {"rebalance_threshold": 1.0},
    ],
)
def test_vol_target_config_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        VolTargetConfig(**kwargs)


def test_exposure_uses_only_past_returns() -> None:
    returns = _returns(1)
    config = VolTargetConfig(lookback_days=20, rebalance_threshold=0.0)
    exposure = volatility_target_exposure(returns, config)

    shocked = returns.copy()
    shocked.iloc[200] = 0.30  # shock oggi
    shocked_exposure = volatility_target_exposure(shocked, config)

    # L'esposizione di oggi non vede lo shock di oggi; quella di domani sì.
    assert shocked_exposure.iloc[200] == exposure.iloc[200]
    assert shocked_exposure.iloc[201] < exposure.iloc[201]


def test_exposure_scales_inversely_with_volatility() -> None:
    config = VolTargetConfig(target_volatility=0.10, lookback_days=60, rebalance_threshold=0.0)
    calm = volatility_target_exposure(_returns(2, vol=0.004), config)
    wild = volatility_target_exposure(_returns(3, vol=0.03), config)

    assert calm.iloc[-1] == pytest.approx(1.0)  # cappata: niente leva
    assert wild.iloc[-1] == pytest.approx(0.10 / (0.03 * sqrt(252)), rel=0.2)


def test_warmup_and_threshold() -> None:
    config = VolTargetConfig(lookback_days=30, warmup_exposure=0.5, rebalance_threshold=0.2)
    exposure = volatility_target_exposure(_returns(4, vol=0.02), config)

    assert (exposure.iloc[:30] == 0.5).all()
    # Con isteresi l'esposizione cambia raramente.
    assert exposure.diff().ne(0).sum() < 20


def test_policy_portfolio_static_weights_and_cash() -> None:
    passive = _returns(5)
    result = run_policy_portfolio({"core": passive}, {"core": 0.6}, cost_per_side=0.0)

    pd.testing.assert_series_equal(result.returns, 0.6 * passive, check_names=False)
    assert result.metrics["mean_sleeve_allocation"] == pytest.approx(0.6)


def test_policy_portfolio_cash_rate() -> None:
    zero = pd.Series(0.0, index=INDEX)
    result = run_policy_portfolio(
        {"core": zero}, {"core": 0.5}, cash_rate_annual=0.04, cost_per_side=0.0
    )

    daily = 1.04 ** (1 / 252) - 1
    assert result.returns.iloc[0] == pytest.approx(0.5 * daily)


def test_policy_portfolio_vol_target_reduces_volatility() -> None:
    risky = _returns(6, vol=0.025)
    base = run_policy_portfolio({"core": risky}, {"core": 1.0})
    targeted = run_policy_portfolio(
        {"core": risky},
        {"core": 1.0},
        vol_target=VolTargetConfig(target_volatility=0.10, lookback_days=40),
    )

    assert targeted.metrics["annualized_volatility"] < base.metrics["annualized_volatility"]
    assert targeted.metrics["mean_sleeve_allocation"] < 1.0


def test_policy_portfolio_charges_exposure_changes() -> None:
    risky = _returns(7, vol=0.02)
    config = VolTargetConfig(lookback_days=20, rebalance_threshold=0.0)

    free = run_policy_portfolio({"core": risky}, {"core": 1.0}, vol_target=config, cost_per_side=0)
    costly = run_policy_portfolio(
        {"core": risky}, {"core": 1.0}, vol_target=config, cost_per_side=0.001
    )

    assert costly.metrics["total_return"] < free.metrics["total_return"]


def test_policy_portfolio_validation() -> None:
    passive = _returns(8)

    with pytest.raises(ValueError, match="almeno una"):
        run_policy_portfolio({}, {})

    with pytest.raises(ValueError, match="stessi nomi"):
        run_policy_portfolio({"core": passive}, {"other": 1.0})

    with pytest.raises(ValueError, match="leva"):
        run_policy_portfolio({"a": passive, "b": passive}, {"a": 0.7, "b": 0.5})

    with pytest.raises(ValueError, match="Peso non valido"):
        run_policy_portfolio({"core": passive}, {"core": -0.1})

    misaligned = passive.iloc[10:]
    with pytest.raises(ValueError, match="allineate"):
        run_policy_portfolio({"a": passive, "b": misaligned}, {"a": 0.5, "b": 0.5})
