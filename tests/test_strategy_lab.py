"""Test del motore del torneo delle strategie."""

import numpy as np
import pandas as pd
import pytest

from project_alpha import strategy_lab as lab


def _closes(days=400, seed=0, drift=0.0003):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=days)
    data = {symbol: 100 * np.cumprod(1 + rng.normal(drift, 0.01, days)) for symbol in lab.UNIVERSE}
    return pd.DataFrame(data, index=index)


def test_decisions_are_month_ends_with_enough_history() -> None:
    closes = _closes()
    dates = lab.decision_dates(closes.index)

    assert dates[0] >= closes.index[lab.LOOKBACK_12M]
    assert all(d.month != (d + pd.offsets.BDay(1)).month for d in dates)


def test_buy_and_hold_one_asset_matches_its_returns_after_one_day_lag() -> None:
    closes = _closes()
    start = lab.decision_dates(closes.index)[0]

    result = lab.run_strategy(closes, lambda h: {"SPY": 1.0}, start=start, cost_per_side=0.0)

    spy = closes["SPY"].pct_change()
    days = result.returns.index
    # Primo giorno: ancora in liquidità (il ribilanciamento avviene alla chiusura).
    assert result.returns.iloc[0] == 0.0
    np.testing.assert_allclose(result.returns.iloc[1:], spy.loc[days[1:]], atol=1e-12)


def test_costs_are_charged_on_turnover() -> None:
    closes = _closes()
    start = lab.decision_dates(closes.index)[0]

    result = lab.run_strategy(closes, lambda h: {"SPY": 1.0}, start=start, cost_per_side=0.001)

    assert result.returns.iloc[0] == pytest.approx(-0.001)
    assert result.turnover.iloc[0] == pytest.approx(1.0)


def test_weights_drift_between_rebalances() -> None:
    closes = _closes(drift=0.002)
    start = lab.decision_dates(closes.index)[0]

    result = lab.run_strategy(
        closes, lambda h: {"SPY": 0.5, "TLT": 0.5}, start=start, cost_per_side=0.0
    )

    # Ogni mese si torna a 50/50: il turnover dopo il primo è piccolo ma non nullo.
    assert result.turnover.iloc[0] == pytest.approx(1.0)
    assert 0 < result.turnover.iloc[1:].mean() < 0.2


def test_cash_earns_the_cash_rate() -> None:
    closes = _closes()
    start = lab.decision_dates(closes.index)[0]

    result = lab.run_strategy(closes, lambda h: {}, start=start, cash_rate_annual=0.02)

    expected = (1.02) ** (1 / lab.TRADING_DAYS) - 1
    np.testing.assert_allclose(result.returns, expected)


def test_strategies_respect_long_only_and_no_leverage() -> None:
    closes = _closes(days=600)
    history = closes.iloc[:400]

    for name, strategy in lab.STRATEGIES.items():
        weights = strategy(history)
        assert all(w >= 0 for w in weights.values()), name
        assert sum(weights.values()) <= 1.0 + 1e-9, name
        assert set(weights) <= set(lab.UNIVERSE), name


def test_dual_momentum_rules() -> None:
    index = pd.bdate_range("2020-01-01", periods=300)
    base = {s: np.linspace(100, 100, 300) for s in lab.UNIVERSE}
    up = dict(base, SPY=np.linspace(100, 130, 300), EFA=np.linspace(100, 120, 300))
    assert lab.dual_momentum(pd.DataFrame(up, index=index)) == {"SPY": 1.0}
    down = dict(base, SPY=np.linspace(100, 90, 300), EFA=np.linspace(100, 120, 300))
    assert lab.dual_momentum(pd.DataFrame(down, index=index)) == {"IEF": 1.0}


def test_bootstrap_detects_a_clearly_better_strategy() -> None:
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2020-01-01", periods=1500)
    noise = rng.normal(0, 0.01, 1500)
    reference = pd.Series(noise, index=index)
    better = pd.Series(noise + 0.001, index=index)

    assert lab.bootstrap_sharpe_difference(better, reference, samples=500) < 0.01
    assert lab.bootstrap_sharpe_difference(reference, better, samples=500) > 0.99


def test_tournament_table_applies_the_criteria() -> None:
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2017-01-02", "2026-06-30")
    noise = rng.normal(0.0002, 0.008, len(index))

    def result(values):
        return lab.LabResult(
            returns=pd.Series(values, index=index),
            weights=pd.DataFrame({"SPY": [1.0]}),
            turnover=pd.Series([1.0]),
        )

    table = lab.evaluate_tournament(
        {
            lab.REFERENCE: result(noise),
            "migliore": result(noise + 0.0008),
            "uguale": result(noise.copy()),
        },
        samples=300,
    )

    assert table.loc[lab.REFERENCE, "esito"] == "riferimento"
    assert table.loc["migliore", "esito"] == "BATTE IL RIFERIMENTO"
    assert table.loc["uguale", "esito"] == "scartata"
