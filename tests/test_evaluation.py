"""Test del modulo di valutazione (blocco 3)."""

from math import sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from project_alpha.evaluation import (
    TRADING_DAYS,
    decompose_gross_returns,
    deflated_sharpe_ratio,
    evaluate_walk_forward_portfolio,
    expected_maximum_sharpe,
    exposure_matched_comparison,
    minimum_track_record_length,
    ols_newey_west,
    on_off_returns,
    performance_summary,
    probabilistic_sharpe_ratio,
    timing_permutation_test,
)
from project_alpha.io_utils import dumps_json


def _returns(seed: int = 0, size: int = 1_000, mean: float = 0.0004) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, 0.01, size)


# --------------------------------------------------------------------------
# Statistiche di base e regressioni
# --------------------------------------------------------------------------


def test_performance_summary_hand_example() -> None:
    summary = performance_summary([0.10, -0.10, 0.10])

    assert summary["total_return"] == pytest.approx(1.1 * 0.9 * 1.1 - 1)
    assert summary["max_drawdown"] == pytest.approx(-0.10)
    assert summary["annualized_volatility"] == pytest.approx(
        np.std([0.10, -0.10, 0.10], ddof=1) * sqrt(TRADING_DAYS)
    )


def test_performance_summary_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="almeno due"):
        performance_summary([0.01])

    with pytest.raises(ValueError, match="non finiti"):
        performance_summary([0.01, float("nan")])


def test_ols_recovers_known_alpha_and_beta() -> None:
    benchmark = _returns(1)
    strategy = 0.0002 + 0.5 * benchmark

    result = ols_newey_west(strategy, benchmark, lags=5)

    assert result.alpha_daily == pytest.approx(0.0002)
    assert result.coefficients[1] == pytest.approx(0.5)
    assert result.r_squared == pytest.approx(1.0)
    assert result.observations == benchmark.size


def test_ols_t_statistic_is_small_for_noise() -> None:
    benchmark = _returns(2)
    strategy = 0.3 * benchmark + np.random.default_rng(3).normal(0, 0.005, benchmark.size)

    result = ols_newey_west(strategy, benchmark, lags=10)

    assert abs(result.alpha_t) < 3
    assert result.coefficients[1] == pytest.approx(0.3, abs=0.05)


def test_ols_validates_inputs() -> None:
    with pytest.raises(ValueError, match="lunghezze"):
        ols_newey_west(np.zeros(10) + 0.01, np.zeros(9))

    with pytest.raises(ValueError, match="lag"):
        ols_newey_west(_returns(size=50), _returns(1, size=50), lags=-1)


# --------------------------------------------------------------------------
# Pari esposizione e decomposizione
# --------------------------------------------------------------------------


def test_static_exposure_benchmark_has_same_sharpe_as_benchmark() -> None:
    benchmark = _returns(4)
    exposure = np.full(benchmark.size, 0.25)
    strategy = exposure * benchmark

    comparison = exposure_matched_comparison(strategy, benchmark, exposure)

    assert comparison["benchmark_static_exposure"]["exposure"] == pytest.approx(0.25)
    assert comparison["benchmark_static_exposure"]["annualized_sharpe"] == pytest.approx(
        comparison["benchmark"]["annualized_sharpe"]
    )
    # Una strategia che è solo "25% del benchmark" coincide con il benchmark statico.
    assert comparison["strategy"]["cagr"] == pytest.approx(
        comparison["benchmark_static_exposure"]["cagr"]
    )
    assert comparison["benchmark_static_beta"]["exposure"] == pytest.approx(0.25)


def test_decomposition_components_sum_to_gross_return() -> None:
    rng = np.random.default_rng(5)
    positions = (rng.random((500, 3)) > 0.7).astype(float)
    returns = rng.normal(0.0005, 0.01, (500, 3))
    weights = np.full(3, 1 / 3)

    parts = decompose_gross_returns(positions, returns, weights)

    assert (
        parts["exposure_component"]
        + parts["timing_component"]
        + parts["selection_component"]
    ) == pytest.approx(parts["gross_return_arithmetic"])


def test_decomposition_of_constant_full_exposure_is_pure_exposure() -> None:
    returns = np.random.default_rng(6).normal(0.0005, 0.01, (300, 2))
    positions = np.ones_like(returns)

    parts = decompose_gross_returns(positions, returns, np.array([0.5, 0.5]))

    assert parts["timing_component"] == pytest.approx(0.0, abs=1e-12)
    assert parts["selection_component"] == pytest.approx(0.0, abs=1e-12)


def test_decomposition_validates_shapes() -> None:
    with pytest.raises(ValueError, match="forme"):
        decompose_gross_returns(np.ones((5, 2)), np.ones((5, 3)), np.ones(2))

    with pytest.raises(ValueError, match="pesi"):
        decompose_gross_returns(np.ones((5, 2)), np.ones((5, 2)), np.ones(3))


def test_on_off_returns_hand_example() -> None:
    curves = {
        "AAA": pd.DataFrame(
            {
                "executed_position": [0, 1, 1, 0],
                "asset_return": [0.01, 0.02, 0.04, -0.01],
            }
        )
    }

    table = on_off_returns(curves)

    assert table.loc[0, "on_days"] == 2
    assert table.loc[0, "on_return_annual"] == pytest.approx(0.03 * TRADING_DAYS)
    assert table.loc[0, "off_return_annual"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Permutazioni
# --------------------------------------------------------------------------


def test_permutation_detects_perfect_timing() -> None:
    returns = np.random.default_rng(7).normal(0.0, 0.01, (400, 1))
    positions = (returns > 0).astype(float)

    result = timing_permutation_test(positions, returns, np.ones(1), permutations=500, seed=1)

    assert result["gross_return_annual"].p_value < 0.01
    assert result["gross_return_annual"].actual > result["gross_return_annual"].permutation_p95


def test_permutation_of_random_timing_is_not_significant() -> None:
    rng = np.random.default_rng(8)
    returns = rng.normal(0.0, 0.01, (400, 2))
    positions = (rng.random((400, 2)) > 0.5).astype(float)

    result = timing_permutation_test(positions, returns, np.full(2, 0.5), permutations=500, seed=2)

    assert result["gross_return_annual"].p_value > 0.05


def test_permutation_is_reproducible_with_seed() -> None:
    rng = np.random.default_rng(9)
    returns = rng.normal(0.0, 0.01, (200, 2))
    positions = (rng.random((200, 2)) > 0.5).astype(float)

    first = timing_permutation_test(positions, returns, np.full(2, 0.5), permutations=200, seed=3)
    second = timing_permutation_test(positions, returns, np.full(2, 0.5), permutations=200, seed=3)

    assert first == second


def test_permutation_preserves_exposure() -> None:
    """Uno shift circolare non cambia l'esposizione: con rendimenti costanti
    tutte le permutazioni danno lo stesso risultato della realtà."""
    positions = np.array([[1.0], [1.0], [0.0], [0.0], [1.0]])
    returns = np.full((5, 1), 0.01)

    result = timing_permutation_test(positions, returns, np.ones(1), permutations=50)

    assert result["gross_return_annual"].permutation_median == pytest.approx(
        result["gross_return_annual"].actual
    )


def test_permutation_validates_inputs() -> None:
    with pytest.raises(ValueError, match="almeno una"):
        timing_permutation_test(np.ones((5, 1)), np.ones((5, 1)), np.ones(1), permutations=0)

    with pytest.raises(ValueError, match="insufficienti"):
        timing_permutation_test(np.ones((2, 1)), np.ones((2, 1)), np.ones(1))


# --------------------------------------------------------------------------
# PSR, DSR, MinTRL
# --------------------------------------------------------------------------


def test_psr_matches_normal_formula_for_gaussian_returns() -> None:
    returns = _returns(10, size=2_000)
    sharpe = returns.mean() / returns.std(ddof=1)
    centered = returns - returns.mean()
    m2 = (centered**2).mean()
    skew = (centered**3).mean() / m2**1.5
    kurt = (centered**4).mean() / m2**2
    expected = NormalDist().cdf(
        sharpe * sqrt(returns.size - 1) / sqrt(1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2)
    )

    assert probabilistic_sharpe_ratio(returns) == pytest.approx(expected)


def test_psr_is_half_when_benchmark_equals_observed_sharpe() -> None:
    returns = _returns(11)
    sharpe = returns.mean() / returns.std(ddof=1)

    assert probabilistic_sharpe_ratio(returns, benchmark_sharpe=sharpe) == pytest.approx(0.5)


def test_expected_max_sharpe_grows_with_trials() -> None:
    one = expected_maximum_sharpe(trials=1, trials_sharpe_std=0.02)
    ten = expected_maximum_sharpe(trials=10, trials_sharpe_std=0.02)
    hundred = expected_maximum_sharpe(trials=100, trials_sharpe_std=0.02)

    assert one == 0.0
    assert 0 < ten < hundred

    with pytest.raises(ValueError):
        expected_maximum_sharpe(trials=0, trials_sharpe_std=0.02)


def test_dsr_is_lower_than_psr_with_many_trials() -> None:
    returns = _returns(12)
    psr = probabilistic_sharpe_ratio(returns)
    dsr = deflated_sharpe_ratio(returns, trials=50, trials_sharpe_std=0.02)

    assert dsr < psr


def test_min_track_record_length() -> None:
    returns = _returns(13, mean=0.001)
    length = minimum_track_record_length(returns)

    assert length is not None and length > 1
    # Con uno Sharpe più basso serve un track record più lungo.
    weaker = minimum_track_record_length(_returns(13, mean=0.0005))
    assert weaker is not None and weaker > length

    assert minimum_track_record_length(_returns(14, mean=-0.001)) is None

    with pytest.raises(ValueError, match="confidenza"):
        minimum_track_record_length(returns, confidence=1.5)


def test_psr_rejects_constant_returns() -> None:
    with pytest.raises(ValueError, match="varianza nulla"):
        probabilistic_sharpe_ratio([0.001] * 10)


# --------------------------------------------------------------------------
# Report completo su un portafoglio walk-forward
# --------------------------------------------------------------------------


def _market_frame(symbol: str, seed: int, periods: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.015, periods))
    open_ = close / (1.0 + rng.normal(0.0, 0.006, periods))

    return pd.DataFrame(
        {
            "symbol": [symbol] * periods,
            "timestamp": pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 3_000, periods).astype(float),
        }
    )


def test_evaluate_walk_forward_portfolio_end_to_end() -> None:
    from project_alpha.batch_optimization import run_batch_walk_forward_optimization
    from project_alpha.portfolio_walk_forward import run_walk_forward_portfolio_backtest

    frame = pd.concat(
        [_market_frame("AAA", 1), _market_frame("BBB", 2)],
        ignore_index=True,
    )
    grid = {
        "fast_ema_period": [3, 4],
        "slow_ema_period": [6],
        "trend_ema_period": [12],
        "rsi_period": [5],
        "rsi_lower": [40.0],
        "rsi_upper": [80.0],
        "atr_period": [5],
        "volume_period": [5],
        "breakout_period": [5],
    }
    batch = run_batch_walk_forward_optimization(
        frame,
        ["AAA", "BBB"],
        grid,
        train_size=60,
        test_size=40,
        score_metric="total_return",
        minimum_completed_trades=0,
    )
    portfolio = run_walk_forward_portfolio_backtest(frame, batch)

    report = evaluate_walk_forward_portfolio(portfolio, permutations=100, seed=0)
    summary = report.summary()

    metrics = portfolio.portfolio.metrics
    assert summary["strategy_cagr"] == pytest.approx(metrics["cagr"])
    assert summary["mean_exposure"] == pytest.approx(metrics["mean_gross_exposure"])
    assert 0.0 < summary["permutation_p_value_return"] <= 1.0
    assert summary["trials"] == 1
    assert summary["deflated_sharpe"] == summary["probabilistic_sharpe"]
    assert len(report.by_window) == len(batch.results["AAA"].windows)
    assert set(report.on_off["symbol"]) == {"AAA", "BBB"}

    # Il report completo deve essere serializzabile in JSON standard.
    assert '"summary"' in dumps_json(report.to_dict())

    # Più prove senza dispersione nota: DSR non calcolabile.
    unknown = evaluate_walk_forward_portfolio(portfolio, trials=10, permutations=10)
    assert unknown.deflated_sharpe is None

    known = evaluate_walk_forward_portfolio(
        portfolio,
        trials=10,
        trials_sharpe_std=0.02,
        permutations=10,
    )
    if known.probabilistic_sharpe is not None:
        assert known.deflated_sharpe <= known.probabilistic_sharpe
