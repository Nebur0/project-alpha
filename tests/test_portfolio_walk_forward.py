import numpy as np
import pandas as pd
import pytest

from project_alpha.batch_optimization import (
    run_batch_walk_forward_optimization,
)
from project_alpha.optimization import (
    run_walk_forward_optimization,
)
from project_alpha.portfolio_walk_forward import (
    reconstruct_walk_forward_oos_curve,
    run_walk_forward_portfolio_backtest,
)


def _sample_market_frame(
    symbol: str,
    periods: int = 100,
    phase: float = 0.0,
) -> pd.DataFrame:
    index = np.arange(periods)

    close = 100.0 + index * 0.30 + np.sin(index / 3.0 + phase) * 2.0

    return pd.DataFrame(
        {
            "symbol": [symbol] * periods,
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=periods,
                freq="D",
                tz="UTC",
            ),
            "open": close - 0.50,
            "high": close + 1.00,
            "low": close - 1.00,
            "close": close,
            "volume": (1_000.0 + (index % 10) * 100.0),
        }
    )


def _small_parameter_grid() -> dict[
    str,
    list[int | float],
]:
    return {
        "fast_ema_period": [
            3,
            4,
        ],
        "slow_ema_period": [
            6,
        ],
        "trend_ema_period": [
            12,
        ],
        "rsi_period": [
            5,
        ],
        "rsi_lower": [
            40.0,
        ],
        "rsi_upper": [
            80.0,
        ],
        "atr_period": [
            5,
        ],
        "volume_period": [
            5,
        ],
        "breakout_period": [
            5,
        ],
    }


def test_reconstructed_curve_matches_window_returns() -> None:
    frame = _sample_market_frame(
        "AAA",
    )

    result = run_walk_forward_optimization(
        frame,
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    curve = reconstruct_walk_forward_oos_curve(
        frame,
        result,
    )

    assert len(curve) == (len(result.windows) * result.test_size)

    assert not curve["timestamp"].duplicated().any()

    for window in result.windows:
        selected = curve.loc[curve["window_number"].eq(window.window_number)]

        reconstructed_return = float((1.0 + selected["strategy_return"]).prod() - 1.0)

        assert reconstructed_return == (pytest.approx(window.out_of_sample_metrics["total_return"]))


def test_reconstructed_curve_compounds_across_windows() -> None:
    frame = _sample_market_frame(
        "AAA",
    )

    result = run_walk_forward_optimization(
        frame,
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    curve = reconstruct_walk_forward_oos_curve(
        frame,
        result,
        initial_capital=1000.0,
    )

    expected = 1000.0 * (1.0 + curve["strategy_return"]).prod()

    assert curve["walk_forward_equity"].iloc[-1] == pytest.approx(expected)


def test_reconstruction_rejects_overlapping_windows() -> None:
    frame = _sample_market_frame(
        "AAA",
        periods=90,
    )

    result = run_walk_forward_optimization(
        frame,
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    with pytest.raises(
        ValueError,
        match="sovrapposte",
    ):
        reconstruct_walk_forward_oos_curve(
            frame,
            result,
        )


def test_walk_forward_portfolio_combines_symbol_curves() -> None:
    first = _sample_market_frame(
        "AAA",
        phase=0.0,
    )

    second = _sample_market_frame(
        "BBB",
        phase=0.7,
    )

    frame = pd.concat(
        [
            first,
            second,
        ],
        ignore_index=True,
    )

    batch_result = run_batch_walk_forward_optimization(
        frame,
        [
            "AAA",
            "BBB",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    result = run_walk_forward_portfolio_backtest(
        frame,
        batch_result,
        initial_capital=1000.0,
    )

    portfolio_curve = result.portfolio.equity_curve

    first_returns = result.symbol_curves["AAA"]["strategy_return"].reset_index(drop=True)

    second_returns = result.symbol_curves["BBB"]["strategy_return"].reset_index(drop=True)

    expected_return = first_returns * 0.5 + second_returns * 0.5

    pd.testing.assert_series_equal(
        portfolio_curve["portfolio_return"].reset_index(drop=True),
        expected_return,
        check_names=False,
    )

    expected_final_equity = 1000.0 * (1.0 + expected_return).prod()

    assert result.portfolio.metrics["final_equity"] == pytest.approx(expected_final_equity)


def test_walk_forward_portfolio_supports_custom_weights() -> None:
    first = _sample_market_frame(
        "AAA",
        phase=0.0,
    )

    second = _sample_market_frame(
        "BBB",
        phase=1.2,
    )

    frame = pd.concat(
        [
            first,
            second,
        ],
        ignore_index=True,
    )

    batch_result = run_batch_walk_forward_optimization(
        frame,
        [
            "AAA",
            "BBB",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    result = run_walk_forward_portfolio_backtest(
        frame,
        batch_result,
        weights={
            "AAA": 0.75,
            "BBB": 0.25,
        },
    )

    assert result.portfolio.weights == {
        "AAA": pytest.approx(0.75),
        "BBB": pytest.approx(0.25),
    }
