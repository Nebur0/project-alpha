import pandas as pd
import pytest

from project_alpha.portfolio import (
    run_fixed_weight_portfolio_backtest,
)


def _curve(
    strategy_returns: list[float],
    asset_returns: list[float] | None = None,
    executed_positions: list[float] | None = None,
) -> pd.DataFrame:
    periods = len(strategy_returns)

    if asset_returns is None:
        asset_returns = strategy_returns

    if executed_positions is None:
        executed_positions = [1.0 for _ in range(periods)]

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=periods,
                freq="D",
                tz="UTC",
            ),
            "asset_return": asset_returns,
            "executed_position": (executed_positions),
            "strategy_return": (strategy_returns),
        }
    )


def test_equal_weight_portfolio_combines_daily_returns() -> None:
    curves = {
        "AAA": _curve(
            [0.0, 0.10, 0.0],
        ),
        "BBB": _curve(
            [0.0, 0.0, 0.10],
        ),
    }

    result = run_fixed_weight_portfolio_backtest(
        curves,
        initial_capital=1000.0,
    )

    curve = result.equity_curve

    assert result.weights == {
        "AAA": pytest.approx(0.5),
        "BBB": pytest.approx(0.5),
    }

    assert curve.loc[
        1,
        "portfolio_return",
    ] == pytest.approx(0.05)

    assert curve.loc[
        2,
        "portfolio_return",
    ] == pytest.approx(0.05)

    assert result.metrics["final_equity"] == pytest.approx(1102.5)

    assert result.metrics["total_return"] == pytest.approx(0.1025)


def test_custom_weights_are_applied() -> None:
    curves = {
        "AAA": _curve(
            [0.0, 0.10],
        ),
        "BBB": _curve(
            [0.0, -0.10],
        ),
    }

    result = run_fixed_weight_portfolio_backtest(
        curves,
        weights={
            "AAA": 0.75,
            "BBB": 0.25,
        },
        initial_capital=1000.0,
    )

    assert result.equity_curve.loc[
        1,
        "portfolio_return",
    ] == pytest.approx(0.05)

    assert result.metrics["final_equity"] == pytest.approx(1050.0)


def test_portfolio_tracks_weighted_benchmark_and_exposure() -> None:
    curves = {
        "AAA": _curve(
            [0.0, 0.0],
            asset_returns=[
                0.0,
                0.10,
            ],
            executed_positions=[
                0.0,
                1.0,
            ],
        ),
        "BBB": _curve(
            [0.0, 0.0],
            asset_returns=[
                0.0,
                -0.10,
            ],
            executed_positions=[
                0.0,
                0.0,
            ],
        ),
    }

    result = run_fixed_weight_portfolio_backtest(
        curves,
    )

    curve = result.equity_curve

    assert curve.loc[
        1,
        "benchmark_return",
    ] == pytest.approx(0.0)

    assert curve.loc[
        1,
        "gross_exposure",
    ] == pytest.approx(0.5)

    assert result.metrics["benchmark_total_return"] == pytest.approx(0.0)


def test_benchmark_risk_metrics_are_reported() -> None:
    curves = {
        "AAA": _curve(
            [
                0.0,
                0.0,
                0.0,
            ],
            asset_returns=[
                0.0,
                0.10,
                -0.05,
            ],
            executed_positions=[
                0.0,
                0.0,
                0.0,
            ],
        ),
    }

    result = run_fixed_weight_portfolio_backtest(
        curves,
        initial_capital=1000.0,
    )

    benchmark_returns = pd.Series(
        [
            0.0,
            0.10,
            -0.05,
        ],
        dtype=float,
    )

    expected_volatility = benchmark_returns.std(ddof=1) * (252.0**0.5)

    expected_sharpe = benchmark_returns.mean() / benchmark_returns.std(ddof=1) * (252.0**0.5)

    assert result.metrics["benchmark_final_equity"] == pytest.approx(1045.0)

    assert result.metrics["benchmark_total_return"] == pytest.approx(0.045)

    assert result.metrics["benchmark_max_drawdown"] == pytest.approx(-0.05)

    assert result.metrics["benchmark_annualized_volatility"] == pytest.approx(expected_volatility)

    assert result.metrics["benchmark_annualized_sharpe"] == pytest.approx(expected_sharpe)

    assert result.metrics["benchmark_cagr"] is not None


def test_mismatched_timestamps_are_rejected() -> None:
    first = _curve(
        [0.0, 0.01],
    )

    second = _curve(
        [0.0, 0.01],
    )

    second.loc[
        1,
        "timestamp",
    ] = pd.Timestamp(
        "2025-01-03",
        tz="UTC",
    )

    with pytest.raises(
        ValueError,
        match="stessi timestamp",
    ):
        run_fixed_weight_portfolio_backtest(
            {
                "AAA": first,
                "BBB": second,
            }
        )


def test_weights_must_match_symbols_and_sum_to_one() -> None:
    curves = {
        "AAA": _curve(
            [0.0, 0.01],
        ),
        "BBB": _curve(
            [0.0, 0.01],
        ),
    }

    with pytest.raises(
        ValueError,
        match="esattamente",
    ):
        run_fixed_weight_portfolio_backtest(
            curves,
            weights={
                "AAA": 1.0,
            },
        )

    with pytest.raises(
        ValueError,
        match="somma dei pesi",
    ):
        run_fixed_weight_portfolio_backtest(
            curves,
            weights={
                "AAA": 0.60,
                "BBB": 0.30,
            },
        )
