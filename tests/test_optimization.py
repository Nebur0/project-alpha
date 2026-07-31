import numpy as np
import pandas as pd
import pytest

from project_alpha.optimization import (
    generate_strategy_parameters,
    grid_search_results_to_frame,
    run_parameter_grid_search,
    run_walk_forward_optimization,
    walk_forward_optimization_to_dict,
    walk_forward_optimization_to_frame,
)
from project_alpha.validation import (
    generate_walk_forward_splits,
)


def _sample_market_frame(
    symbol: str = "TEST",
    periods: int = 80,
) -> pd.DataFrame:
    index = np.arange(periods)

    close = (
        100.0
        + index * 0.30
        + np.sin(index / 3.0) * 2.0
    )

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
            "volume": (
                1_000.0
                + (index % 10) * 100.0
            ),
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


def test_generate_strategy_parameters_creates_combinations(
) -> None:
    parameters = generate_strategy_parameters(
        {
            "fast_ema_period": [
                5,
                10,
            ],
            "slow_ema_period": [
                20,
            ],
            "trend_ema_period": [
                100,
                200,
            ],
        }
    )

    assert len(parameters) == 4

    combinations = {
        (
            item.fast_ema_period,
            item.slow_ema_period,
            item.trend_ema_period,
        )
        for item in parameters
    }

    assert combinations == {
        (
            5,
            20,
            100,
        ),
        (
            5,
            20,
            200,
        ),
        (
            10,
            20,
            100,
        ),
        (
            10,
            20,
            200,
        ),
    }


def test_generate_strategy_parameters_uses_defaults_for_missing_fields(
) -> None:
    parameters = generate_strategy_parameters(
        {
            "fast_ema_period": [
                10,
            ],
            "slow_ema_period": [
                30,
            ],
            "trend_ema_period": [
                100,
            ],
        }
    )

    result = parameters[0]

    assert result.rsi_period == 14
    assert result.rsi_lower == pytest.approx(
        50.0
    )
    assert result.rsi_upper == pytest.approx(
        70.0
    )
    assert result.atr_period == 14
    assert result.volume_period == 20
    assert result.breakout_period == 20


def test_generate_strategy_parameters_rejects_invalid_grid(
) -> None:
    with pytest.raises(
        ValueError,
        match="griglia dei parametri è vuota",
    ):
        generate_strategy_parameters({})

    with pytest.raises(
        ValueError,
        match="Parametri strategici sconosciuti",
    ):
        generate_strategy_parameters(
            {
                "unknown_parameter": [
                    10,
                ],
            }
        )

    with pytest.raises(
        ValueError,
        match="è vuota",
    ):
        generate_strategy_parameters(
            {
                "fast_ema_period": [],
            }
        )


def test_run_parameter_grid_search_returns_ranked_results(
) -> None:
    frame = _sample_market_frame()

    results = run_parameter_grid_search(
        frame,
        _small_parameter_grid(),
        score_metric="total_return",
    )

    assert len(results) == 2

    assert all(
        result.score_metric
        == "total_return"
        for result in results
    )

    assert {
        result.parameters.fast_ema_period
        for result in results
    } == {
        3,
        4,
    }

    scores = [
        result.score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    assert all(
        "total_return"
        in result.metrics
        for result in results
    )


def test_run_parameter_grid_search_rejects_multiple_symbols(
) -> None:
    first = _sample_market_frame(
        symbol="SPY",
    )

    second = _sample_market_frame(
        symbol="QQQ",
    )

    frame = pd.concat(
        [
            first,
            second,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="un solo simbolo",
    ):
        run_parameter_grid_search(
            frame,
            _small_parameter_grid(),
        )


def test_grid_search_results_to_frame_creates_ranked_table(
) -> None:
    results = run_parameter_grid_search(
        _sample_market_frame(),
        _small_parameter_grid(),
        score_metric="total_return",
    )

    frame = grid_search_results_to_frame(
        results
    )

    assert frame["rank"].tolist() == [
        1,
        2,
    ]

    expected_columns = {
        "rank",
        "fast_ema_period",
        "slow_ema_period",
        "trend_ema_period",
        "rsi_period",
        "rsi_lower",
        "rsi_upper",
        "atr_period",
        "volume_period",
        "breakout_period",
        "score_metric",
        "score",
        "total_return",
        "annualized_sharpe",
        "max_drawdown",
    }

    assert expected_columns.issubset(
        frame.columns
    )


def test_walk_forward_optimization_selects_best_training_result(
) -> None:
    frame = _sample_market_frame(
        periods=100,
    )

    grid = _small_parameter_grid()

    result = run_walk_forward_optimization(
        frame,
        grid,
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    splits = generate_walk_forward_splits(
        frame=frame,
        train_size=40,
        test_size=20,
        step_size=20,
    )

    first_training_results = (
        run_parameter_grid_search(
            splits[0].in_sample,
            grid,
            score_metric="total_return",
        )
    )

    expected = first_training_results[0]
    first_window = result.windows[0]

    assert first_window.selected_parameters == (
        expected.parameters
    )

    assert (
        first_window.selected_score
        == pytest.approx(
            expected.score
        )
    )

    assert (
        first_window.in_sample_metrics
        == expected.metrics
    )

    assert (
        "total_return"
        in first_window.out_of_sample_metrics
    )


def test_walk_forward_optimization_reports_window_metadata(
) -> None:
    result = run_walk_forward_optimization(
        _sample_market_frame(
            periods=100,
        ),
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    assert result.symbol == "TEST"
    assert result.score_metric == "total_return"
    assert result.train_size == 40
    assert result.test_size == 20
    assert result.step_size == 20
    assert result.minimum_completed_trades == 0
    assert result.windows_overlap is False
    assert result.compounding_enabled is True
    assert len(result.windows) > 0

    window_numbers = [
        window.window_number
        for window in result.windows
    ]

    assert window_numbers == list(
        range(
            1,
            len(result.windows) + 1,
        )
    )

    assert "window_count" in result.summary
    assert "positive_window_count" in (
        result.summary
    )


def test_walk_forward_optimization_detects_overlapping_windows(
) -> None:
    result = run_walk_forward_optimization(
        _sample_market_frame(
            periods=90,
        ),
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    assert result.step_size == 10
    assert result.windows_overlap is True
    assert result.compounding_enabled is False


@pytest.mark.parametrize(
    "minimum_completed_trades",
    [
        -1,
        True,
        1.5,
    ],
)
def test_walk_forward_optimization_rejects_invalid_minimum_trades(
    minimum_completed_trades: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "minimum_completed_trades deve "
            "essere un intero non negativo"
        ),
    ):
        run_walk_forward_optimization(
            _sample_market_frame(),
            _small_parameter_grid(),
            train_size=40,
            test_size=20,
            score_metric="total_return",
            minimum_completed_trades=(
                minimum_completed_trades
            ),
        )


def test_walk_forward_optimization_rejects_candidates_below_trade_limit(
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Nessuna combinazione soddisfa"
        ),
    ):
        run_walk_forward_optimization(
            _sample_market_frame(),
            _small_parameter_grid(),
            train_size=40,
            test_size=20,
            score_metric="total_return",
            minimum_completed_trades=10_000,
        )


def test_walk_forward_optimization_serializers(
) -> None:
    result = run_walk_forward_optimization(
        _sample_market_frame(
            periods=100,
        ),
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    report = walk_forward_optimization_to_dict(
        result
    )

    assert report["symbol"] == "TEST"
    assert report["score_metric"] == (
        "total_return"
    )
    assert report["window_count"] == len(
        result.windows
    )
    assert report["windows_overlap"] is False
    assert report["compounding_enabled"] is True

    windows = report["windows"]

    assert isinstance(
        windows,
        list,
    )
    assert len(windows) == len(
        result.windows
    )

    first_window = windows[0]

    assert "selected_parameters" in (
        first_window
    )
    assert "in_sample" in first_window
    assert "out_of_sample" in first_window

    frame = walk_forward_optimization_to_frame(
        result
    )

    assert len(frame) == len(
        result.windows
    )

    expected_columns = {
        "symbol",
        "window_number",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "fast_ema_period",
        "slow_ema_period",
        "trend_ema_period",
        "rsi_period",
        "rsi_lower",
        "rsi_upper",
        "atr_period",
        "volume_period",
        "breakout_period",
        "score_metric",
        "selected_score",
        "in_sample_total_return",
        "out_of_sample_total_return",
        "out_of_sample_annualized_sharpe",
        "out_of_sample_max_drawdown",
    }

    assert expected_columns.issubset(
        frame.columns
    )