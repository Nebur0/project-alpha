from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import project_alpha.optimization as optimization_module
from project_alpha.optimization import (
    generate_strategy_parameters,
    grid_search_results_to_frame,
    run_parameter_grid_search,
    run_walk_forward_optimization,
    walk_forward_optimization_to_dict,
    walk_forward_optimization_to_frame,
)
from project_alpha.scoring import (
    RobustScoreConfig,
)
from project_alpha.validation import (
    generate_walk_forward_splits,
)


def _sample_market_frame(
    symbol: str = "TEST",
    periods: int = 80,
) -> pd.DataFrame:
    index = np.arange(periods)

    close = 100.0 + index * 0.30 + np.sin(index / 3.0) * 2.0

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


def test_generate_strategy_parameters_creates_combinations() -> None:
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


def test_generate_strategy_parameters_uses_defaults_for_missing_fields() -> None:
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
    assert result.rsi_lower == pytest.approx(50.0)
    assert result.rsi_upper == pytest.approx(70.0)
    assert result.atr_period == 14
    assert result.volume_period == 20
    assert result.breakout_period == 20


def test_generate_strategy_parameters_rejects_invalid_grid() -> None:
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


def test_run_parameter_grid_search_returns_ranked_results() -> None:
    frame = _sample_market_frame()

    results = run_parameter_grid_search(
        frame,
        _small_parameter_grid(),
        score_metric="total_return",
    )

    assert len(results) == 2

    assert all(result.score_metric == "total_return" for result in results)

    assert {result.parameters.fast_ema_period for result in results} == {
        3,
        4,
    }

    scores = [result.score for result in results]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    assert all("total_return" in result.metrics for result in results)


def test_run_parameter_grid_search_rejects_multiple_symbols() -> None:
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


def test_grid_search_results_to_frame_creates_ranked_table() -> None:
    results = run_parameter_grid_search(
        _sample_market_frame(),
        _small_parameter_grid(),
        score_metric="total_return",
    )

    frame = grid_search_results_to_frame(results)

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

    assert expected_columns.issubset(frame.columns)


def test_walk_forward_optimization_selects_best_training_result() -> None:
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

    first_training_results = run_parameter_grid_search(
        splits[0].in_sample,
        grid,
        score_metric="total_return",
    )

    expected = first_training_results[0]
    first_window = result.windows[0]

    assert first_window.selected_parameters == (expected.parameters)

    assert first_window.selected_score == pytest.approx(expected.score)

    assert first_window.in_sample_metrics == expected.metrics

    assert "total_return" in first_window.out_of_sample_metrics


def test_walk_forward_optimization_reports_window_metadata() -> None:
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

    window_numbers = [window.window_number for window in result.windows]

    assert window_numbers == list(
        range(
            1,
            len(result.windows) + 1,
        )
    )

    assert "window_count" in result.summary
    assert "positive_window_count" in (result.summary)


def test_walk_forward_optimization_detects_overlapping_windows() -> None:
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
        match=("minimum_completed_trades deve essere un intero non negativo"),
    ):
        run_walk_forward_optimization(
            _sample_market_frame(),
            _small_parameter_grid(),
            train_size=40,
            test_size=20,
            score_metric="total_return",
            minimum_completed_trades=(minimum_completed_trades),
        )


def test_walk_forward_optimization_rejects_candidates_below_trade_limit() -> None:
    with pytest.raises(
        ValueError,
        match=("Nessuna combinazione soddisfa"),
    ):
        run_walk_forward_optimization(
            _sample_market_frame(),
            _small_parameter_grid(),
            train_size=40,
            test_size=20,
            score_metric="total_return",
            minimum_completed_trades=10_000,
        )


def test_walk_forward_optimization_serializers() -> None:
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

    report = walk_forward_optimization_to_dict(result)

    assert report["symbol"] == "TEST"
    assert report["score_metric"] == ("total_return")
    assert report["window_count"] == len(result.windows)
    assert report["windows_overlap"] is False
    assert report["compounding_enabled"] is True

    windows = report["windows"]

    assert isinstance(
        windows,
        list,
    )
    assert len(windows) == len(result.windows)

    first_window = windows[0]

    assert "selected_parameters" in (first_window)
    assert "in_sample" in first_window
    assert "out_of_sample" in first_window

    frame = walk_forward_optimization_to_frame(result)

    assert len(frame) == len(result.windows)

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

    assert expected_columns.issubset(frame.columns)


def _install_deterministic_robust_scoring_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_add_indicators(
        frame: pd.DataFrame,
        parameters: object,
    ) -> pd.DataFrame:
        return frame.copy()

    def fake_add_baseline_position(
        frame: pd.DataFrame,
        parameters: object,
    ) -> pd.DataFrame:
        result = frame.copy()
        result["candidate_fast_ema_period"] = parameters.fast_ema_period
        return result

    def fake_run_long_only_backtest(
        frame: pd.DataFrame,
        *,
        initial_capital: float,
        cost_per_side: float,
    ) -> SimpleNamespace:
        fast_ema_period = int(frame["candidate_fast_ema_period"].iloc[0])

        if fast_ema_period == 3:
            metrics = {
                "initial_capital": initial_capital,
                "final_equity": (initial_capital * 1.20),
                "total_return": 0.20,
                "benchmark_total_return": 0.30,
                "cagr": 0.20,
                "annualized_sharpe": 1.00,
                "max_drawdown": -0.10,
                "trade_events": 40,
                "completed_trades": 20,
                "winning_trades": 12,
                "losing_trades": 8,
                "win_rate": 0.60,
                "average_trade_return": 0.01,
                "average_win": 0.02,
                "average_loss": -0.01,
                "profit_factor": 1.50,
                "market_exposure": 0.40,
                "cost_per_side": cost_per_side,
            }
        else:
            metrics = {
                "initial_capital": initial_capital,
                "final_equity": (initial_capital * 1.50),
                "total_return": 0.50,
                "benchmark_total_return": 0.30,
                "cagr": 0.50,
                "annualized_sharpe": 0.80,
                "max_drawdown": -0.05,
                "trade_events": 40,
                "completed_trades": 20,
                "winning_trades": 11,
                "losing_trades": 9,
                "win_rate": 0.55,
                "average_trade_return": 0.02,
                "average_win": 0.03,
                "average_loss": -0.01,
                "profit_factor": 1.80,
                "market_exposure": 0.45,
                "cost_per_side": cost_per_side,
            }

        return SimpleNamespace(metrics=metrics)

    monkeypatch.setattr(
        optimization_module,
        "add_indicators",
        fake_add_indicators,
    )
    monkeypatch.setattr(
        optimization_module,
        "add_baseline_position",
        fake_add_baseline_position,
    )
    monkeypatch.setattr(
        optimization_module,
        "run_long_only_backtest",
        fake_run_long_only_backtest,
    )


def test_parameter_grid_search_robust_score_changes_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_robust_scoring_mocks(monkeypatch)

    frame = _sample_market_frame()

    sharpe_results = run_parameter_grid_search(
        frame,
        _small_parameter_grid(),
        score_metric="annualized_sharpe",
    )

    robust_results = run_parameter_grid_search(
        frame,
        _small_parameter_grid(),
        score_metric="robust_score",
    )

    assert sharpe_results[0].parameters.fast_ema_period == 3

    assert robust_results[0].parameters.fast_ema_period == 4

    assert robust_results[0].score_metric == ("robust_score")
    assert robust_results[0].robust_score is not None
    assert robust_results[0].score == pytest.approx(0.95)
    assert robust_results[0].robust_score.excess_return == pytest.approx(0.20)


def test_parameter_grid_search_uses_custom_robust_score_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_robust_scoring_mocks(monkeypatch)

    config = RobustScoreConfig(
        sharpe_weight=10.0,
        excess_return_weight=0.0,
        drawdown_weight=0.0,
        trade_shortfall_weight=0.0,
        target_completed_trades=20,
    )

    results = run_parameter_grid_search(
        _sample_market_frame(),
        _small_parameter_grid(),
        score_metric="robust_score",
        robust_score_config=config,
    )

    assert results[0].parameters.fast_ema_period == 3
    assert results[0].score == pytest.approx(10.0)

    frame = grid_search_results_to_frame(results)

    expected_columns = {
        "robust_score",
        "robust_sharpe",
        "robust_excess_return",
        "robust_max_drawdown",
        "robust_completed_trades",
        "robust_trade_shortfall",
    }

    assert expected_columns.issubset(frame.columns)


def test_parameter_grid_search_rejects_robust_config_with_other_metric() -> None:
    with pytest.raises(
        ValueError,
        match=("robust_score_config può essere usato solo"),
    ):
        run_parameter_grid_search(
            _sample_market_frame(),
            _small_parameter_grid(),
            score_metric="total_return",
            robust_score_config=(RobustScoreConfig()),
        )


def test_walk_forward_optimization_propagates_robust_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_robust_scoring_mocks(monkeypatch)

    config = RobustScoreConfig(
        target_completed_trades=20,
    )

    result = run_walk_forward_optimization(
        _sample_market_frame(
            periods=100,
        ),
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="robust_score",
        minimum_completed_trades=0,
        robust_score_config=config,
    )

    assert result.score_metric == "robust_score"
    assert result.robust_score_config == config
    assert len(result.windows) > 0

    first_window = result.windows[0]

    assert first_window.selected_parameters.fast_ema_period == 4
    assert first_window.robust_score is not None
    assert first_window.selected_score == (pytest.approx(first_window.robust_score.score))

    report = walk_forward_optimization_to_dict(result)

    assert report["robust_score_config"] == {
        "sharpe_weight": 1.0,
        "excess_return_weight": 1.0,
        "drawdown_weight": 1.0,
        "trade_shortfall_weight": 0.5,
        "target_completed_trades": 20,
    }

    report_windows = report["windows"]

    assert isinstance(
        report_windows,
        list,
    )
    assert report_windows[0]["selected_robust_score"] is not None

    frame = walk_forward_optimization_to_frame(result)

    expected_columns = {
        "selected_robust_score",
        "selected_robust_excess_return",
        "selected_robust_trade_shortfall",
    }

    assert expected_columns.issubset(frame.columns)
