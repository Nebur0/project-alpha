from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import project_alpha.optimization as optimization_module
from project_alpha.batch_optimization import (
    batch_walk_forward_comparison_to_frame,
    batch_walk_forward_optimization_to_dict,
    batch_walk_forward_optimization_to_frame,
    run_batch_walk_forward_optimization,
)
from project_alpha.scoring import (
    RobustScoreConfig,
    RobustScoreV3Config,
)


def _sample_symbol_frame(
    symbol: str,
    periods: int = 100,
    *,
    offset: float = 0.0,
) -> pd.DataFrame:
    index = np.arange(periods)

    close = (
        100.0
        + offset
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


def _sample_batch_frame() -> pd.DataFrame:
    return pd.concat(
        [
            _sample_symbol_frame(
                "SPY",
                offset=0.0,
            ),
            _sample_symbol_frame(
                "QQQ",
                offset=10.0,
            ),
        ],
        ignore_index=True,
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

        result["candidate_fast_ema_period"] = (
            parameters.fast_ema_period
        )

        return result

    def fake_run_long_only_backtest(
        frame: pd.DataFrame,
        *,
        initial_capital: float,
        cost_per_side: float,
    ) -> SimpleNamespace:
        fast_ema_period = int(
            frame[
                "candidate_fast_ema_period"
            ].iloc[0]
        )

        if fast_ema_period == 3:
            metrics = {
                "initial_capital": initial_capital,
                "final_equity": (
                    initial_capital * 1.20
                ),
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
                "final_equity": (
                    initial_capital * 1.50
                ),
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

        return SimpleNamespace(
            metrics=metrics
        )

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


def test_batch_walk_forward_optimization_runs_multiple_symbols(
) -> None:
    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
            "QQQ",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    assert result.symbols == (
        "SPY",
        "QQQ",
    )
    assert result.score_metric == (
        "total_return"
    )
    assert result.train_size == 40
    assert result.test_size == 20
    assert result.step_size == 20
    assert result.minimum_completed_trades == 0
    assert result.robust_score_config is None
    assert result.robust_score_v3_config is None
    assert set(result.results) == {
        "SPY",
        "QQQ",
    }
    assert (
        result.results["SPY"].symbol
        == "SPY"
    )
    assert (
        result.results["QQQ"].symbol
        == "QQQ"
    )


def test_batch_walk_forward_optimization_normalizes_symbols(
) -> None:
    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "spy",
            "QQQ",
            "SPY",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    assert result.symbols == (
        "SPY",
        "QQQ",
    )


def test_batch_walk_forward_optimization_uses_default_step_size(
) -> None:
    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    assert result.step_size == 20


def test_batch_walk_forward_optimization_rejects_empty_frame(
) -> None:
    with pytest.raises(
        ValueError,
        match="dataset batch è vuoto",
    ):
        run_batch_walk_forward_optimization(
            pd.DataFrame(),
            [
                "SPY",
            ],
            _small_parameter_grid(),
            train_size=40,
            test_size=20,
        )


def test_batch_walk_forward_optimization_requires_symbol_column(
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=10,
                freq="D",
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="colonna 'symbol'",
    ):
        run_batch_walk_forward_optimization(
            frame,
            [
                "SPY",
            ],
            _small_parameter_grid(),
            train_size=5,
            test_size=2,
        )


def test_batch_walk_forward_optimization_rejects_empty_symbols(
) -> None:
    with pytest.raises(
        ValueError,
        match="almeno un simbolo",
    ):
        run_batch_walk_forward_optimization(
            _sample_batch_frame(),
            [],
            _small_parameter_grid(),
            train_size=40,
            test_size=20,
        )


def test_batch_walk_forward_optimization_rejects_missing_symbols(
) -> None:
    with pytest.raises(
        ValueError,
        match="Simboli non trovati",
    ):
        run_batch_walk_forward_optimization(
            _sample_batch_frame(),
            [
                "SPY",
                "AAPL",
            ],
            _small_parameter_grid(),
            train_size=40,
            test_size=20,
        )


def test_batch_walk_forward_optimization_to_dict(
) -> None:
    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
            "QQQ",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    report = (
        batch_walk_forward_optimization_to_dict(
            result
        )
    )

    assert report["symbols"] == [
        "SPY",
        "QQQ",
    ]
    assert report["symbol_count"] == 2
    assert report["train_size"] == 40
    assert report["test_size"] == 20
    assert report["step_size"] == 20
    assert report["robust_score_config"] is None
    assert report["robust_score_v3_config"] is None

    results = report["results"]

    assert isinstance(
        results,
        dict,
    )
    assert set(results) == {
        "SPY",
        "QQQ",
    }
    assert (
        results["SPY"]["symbol"]
        == "SPY"
    )
    assert "summary" in results["SPY"]
    assert "windows" in results["SPY"]


def test_batch_walk_forward_optimization_to_frame(
) -> None:
    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
            "QQQ",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    frame = (
        batch_walk_forward_optimization_to_frame(
            result
        )
    )

    assert not frame.empty
    assert set(
        frame["symbol"].unique()
    ) == {
        "SPY",
        "QQQ",
    }

    expected_columns = {
        "symbol",
        "window_number",
        "fast_ema_period",
        "slow_ema_period",
        "trend_ema_period",
        "selected_score",
        "in_sample_total_return",
        "out_of_sample_total_return",
    }

    assert expected_columns.issubset(
        frame.columns
    )


def test_batch_walk_forward_comparison_to_frame(
) -> None:
    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
            "QQQ",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    frame = (
        batch_walk_forward_comparison_to_frame(
            result
        )
    )

    assert frame["symbol"].tolist() == [
        "SPY",
        "QQQ",
    ]

    expected_columns = {
        "symbol",
        "score_metric",
        "train_size",
        "test_size",
        "step_size",
        "minimum_completed_trades",
        "window_count",
        "windows_overlap",
        "compounding_enabled",
        "positive_window_count",
        "positive_window_rate",
        "mean_window_annualized_sharpe",
        "worst_max_drawdown",
        "total_completed_trades",
    }

    assert expected_columns.issubset(
        frame.columns
    )


def test_batch_walk_forward_optimization_supports_robust_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_robust_scoring_mocks(
        monkeypatch
    )

    config = RobustScoreConfig(
        sharpe_weight=1.5,
        excess_return_weight=2.0,
        drawdown_weight=0.75,
        trade_shortfall_weight=0.25,
        target_completed_trades=5,
    )

    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
            "QQQ",
        ],
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

    for symbol in result.symbols:
        symbol_result = result.results[symbol]

        assert (
            symbol_result.score_metric
            == "robust_score"
        )
        assert (
            symbol_result.robust_score_config
            == config
        )
        assert len(symbol_result.windows) > 0

        for window in symbol_result.windows:
            assert window.robust_score is not None
            assert (
                window.selected_score
                == pytest.approx(
                    window.robust_score.score
                )
            )


def test_batch_walk_forward_optimization_robust_config_is_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_robust_scoring_mocks(
        monkeypatch
    )

    config = RobustScoreConfig(
        sharpe_weight=1.5,
        excess_return_weight=2.0,
        drawdown_weight=0.75,
        trade_shortfall_weight=0.25,
        target_completed_trades=5,
    )

    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="robust_score",
        minimum_completed_trades=0,
        robust_score_config=config,
    )

    report = (
        batch_walk_forward_optimization_to_dict(
            result
        )
    )

    assert report["robust_score_config"] == {
        "sharpe_weight": 1.5,
        "excess_return_weight": 2.0,
        "drawdown_weight": 0.75,
        "trade_shortfall_weight": 0.25,
        "target_completed_trades": 5,
    }


def test_batch_walk_forward_comparison_includes_robust_config_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_robust_scoring_mocks(
        monkeypatch
    )

    config = RobustScoreConfig(
        sharpe_weight=1.5,
        excess_return_weight=2.0,
        drawdown_weight=0.75,
        trade_shortfall_weight=0.25,
        target_completed_trades=5,
    )

    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
            "QQQ",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="robust_score",
        minimum_completed_trades=0,
        robust_score_config=config,
    )

    frame = (
        batch_walk_forward_comparison_to_frame(
            result
        )
    )

    expected_columns = {
        "robust_sharpe_weight",
        "robust_excess_return_weight",
        "robust_drawdown_weight",
        "robust_trade_shortfall_weight",
        "robust_target_completed_trades",
    }

    assert expected_columns.issubset(
        frame.columns
    )

    assert set(
        frame["robust_target_completed_trades"]
    ) == {
        5,
    }


def test_batch_walk_forward_optimization_supports_robust_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_robust_scoring_mocks(
        monkeypatch
    )

    base_config = RobustScoreConfig(
        sharpe_weight=1.5,
        excess_return_weight=2.0,
        drawdown_weight=0.75,
        trade_shortfall_weight=0.25,
        target_completed_trades=5,
    )

    v3_config = RobustScoreV3Config(
        base_score_weight=0.5,
        neighborhood_score_weight=0.5,
    )

    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
            "QQQ",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="robust_score_v3",
        minimum_completed_trades=0,
        robust_score_config=base_config,
        robust_score_v3_config=v3_config,
    )

    assert (
        result.score_metric
        == "robust_score_v3"
    )
    assert (
        result.robust_score_config
        == base_config
    )
    assert (
        result.robust_score_v3_config
        == v3_config
    )

    for symbol in result.symbols:
        symbol_result = result.results[symbol]

        assert (
            symbol_result.score_metric
            == "robust_score_v3"
        )
        assert (
            symbol_result.robust_score_config
            == base_config
        )
        assert (
            symbol_result
            .robust_score_v3_config
            == v3_config
        )
        assert len(symbol_result.windows) > 0

        for window in symbol_result.windows:
            assert (
                window.robust_score
                is not None
            )
            assert (
                window.robust_score_v3
                is not None
            )
            assert (
                window.selected_score
                == pytest.approx(
                    window
                    .robust_score_v3
                    .score
                )
            )


def test_batch_walk_forward_optimization_v3_config_is_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_robust_scoring_mocks(
        monkeypatch
    )

    base_config = RobustScoreConfig(
        sharpe_weight=1.5,
        excess_return_weight=2.0,
        drawdown_weight=0.75,
        trade_shortfall_weight=0.25,
        target_completed_trades=5,
    )

    v3_config = RobustScoreV3Config(
        base_score_weight=0.25,
        neighborhood_score_weight=0.75,
    )

    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="robust_score_v3",
        minimum_completed_trades=0,
        robust_score_config=base_config,
        robust_score_v3_config=v3_config,
    )

    report = (
        batch_walk_forward_optimization_to_dict(
            result
        )
    )

    assert report["robust_score_config"] == {
        "sharpe_weight": 1.5,
        "excess_return_weight": 2.0,
        "drawdown_weight": 0.75,
        "trade_shortfall_weight": 0.25,
        "target_completed_trades": 5,
    }

    assert report["robust_score_v3_config"] == {
        "base_score_weight": 0.25,
        "neighborhood_score_weight": 0.75,
    }

    symbol_report = report["results"]["SPY"]

    assert (
        symbol_report[
            "robust_score_v3_config"
        ]
        == {
            "base_score_weight": 0.25,
            "neighborhood_score_weight": 0.75,
        }
    )


def test_batch_walk_forward_frame_includes_robust_v3_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_robust_scoring_mocks(
        monkeypatch
    )

    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
            "QQQ",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="robust_score_v3",
        minimum_completed_trades=0,
        robust_score_config=RobustScoreConfig(),
        robust_score_v3_config=(
            RobustScoreV3Config()
        ),
    )

    frame = (
        batch_walk_forward_optimization_to_frame(
            result
        )
    )

    expected_columns = {
        "selected_robust_v3_score",
        "selected_robust_v3_base_score",
        "selected_robust_v3_neighborhood_score",
        "selected_robust_v3_neighbor_count",
    }

    assert expected_columns.issubset(
        frame.columns
    )


def test_batch_walk_forward_comparison_includes_robust_v3_config_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_robust_scoring_mocks(
        monkeypatch
    )

    v3_config = RobustScoreV3Config(
        base_score_weight=0.25,
        neighborhood_score_weight=0.75,
    )

    result = run_batch_walk_forward_optimization(
        _sample_batch_frame(),
        [
            "SPY",
            "QQQ",
        ],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="robust_score_v3",
        minimum_completed_trades=0,
        robust_score_config=RobustScoreConfig(),
        robust_score_v3_config=v3_config,
    )

    frame = (
        batch_walk_forward_comparison_to_frame(
            result
        )
    )

    expected_columns = {
        "robust_v3_base_score_weight",
        "robust_v3_neighborhood_score_weight",
    }

    assert expected_columns.issubset(
        frame.columns
    )

    assert set(
        frame[
            "robust_v3_base_score_weight"
        ]
    ) == {
        0.25,
    }

    assert set(
        frame[
            "robust_v3_neighborhood_score_weight"
        ]
    ) == {
        0.75,
    }

