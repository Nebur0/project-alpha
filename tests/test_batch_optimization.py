import numpy as np
import pandas as pd
import pytest

from project_alpha.batch_optimization import (
    batch_walk_forward_comparison_to_frame,
    batch_walk_forward_optimization_to_dict,
    batch_walk_forward_optimization_to_frame,
    run_batch_walk_forward_optimization,
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