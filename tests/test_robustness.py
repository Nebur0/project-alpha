import pandas as pd
import pytest

from project_alpha.robustness import (
    REGIME_DOWN_HIGH_VOL,
    REGIME_DOWN_LOW_VOL,
    REGIME_UP_HIGH_VOL,
    REGIME_UP_LOW_VOL,
    RegimeAnalysisConfig,
    analyze_walk_forward_regimes,
    calculate_annualized_realized_volatility,
    classify_market_regime,
    regime_summaries_to_frame,
    regime_window_results_to_frame,
    summarize_regime_results,
)


def _market_frame(
    *,
    symbol: str = "SPY",
    closes: list[float] | None = None,
) -> pd.DataFrame:
    effective_closes = (
        closes
        if closes is not None
        else [
            100.0,
            101.0,
            102.0,
            101.0,
            103.0,
            104.0,
        ]
    )

    return pd.DataFrame(
        {
            "symbol": [
                symbol
            ] * len(
                effective_closes
            ),
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=len(
                    effective_closes
                ),
                freq="D",
                tz="UTC",
            ),
            "close": effective_closes,
        }
    )


def _optimization_frame(
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "SPY",
                "SPY",
            ],
            "window_number": [
                1,
                2,
            ],
            "test_start": [
                "2024-01-01T00:00:00+00:00",
                "2024-01-04T00:00:00+00:00",
            ],
            "test_end": [
                "2024-01-03T00:00:00+00:00",
                "2024-01-06T00:00:00+00:00",
            ],
            "out_of_sample_total_return": [
                0.05,
                -0.02,
            ],
            (
                "out_of_sample_"
                "benchmark_total_return"
            ): [
                0.03,
                -0.04,
            ],
            (
                "out_of_sample_"
                "annualized_sharpe"
            ): [
                1.25,
                -0.50,
            ],
            (
                "out_of_sample_"
                "max_drawdown"
            ): [
                -0.04,
                -0.08,
            ],
            (
                "out_of_sample_"
                "completed_trades"
            ): [
                3,
                2,
            ],
        }
    )


def test_regime_analysis_config_uses_frozen_defaults(
) -> None:
    config = RegimeAnalysisConfig()

    assert config.annualization_factor == 252
    assert (
        config.high_volatility_threshold
        == pytest.approx(
            0.25
        )
    )


@pytest.mark.parametrize(
    (
        "kwargs",
        "message",
    ),
    [
        (
            {
                "annualization_factor": 0,
            },
            "annualization_factor",
        ),
        (
            {
                "annualization_factor": True,
            },
            "annualization_factor",
        ),
        (
            {
                "high_volatility_threshold": -0.01,
            },
            "high_volatility_threshold",
        ),
        (
            {
                "high_volatility_threshold": float(
                    "inf"
                ),
            },
            "high_volatility_threshold",
        ),
    ],
)
def test_regime_analysis_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        RegimeAnalysisConfig(
            **kwargs
        )


def test_realized_volatility_is_zero_for_constant_prices(
) -> None:
    volatility = (
        calculate_annualized_realized_volatility(
            _market_frame(
                closes=[
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            )
        )
    )

    assert volatility == pytest.approx(
        0.0
    )


def test_realized_volatility_matches_manual_calculation(
) -> None:
    frame = _market_frame(
        closes=[
            100.0,
            102.0,
            101.0,
            104.0,
            103.0,
        ]
    )

    expected = (
        frame["close"]
        .pct_change()
        .dropna()
        .std(
            ddof=1
        )
        * (252 ** 0.5)
    )

    result = (
        calculate_annualized_realized_volatility(
            frame
        )
    )

    assert result == pytest.approx(
        expected
    )


def test_realized_volatility_rejects_invalid_market_frame(
) -> None:
    with pytest.raises(
        ValueError,
        match="vuota",
    ):
        calculate_annualized_realized_volatility(
            pd.DataFrame()
        )

    with pytest.raises(
        ValueError,
        match="close",
    ):
        calculate_annualized_realized_volatility(
            pd.DataFrame(
                {
                    "price": [
                        100.0,
                        101.0,
                    ],
                }
            )
        )

    with pytest.raises(
        ValueError,
        match="positivi",
    ):
        calculate_annualized_realized_volatility(
            pd.DataFrame(
                {
                    "close": [
                        100.0,
                        0.0,
                        101.0,
                    ],
                }
            )
        )


@pytest.mark.parametrize(
    (
        "benchmark_return",
        "volatility",
        "expected_regime",
        "expected_direction",
        "expected_volatility_regime",
    ),
    [
        (
            0.10,
            0.20,
            REGIME_UP_LOW_VOL,
            "UP",
            "LOW_VOL",
        ),
        (
            0.10,
            0.25,
            REGIME_UP_HIGH_VOL,
            "UP",
            "HIGH_VOL",
        ),
        (
            -0.10,
            0.20,
            REGIME_DOWN_LOW_VOL,
            "DOWN",
            "LOW_VOL",
        ),
        (
            -0.10,
            0.30,
            REGIME_DOWN_HIGH_VOL,
            "DOWN",
            "HIGH_VOL",
        ),
    ],
)
def test_classify_market_regime_uses_frozen_rules(
    benchmark_return: float,
    volatility: float,
    expected_regime: str,
    expected_direction: str,
    expected_volatility_regime: str,
) -> None:
    result = classify_market_regime(
        benchmark_return,
        volatility,
    )

    assert result == (
        expected_regime,
        expected_direction,
        expected_volatility_regime,
    )


def test_zero_benchmark_return_is_classified_as_up(
) -> None:
    regime, direction, volatility = (
        classify_market_regime(
            0.0,
            0.10,
        )
    )

    assert regime == REGIME_UP_LOW_VOL
    assert direction == "UP"
    assert volatility == "LOW_VOL"


def test_classify_market_regime_rejects_negative_volatility(
) -> None:
    with pytest.raises(
        ValueError,
        match="non può essere negativa",
    ):
        classify_market_regime(
            0.10,
            -0.01,
        )


def test_analyze_walk_forward_regimes_maps_windows_to_market_data(
) -> None:
    config = RegimeAnalysisConfig(
        high_volatility_threshold=10.0,
    )

    results = analyze_walk_forward_regimes(
        _market_frame(),
        _optimization_frame(),
        config,
    )

    assert len(results) == 2

    first = results[0]
    second = results[1]

    assert first.symbol == "SPY"
    assert first.window_number == 1
    assert (
        first.regime
        == REGIME_UP_LOW_VOL
    )
    assert (
        first.strategy_total_return
        == pytest.approx(
            0.05
        )
    )
    assert (
        first.benchmark_total_return
        == pytest.approx(
            0.03
        )
    )
    assert (
        first.excess_return
        == pytest.approx(
            0.02
        )
    )
    assert (
        first.annualized_sharpe
        == pytest.approx(
            1.25
        )
    )
    assert (
        first.max_drawdown
        == pytest.approx(
            -0.04
        )
    )
    assert first.completed_trades == 3

    assert (
        second.regime
        == REGIME_DOWN_LOW_VOL
    )
    assert (
        second.strategy_total_return
        == pytest.approx(
            -0.02
        )
    )
    assert (
        second.benchmark_total_return
        == pytest.approx(
            -0.04
        )
    )
    assert (
        second.excess_return
        == pytest.approx(
            0.02
        )
    )


def test_analyze_walk_forward_regimes_normalizes_symbol_case(
) -> None:
    optimization = (
        _optimization_frame()
    )
    optimization["symbol"] = "spy"

    results = analyze_walk_forward_regimes(
        _market_frame(
            symbol="SPY"
        ),
        optimization,
        RegimeAnalysisConfig(
            high_volatility_threshold=10.0,
        ),
    )

    assert {
        result.symbol
        for result in results
    } == {
        "SPY",
    }


def test_analyze_walk_forward_regimes_rejects_missing_symbol(
) -> None:
    optimization = (
        _optimization_frame()
    )
    optimization["symbol"] = "QQQ"

    with pytest.raises(
        ValueError,
        match="QQQ",
    ):
        analyze_walk_forward_regimes(
            _market_frame(
                symbol="SPY"
            ),
            optimization,
        )


def test_regime_window_results_to_frame_contains_audit_columns(
) -> None:
    results = analyze_walk_forward_regimes(
        _market_frame(),
        _optimization_frame(),
        RegimeAnalysisConfig(
            high_volatility_threshold=10.0,
        ),
    )

    frame = regime_window_results_to_frame(
        results
    )

    expected_columns = {
        "symbol",
        "window_number",
        "test_start",
        "test_end",
        "regime",
        "market_direction",
        "volatility_regime",
        "benchmark_total_return",
        "annualized_realized_volatility",
        "strategy_total_return",
        "excess_return",
        "annualized_sharpe",
        "max_drawdown",
        "completed_trades",
    }

    assert expected_columns.issubset(
        frame.columns
    )


def test_summarize_regime_results_aggregates_per_regime(
) -> None:
    results = analyze_walk_forward_regimes(
        _market_frame(),
        _optimization_frame(),
        RegimeAnalysisConfig(
            high_volatility_threshold=10.0,
        ),
    )

    summaries = summarize_regime_results(
        results
    )

    by_regime = {
        summary.regime: summary
        for summary in summaries
    }

    up = by_regime[
        REGIME_UP_LOW_VOL
    ]
    down = by_regime[
        REGIME_DOWN_LOW_VOL
    ]

    assert up.window_count == 1
    assert up.positive_window_count == 1
    assert (
        up.positive_window_rate
        == pytest.approx(
            1.0
        )
    )
    assert up.benchmark_beat_count == 1
    assert (
        up.benchmark_beat_rate
        == pytest.approx(
            1.0
        )
    )
    assert (
        up.mean_strategy_total_return
        == pytest.approx(
            0.05
        )
    )
    assert (
        up.mean_excess_return
        == pytest.approx(
            0.02
        )
    )
    assert (
        up.mean_annualized_sharpe
        == pytest.approx(
            1.25
        )
    )
    assert (
        up.worst_max_drawdown
        == pytest.approx(
            -0.04
        )
    )
    assert up.total_completed_trades == 3

    assert down.window_count == 1
    assert down.positive_window_count == 0
    assert (
        down.positive_window_rate
        == pytest.approx(
            0.0
        )
    )
    assert down.benchmark_beat_count == 1
    assert (
        down.benchmark_beat_rate
        == pytest.approx(
            1.0
        )
    )
    assert (
        down.mean_strategy_total_return
        == pytest.approx(
            -0.02
        )
    )
    assert (
        down.mean_excess_return
        == pytest.approx(
            0.02
        )
    )
    assert down.total_completed_trades == 2


def test_summarize_regime_results_returns_empty_tuple_for_empty_input(
) -> None:
    assert summarize_regime_results(
        []
    ) == ()


def test_regime_summaries_to_frame_contains_summary_columns(
) -> None:
    results = analyze_walk_forward_regimes(
        _market_frame(),
        _optimization_frame(),
        RegimeAnalysisConfig(
            high_volatility_threshold=10.0,
        ),
    )

    summaries = summarize_regime_results(
        results
    )

    frame = regime_summaries_to_frame(
        summaries
    )

    expected_columns = {
        "regime",
        "window_count",
        "positive_window_count",
        "positive_window_rate",
        "benchmark_beat_count",
        "benchmark_beat_rate",
        "mean_strategy_total_return",
        "median_strategy_total_return",
        "mean_benchmark_total_return",
        "mean_excess_return",
        "mean_annualized_sharpe",
        "worst_max_drawdown",
        "total_completed_trades",
    }

    assert expected_columns.issubset(
        frame.columns
    )
