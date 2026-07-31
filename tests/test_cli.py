from datetime import date
from pathlib import Path

import pytest

from project_alpha.cli import build_parser


def test_batch_validate_parser_reads_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "batch-validate",
            "--input",
            "data/processed/features.csv",
            "--symbols",
            "SPY",
            "QQQ",
            "AAPL",
            "--split-date",
            "2023-01-01",
            "--initial-capital",
            "15000",
            "--cost-per-side",
            "0.002",
        ]
    )

    assert args.command == "batch-validate"
    assert args.input == Path(
        "data/processed/features.csv"
    )
    assert args.symbols == [
        "SPY",
        "QQQ",
        "AAPL",
    ]
    assert args.split_date == date(
        2023,
        1,
        1,
    )
    assert args.initial_capital == pytest.approx(
        15_000.0
    )
    assert args.cost_per_side == pytest.approx(
        0.002
    )


def test_batch_validate_requires_symbols() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "batch-validate",
                "--split-date",
                "2023-01-01",
            ]
        )


def test_walk_forward_parser_reads_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "walk-forward",
            "--input",
            "data/processed/features.csv",
            "--symbol",
            "SPY",
            "--train-size",
            "1000",
            "--test-size",
            "252",
            "--step-size",
            "126",
            "--initial-capital",
            "20000",
            "--cost-per-side",
            "0.0015",
        ]
    )

    assert args.command == "walk-forward"
    assert args.input == Path(
        "data/processed/features.csv"
    )
    assert args.symbol == "SPY"
    assert args.train_size == 1000
    assert args.test_size == 252
    assert args.step_size == 126
    assert args.initial_capital == pytest.approx(
        20_000.0
    )
    assert args.cost_per_side == pytest.approx(
        0.0015
    )


def test_walk_forward_requires_window_sizes() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "walk-forward",
                "--symbol",
                "SPY",
            ]
        )


def test_batch_walk_forward_parser_reads_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "batch-walk-forward",
            "--input",
            "data/processed/features.csv",
            "--symbols",
            "SPY",
            "QQQ",
            "AAPL",
            "--train-size",
            "1000",
            "--test-size",
            "252",
            "--step-size",
            "126",
            "--initial-capital",
            "25000",
            "--cost-per-side",
            "0.002",
        ]
    )

    assert args.command == "batch-walk-forward"
    assert args.input == Path(
        "data/processed/features.csv"
    )
    assert args.symbols == [
        "SPY",
        "QQQ",
        "AAPL",
    ]
    assert args.train_size == 1000
    assert args.test_size == 252
    assert args.step_size == 126
    assert args.initial_capital == pytest.approx(
        25_000.0
    )
    assert args.cost_per_side == pytest.approx(
        0.002
    )


def test_batch_walk_forward_requires_arguments() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "batch-walk-forward",
                "--train-size",
                "1000",
                "--test-size",
                "252",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "batch-walk-forward",
                "--symbols",
                "SPY",
                "QQQ",
            ]
        )


def test_analyze_parser_reads_custom_strategy_parameters() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "analyze",
            "--input",
            "data/raw/bars.csv",
            "--output",
            "data/processed/custom.csv",
            "--fast-ema-period",
            "10",
            "--slow-ema-period",
            "30",
            "--trend-ema-period",
            "100",
            "--rsi-period",
            "7",
            "--rsi-lower",
            "40",
            "--rsi-upper",
            "80",
            "--atr-period",
            "10",
            "--volume-period",
            "15",
            "--breakout-period",
            "25",
        ]
    )

    assert args.command == "analyze"
    assert args.input == Path(
        "data/raw/bars.csv"
    )
    assert args.output == Path(
        "data/processed/custom.csv"
    )
    assert args.fast_ema_period == 10
    assert args.slow_ema_period == 30
    assert args.trend_ema_period == 100
    assert args.rsi_period == 7
    assert args.rsi_lower == pytest.approx(
        40.0
    )
    assert args.rsi_upper == pytest.approx(
        80.0
    )
    assert args.atr_period == 10
    assert args.volume_period == 15
    assert args.breakout_period == 25


def test_run_parser_uses_default_strategy_parameters() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "--symbols",
            "SPY",
            "--start",
            "2020-01-01",
        ]
    )

    assert args.command == "run"
    assert args.fast_ema_period == 20
    assert args.slow_ema_period == 50
    assert args.trend_ema_period == 200
    assert args.rsi_period == 14
    assert args.rsi_lower == pytest.approx(
        50.0
    )
    assert args.rsi_upper == pytest.approx(
        70.0
    )
    assert args.atr_period == 14
    assert args.volume_period == 20
    assert args.breakout_period == 20


def test_grid_search_parser_reads_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "grid-search",
            "--input",
            "data/raw/bars.csv",
            "--symbol",
            "SPY",
            "--fast-ema-periods",
            "10",
            "20",
            "--slow-ema-periods",
            "30",
            "50",
            "--trend-ema-periods",
            "100",
            "200",
            "--rsi-periods",
            "7",
            "14",
            "--rsi-lowers",
            "40",
            "50",
            "--rsi-uppers",
            "70",
            "80",
            "--atr-periods",
            "10",
            "14",
            "--volume-periods",
            "15",
            "20",
            "--breakout-periods",
            "20",
            "25",
            "--score-metric",
            "total_return",
            "--top",
            "5",
            "--initial-capital",
            "15000",
            "--cost-per-side",
            "0.002",
        ]
    )

    assert args.command == "grid-search"
    assert args.input == Path(
        "data/raw/bars.csv"
    )
    assert args.symbol == "SPY"
    assert args.fast_ema_periods == [
        10,
        20,
    ]
    assert args.slow_ema_periods == [
        30,
        50,
    ]
    assert args.trend_ema_periods == [
        100,
        200,
    ]
    assert args.rsi_periods == [
        7,
        14,
    ]
    assert args.rsi_lowers == pytest.approx(
        [
            40.0,
            50.0,
        ]
    )
    assert args.rsi_uppers == pytest.approx(
        [
            70.0,
            80.0,
        ]
    )
    assert args.atr_periods == [
        10,
        14,
    ]
    assert args.volume_periods == [
        15,
        20,
    ]
    assert args.breakout_periods == [
        20,
        25,
    ]
    assert args.score_metric == "total_return"
    assert args.top == 5
    assert args.initial_capital == pytest.approx(
        15_000.0
    )
    assert args.cost_per_side == pytest.approx(
        0.002
    )


def test_grid_search_parser_uses_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "grid-search",
            "--symbol",
            "SPY",
        ]
    )

    assert args.command == "grid-search"
    assert args.input == Path(
        "data/raw/bars.csv"
    )
    assert args.symbol == "SPY"
    assert args.fast_ema_periods == [
        20,
    ]
    assert args.slow_ema_periods == [
        50,
    ]
    assert args.trend_ema_periods == [
        200,
    ]
    assert args.rsi_periods == [
        14,
    ]
    assert args.rsi_lowers == pytest.approx(
        [
            50.0,
        ]
    )
    assert args.rsi_uppers == pytest.approx(
        [
            70.0,
        ]
    )
    assert args.atr_periods == [
        14,
    ]
    assert args.volume_periods == [
        20,
    ]
    assert args.breakout_periods == [
        20,
    ]
    assert (
        args.score_metric
        == "annualized_sharpe"
    )
    assert args.top == 10
    assert args.initial_capital == pytest.approx(
        10_000.0
    )
    assert args.cost_per_side == pytest.approx(
        0.001
    )


def test_grid_search_requires_symbol() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "grid-search",
            ]
        )