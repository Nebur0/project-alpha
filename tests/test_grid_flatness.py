from pathlib import Path

import pandas as pd
import pytest

from project_alpha.config import (
    StrategyParameters,
)
from project_alpha.grid_flatness import (
    _are_immediate_neighbors,
    _normalized_grid_axes,
    _unique_return_count,
    analyze_grid_flatness,
    grid_flatness_summary_to_frame,
    grid_flatness_to_frame,
    run_grid_flatness_batch,
    run_grid_flatness_from_csv,
    summarize_grid_flatness,
)


def _market_frame(
    symbol: str = "SPY",
) -> pd.DataFrame:
    closes = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
        108.0,
        109.0,
        110.0,
        111.0,
        112.0,
        113.0,
        114.0,
        115.0,
        116.0,
        117.0,
        118.0,
        119.0,
        120.0,
        121.0,
        122.0,
        123.0,
        124.0,
        125.0,
        126.0,
        127.0,
        128.0,
        129.0,
        130.0,
        131.0,
        132.0,
        133.0,
        134.0,
        135.0,
        136.0,
        137.0,
        138.0,
        139.0,
    ]

    return pd.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=len(closes),
                freq="D",
                tz="UTC",
            ),
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        }
    )


def _small_grid() -> dict[
    str,
    list[int],
]:
    return {
        "fast_ema_period": [
            2,
            3,
        ],
        "slow_ema_period": [
            4,
            5,
        ],
        "trend_ema_period": [
            6,
            7,
        ],
    }


def test_normalized_grid_axes_returns_sorted_unique_values() -> None:
    axes = _normalized_grid_axes(
        {
            "fast_ema_period": [
                3,
                2,
                3,
            ],
            "slow_ema_period": [
                5,
                4,
            ],
            "trend_ema_period": [
                7,
                6,
            ],
        }
    )

    assert axes == {
        "fast_ema_period": (
            2.0,
            3.0,
        ),
        "slow_ema_period": (
            4.0,
            5.0,
        ),
        "trend_ema_period": (
            6.0,
            7.0,
        ),
    }


def test_normalized_grid_axes_rejects_missing_parameter() -> None:
    with pytest.raises(
        ValueError,
        match="Mancanti",
    ):
        _normalized_grid_axes(
            {
                "fast_ema_period": [
                    2,
                    3,
                ],
                "slow_ema_period": [
                    4,
                    5,
                ],
            }
        )


def test_are_immediate_neighbors_true_when_one_axis_moves_one_step() -> None:
    axes = _normalized_grid_axes(_small_grid())

    first = StrategyParameters(
        fast_ema_period=2,
        slow_ema_period=4,
        trend_ema_period=6,
    )

    second = StrategyParameters(
        fast_ema_period=3,
        slow_ema_period=4,
        trend_ema_period=6,
    )

    assert _are_immediate_neighbors(
        first,
        second,
        axes,
    )


def test_are_immediate_neighbors_false_when_two_axes_change() -> None:
    axes = _normalized_grid_axes(_small_grid())

    first = StrategyParameters(
        fast_ema_period=2,
        slow_ema_period=4,
        trend_ema_period=6,
    )

    second = StrategyParameters(
        fast_ema_period=3,
        slow_ema_period=5,
        trend_ema_period=6,
    )

    assert not _are_immediate_neighbors(
        first,
        second,
        axes,
    )


def test_are_immediate_neighbors_false_for_same_parameters() -> None:
    axes = _normalized_grid_axes(_small_grid())

    parameters = StrategyParameters(
        fast_ema_period=2,
        slow_ema_period=4,
        trend_ema_period=6,
    )

    assert not _are_immediate_neighbors(
        parameters,
        parameters,
        axes,
    )


def test_unique_return_count_merges_only_exact_tolerance_matches() -> None:
    result = _unique_return_count(
        [
            0.10,
            0.10,
            0.1000000000005,
            0.11,
        ]
    )

    assert result == 2


def test_analyze_grid_flatness_returns_expected_window_count() -> None:
    windows = analyze_grid_flatness(
        _market_frame(),
        _small_grid(),
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    assert len(windows) == 2

    for window in windows:
        assert window.candidate_count == 8
        assert 1 <= window.unique_oos_return_count <= 8
        assert 1 <= window.selected_equivalent_count <= 8
        assert 0.0 < window.selected_equivalent_rate <= 1.0
        assert 0.0 < window.within_1pp_best_rate <= 1.0
        assert 0.0 < window.within_2pp_best_rate <= 1.0
        assert window.selected_regret >= 0.0
        assert window.selected_neighbor_count >= 1


def test_analyze_grid_flatness_preserves_symbol() -> None:
    windows = analyze_grid_flatness(
        _market_frame(symbol="spy"),
        _small_grid(),
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    assert {window.symbol for window in windows} == {
        "SPY",
    }


def test_analyze_grid_flatness_rejects_multiple_symbols() -> None:
    frame = pd.concat(
        [
            _market_frame(symbol="SPY"),
            _market_frame(symbol="QQQ"),
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="un solo simbolo",
    ):
        analyze_grid_flatness(
            frame,
            _small_grid(),
            train_size=20,
            test_size=10,
            step_size=10,
            score_metric="total_return",
            minimum_completed_trades=0,
        )


def test_grid_flatness_to_frame_contains_expected_columns() -> None:
    windows = analyze_grid_flatness(
        _market_frame(),
        _small_grid(),
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    frame = grid_flatness_to_frame(windows)

    expected_columns = {
        "symbol",
        "window_number",
        "candidate_count",
        "selected_oos_total_return",
        "best_oos_total_return",
        "std_oos_total_return",
        "iqr_oos_total_return",
        "range_oos_total_return",
        "unique_oos_return_count",
        "selected_equivalent_count",
        "selected_equivalent_rate",
        "within_1pp_best_rate",
        "within_2pp_best_rate",
        "selected_regret",
        "selected_neighbor_count",
        "selected_mean_abs_neighbor_delta",
    }

    assert expected_columns.issubset(frame.columns)


def test_summarize_grid_flatness_matches_windows() -> None:
    windows = analyze_grid_flatness(
        _market_frame(),
        _small_grid(),
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    summary = summarize_grid_flatness(windows)

    assert summary.symbol_count == 1
    assert summary.window_count == 2
    assert 0.0 < summary.mean_selected_equivalent_rate <= 1.0
    assert 0.0 < summary.mean_within_1pp_best_rate <= 1.0
    assert 0.0 < summary.mean_within_2pp_best_rate <= 1.0
    assert 0.0 <= summary.selected_within_1pp_best_rate <= 1.0
    assert 0.0 <= summary.selected_within_2pp_best_rate <= 1.0
    assert summary.mean_selected_regret >= 0.0


def test_summarize_grid_flatness_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="almeno una finestra",
    ):
        summarize_grid_flatness([])


def test_grid_flatness_summary_to_frame_contains_expected_columns() -> None:
    windows = analyze_grid_flatness(
        _market_frame(),
        _small_grid(),
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    summary = summarize_grid_flatness(windows)

    frame = grid_flatness_summary_to_frame(summary)

    expected_columns = {
        "symbol_count",
        "window_count",
        "mean_unique_oos_return_count",
        "mean_selected_equivalent_rate",
        "mean_within_1pp_best_rate",
        "mean_within_2pp_best_rate",
        "selected_within_1pp_best_rate",
        "selected_within_2pp_best_rate",
        "mean_std_oos_total_return",
        "mean_iqr_oos_total_return",
        "mean_range_oos_total_return",
        "mean_selected_regret",
        "mean_selected_mean_abs_neighbor_delta",
    }

    assert expected_columns.issubset(frame.columns)


def test_run_grid_flatness_batch_combines_symbols() -> None:
    frame = pd.concat(
        [
            _market_frame(symbol="SPY"),
            _market_frame(symbol="QQQ"),
        ],
        ignore_index=True,
    )

    windows = run_grid_flatness_batch(
        frame,
        [
            "SPY",
            "QQQ",
        ],
        _small_grid(),
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    assert len(windows) == 4

    assert {window.symbol for window in windows} == {
        "SPY",
        "QQQ",
    }


def test_run_grid_flatness_batch_rejects_missing_symbol() -> None:
    with pytest.raises(
        ValueError,
        match="QQQ",
    ):
        run_grid_flatness_batch(
            _market_frame(symbol="SPY"),
            [
                "QQQ",
            ],
            _small_grid(),
            train_size=20,
            test_size=10,
            step_size=10,
            score_metric="total_return",
            minimum_completed_trades=0,
        )


def test_run_grid_flatness_from_csv_creates_reports(
    tmp_path: Path,
) -> None:
    market_path = tmp_path / "market.csv"

    output_dir = tmp_path / "grid_flatness"

    _market_frame().to_csv(
        market_path,
        index=False,
    )

    result = run_grid_flatness_from_csv(
        market_path,
        [
            "SPY",
        ],
        _small_grid(),
        output_dir,
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    windows_path = output_dir / "grid_flatness_windows.csv"

    summary_path = output_dir / "grid_flatness_summary.csv"

    assert windows_path.exists()
    assert summary_path.exists()

    assert result["windows_path"] == windows_path

    assert result["summary_path"] == summary_path

    assert len(result["windows"]) == 2


def test_run_grid_flatness_from_csv_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Market dataset",
    ):
        run_grid_flatness_from_csv(
            tmp_path / "missing.csv",
            [
                "SPY",
            ],
            _small_grid(),
            tmp_path / "out",
            train_size=20,
            test_size=10,
            step_size=10,
            score_metric="total_return",
            minimum_completed_trades=0,
        )
