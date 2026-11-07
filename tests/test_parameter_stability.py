from pathlib import Path

import pandas as pd
import pytest

from project_alpha.parameter_stability import (
    analyze_symbol_parameter_stability,
    combination_frequencies_to_frame,
    parameter_combination_frequencies,
    parameter_stability_summary_to_frame,
    run_parameter_stability_from_csv,
    summarize_parameter_stability,
    symbol_stability_to_frame,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "SPY",
                "SPY",
                "SPY",
                "SPY",
                "QQQ",
                "QQQ",
                "QQQ",
                "QQQ",
            ],
            "window_number": [
                1,
                2,
                3,
                4,
                1,
                2,
                3,
                4,
            ],
            "fast_ema_period": [
                10,
                10,
                20,
                20,
                10,
                10,
                10,
                20,
            ],
            "slow_ema_period": [
                30,
                30,
                40,
                40,
                30,
                30,
                30,
                40,
            ],
            "trend_ema_period": [
                100,
                100,
                100,
                150,
                100,
                100,
                100,
                150,
            ],
        }
    )


def test_parameter_combination_frequencies_counts_correctly() -> None:
    results = parameter_combination_frequencies(_frame())

    assert len(results) == 3

    first = results[0]

    assert first.fast_ema_period == 10
    assert first.slow_ema_period == 30
    assert first.trend_ema_period == 100
    assert first.selection_count == 5
    assert first.selection_rate == pytest.approx(5 / 8)


def test_parameter_combination_frequencies_are_sorted_by_frequency() -> None:
    results = parameter_combination_frequencies(_frame())

    counts = [result.selection_count for result in results]

    assert counts == sorted(
        counts,
        reverse=True,
    )


def test_symbol_parameter_stability_calculates_spy_changes() -> None:
    results = analyze_symbol_parameter_stability(_frame())

    by_symbol = {result.symbol: result for result in results}

    spy = by_symbol["SPY"]

    assert spy.window_count == 4
    assert spy.unique_combination_count == 3

    assert spy.modal_fast_ema_period == 10
    assert spy.modal_slow_ema_period == 30
    assert spy.modal_trend_ema_period == 100

    assert spy.modal_combination_count == 2
    assert spy.modal_combination_rate == pytest.approx(0.5)

    assert spy.adjacent_transition_count == 3
    assert spy.adjacent_change_count == 2
    assert spy.adjacent_change_rate == pytest.approx(2 / 3)

    assert spy.fast_change_count == 1
    assert spy.fast_change_rate == pytest.approx(1 / 3)

    assert spy.slow_change_count == 1
    assert spy.slow_change_rate == pytest.approx(1 / 3)

    assert spy.trend_change_count == 1
    assert spy.trend_change_rate == pytest.approx(1 / 3)


def test_symbol_parameter_stability_calculates_qqq_changes() -> None:
    results = analyze_symbol_parameter_stability(_frame())

    by_symbol = {result.symbol: result for result in results}

    qqq = by_symbol["QQQ"]

    assert qqq.window_count == 4
    assert qqq.unique_combination_count == 2

    assert qqq.modal_combination_count == 3
    assert qqq.modal_combination_rate == pytest.approx(0.75)

    assert qqq.adjacent_transition_count == 3
    assert qqq.adjacent_change_count == 1
    assert qqq.adjacent_change_rate == pytest.approx(1 / 3)


def test_single_window_symbol_has_zero_change_rate() -> None:
    frame = pd.DataFrame(
        {
            "symbol": [
                "SPY",
            ],
            "window_number": [
                1,
            ],
            "fast_ema_period": [
                20,
            ],
            "slow_ema_period": [
                40,
            ],
            "trend_ema_period": [
                150,
            ],
        }
    )

    result = analyze_symbol_parameter_stability(frame)[0]

    assert result.adjacent_transition_count == 0
    assert result.adjacent_change_count == 0
    assert result.adjacent_change_rate == pytest.approx(0.0)
    assert result.normalized_combination_entropy == pytest.approx(0.0)


def test_normalized_entropy_is_between_zero_and_one() -> None:
    results = analyze_symbol_parameter_stability(_frame())

    for result in results:
        assert 0.0 <= result.normalized_combination_entropy <= 1.0


def test_summary_calculates_global_modal_combination() -> None:
    frame = _frame()

    summary = summarize_parameter_stability(frame)

    assert summary.symbol_count == 2
    assert summary.total_window_count == 8
    assert summary.unique_combination_count == 3

    assert summary.global_modal_fast_ema_period == 10
    assert summary.global_modal_slow_ema_period == 30
    assert summary.global_modal_trend_ema_period == 100

    assert summary.global_modal_combination_count == 5
    assert summary.global_modal_combination_rate == pytest.approx(5 / 8)


def test_summary_aggregates_transition_counts() -> None:
    summary = summarize_parameter_stability(_frame())

    assert summary.total_adjacent_transition_count == 6
    assert summary.total_adjacent_change_count == 3
    assert summary.overall_adjacent_change_rate == pytest.approx(0.5)

    assert summary.mean_symbol_adjacent_change_rate == pytest.approx(0.5)
    assert summary.median_symbol_adjacent_change_rate == pytest.approx(0.5)


def test_summary_modal_rates_match_symbol_results() -> None:
    summary = summarize_parameter_stability(_frame())

    assert summary.mean_symbol_modal_combination_rate == pytest.approx(0.625)
    assert summary.median_symbol_modal_combination_rate == pytest.approx(0.625)


def test_frame_helpers_contain_expected_columns() -> None:
    frame = _frame()

    frequencies = parameter_combination_frequencies(frame)
    symbol_results = analyze_symbol_parameter_stability(frame)
    summary = summarize_parameter_stability(
        frame,
        symbol_results,
    )

    frequency_frame = combination_frequencies_to_frame(frequencies)
    symbol_frame = symbol_stability_to_frame(symbol_results)
    summary_frame = parameter_stability_summary_to_frame(summary)

    assert {
        "fast_ema_period",
        "slow_ema_period",
        "trend_ema_period",
        "selection_count",
        "selection_rate",
    }.issubset(frequency_frame.columns)

    assert {
        "symbol",
        "window_count",
        "unique_combination_count",
        "modal_combination_rate",
        "normalized_combination_entropy",
        "adjacent_change_rate",
    }.issubset(symbol_frame.columns)

    assert {
        "symbol_count",
        "total_window_count",
        "unique_combination_count",
        "global_modal_combination_rate",
        "normalized_combination_entropy",
        "overall_adjacent_change_rate",
    }.issubset(summary_frame.columns)


@pytest.mark.parametrize(
    "column",
    [
        "symbol",
        "window_number",
        "fast_ema_period",
        "slow_ema_period",
        "trend_ema_period",
    ],
)
def test_rejects_missing_required_columns(
    column: str,
) -> None:
    frame = _frame().drop(
        columns=[
            column,
        ]
    )

    with pytest.raises(
        ValueError,
        match="Colonne mancanti",
    ):
        parameter_combination_frequencies(frame)


def test_rejects_empty_frame() -> None:
    frame = _frame().iloc[0:0]

    with pytest.raises(
        ValueError,
        match="vuoto",
    ):
        analyze_symbol_parameter_stability(frame)


def test_rejects_duplicate_symbol_window() -> None:
    frame = pd.concat(
        [
            _frame(),
            _frame().iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        analyze_symbol_parameter_stability(frame)


@pytest.mark.parametrize(
    (
        "column",
        "value",
    ),
    [
        (
            "window_number",
            0,
        ),
        (
            "fast_ema_period",
            0,
        ),
        (
            "slow_ema_period",
            -1,
        ),
        (
            "trend_ema_period",
            1.5,
        ),
    ],
)
def test_rejects_invalid_positive_integer_fields(
    column: str,
    value: float,
) -> None:
    frame = _frame()

    if (
        isinstance(
            value,
            float,
        )
        and not value.is_integer()
    ):
        frame[column] = frame[column].astype(float)

    frame.loc[
        0,
        column,
    ] = value

    with pytest.raises(
        ValueError,
        match=column,
    ):
        analyze_symbol_parameter_stability(frame)


def test_normalizes_symbol_case() -> None:
    frame = _frame()
    frame["symbol"] = [
        "spy",
        "spy",
        "spy",
        "spy",
        "qqq",
        "qqq",
        "qqq",
        "qqq",
    ]

    results = analyze_symbol_parameter_stability(frame)

    assert {result.symbol for result in results} == {
        "QQQ",
        "SPY",
    }


def test_run_parameter_stability_from_csv_creates_reports(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "walk_forward.csv"
    output_dir = tmp_path / "parameter_stability"

    _frame().to_csv(
        input_path,
        index=False,
    )

    result = run_parameter_stability_from_csv(
        input_path,
        output_dir,
    )

    frequencies_path = output_dir / "parameter_combination_frequencies.csv"
    symbol_path = output_dir / "parameter_stability_by_symbol.csv"
    summary_path = output_dir / "parameter_stability_summary.csv"

    assert frequencies_path.exists()
    assert symbol_path.exists()
    assert summary_path.exists()

    assert result["frequencies_path"] == frequencies_path
    assert result["symbol_path"] == symbol_path
    assert result["summary_path"] == summary_path


def test_run_parameter_stability_from_csv_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Report walk-forward",
    ):
        run_parameter_stability_from_csv(
            tmp_path / "missing.csv",
            tmp_path / "out",
        )
