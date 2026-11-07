from pathlib import Path

import pandas as pd
import pytest

from project_alpha.selection_skill import (
    _oos_percentile,
    _oos_rank,
    _spearman_rank_correlation,
    analyze_selection_skill,
    run_selection_skill_batch,
    run_selection_skill_from_csv,
    selection_skill_summary_to_frame,
    selection_skill_to_frame,
    summarize_selection_skill,
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


def _small_grid() -> dict[str, list[int]]:
    return {
        "fast_ema_period": [2, 3],
        "slow_ema_period": [4],
        "trend_ema_period": [5],
    }


def test_oos_rank_returns_one_for_best_value() -> None:
    assert _oos_rank(
        [0.10, 0.20, 0.05],
        1,
    ) == 1


def test_oos_rank_uses_competition_rank_for_ties() -> None:
    assert _oos_rank(
        [0.20, 0.20, 0.10],
        1,
    ) == 1

    assert _oos_rank(
        [0.20, 0.10, 0.10],
        1,
    ) == 2


@pytest.mark.parametrize(
    ("rank", "candidate_count", "expected"),
    [
        (1, 5, 1.0),
        (3, 5, 0.5),
        (5, 5, 0.0),
        (1, 1, 1.0),
    ],
)
def test_oos_percentile_matches_rank(
    rank: int,
    candidate_count: int,
    expected: float,
) -> None:
    assert _oos_percentile(
        rank,
        candidate_count,
    ) == pytest.approx(expected)


def test_oos_percentile_rejects_invalid_inputs() -> None:
    with pytest.raises(
        ValueError,
        match="candidate_count",
    ):
        _oos_percentile(
            1,
            0,
        )

    with pytest.raises(
        ValueError,
        match="rank",
    ):
        _oos_percentile(
            0,
            5,
        )


def test_spearman_rank_correlation_positive_monotonic() -> None:
    result = _spearman_rank_correlation(
        [1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3],
    )

    assert result == pytest.approx(1.0)


def test_spearman_rank_correlation_negative_monotonic() -> None:
    result = _spearman_rank_correlation(
        [1.0, 2.0, 3.0],
        [0.3, 0.2, 0.1],
    )

    assert result == pytest.approx(-1.0)


def test_spearman_rank_correlation_returns_none_for_constant_series() -> None:
    result = _spearman_rank_correlation(
        [1.0, 1.0, 1.0],
        [0.1, 0.2, 0.3],
    )

    assert result is None


def test_analyze_selection_skill_returns_expected_window_count() -> None:
    windows = analyze_selection_skill(
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
        assert window.candidate_count == 2
        assert 1 <= window.selected_oos_rank <= 2
        assert 0.0 <= window.selected_oos_percentile <= 1.0
        assert window.oracle_regret >= 0.0


def test_analyze_selection_skill_preserves_symbol() -> None:
    windows = analyze_selection_skill(
        _market_frame(symbol="spy"),
        _small_grid(),
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    assert {
        window.symbol
        for window in windows
    } == {"SPY"}


def test_analyze_selection_skill_rejects_multiple_symbols() -> None:
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
        analyze_selection_skill(
            frame,
            _small_grid(),
            train_size=20,
            test_size=10,
            step_size=10,
            score_metric="total_return",
            minimum_completed_trades=0,
        )


def test_selection_skill_to_frame_contains_expected_columns() -> None:
    windows = analyze_selection_skill(
        _market_frame(),
        _small_grid(),
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    frame = selection_skill_to_frame(windows)

    expected_columns = {
        "symbol",
        "window_number",
        "candidate_count",
        "selected_oos_rank",
        "selected_oos_percentile",
        "selected_is_top_quartile",
        "selected_is_top_half",
        "selected_oos_total_return",
        "best_oos_total_return",
        "mean_oos_total_return",
        "median_oos_total_return",
        "selected_minus_mean_oos_return",
        "selected_minus_median_oos_return",
        "oracle_regret",
        "training_score_oos_return_spearman",
    }

    assert expected_columns.issubset(frame.columns)


def test_summarize_selection_skill_matches_windows() -> None:
    windows = analyze_selection_skill(
        _market_frame(),
        _small_grid(),
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    summary = summarize_selection_skill(windows)

    assert summary.symbol_count == 1
    assert summary.window_count == 2
    assert 0.0 <= summary.top_quartile_rate <= 1.0
    assert 0.0 <= summary.top_half_rate <= 1.0
    assert summary.mean_oracle_regret >= 0.0
    assert summary.median_oracle_regret >= 0.0


def test_summarize_selection_skill_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="almeno una finestra",
    ):
        summarize_selection_skill([])


def test_selection_skill_summary_to_frame_contains_expected_columns() -> None:
    windows = analyze_selection_skill(
        _market_frame(),
        _small_grid(),
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    summary = summarize_selection_skill(windows)
    frame = selection_skill_summary_to_frame(summary)

    expected_columns = {
        "symbol_count",
        "window_count",
        "mean_selected_oos_rank",
        "median_selected_oos_rank",
        "mean_selected_oos_percentile",
        "median_selected_oos_percentile",
        "top_quartile_rate",
        "top_half_rate",
        "mean_oracle_regret",
        "median_oracle_regret",
        "mean_training_score_oos_return_spearman",
        "median_training_score_oos_return_spearman",
    }

    assert expected_columns.issubset(frame.columns)


def test_run_selection_skill_batch_combines_symbols() -> None:
    frame = pd.concat(
        [
            _market_frame(symbol="SPY"),
            _market_frame(symbol="QQQ"),
        ],
        ignore_index=True,
    )

    windows = run_selection_skill_batch(
        frame,
        ["SPY", "QQQ"],
        _small_grid(),
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    assert len(windows) == 4
    assert {
        window.symbol
        for window in windows
    } == {"SPY", "QQQ"}


def test_run_selection_skill_batch_rejects_missing_symbol() -> None:
    with pytest.raises(
        ValueError,
        match="QQQ",
    ):
        run_selection_skill_batch(
            _market_frame(symbol="SPY"),
            ["QQQ"],
            _small_grid(),
            train_size=20,
            test_size=10,
            step_size=10,
            score_metric="total_return",
            minimum_completed_trades=0,
        )


def test_run_selection_skill_from_csv_creates_reports(
    tmp_path: Path,
) -> None:
    market_path = tmp_path / "market.csv"
    output_dir = tmp_path / "selection_skill"

    _market_frame().to_csv(
        market_path,
        index=False,
    )

    result = run_selection_skill_from_csv(
        market_path,
        ["SPY"],
        _small_grid(),
        output_dir,
        train_size=20,
        test_size=10,
        step_size=10,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    windows_path = (
        output_dir
        / "selection_skill_windows.csv"
    )

    summary_path = (
        output_dir
        / "selection_skill_summary.csv"
    )

    assert windows_path.exists()
    assert summary_path.exists()
    assert result["windows_path"] == windows_path
    assert result["summary_path"] == summary_path
    assert len(result["windows"]) == 2


def test_run_selection_skill_from_csv_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Market dataset",
    ):
        run_selection_skill_from_csv(
            tmp_path / "missing.csv",
            ["SPY"],
            _small_grid(),
            tmp_path / "out",
            train_size=20,
            test_size=10,
            step_size=10,
            score_metric="total_return",
            minimum_completed_trades=0,
        )
