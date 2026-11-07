from pathlib import Path

import pandas as pd
import pytest

from project_alpha.temporal_robustness import (
    analyze_temporal_clusters,
    run_temporal_robustness_from_csv,
    summarize_temporal_robustness,
    temporal_clusters_to_frame,
    temporal_summary_to_frame,
)


def _optimization_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "SPY",
                "QQQ",
                "SPY",
                "QQQ",
            ],
            "window_number": [
                1,
                1,
                2,
                2,
            ],
            "test_start": [
                "2022-01-01T00:00:00+00:00",
                "2022-01-01T00:00:00+00:00",
                "2023-01-01T00:00:00+00:00",
                "2023-01-01T00:00:00+00:00",
            ],
            "test_end": [
                "2022-12-31T00:00:00+00:00",
                "2022-12-31T00:00:00+00:00",
                "2023-12-31T00:00:00+00:00",
                "2023-12-31T00:00:00+00:00",
            ],
            "out_of_sample_total_return": [
                -0.02,
                0.01,
                0.10,
                0.20,
            ],
            (
                "out_of_sample_"
                "benchmark_total_return"
            ): [
                -0.15,
                -0.20,
                0.25,
                0.30,
            ],
            (
                "out_of_sample_"
                "annualized_sharpe"
            ): [
                -0.20,
                0.10,
                0.80,
                1.20,
            ],
            (
                "out_of_sample_"
                "max_drawdown"
            ): [
                -0.08,
                -0.05,
                -0.04,
                -0.06,
            ],
            (
                "out_of_sample_"
                "completed_trades"
            ): [
                2,
                1,
                3,
                4,
            ],
        }
    )


def test_analyze_temporal_clusters_groups_same_period(
) -> None:
    clusters = analyze_temporal_clusters(
        _optimization_frame()
    )

    assert len(clusters) == 2
    assert clusters[0].symbol_count == 2
    assert clusters[1].symbol_count == 2


def test_analyze_temporal_clusters_calculates_first_cluster_metrics(
) -> None:
    clusters = analyze_temporal_clusters(
        _optimization_frame()
    )

    first = clusters[0]

    assert (
        first.positive_symbol_count
        == 1
    )
    assert (
        first.positive_symbol_rate
        == pytest.approx(
            0.5
        )
    )

    assert (
        first.benchmark_beat_count
        == 2
    )
    assert (
        first.benchmark_beat_rate
        == pytest.approx(
            1.0
        )
    )

    assert (
        first.mean_strategy_total_return
        == pytest.approx(
            -0.005
        )
    )
    assert (
        first.median_strategy_total_return
        == pytest.approx(
            -0.005
        )
    )

    assert (
        first.mean_benchmark_total_return
        == pytest.approx(
            -0.175
        )
    )

    assert (
        first.mean_excess_return
        == pytest.approx(
            0.17
        )
    )

    assert (
        first.mean_annualized_sharpe
        == pytest.approx(
            -0.05
        )
    )

    assert (
        first.worst_max_drawdown
        == pytest.approx(
            -0.08
        )
    )

    assert (
        first.total_completed_trades
        == 3
    )


def test_analyze_temporal_clusters_calculates_second_cluster_metrics(
) -> None:
    clusters = analyze_temporal_clusters(
        _optimization_frame()
    )

    second = clusters[1]

    assert (
        second.positive_symbol_count
        == 2
    )
    assert (
        second.positive_symbol_rate
        == pytest.approx(
            1.0
        )
    )

    assert (
        second.benchmark_beat_count
        == 0
    )
    assert (
        second.benchmark_beat_rate
        == pytest.approx(
            0.0
        )
    )

    assert (
        second.mean_strategy_total_return
        == pytest.approx(
            0.15
        )
    )

    assert (
        second.mean_benchmark_total_return
        == pytest.approx(
            0.275
        )
    )

    assert (
        second.mean_excess_return
        == pytest.approx(
            -0.125
        )
    )

    assert (
        second.total_completed_trades
        == 7
    )


def test_analyze_temporal_clusters_normalizes_symbol_case(
) -> None:
    frame = _optimization_frame()
    frame["symbol"] = [
        "spy",
        "qqq",
        "SPY",
        "QQQ",
    ]

    clusters = analyze_temporal_clusters(
        frame
    )

    assert len(clusters) == 2


def test_analyze_temporal_clusters_rejects_missing_columns(
) -> None:
    frame = _optimization_frame().drop(
        columns=[
            "out_of_sample_total_return"
        ]
    )

    with pytest.raises(
        ValueError,
        match="Colonne mancanti",
    ):
        analyze_temporal_clusters(
            frame
        )


def test_analyze_temporal_clusters_rejects_empty_frame(
) -> None:
    frame = _optimization_frame().iloc[
        0:0
    ]

    with pytest.raises(
        ValueError,
        match="vuoto",
    ):
        analyze_temporal_clusters(
            frame
        )


def test_analyze_temporal_clusters_rejects_invalid_dates(
) -> None:
    frame = _optimization_frame()
    frame.loc[
        0,
        "test_end",
    ] = "2021-12-31T00:00:00+00:00"

    with pytest.raises(
        ValueError,
        match="test_end",
    ):
        analyze_temporal_clusters(
            frame
        )


def test_analyze_temporal_clusters_rejects_duplicate_symbol_period(
) -> None:
    frame = pd.concat(
        [
            _optimization_frame(),
            _optimization_frame().iloc[
                [0]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        analyze_temporal_clusters(
            frame
        )


def test_analyze_temporal_clusters_rejects_negative_completed_trades(
) -> None:
    frame = _optimization_frame()

    frame.loc[
        0,
        "out_of_sample_completed_trades",
    ] = -1

    with pytest.raises(
        ValueError,
        match="completed_trades",
    ):
        analyze_temporal_clusters(
            frame
        )


def test_summarize_temporal_robustness_calculates_cluster_level_stats(
) -> None:
    clusters = analyze_temporal_clusters(
        _optimization_frame()
    )

    summary = summarize_temporal_robustness(
        clusters
    )

    assert summary.cluster_count == 2
    assert summary.total_symbol_windows == 4

    assert (
        summary.mean_cluster_strategy_return
        == pytest.approx(
            0.0725
        )
    )

    assert (
        summary.median_cluster_strategy_return
        == pytest.approx(
            0.0725
        )
    )

    assert (
        summary.mean_cluster_benchmark_return
        == pytest.approx(
            0.05
        )
    )

    assert (
        summary.mean_cluster_excess_return
        == pytest.approx(
            0.0225
        )
    )

    assert (
        summary.positive_cluster_count
        == 1
    )
    assert (
        summary.positive_cluster_rate
        == pytest.approx(
            0.5
        )
    )

    assert (
        summary.benchmark_beat_cluster_count
        == 1
    )
    assert (
        summary.benchmark_beat_cluster_rate
        == pytest.approx(
            0.5
        )
    )


def test_summarize_temporal_robustness_rejects_empty_input(
) -> None:
    with pytest.raises(
        ValueError,
        match="almeno un cluster",
    ):
        summarize_temporal_robustness(
            []
        )


def test_temporal_clusters_to_frame_contains_expected_columns(
) -> None:
    clusters = analyze_temporal_clusters(
        _optimization_frame()
    )

    frame = temporal_clusters_to_frame(
        clusters
    )

    expected_columns = {
        "test_start",
        "test_end",
        "symbol_count",
        "positive_symbol_count",
        "positive_symbol_rate",
        "benchmark_beat_count",
        "benchmark_beat_rate",
        "mean_strategy_total_return",
        "median_strategy_total_return",
        "mean_benchmark_total_return",
        "median_benchmark_total_return",
        "mean_excess_return",
        "median_excess_return",
        "mean_annualized_sharpe",
        "worst_max_drawdown",
        "total_completed_trades",
    }

    assert expected_columns.issubset(
        frame.columns
    )


def test_temporal_summary_to_frame_contains_expected_columns(
) -> None:
    clusters = analyze_temporal_clusters(
        _optimization_frame()
    )

    summary = summarize_temporal_robustness(
        clusters
    )

    frame = temporal_summary_to_frame(
        summary
    )

    expected_columns = {
        "cluster_count",
        "total_symbol_windows",
        "mean_cluster_strategy_return",
        "median_cluster_strategy_return",
        "mean_cluster_benchmark_return",
        "median_cluster_benchmark_return",
        "mean_cluster_excess_return",
        "median_cluster_excess_return",
        "positive_cluster_count",
        "positive_cluster_rate",
        "benchmark_beat_cluster_count",
        "benchmark_beat_cluster_rate",
    }

    assert expected_columns.issubset(
        frame.columns
    )


def test_run_temporal_robustness_from_csv_creates_reports(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path
        / "walk_forward.csv"
    )

    output_dir = (
        tmp_path
        / "temporal"
    )

    _optimization_frame().to_csv(
        input_path,
        index=False,
    )

    result = run_temporal_robustness_from_csv(
        input_path,
        output_dir,
    )

    clusters_path = (
        output_dir
        / "temporal_clusters.csv"
    )

    summary_path = (
        output_dir
        / "temporal_summary.csv"
    )

    assert clusters_path.exists()
    assert summary_path.exists()

    assert (
        result["clusters_path"]
        == clusters_path
    )

    assert (
        result["summary_path"]
        == summary_path
    )

    assert len(
        result["clusters"]
    ) == 2

    assert (
        result["summary"].cluster_count
        == 2
    )


def test_run_temporal_robustness_from_csv_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Report walk-forward",
    ):
        run_temporal_robustness_from_csv(
            tmp_path / "missing.csv",
            tmp_path / "out",
        )
