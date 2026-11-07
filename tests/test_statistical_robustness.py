from pathlib import Path

import pandas as pd
import pytest

from project_alpha.statistical_robustness import (
    DEFAULT_METRICS,
    exact_bootstrap_distribution_to_frame,
    exact_cluster_bootstrap_distribution,
    leave_one_cluster_out,
    leave_one_cluster_out_to_frame,
    run_statistical_robustness_from_csv,
    statistical_robustness_summary_to_frame,
    summarize_statistical_robustness,
)


def _temporal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "test_start": [
                "2021-01-01",
                "2022-01-01",
                "2023-01-01",
                "2024-01-01",
            ],
            "test_end": [
                "2021-12-31",
                "2022-12-31",
                "2023-12-31",
                "2024-12-31",
            ],
            "mean_strategy_total_return": [
                -0.10,
                0.10,
                0.20,
                0.00,
            ],
            "mean_excess_return": [
                0.20,
                -0.30,
                -0.20,
                -0.10,
            ],
            "positive_symbol_rate": [
                0.40,
                0.60,
                0.70,
                0.50,
            ],
            "benchmark_beat_rate": [
                0.70,
                0.20,
                0.30,
                0.20,
            ],
            "mean_annualized_sharpe": [
                -0.20,
                0.50,
                0.60,
                -0.10,
            ],
        }
    )


def test_exact_cluster_bootstrap_distribution_enumerates_all_samples() -> None:
    distribution = exact_cluster_bootstrap_distribution(
        [
            0.0,
            1.0,
        ]
    )

    assert len(distribution) == 4

    assert sorted(distribution) == pytest.approx(
        [
            0.0,
            0.5,
            0.5,
            1.0,
        ]
    )


def test_exact_cluster_bootstrap_distribution_rejects_single_cluster() -> None:
    with pytest.raises(
        ValueError,
        match="almeno due",
    ):
        exact_cluster_bootstrap_distribution(
            [
                1.0,
            ]
        )


def test_exact_bootstrap_distribution_has_n_power_n_rows() -> None:
    frame = _temporal_frame()

    result = exact_bootstrap_distribution_to_frame(frame)

    assert len(result) == 256

    assert {
        "bootstrap_sample",
        *DEFAULT_METRICS,
    }.issubset(result.columns)


def test_exact_bootstrap_distribution_preserves_observed_mean() -> None:
    frame = _temporal_frame()

    result = exact_bootstrap_distribution_to_frame(frame)

    for metric in DEFAULT_METRICS:
        assert float(result[metric].mean()) == pytest.approx(float(frame[metric].mean()))


def test_leave_one_cluster_out_returns_one_result_per_cluster() -> None:
    results = leave_one_cluster_out(_temporal_frame())

    assert len(results) == 4

    assert all(result.remaining_cluster_count == 3 for result in results)


def test_leave_one_cluster_out_mean_matches_manual_calculation() -> None:
    frame = _temporal_frame()

    results = leave_one_cluster_out(frame)

    first = results[0]

    expected = float(frame.iloc[1:]["mean_strategy_total_return"].mean())

    assert first.mean_strategy_total_return == pytest.approx(expected)


def test_leave_one_cluster_out_to_frame_contains_expected_columns() -> None:
    results = leave_one_cluster_out(_temporal_frame())

    frame = leave_one_cluster_out_to_frame(results)

    expected_columns = {
        "omitted_test_start",
        "omitted_test_end",
        "remaining_cluster_count",
        "mean_strategy_total_return",
        "mean_excess_return",
        "positive_symbol_rate",
        "benchmark_beat_rate",
        "mean_annualized_sharpe",
    }

    assert expected_columns.issubset(frame.columns)


def test_summarize_statistical_robustness_returns_one_row_per_metric() -> None:
    results = summarize_statistical_robustness(_temporal_frame())

    assert len(results) == len(DEFAULT_METRICS)

    assert {result.metric for result in results} == set(DEFAULT_METRICS)


def test_summarize_statistical_robustness_bootstrap_count_is_exact() -> None:
    results = summarize_statistical_robustness(_temporal_frame())

    assert all(result.bootstrap_sample_count == 256 for result in results)


def test_summarize_statistical_robustness_observed_and_bootstrap_means_match() -> None:
    results = summarize_statistical_robustness(_temporal_frame())

    for result in results:
        assert result.bootstrap_mean == pytest.approx(result.observed_mean)


def test_summarize_statistical_robustness_confidence_interval_is_ordered() -> None:
    results = summarize_statistical_robustness(_temporal_frame())

    for result in results:
        assert (
            result.bootstrap_ci_lower_95 <= result.bootstrap_median <= result.bootstrap_ci_upper_95
        )


def test_rate_metrics_include_probability_above_half() -> None:
    results = summarize_statistical_robustness(_temporal_frame())

    by_metric = {result.metric: result for result in results}

    assert by_metric["positive_symbol_rate"].bootstrap_probability_above_half is not None

    assert by_metric["benchmark_beat_rate"].bootstrap_probability_above_half is not None

    assert by_metric["mean_strategy_total_return"].bootstrap_probability_above_half is None


def test_probability_fields_are_bounded() -> None:
    results = summarize_statistical_robustness(_temporal_frame())

    for result in results:
        assert 0.0 <= result.bootstrap_probability_above_zero <= 1.0

        if result.bootstrap_probability_above_half is not None:
            assert 0.0 <= result.bootstrap_probability_above_half <= 1.0


def test_leave_one_out_range_matches_min_and_max() -> None:
    results = summarize_statistical_robustness(_temporal_frame())

    for result in results:
        assert result.leave_one_out_range == pytest.approx(
            result.leave_one_out_max_mean - result.leave_one_out_min_mean
        )


def test_statistical_robustness_summary_to_frame_contains_expected_columns() -> None:
    results = summarize_statistical_robustness(_temporal_frame())

    frame = statistical_robustness_summary_to_frame(results)

    expected_columns = {
        "metric",
        "cluster_count",
        "observed_mean",
        "bootstrap_sample_count",
        "bootstrap_mean",
        "bootstrap_median",
        "bootstrap_ci_lower_95",
        "bootstrap_ci_upper_95",
        "bootstrap_probability_above_zero",
        "bootstrap_probability_above_half",
        "leave_one_out_min_mean",
        "leave_one_out_max_mean",
        "leave_one_out_range",
    }

    assert expected_columns.issubset(frame.columns)


def test_summarize_rejects_missing_metric_column() -> None:
    frame = _temporal_frame().drop(
        columns=[
            "mean_excess_return",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Colonne mancanti",
    ):
        summarize_statistical_robustness(frame)


def test_summarize_rejects_non_numeric_metric() -> None:
    frame = _temporal_frame()

    frame["mean_excess_return"] = frame["mean_excess_return"].astype(object)

    frame.loc[
        0,
        "mean_excess_return",
    ] = "not-a-number"

    with pytest.raises(
        ValueError,
        match="mean_excess_return",
    ):
        summarize_statistical_robustness(frame)


def test_run_statistical_robustness_from_csv_creates_all_reports(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "temporal_clusters.csv"

    output_dir = tmp_path / "statistical"

    _temporal_frame().to_csv(
        input_path,
        index=False,
    )

    result = run_statistical_robustness_from_csv(
        input_path,
        output_dir,
    )

    bootstrap_path = output_dir / "time_cluster_bootstrap_distribution.csv"

    leave_one_out_path = output_dir / "time_cluster_leave_one_out.csv"

    summary_path = output_dir / "statistical_robustness_summary.csv"

    assert bootstrap_path.exists()
    assert leave_one_out_path.exists()
    assert summary_path.exists()

    assert result["bootstrap_path"] == bootstrap_path

    assert result["leave_one_out_path"] == leave_one_out_path

    assert result["summary_path"] == summary_path

    assert len(result["bootstrap_distribution"]) == 256

    assert len(result["leave_one_out"]) == 4

    assert len(result["summary"]) == len(DEFAULT_METRICS)


def test_run_statistical_robustness_from_csv_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Temporal clusters report",
    ):
        run_statistical_robustness_from_csv(
            tmp_path / "missing.csv",
            tmp_path / "out",
        )
