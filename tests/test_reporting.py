from pathlib import Path

import pandas as pd
import pytest

from project_alpha.reporting import (
    save_batch_summary_csv,
    summarize_walk_forward_metrics,
)


def test_save_batch_summary_csv(
    tmp_path: Path,
) -> None:
    results = {
        "SPY": {
            "initial_capital": 10_000.0,
            "total_return": 0.10,
        },
        "QQQ": {
            "initial_capital": 10_000.0,
            "total_return": 0.25,
        },
    }

    output_path = (
        tmp_path / "batch_metrics.csv"
    )

    returned_path = save_batch_summary_csv(
        results,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()

    frame = pd.read_csv(
        output_path
    )

    assert frame["symbol"].tolist() == [
        "SPY",
        "QQQ",
    ]

    assert frame.loc[
        0,
        "total_return",
    ] == pytest.approx(0.10)

    assert frame.loc[
        1,
        "total_return",
    ] == pytest.approx(0.25)


def test_save_batch_summary_csv_rejects_empty_results(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path / "batch_metrics.csv"
    )

    with pytest.raises(
        ValueError,
        match="Nessun risultato",
    ):
        save_batch_summary_csv(
            {},
            output_path,
        )


def test_summarize_walk_forward_metrics() -> None:
    results = [
        {
            "total_return": 0.10,
            "benchmark_total_return": 0.05,
            "annualized_sharpe": 1.20,
            "max_drawdown": -0.08,
            "completed_trades": 3,
        },
        {
            "total_return": -0.05,
            "benchmark_total_return": 0.02,
            "annualized_sharpe": 0.40,
            "max_drawdown": -0.12,
            "completed_trades": 2,
        },
    ]

    summary = (
        summarize_walk_forward_metrics(
            results
        )
    )

    assert summary["window_count"] == 2

    assert (
        summary["positive_window_count"]
        == 1
    )

    assert summary[
        "positive_window_rate"
    ] == pytest.approx(0.50)

    assert summary[
        "compounded_total_return"
    ] == pytest.approx(0.045)

    assert summary[
        "compounded_benchmark_total_return"
    ] == pytest.approx(0.071)

    assert summary[
        "compounded_excess_return"
    ] == pytest.approx(-0.026)

    assert summary[
        "mean_window_annualized_sharpe"
    ] == pytest.approx(0.80)

    assert summary[
        "worst_max_drawdown"
    ] == pytest.approx(-0.12)

    assert summary[
        "total_completed_trades"
    ] == 5


def test_summarize_walk_forward_metrics_can_disable_compounding(
) -> None:
    results = [
        {
            "total_return": 0.10,
            "benchmark_total_return": 0.05,
            "annualized_sharpe": 1.00,
            "max_drawdown": -0.10,
            "completed_trades": 4,
        }
    ]

    summary = (
        summarize_walk_forward_metrics(
            results,
            allow_compounding=False,
        )
    )

    assert (
        summary["compounded_total_return"]
        is None
    )

    assert (
        summary[
            "compounded_benchmark_total_return"
        ]
        is None
    )

    assert (
        summary[
            "compounded_excess_return"
        ]
        is None
    )


def test_summarize_walk_forward_metrics_rejects_empty_results(
) -> None:
    with pytest.raises(
        ValueError,
        match="Nessun risultato out-of-sample",
    ):
        summarize_walk_forward_metrics([])


def test_summarize_walk_forward_metrics_rejects_invalid_metric(
) -> None:
    results = [
        {
            "total_return": 0.10,
            "benchmark_total_return": 0.05,
            "annualized_sharpe": 1.00,
            "max_drawdown": -0.10,
            "completed_trades": 2.5,
        }
    ]

    with pytest.raises(
        ValueError,
        match="intero non negativo",
    ):
        summarize_walk_forward_metrics(
            results
        )