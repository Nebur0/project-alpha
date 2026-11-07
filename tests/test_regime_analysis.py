import json
from pathlib import Path

import pandas as pd
import pytest

from project_alpha.regime_analysis import (
    build_parser,
    run_regime_analysis,
)


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "SPY",
                "SPY",
                "SPY",
                "SPY",
                "SPY",
                "SPY",
            ],
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=6,
                freq="D",
                tz="UTC",
            ),
            "close": [
                100.0,
                101.0,
                102.0,
                101.0,
                103.0,
                104.0,
            ],
        }
    )


def _optimization_frame() -> pd.DataFrame:
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


def _write_inputs(
    tmp_path: Path,
) -> tuple[Path, Path]:
    market_path = (
        tmp_path / "market.csv"
    )
    optimization_path = (
        tmp_path / "optimization.csv"
    )

    _market_frame().to_csv(
        market_path,
        index=False,
    )
    _optimization_frame().to_csv(
        optimization_path,
        index=False,
    )

    return (
        market_path,
        optimization_path,
    )


def test_build_parser_reads_arguments(
) -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--market",
            "data/raw/bars.csv",
            "--optimization-report",
            (
                "reports/"
                "batch_walk_forward_"
                "optimization.csv"
            ),
            "--output-dir",
            "reports/custom_regimes",
            "--annualization-factor",
            "365",
            "--high-volatility-threshold",
            "0.30",
        ]
    )

    assert args.market == Path(
        "data/raw/bars.csv"
    )
    assert args.optimization_report == Path(
        "reports/"
        "batch_walk_forward_"
        "optimization.csv"
    )
    assert args.output_dir == Path(
        "reports/custom_regimes"
    )
    assert args.annualization_factor == 365
    assert (
        args.high_volatility_threshold
        == pytest.approx(
            0.30
        )
    )


def test_build_parser_uses_frozen_defaults(
) -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--market",
            "market.csv",
            "--optimization-report",
            "optimization.csv",
        ]
    )

    assert args.output_dir == Path(
        "reports/advanced_robustness"
    )
    assert args.annualization_factor == 252
    assert (
        args.high_volatility_threshold
        == pytest.approx(
            0.25
        )
    )


def test_run_regime_analysis_creates_reports(
    tmp_path: Path,
) -> None:
    (
        market_path,
        optimization_path,
    ) = _write_inputs(
        tmp_path
    )

    output_dir = (
        tmp_path / "reports"
    )

    result = run_regime_analysis(
        market_path,
        optimization_path,
        output_dir,
        high_volatility_threshold=10.0,
    )

    assert len(
        result["window_results"]
    ) == 2
    assert len(
        result["summaries"]
    ) == 2

    window_csv_path = (
        output_dir
        / "regime_windows.csv"
    )
    summary_csv_path = (
        output_dir
        / "regime_summary.csv"
    )
    json_path = (
        output_dir
        / "regime_analysis.json"
    )

    assert window_csv_path.exists()
    assert summary_csv_path.exists()
    assert json_path.exists()

    window_frame = pd.read_csv(
        window_csv_path
    )
    summary_frame = pd.read_csv(
        summary_csv_path
    )

    assert len(window_frame) == 2
    assert set(
        summary_frame["regime"]
    ) == {
        "UP_LOW_VOL",
        "DOWN_LOW_VOL",
    }


def test_run_regime_analysis_serializes_config(
    tmp_path: Path,
) -> None:
    (
        market_path,
        optimization_path,
    ) = _write_inputs(
        tmp_path
    )

    output_dir = (
        tmp_path / "reports"
    )

    run_regime_analysis(
        market_path,
        optimization_path,
        output_dir,
        annualization_factor=365,
        high_volatility_threshold=0.30,
    )

    report = json.loads(
        (
            output_dir
            / "regime_analysis.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert report["config"] == {
        "annualization_factor": 365,
        "high_volatility_threshold": 0.30,
    }
    assert report["window_count"] == 2
    assert report["regime_count"] >= 1
    assert report["market_path"] == str(
        market_path
    )
    assert (
        report[
            "optimization_report_path"
        ]
        == str(
            optimization_path
        )
    )


def test_run_regime_analysis_returns_output_paths(
    tmp_path: Path,
) -> None:
    (
        market_path,
        optimization_path,
    ) = _write_inputs(
        tmp_path
    )

    output_dir = (
        tmp_path / "reports"
    )

    result = run_regime_analysis(
        market_path,
        optimization_path,
        output_dir,
    )

    assert result[
        "window_csv_path"
    ] == (
        output_dir
        / "regime_windows.csv"
    )
    assert result[
        "summary_csv_path"
    ] == (
        output_dir
        / "regime_summary.csv"
    )
    assert result[
        "json_path"
    ] == (
        output_dir
        / "regime_analysis.json"
    )


def test_run_regime_analysis_rejects_missing_market_file(
    tmp_path: Path,
) -> None:
    optimization_path = (
        tmp_path
        / "optimization.csv"
    )

    _optimization_frame().to_csv(
        optimization_path,
        index=False,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Market dataset",
    ):
        run_regime_analysis(
            tmp_path / "missing.csv",
            optimization_path,
            tmp_path / "reports",
        )


def test_run_regime_analysis_rejects_missing_optimization_report(
    tmp_path: Path,
) -> None:
    market_path = (
        tmp_path / "market.csv"
    )

    _market_frame().to_csv(
        market_path,
        index=False,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Report walk-forward",
    ):
        run_regime_analysis(
            market_path,
            tmp_path / "missing.csv",
            tmp_path / "reports",
        )


def test_run_regime_analysis_rejects_invalid_config(
    tmp_path: Path,
) -> None:
    (
        market_path,
        optimization_path,
    ) = _write_inputs(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match=(
            "high_volatility_threshold"
        ),
    ):
        run_regime_analysis(
            market_path,
            optimization_path,
            tmp_path / "reports",
            high_volatility_threshold=(
                -0.01
            ),
        )
