from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Final

import pandas as pd

DEFAULT_METRICS: Final[tuple[str, ...]] = (
    "mean_strategy_total_return",
    "mean_excess_return",
    "positive_symbol_rate",
    "benchmark_beat_rate",
    "mean_annualized_sharpe",
)

RATE_METRICS: Final[frozenset[str]] = frozenset(
    {
        "positive_symbol_rate",
        "benchmark_beat_rate",
    }
)


@dataclass(frozen=True)
class StatisticalRobustnessMetric:
    metric: str
    cluster_count: int
    observed_mean: float
    bootstrap_sample_count: int
    bootstrap_mean: float
    bootstrap_median: float
    bootstrap_ci_lower_95: float
    bootstrap_ci_upper_95: float
    bootstrap_probability_above_zero: float
    bootstrap_probability_above_half: float | None
    leave_one_out_min_mean: float
    leave_one_out_max_mean: float
    leave_one_out_range: float


@dataclass(frozen=True)
class LeaveOneClusterOutResult:
    omitted_test_start: str
    omitted_test_end: str
    remaining_cluster_count: int
    mean_strategy_total_return: float
    mean_excess_return: float
    positive_symbol_rate: float
    benchmark_beat_rate: float
    mean_annualized_sharpe: float


def _required_float(
    value: object,
    *,
    name: str,
) -> float:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} deve essere numerico."
        )

    numeric_value = float(value)

    if not isfinite(numeric_value):
        raise ValueError(
            f"{name} deve essere finito."
        )

    return numeric_value


def _validate_temporal_clusters_frame(
    frame: pd.DataFrame,
    metrics: tuple[str, ...],
) -> None:
    if frame.empty:
        raise ValueError(
            "Il report temporale è vuoto."
        )

    required_columns = {
        "test_start",
        "test_end",
        *metrics,
    }

    missing = sorted(
        required_columns
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Colonne mancanti nel report "
            f"temporale: {missing}."
        )

    if len(frame) < 2:
        raise ValueError(
            "Servono almeno due cluster "
            "temporali."
        )

    for metric in metrics:
        numeric = pd.to_numeric(
            frame[metric],
            errors="coerce",
        )

        if numeric.isna().any():
            raise ValueError(
                f"La metrica '{metric}' contiene "
                "valori non numerici o mancanti."
            )

        if not numeric.map(
            lambda value: isfinite(
                float(value)
            )
        ).all():
            raise ValueError(
                f"La metrica '{metric}' contiene "
                "valori non finiti."
            )


def exact_cluster_bootstrap_distribution(
    values: list[float],
) -> tuple[float, ...]:
    if len(values) < 2:
        raise ValueError(
            "Servono almeno due valori "
            "cluster-level."
        )

    normalized = [
        _required_float(
            value,
            name="cluster_value",
        )
        for value in values
    ]

    cluster_count = len(
        normalized
    )

    means: list[float] = []

    for indices in product(
        range(cluster_count),
        repeat=cluster_count,
    ):
        sample_mean = sum(
            normalized[index]
            for index in indices
        ) / cluster_count

        means.append(
            float(sample_mean)
        )

    return tuple(
        means
    )


def _percentile(
    values: tuple[float, ...],
    percentile: float,
) -> float:
    if not values:
        raise ValueError(
            "La distribuzione bootstrap "
            "è vuota."
        )

    if not 0.0 <= percentile <= 1.0:
        raise ValueError(
            "percentile deve essere compreso "
            "tra 0 e 1."
        )

    series = pd.Series(
        values,
        dtype=float,
    )

    return float(
        series.quantile(
            percentile,
            interpolation="linear",
        )
    )


def leave_one_cluster_out(
    frame: pd.DataFrame,
) -> tuple[
    LeaveOneClusterOutResult,
    ...,
]:
    _validate_temporal_clusters_frame(
        frame,
        DEFAULT_METRICS,
    )

    results: list[
        LeaveOneClusterOutResult
    ] = []

    for omitted_index in frame.index:
        omitted = frame.loc[
            omitted_index
        ]

        remaining = frame.drop(
            index=omitted_index
        )

        results.append(
            LeaveOneClusterOutResult(
                omitted_test_start=str(
                    omitted[
                        "test_start"
                    ]
                ),
                omitted_test_end=str(
                    omitted[
                        "test_end"
                    ]
                ),
                remaining_cluster_count=len(
                    remaining
                ),
                mean_strategy_total_return=float(
                    pd.to_numeric(
                        remaining[
                            "mean_strategy_total_return"
                        ],
                        errors="raise",
                    ).mean()
                ),
                mean_excess_return=float(
                    pd.to_numeric(
                        remaining[
                            "mean_excess_return"
                        ],
                        errors="raise",
                    ).mean()
                ),
                positive_symbol_rate=float(
                    pd.to_numeric(
                        remaining[
                            "positive_symbol_rate"
                        ],
                        errors="raise",
                    ).mean()
                ),
                benchmark_beat_rate=float(
                    pd.to_numeric(
                        remaining[
                            "benchmark_beat_rate"
                        ],
                        errors="raise",
                    ).mean()
                ),
                mean_annualized_sharpe=float(
                    pd.to_numeric(
                        remaining[
                            "mean_annualized_sharpe"
                        ],
                        errors="raise",
                    ).mean()
                ),
            )
        )

    return tuple(
        results
    )


def leave_one_cluster_out_to_frame(
    results: tuple[
        LeaveOneClusterOutResult,
        ...,
    ],
) -> pd.DataFrame:
    return pd.DataFrame(
        asdict(
            result
        )
        for result in results
    )


def exact_bootstrap_distribution_to_frame(
    frame: pd.DataFrame,
    metrics: tuple[str, ...] = (
        DEFAULT_METRICS
    ),
) -> pd.DataFrame:
    _validate_temporal_clusters_frame(
        frame,
        metrics,
    )

    distributions = {
        metric: (
            exact_cluster_bootstrap_distribution(
                [
                    float(value)
                    for value
                    in pd.to_numeric(
                        frame[metric],
                        errors="raise",
                    )
                ]
            )
        )
        for metric in metrics
    }

    expected_count = (
        len(frame) ** len(frame)
    )

    for metric, distribution in (
        distributions.items()
    ):
        if (
            len(distribution)
            != expected_count
        ):
            raise RuntimeError(
                "Dimensione bootstrap inattesa "
                f"per '{metric}'."
            )

    return pd.DataFrame(
        {
            "bootstrap_sample": list(
                range(
                    1,
                    expected_count + 1,
                )
            ),
            **{
                metric: distribution
                for metric, distribution
                in distributions.items()
            },
        }
    )


def summarize_statistical_robustness(
    frame: pd.DataFrame,
    metrics: tuple[str, ...] = (
        DEFAULT_METRICS
    ),
) -> tuple[
    StatisticalRobustnessMetric,
    ...,
]:
    _validate_temporal_clusters_frame(
        frame,
        metrics,
    )

    bootstrap_frame = (
        exact_bootstrap_distribution_to_frame(
            frame,
            metrics,
        )
    )

    loo_results = (
        leave_one_cluster_out(
            frame
        )
    )

    loo_frame = (
        leave_one_cluster_out_to_frame(
            loo_results
        )
    )

    summaries: list[
        StatisticalRobustnessMetric
    ] = []

    for metric in metrics:
        observed_values = pd.to_numeric(
            frame[metric],
            errors="raise",
        ).astype(float)

        bootstrap_values = tuple(
            float(value)
            for value in bootstrap_frame[
                metric
            ]
        )

        loo_values = pd.to_numeric(
            loo_frame[metric],
            errors="raise",
        ).astype(float)

        probability_above_zero = (
            sum(
                value > 0.0
                for value in bootstrap_values
            )
            / len(
                bootstrap_values
            )
        )

        probability_above_half: (
            float | None
        )

        if metric in RATE_METRICS:
            probability_above_half = (
                sum(
                    value > 0.5
                    for value
                    in bootstrap_values
                )
                / len(
                    bootstrap_values
                )
            )
        else:
            probability_above_half = None

        loo_min = float(
            loo_values.min()
        )

        loo_max = float(
            loo_values.max()
        )

        summaries.append(
            StatisticalRobustnessMetric(
                metric=metric,
                cluster_count=len(
                    frame
                ),
                observed_mean=float(
                    observed_values.mean()
                ),
                bootstrap_sample_count=len(
                    bootstrap_values
                ),
                bootstrap_mean=float(
                    pd.Series(
                        bootstrap_values,
                        dtype=float,
                    ).mean()
                ),
                bootstrap_median=float(
                    pd.Series(
                        bootstrap_values,
                        dtype=float,
                    ).median()
                ),
                bootstrap_ci_lower_95=(
                    _percentile(
                        bootstrap_values,
                        0.025,
                    )
                ),
                bootstrap_ci_upper_95=(
                    _percentile(
                        bootstrap_values,
                        0.975,
                    )
                ),
                bootstrap_probability_above_zero=(
                    probability_above_zero
                ),
                bootstrap_probability_above_half=(
                    probability_above_half
                ),
                leave_one_out_min_mean=(
                    loo_min
                ),
                leave_one_out_max_mean=(
                    loo_max
                ),
                leave_one_out_range=(
                    loo_max
                    - loo_min
                ),
            )
        )

    return tuple(
        summaries
    )


def statistical_robustness_summary_to_frame(
    results: tuple[
        StatisticalRobustnessMetric,
        ...,
    ],
) -> pd.DataFrame:
    return pd.DataFrame(
        asdict(
            result
        )
        for result in results
    )


def run_statistical_robustness_from_csv(
    temporal_clusters_path: Path,
    output_dir: Path,
    *,
    metrics: tuple[str, ...] = (
        DEFAULT_METRICS
    ),
) -> dict[str, object]:
    if not temporal_clusters_path.exists():
        raise FileNotFoundError(
            "Temporal clusters report non "
            f"trovato: {temporal_clusters_path}"
        )

    frame = pd.read_csv(
        temporal_clusters_path
    )

    _validate_temporal_clusters_frame(
        frame,
        metrics,
    )

    bootstrap_frame = (
        exact_bootstrap_distribution_to_frame(
            frame,
            metrics,
        )
    )

    leave_one_out_results = (
        leave_one_cluster_out(
            frame
        )
    )

    summary_results = (
        summarize_statistical_robustness(
            frame,
            metrics,
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bootstrap_path = (
        output_dir
        / "time_cluster_bootstrap_distribution.csv"
    )

    leave_one_out_path = (
        output_dir
        / "time_cluster_leave_one_out.csv"
    )

    summary_path = (
        output_dir
        / "statistical_robustness_summary.csv"
    )

    bootstrap_frame.to_csv(
        bootstrap_path,
        index=False,
    )

    leave_one_cluster_out_to_frame(
        leave_one_out_results
    ).to_csv(
        leave_one_out_path,
        index=False,
    )

    statistical_robustness_summary_to_frame(
        summary_results
    ).to_csv(
        summary_path,
        index=False,
    )

    return {
        "bootstrap_distribution": (
            bootstrap_frame
        ),
        "leave_one_out": (
            leave_one_out_results
        ),
        "summary": (
            summary_results
        ),
        "bootstrap_path": (
            bootstrap_path
        ),
        "leave_one_out_path": (
            leave_one_out_path
        ),
        "summary_path": (
            summary_path
        ),
    }
