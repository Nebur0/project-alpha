from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TemporalClusterResult:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    symbol_count: int
    positive_symbol_count: int
    positive_symbol_rate: float
    benchmark_beat_count: int
    benchmark_beat_rate: float
    mean_strategy_total_return: float
    median_strategy_total_return: float
    mean_benchmark_total_return: float
    median_benchmark_total_return: float
    mean_excess_return: float
    median_excess_return: float
    mean_annualized_sharpe: float | None
    worst_max_drawdown: float | None
    total_completed_trades: int


@dataclass(frozen=True)
class TemporalRobustnessSummary:
    cluster_count: int
    total_symbol_windows: int
    mean_cluster_strategy_return: float
    median_cluster_strategy_return: float
    mean_cluster_benchmark_return: float
    median_cluster_benchmark_return: float
    mean_cluster_excess_return: float
    median_cluster_excess_return: float
    positive_cluster_count: int
    positive_cluster_rate: float
    benchmark_beat_cluster_count: int
    benchmark_beat_cluster_rate: float


def _required_numeric(
    value: object,
    *,
    name: str,
) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} deve essere numerico.")

    numeric_value = float(value)

    if not isfinite(numeric_value):
        raise ValueError(f"{name} deve essere finito.")

    return numeric_value


def _validate_optimization_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "symbol",
        "window_number",
        "test_start",
        "test_end",
        "out_of_sample_total_return",
        "out_of_sample_benchmark_total_return",
        "out_of_sample_annualized_sharpe",
        "out_of_sample_max_drawdown",
        "out_of_sample_completed_trades",
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(f"Colonne mancanti nel report walk-forward: {sorted(missing_columns)}")

    if frame.empty:
        raise ValueError("Il report walk-forward è vuoto.")

    result = frame.copy()

    result["symbol"] = result["symbol"].astype(str).str.upper()

    result["test_start"] = pd.to_datetime(
        result["test_start"],
        utc=True,
    )
    result["test_end"] = pd.to_datetime(
        result["test_end"],
        utc=True,
    )

    if (result["test_end"] < result["test_start"]).any():
        raise ValueError("Sono presenti finestre con test_end precedente a test_start.")

    duplicate_mask = result.duplicated(
        subset=[
            "symbol",
            "test_start",
            "test_end",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = result.loc[
            duplicate_mask,
            [
                "symbol",
                "test_start",
                "test_end",
            ],
        ]

        raise ValueError(
            "Sono presenti finestre duplicate "
            "per simbolo e periodo: "
            f"{duplicates.to_dict('records')}"
        )

    return result.sort_values(
        [
            "test_start",
            "test_end",
            "symbol",
        ],
        kind="stable",
    ).reset_index(drop=True)


def analyze_temporal_clusters(
    optimization_frame: pd.DataFrame,
) -> tuple[TemporalClusterResult, ...]:
    frame = _validate_optimization_frame(optimization_frame)

    results: list[TemporalClusterResult] = []

    grouped = frame.groupby(
        [
            "test_start",
            "test_end",
        ],
        sort=True,
        dropna=False,
    )

    for (
        test_start,
        test_end,
    ), group in grouped:
        strategy_returns = pd.to_numeric(
            group["out_of_sample_total_return"],
            errors="raise",
        )

        benchmark_returns = pd.to_numeric(
            group[("out_of_sample_benchmark_total_return")],
            errors="raise",
        )

        excess_returns = strategy_returns - benchmark_returns

        positive_symbol_count = int((strategy_returns > 0.0).sum())

        benchmark_beat_count = int((excess_returns > 0.0).sum())

        symbol_count = len(group)

        sharpe_values = pd.to_numeric(
            group[("out_of_sample_annualized_sharpe")],
            errors="coerce",
        ).dropna()

        drawdown_values = pd.to_numeric(
            group[("out_of_sample_max_drawdown")],
            errors="coerce",
        ).dropna()

        trade_values = pd.to_numeric(
            group[("out_of_sample_completed_trades")],
            errors="coerce",
        ).dropna()

        if not trade_values.empty and (trade_values < 0.0).any():
            raise ValueError("completed_trades non può contenere valori negativi.")

        results.append(
            TemporalClusterResult(
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),
                symbol_count=(symbol_count),
                positive_symbol_count=(positive_symbol_count),
                positive_symbol_rate=(positive_symbol_count / symbol_count),
                benchmark_beat_count=(benchmark_beat_count),
                benchmark_beat_rate=(benchmark_beat_count / symbol_count),
                mean_strategy_total_return=(
                    _required_numeric(
                        strategy_returns.mean(),
                        name=("mean_strategy_total_return"),
                    )
                ),
                median_strategy_total_return=(
                    _required_numeric(
                        strategy_returns.median(),
                        name=("median_strategy_total_return"),
                    )
                ),
                mean_benchmark_total_return=(
                    _required_numeric(
                        benchmark_returns.mean(),
                        name=("mean_benchmark_total_return"),
                    )
                ),
                median_benchmark_total_return=(
                    _required_numeric(
                        benchmark_returns.median(),
                        name=("median_benchmark_total_return"),
                    )
                ),
                mean_excess_return=(
                    _required_numeric(
                        excess_returns.mean(),
                        name=("mean_excess_return"),
                    )
                ),
                median_excess_return=(
                    _required_numeric(
                        excess_returns.median(),
                        name=("median_excess_return"),
                    )
                ),
                mean_annualized_sharpe=(
                    float(sharpe_values.mean()) if not sharpe_values.empty else None
                ),
                worst_max_drawdown=(
                    float(drawdown_values.min()) if not drawdown_values.empty else None
                ),
                total_completed_trades=(int(trade_values.sum()) if not trade_values.empty else 0),
            )
        )

    return tuple(results)


def summarize_temporal_robustness(
    clusters: Iterable[TemporalClusterResult],
) -> TemporalRobustnessSummary:
    items = tuple(clusters)

    if not items:
        raise ValueError("Servono almeno un cluster temporale.")

    strategy_means = pd.Series(
        [item.mean_strategy_total_return for item in items],
        dtype=float,
    )

    benchmark_means = pd.Series(
        [item.mean_benchmark_total_return for item in items],
        dtype=float,
    )

    excess_means = pd.Series(
        [item.mean_excess_return for item in items],
        dtype=float,
    )

    positive_cluster_count = int((strategy_means > 0.0).sum())

    benchmark_beat_cluster_count = int((excess_means > 0.0).sum())

    cluster_count = len(items)

    return TemporalRobustnessSummary(
        cluster_count=(cluster_count),
        total_symbol_windows=(sum(item.symbol_count for item in items)),
        mean_cluster_strategy_return=(float(strategy_means.mean())),
        median_cluster_strategy_return=(float(strategy_means.median())),
        mean_cluster_benchmark_return=(float(benchmark_means.mean())),
        median_cluster_benchmark_return=(float(benchmark_means.median())),
        mean_cluster_excess_return=(float(excess_means.mean())),
        median_cluster_excess_return=(float(excess_means.median())),
        positive_cluster_count=(positive_cluster_count),
        positive_cluster_rate=(positive_cluster_count / cluster_count),
        benchmark_beat_cluster_count=(benchmark_beat_cluster_count),
        benchmark_beat_cluster_rate=(benchmark_beat_cluster_count / cluster_count),
    )


def temporal_clusters_to_frame(
    clusters: Iterable[TemporalClusterResult],
) -> pd.DataFrame:
    return pd.DataFrame(asdict(cluster) for cluster in clusters)


def temporal_summary_to_frame(
    summary: TemporalRobustnessSummary,
) -> pd.DataFrame:
    return pd.DataFrame([asdict(summary)])


def run_temporal_robustness_from_csv(
    optimization_report_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if not optimization_report_path.exists():
        raise FileNotFoundError(f"Report walk-forward non trovato: {optimization_report_path}")

    frame = pd.read_csv(
        optimization_report_path,
        parse_dates=[
            "test_start",
            "test_end",
        ],
    )

    clusters = analyze_temporal_clusters(frame)

    summary = summarize_temporal_robustness(clusters)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clusters_path = output_dir / "temporal_clusters.csv"

    summary_path = output_dir / "temporal_summary.csv"

    temporal_clusters_to_frame(clusters).to_csv(
        clusters_path,
        index=False,
    )

    temporal_summary_to_frame(summary).to_csv(
        summary_path,
        index=False,
    )

    return {
        "clusters": clusters,
        "summary": summary,
        "clusters_path": (clusters_path),
        "summary_path": (summary_path),
    }
