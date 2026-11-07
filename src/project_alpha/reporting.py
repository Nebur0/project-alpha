from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

MetricValue = float | int | None


def _required_float(
    metrics: Mapping[str, MetricValue],
    metric_name: str,
) -> float:
    value = metrics.get(metric_name)

    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"La metrica '{metric_name}' è assente "
            "o non è numerica."
        )

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"La metrica '{metric_name}' "
            "deve essere finita."
        )

    return numeric_value


def _optional_finite_float(
    metrics: Mapping[str, MetricValue],
    metric_name: str,
) -> float | None:
    value = metrics.get(metric_name)

    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"La metrica '{metric_name}' "
            "non è numerica."
        )

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def _required_non_negative_int(
    metrics: Mapping[str, MetricValue],
    metric_name: str,
) -> int:
    numeric_value = _required_float(
        metrics,
        metric_name,
    )

    if (
        numeric_value < 0
        or not numeric_value.is_integer()
    ):
        raise ValueError(
            f"La metrica '{metric_name}' deve "
            "essere un intero non negativo."
        )

    return int(numeric_value)


def sanitize_metric_values(
    metrics: Mapping[str, MetricValue],
) -> dict[str, MetricValue]:
    sanitized: dict[str, MetricValue] = {}

    for metric_name, value in metrics.items():
        if value is None:
            sanitized[metric_name] = None
            continue

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
        ):
            raise ValueError(
                f"La metrica '{metric_name}' "
                "non è numerica."
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            sanitized[metric_name] = None
        elif isinstance(value, Integral):
            sanitized[metric_name] = int(value)
        else:
            sanitized[metric_name] = numeric_value

    return sanitized


def summarize_walk_forward_metrics(
    out_of_sample_results: Sequence[
        Mapping[str, MetricValue]
    ],
    *,
    allow_compounding: bool = True,
) -> dict[str, MetricValue]:
    if not out_of_sample_results:
        raise ValueError(
            "Nessun risultato out-of-sample "
            "disponibile per il riepilogo."
        )

    total_returns = [
        _required_float(
            metrics,
            "total_return",
        )
        for metrics in out_of_sample_results
    ]

    benchmark_returns = [
        _required_float(
            metrics,
            "benchmark_total_return",
        )
        for metrics in out_of_sample_results
    ]

    sharpes = [
        sharpe
        for metrics in out_of_sample_results
        if (
            sharpe := _optional_finite_float(
                metrics,
                "annualized_sharpe",
            )
        )
        is not None
    ]

    drawdowns = [
        _required_float(
            metrics,
            "max_drawdown",
        )
        for metrics in out_of_sample_results
    ]

    completed_trades = [
        _required_non_negative_int(
            metrics,
            "completed_trades",
        )
        for metrics in out_of_sample_results
    ]

    window_count = len(
        out_of_sample_results
    )

    positive_window_count = sum(
        total_return > 0.0
        for total_return in total_returns
    )

    valid_sharpe_window_count = len(
        sharpes
    )

    mean_sharpe: float | None = None

    if sharpes:
        mean_sharpe = (
            sum(sharpes)
            / valid_sharpe_window_count
        )

    compounded_total_return: float | None = None
    compounded_benchmark_return: float | None = None
    compounded_excess_return: float | None = None

    if allow_compounding:
        compounded_total_return = (
            math.prod(
                1.0 + total_return
                for total_return in total_returns
            )
            - 1.0
        )

        compounded_benchmark_return = (
            math.prod(
                1.0 + benchmark_return
                for benchmark_return
                in benchmark_returns
            )
            - 1.0
        )

        compounded_excess_return = (
            compounded_total_return
            - compounded_benchmark_return
        )

    return {
        "window_count": window_count,
        "positive_window_count": (
            positive_window_count
        ),
        "positive_window_rate": (
            positive_window_count
            / window_count
        ),
        "compounded_total_return": (
            compounded_total_return
        ),
        "compounded_benchmark_total_return": (
            compounded_benchmark_return
        ),
        "compounded_excess_return": (
            compounded_excess_return
        ),
        "valid_sharpe_window_count": (
            valid_sharpe_window_count
        ),
        "mean_window_annualized_sharpe": (
            mean_sharpe
        ),
        "worst_max_drawdown": min(
            drawdowns
        ),
        "total_completed_trades": sum(
            completed_trades
        ),
    }


def save_batch_summary_csv(
    results: Mapping[
        str,
        Mapping[str, MetricValue],
    ],
    output_path: Path = Path(
        "reports/batch_metrics.csv"
    ),
) -> Path:
    if not results:
        raise ValueError(
            "Nessun risultato disponibile "
            "per creare il report CSV."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.DataFrame.from_dict(
        results,
        orient="index",
    )

    frame.index.name = "symbol"
    frame = frame.reset_index()

    frame.to_csv(
        output_path,
        index=False,
    )

    return output_path