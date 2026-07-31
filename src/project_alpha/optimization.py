from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from itertools import product
from numbers import Real

import pandas as pd

from project_alpha.backtest import (
    run_long_only_backtest,
)
from project_alpha.config import StrategyParameters
from project_alpha.indicators import add_indicators
from project_alpha.reporting import (
    sanitize_metric_values,
)
from project_alpha.strategy import (
    add_baseline_position,
)


GridValue = int | float
MetricValue = float | int | None


@dataclass(frozen=True)
class GridSearchResult:
    parameters: StrategyParameters
    metrics: dict[str, MetricValue]
    score_metric: str
    score: float | None


def generate_strategy_parameters(
    grid: Mapping[
        str,
        Sequence[GridValue],
    ],
) -> list[StrategyParameters]:
    if not grid:
        raise ValueError(
            "La griglia dei parametri è vuota."
        )

    allowed_fields = {
        field.name
        for field in fields(StrategyParameters)
    }

    unknown_fields = (
        set(grid)
        - allowed_fields
    )

    if unknown_fields:
        raise ValueError(
            "Parametri strategici sconosciuti: "
            f"{sorted(unknown_fields)}"
        )

    parameter_names = sorted(grid)
    parameter_values: list[
        tuple[GridValue, ...]
    ] = []

    for parameter_name in parameter_names:
        values = tuple(
            dict.fromkeys(
                grid[parameter_name]
            )
        )

        if not values:
            raise ValueError(
                "La griglia del parametro "
                f"'{parameter_name}' è vuota."
            )

        parameter_values.append(values)

    default_values = asdict(
        StrategyParameters()
    )

    combinations: list[
        StrategyParameters
    ] = []

    for values in product(
        *parameter_values
    ):
        candidate_values = (
            default_values.copy()
        )

        candidate_values.update(
            dict(
                zip(
                    parameter_names,
                    values,
                    strict=True,
                )
            )
        )

        combinations.append(
            StrategyParameters(
                **candidate_values
            )
        )

    return combinations


def _metric_as_float(
    metrics: Mapping[
        str,
        MetricValue,
    ],
    metric_name: str,
) -> float | None:
    value = metrics.get(metric_name)

    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        return None

    return float(value)


def run_parameter_grid_search(
    frame: pd.DataFrame,
    grid: Mapping[
        str,
        Sequence[GridValue],
    ],
    *,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
    score_metric: str = (
        "annualized_sharpe"
    ),
) -> list[GridSearchResult]:
    if frame.empty:
        raise ValueError(
            "Il dataset della grid search è vuoto."
        )

    if "symbol" not in frame.columns:
        raise ValueError(
            "La colonna 'symbol' è obbligatoria."
        )

    symbols = (
        frame["symbol"]
        .astype(str)
        .str.upper()
        .unique()
    )

    if len(symbols) != 1:
        raise ValueError(
            "La grid search deve ricevere "
            "un solo simbolo alla volta."
        )

    if initial_capital <= 0.0:
        raise ValueError(
            "initial_capital deve essere positivo."
        )

    if cost_per_side < 0.0:
        raise ValueError(
            "cost_per_side non può essere negativo."
        )

    parameter_sets = (
        generate_strategy_parameters(grid)
    )

    results: list[GridSearchResult] = []

    for parameters in parameter_sets:
        enriched = add_indicators(
            frame,
            parameters,
        )

        positioned = add_baseline_position(
            enriched,
            parameters,
        )

        backtest_result = (
            run_long_only_backtest(
                positioned,
                initial_capital=(
                    initial_capital
                ),
                cost_per_side=(
                    cost_per_side
                ),
            )
        )

        metrics = sanitize_metric_values(
            backtest_result.metrics
        )

        if score_metric not in metrics:
            raise ValueError(
                "Metrica di ordinamento "
                f"'{score_metric}' non disponibile."
            )

        score = _metric_as_float(
            metrics,
            score_metric,
        )

        results.append(
            GridSearchResult(
                parameters=parameters,
                metrics=metrics,
                score_metric=score_metric,
                score=score,
            )
        )

    def sort_key(
        result: GridSearchResult,
    ) -> tuple[float, float, float]:
        score = (
            result.score
            if result.score is not None
            else float("-inf")
        )

        total_return = (
            _metric_as_float(
                result.metrics,
                "total_return",
            )
        )

        max_drawdown = (
            _metric_as_float(
                result.metrics,
                "max_drawdown",
            )
        )

        return (
            score,
            (
                total_return
                if total_return is not None
                else float("-inf")
            ),
            (
                max_drawdown
                if max_drawdown is not None
                else float("-inf")
            ),
        )

    return sorted(
        results,
        key=sort_key,
        reverse=True,
    )


def grid_search_results_to_frame(
    results: Sequence[
        GridSearchResult
    ],
) -> pd.DataFrame:
    rows = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                **asdict(result.parameters),
                "score_metric": (
                    result.score_metric
                ),
                "score": result.score,
                **result.metrics,
            }
        )

    return pd.DataFrame(rows)