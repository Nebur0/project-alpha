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
    summarize_walk_forward_metrics,
)
from project_alpha.scoring import (
    RobustScoreConfig,
    RobustScoreResult,
    calculate_robust_score,
)
from project_alpha.strategy import (
    add_baseline_position,
)
from project_alpha.validation import (
    generate_walk_forward_splits,
)


GridValue = int | float
MetricValue = float | int | None

ROBUST_SCORE_METRIC = "robust_score"


@dataclass(frozen=True)
class GridSearchResult:
    parameters: StrategyParameters
    metrics: dict[str, MetricValue]
    score_metric: str
    score: float | None
    robust_score: RobustScoreResult | None = None


@dataclass(frozen=True)
class WalkForwardOptimizationWindow:
    window_number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected_parameters: StrategyParameters
    selected_score: float
    in_sample_metrics: dict[str, MetricValue]
    out_of_sample_metrics: dict[str, MetricValue]
    robust_score: RobustScoreResult | None = None


@dataclass(frozen=True)
class WalkForwardOptimizationResult:
    symbol: str
    score_metric: str
    train_size: int
    test_size: int
    step_size: int
    minimum_completed_trades: int
    windows_overlap: bool
    compounding_enabled: bool
    robust_score_config: RobustScoreConfig | None
    windows: tuple[
        WalkForwardOptimizationWindow,
        ...,
    ]
    summary: dict[str, MetricValue]


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


def _resolve_robust_score_config(
    score_metric: str,
    robust_score_config: RobustScoreConfig | None,
) -> RobustScoreConfig | None:
    if score_metric == ROBUST_SCORE_METRIC:
        return (
            robust_score_config
            or RobustScoreConfig()
        )

    if robust_score_config is not None:
        raise ValueError(
            "robust_score_config può essere "
            "usato solo con "
            "score_metric='robust_score'."
        )

    return None


def _calculate_candidate_score(
    metrics: Mapping[
        str,
        MetricValue,
    ],
    *,
    score_metric: str,
    robust_score_config: RobustScoreConfig | None,
) -> tuple[
    float | None,
    RobustScoreResult | None,
]:
    if score_metric == ROBUST_SCORE_METRIC:
        if robust_score_config is None:
            raise RuntimeError(
                "Configurazione robust score "
                "non inizializzata."
            )

        try:
            robust_score = calculate_robust_score(
                metrics,
                robust_score_config,
            )
        except ValueError:
            return None, None

        return (
            robust_score.score,
            robust_score,
        )

    if score_metric not in metrics:
        raise ValueError(
            "Metrica di ordinamento "
            f"'{score_metric}' non disponibile."
        )

    return (
        _metric_as_float(
            metrics,
            score_metric,
        ),
        None,
    )


def _validate_single_symbol_frame(
    frame: pd.DataFrame,
) -> str:
    if frame.empty:
        raise ValueError(
            "Il dataset è vuoto."
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
            "L'ottimizzazione deve ricevere "
            "un solo simbolo alla volta."
        )

    return str(symbols[0])


def _validate_backtest_arguments(
    initial_capital: float,
    cost_per_side: float,
) -> None:
    if initial_capital <= 0.0:
        raise ValueError(
            "initial_capital deve essere positivo."
        )

    if cost_per_side < 0.0:
        raise ValueError(
            "cost_per_side non può essere negativo."
        )


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
    robust_score_config: (
        RobustScoreConfig | None
    ) = None,
) -> list[GridSearchResult]:
    _validate_single_symbol_frame(frame)

    _validate_backtest_arguments(
        initial_capital,
        cost_per_side,
    )

    effective_robust_score_config = (
        _resolve_robust_score_config(
            score_metric,
            robust_score_config,
        )
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

        (
            score,
            robust_score,
        ) = _calculate_candidate_score(
            metrics,
            score_metric=score_metric,
            robust_score_config=(
                effective_robust_score_config
            ),
        )

        results.append(
            GridSearchResult(
                parameters=parameters,
                metrics=metrics,
                score_metric=score_metric,
                score=score,
                robust_score=robust_score,
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

        total_return = _metric_as_float(
            result.metrics,
            "total_return",
        )

        max_drawdown = _metric_as_float(
            result.metrics,
            "max_drawdown",
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
        robust_values: dict[
            str,
            MetricValue,
        ] = {}

        if result.robust_score is not None:
            robust_values = {
                "robust_score": (
                    result.robust_score.score
                ),
                "robust_sharpe": (
                    result.robust_score.sharpe
                ),
                "robust_excess_return": (
                    result.robust_score
                    .excess_return
                ),
                "robust_max_drawdown": (
                    result.robust_score
                    .max_drawdown
                ),
                "robust_completed_trades": (
                    result.robust_score
                    .completed_trades
                ),
                "robust_trade_shortfall": (
                    result.robust_score
                    .trade_shortfall
                ),
            }

        rows.append(
            {
                "rank": rank,
                **asdict(result.parameters),
                "score_metric": (
                    result.score_metric
                ),
                "score": result.score,
                **robust_values,
                **result.metrics,
            }
        )

    return pd.DataFrame(rows)


def _select_best_training_result(
    results: Sequence[
        GridSearchResult
    ],
    *,
    minimum_completed_trades: int,
) -> GridSearchResult:
    for result in results:
        completed_trades = _metric_as_float(
            result.metrics,
            "completed_trades",
        )

        if result.score is None:
            continue

        if completed_trades is None:
            continue

        if (
            completed_trades
            < minimum_completed_trades
        ):
            continue

        return result

    raise ValueError(
        "Nessuna combinazione soddisfa i "
        "requisiti minimi nella finestra "
        "in-sample."
    )


def _prepare_out_of_sample_frame(
    in_sample: pd.DataFrame,
    out_of_sample: pd.DataFrame,
    parameters: StrategyParameters,
) -> pd.DataFrame:
    context = pd.concat(
        [
            in_sample,
            out_of_sample,
        ],
        ignore_index=True,
    )

    context = (
        context
        .sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    enriched_context = add_indicators(
        context,
        parameters,
    )

    test_start = pd.Timestamp(
        out_of_sample["timestamp"].min()
    )
    test_end = pd.Timestamp(
        out_of_sample["timestamp"].max()
    )

    timestamp_series = pd.to_datetime(
        enriched_context["timestamp"]
    )

    selected = enriched_context.loc[
        timestamp_series.between(
            test_start,
            test_end,
            inclusive="both",
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            "La finestra out-of-sample "
            "preparata è vuota."
        )

    return add_baseline_position(
        selected,
        parameters,
    )


def run_walk_forward_optimization(
    frame: pd.DataFrame,
    grid: Mapping[
        str,
        Sequence[GridValue],
    ],
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
    score_metric: str = (
        "annualized_sharpe"
    ),
    minimum_completed_trades: int = 1,
    robust_score_config: (
        RobustScoreConfig | None
    ) = None,
) -> WalkForwardOptimizationResult:
    symbol = _validate_single_symbol_frame(
        frame
    )

    _validate_backtest_arguments(
        initial_capital,
        cost_per_side,
    )

    effective_robust_score_config = (
        _resolve_robust_score_config(
            score_metric,
            robust_score_config,
        )
    )

    if (
        isinstance(
            minimum_completed_trades,
            bool,
        )
        or not isinstance(
            minimum_completed_trades,
            int,
        )
        or minimum_completed_trades < 0
    ):
        raise ValueError(
            "minimum_completed_trades deve "
            "essere un intero non negativo."
        )

    splits = generate_walk_forward_splits(
        frame=frame,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )

    effective_step_size = (
        test_size
        if step_size is None
        else step_size
    )

    windows_overlap = (
        effective_step_size
        < test_size
    )

    windows: list[
        WalkForwardOptimizationWindow
    ] = []

    out_of_sample_metrics: list[
        dict[str, MetricValue]
    ] = []

    for split in splits:
        training_results = (
            run_parameter_grid_search(
                split.in_sample,
                grid,
                initial_capital=(
                    initial_capital
                ),
                cost_per_side=(
                    cost_per_side
                ),
                score_metric=score_metric,
                robust_score_config=(
                    effective_robust_score_config
                ),
            )
        )

        selected_result = (
            _select_best_training_result(
                training_results,
                minimum_completed_trades=(
                    minimum_completed_trades
                ),
            )
        )

        positioned_out_of_sample = (
            _prepare_out_of_sample_frame(
                split.in_sample,
                split.out_of_sample,
                selected_result.parameters,
            )
        )

        out_of_sample_result = (
            run_long_only_backtest(
                positioned_out_of_sample,
                initial_capital=(
                    initial_capital
                ),
                cost_per_side=(
                    cost_per_side
                ),
            )
        )

        sanitized_out_of_sample_metrics = (
            sanitize_metric_values(
                out_of_sample_result.metrics
            )
        )

        out_of_sample_metrics.append(
            sanitized_out_of_sample_metrics
        )

        windows.append(
            WalkForwardOptimizationWindow(
                window_number=(
                    split.window_number
                ),
                train_start=(
                    split.train_start
                ),
                train_end=(
                    split.train_end
                ),
                test_start=(
                    split.test_start
                ),
                test_end=(
                    split.test_end
                ),
                selected_parameters=(
                    selected_result.parameters
                ),
                selected_score=(
                    selected_result.score
                ),
                in_sample_metrics=(
                    selected_result.metrics
                ),
                out_of_sample_metrics=(
                    sanitized_out_of_sample_metrics
                ),
                robust_score=(
                    selected_result.robust_score
                ),
            )
        )

    summary = summarize_walk_forward_metrics(
        out_of_sample_metrics,
        allow_compounding=(
            not windows_overlap
        ),
    )

    return WalkForwardOptimizationResult(
        symbol=symbol,
        score_metric=score_metric,
        train_size=train_size,
        test_size=test_size,
        step_size=effective_step_size,
        minimum_completed_trades=(
            minimum_completed_trades
        ),
        windows_overlap=windows_overlap,
        compounding_enabled=(
            not windows_overlap
        ),
        robust_score_config=(
            effective_robust_score_config
        ),
        windows=tuple(windows),
        summary=summary,
    )


def walk_forward_optimization_to_dict(
    result: WalkForwardOptimizationResult,
) -> dict[str, object]:
    windows = []

    for window in result.windows:
        windows.append(
            {
                "window_number": (
                    window.window_number
                ),
                "train_start": (
                    window.train_start
                    .isoformat()
                ),
                "train_end": (
                    window.train_end
                    .isoformat()
                ),
                "test_start": (
                    window.test_start
                    .isoformat()
                ),
                "test_end": (
                    window.test_end
                    .isoformat()
                ),
                "selected_parameters": asdict(
                    window.selected_parameters
                ),
                "selected_score": (
                    window.selected_score
                ),
                "selected_robust_score": (
                    asdict(window.robust_score)
                    if window.robust_score
                    is not None
                    else None
                ),
                "in_sample": (
                    window.in_sample_metrics
                ),
                "out_of_sample": (
                    window.out_of_sample_metrics
                ),
            }
        )

    return {
        "symbol": result.symbol,
        "score_metric": result.score_metric,
        "train_size": result.train_size,
        "test_size": result.test_size,
        "step_size": result.step_size,
        "minimum_completed_trades": (
            result.minimum_completed_trades
        ),
        "robust_score_config": (
            asdict(result.robust_score_config)
            if result.robust_score_config
            is not None
            else None
        ),
        "window_count": len(
            result.windows
        ),
        "windows_overlap": (
            result.windows_overlap
        ),
        "compounding_enabled": (
            result.compounding_enabled
        ),
        "summary": result.summary,
        "windows": windows,
    }


def walk_forward_optimization_to_frame(
    result: WalkForwardOptimizationResult,
) -> pd.DataFrame:
    rows = []

    for window in result.windows:
        rows.append(
            {
                "symbol": result.symbol,
                "window_number": (
                    window.window_number
                ),
                "train_start": (
                    window.train_start
                ),
                "train_end": (
                    window.train_end
                ),
                "test_start": (
                    window.test_start
                ),
                "test_end": (
                    window.test_end
                ),
                **asdict(
                    window.selected_parameters
                ),
                "score_metric": (
                    result.score_metric
                ),
                "selected_score": (
                    window.selected_score
                ),
                "selected_robust_score": (
                    window.robust_score.score
                    if window.robust_score
                    is not None
                    else None
                ),
                "selected_robust_excess_return": (
                    window.robust_score
                    .excess_return
                    if window.robust_score
                    is not None
                    else None
                ),
                "selected_robust_trade_shortfall": (
                    window.robust_score
                    .trade_shortfall
                    if window.robust_score
                    is not None
                    else None
                ),
                **{
                    f"in_sample_{name}": value
                    for name, value in (
                        window
                        .in_sample_metrics
                        .items()
                    )
                },
                **{
                    (
                        "out_of_sample_"
                        f"{name}"
                    ): value
                    for name, value in (
                        window
                        .out_of_sample_metrics
                        .items()
                    )
                },
            }
        )

    return pd.DataFrame(rows)