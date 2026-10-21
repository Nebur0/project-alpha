from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
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
    RobustScoreV4Config,
    RobustScoreV4Result,
    RobustScoreV5Config,
    RobustScoreV5QualityResult,
    RobustScoreV5Result,
    calculate_robust_score,
    calculate_robust_score_v4,
    calculate_robust_score_v5,
    calculate_robust_score_v5_quality,
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
ROBUST_SCORE_V4_METRIC = "robust_score_v4"
ROBUST_SCORE_V5_METRIC = "robust_score_v5"


@dataclass(frozen=True)
class GridSearchResult:
    parameters: StrategyParameters
    metrics: dict[str, MetricValue]
    score_metric: str
    score: float | None
    robust_score: RobustScoreResult | None = None
    robust_score_v4: RobustScoreV4Result | None = None
    robust_score_v5_quality: RobustScoreV5QualityResult | None = None
    robust_score_v5: RobustScoreV5Result | None = None


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
    robust_score_v4: RobustScoreV4Result | None = None
    robust_score_v5_quality: RobustScoreV5QualityResult | None = None
    robust_score_v5: RobustScoreV5Result | None = None


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
    robust_score_v4_config: RobustScoreV4Config | None = None
    robust_score_v5_config: RobustScoreV5Config | None = None


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
    if score_metric in {
        ROBUST_SCORE_METRIC,
        ROBUST_SCORE_V4_METRIC,
    }:
        return (
            robust_score_config
            or RobustScoreConfig()
        )

    if robust_score_config is not None:
        raise ValueError(
            "robust_score_config può essere "
            "usato solo con score_metric="
            "'robust_score' o 'robust_score_v4'."
        )

    return None


def _resolve_robust_score_v4_config(
    score_metric: str,
    robust_score_v4_config: RobustScoreV4Config | None,
) -> RobustScoreV4Config | None:
    if score_metric == ROBUST_SCORE_V4_METRIC:
        return (
            robust_score_v4_config
            or RobustScoreV4Config()
        )

    if robust_score_v4_config is not None:
        raise ValueError(
            "robust_score_v4_config può essere "
            "usato solo con "
            "score_metric='robust_score_v4'."
        )

    return None


def _resolve_robust_score_v5_config(
    score_metric: str,
    robust_score_v5_config: RobustScoreV5Config | None,
) -> RobustScoreV5Config | None:
    if score_metric == ROBUST_SCORE_V5_METRIC:
        return (
            robust_score_v5_config
            or RobustScoreV5Config()
        )

    if robust_score_v5_config is not None:
        raise ValueError(
            "robust_score_v5_config può essere "
            "usato solo con "
            "score_metric='robust_score_v5'."
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


def _split_temporal_folds(
    frame: pd.DataFrame,
    *,
    fold_count: int,
) -> tuple[pd.DataFrame, ...]:
    if len(frame) < fold_count:
        raise ValueError(
            "Il training set deve contenere "
            "almeno una riga per temporal fold."
        )

    ordered = (
        frame
        .sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    base_size = len(ordered) // fold_count
    remainder = len(ordered) % fold_count

    folds: list[pd.DataFrame] = []
    start = 0

    for fold_index in range(fold_count):
        fold_size = (
            base_size
            + (
                1
                if fold_index < remainder
                else 0
            )
        )
        stop = start + fold_size

        folds.append(
            ordered.iloc[
                start:stop
            ].copy()
        )

        start = stop

    return tuple(folds)


def _scaled_fold_robust_score_config(
    config: RobustScoreConfig,
    *,
    fold_size: int,
    training_size: int,
) -> RobustScoreConfig:
    scaled_target = max(
        1,
        int(
            round(
                config.target_completed_trades
                * fold_size
                / training_size
            )
        ),
    )

    return replace(
        config,
        target_completed_trades=(
            scaled_target
        ),
    )


def _calculate_temporal_fold_scores(
    enriched_training_frame: pd.DataFrame,
    parameters: StrategyParameters,
    *,
    initial_capital: float,
    cost_per_side: float,
    robust_score_config: RobustScoreConfig,
    robust_score_v4_config: RobustScoreV4Config,
) -> tuple[float, ...] | None:
    folds = _split_temporal_folds(
        enriched_training_frame,
        fold_count=(
            robust_score_v4_config
            .temporal_fold_count
        ),
    )

    training_size = len(
        enriched_training_frame
    )

    fold_scores: list[float] = []

    for fold in folds:
        positioned_fold = (
            add_baseline_position(
                fold,
                parameters,
            )
        )

        fold_backtest = (
            run_long_only_backtest(
                positioned_fold,
                initial_capital=(
                    initial_capital
                ),
                cost_per_side=(
                    cost_per_side
                ),
            )
        )

        fold_metrics = (
            sanitize_metric_values(
                fold_backtest.metrics
            )
        )

        fold_config = (
            _scaled_fold_robust_score_config(
                robust_score_config,
                fold_size=len(fold),
                training_size=training_size,
            )
        )

        try:
            fold_score = (
                calculate_robust_score(
                    fold_metrics,
                    fold_config,
                )
            )
        except ValueError:
            return None

        fold_scores.append(
            fold_score.score
        )

    return tuple(fold_scores)


def _scaled_fold_robust_score_v5_config(
    config: RobustScoreV5Config,
    *,
    fold_size: int,
    training_size: int,
) -> RobustScoreV5Config:
    scaled_target = max(
        1,
        int(
            round(
                config.target_completed_trades
                * fold_size
                / training_size
            )
        ),
    )

    return replace(
        config,
        target_completed_trades=(
            scaled_target
        ),
    )


def _calculate_temporal_fold_scores_v5(
    enriched_training_frame: pd.DataFrame,
    parameters: StrategyParameters,
    *,
    initial_capital: float,
    cost_per_side: float,
    robust_score_v5_config: RobustScoreV5Config,
) -> tuple[float, ...] | None:
    folds = _split_temporal_folds(
        enriched_training_frame,
        fold_count=(
            robust_score_v5_config
            .temporal_fold_count
        ),
    )

    training_size = len(
        enriched_training_frame
    )

    fold_scores: list[float] = []

    for fold in folds:
        positioned_fold = (
            add_baseline_position(
                fold,
                parameters,
            )
        )

        fold_backtest = (
            run_long_only_backtest(
                positioned_fold,
                initial_capital=(
                    initial_capital
                ),
                cost_per_side=(
                    cost_per_side
                ),
            )
        )

        fold_metrics = (
            sanitize_metric_values(
                fold_backtest.metrics
            )
        )

        fold_config = (
            _scaled_fold_robust_score_v5_config(
                robust_score_v5_config,
                fold_size=len(fold),
                training_size=training_size,
            )
        )

        try:
            fold_quality = (
                calculate_robust_score_v5_quality(
                    fold_metrics,
                    fold_config,
                )
            )
        except ValueError:
            return None

        fold_scores.append(
            fold_quality.score
        )

    return tuple(fold_scores)


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
    robust_score_v4_config: (
        RobustScoreV4Config | None
    ) = None,
    robust_score_v5_config: (
        RobustScoreV5Config | None
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

    effective_robust_score_v4_config = (
        _resolve_robust_score_v4_config(
            score_metric,
            robust_score_v4_config,
        )
    )

    effective_robust_score_v5_config = (
        _resolve_robust_score_v5_config(
            score_metric,
            robust_score_v5_config,
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

        robust_score_v4 = None
        robust_score_v5_quality = None
        robust_score_v5 = None

        if score_metric == ROBUST_SCORE_V4_METRIC:
            if (
                effective_robust_score_config
                is None
                or effective_robust_score_v4_config
                is None
            ):
                raise RuntimeError(
                    "Configurazione robust score v4 "
                    "non inizializzata."
                )

            try:
                robust_score = (
                    calculate_robust_score(
                        metrics,
                        effective_robust_score_config,
                    )
                )
            except ValueError:
                score = None
                robust_score = None
            else:
                temporal_fold_scores = (
                    _calculate_temporal_fold_scores(
                        enriched,
                        parameters,
                        initial_capital=(
                            initial_capital
                        ),
                        cost_per_side=(
                            cost_per_side
                        ),
                        robust_score_config=(
                            effective_robust_score_config
                        ),
                        robust_score_v4_config=(
                            effective_robust_score_v4_config
                        ),
                    )
                )

                if temporal_fold_scores is None:
                    score = None
                else:
                    robust_score_v4 = (
                        calculate_robust_score_v4(
                            robust_score.score,
                            temporal_fold_scores,
                            effective_robust_score_v4_config,
                        )
                    )
                    score = (
                        robust_score_v4.score
                    )
        elif score_metric == ROBUST_SCORE_V5_METRIC:
            robust_score = None

            if effective_robust_score_v5_config is None:
                raise RuntimeError(
                    "Configurazione robust score v5 "
                    "non inizializzata."
                )

            try:
                robust_score_v5_quality = (
                    calculate_robust_score_v5_quality(
                        metrics,
                        effective_robust_score_v5_config,
                    )
                )
            except ValueError:
                score = None
            else:
                temporal_fold_scores = (
                    _calculate_temporal_fold_scores_v5(
                        enriched,
                        parameters,
                        initial_capital=(
                            initial_capital
                        ),
                        cost_per_side=(
                            cost_per_side
                        ),
                        robust_score_v5_config=(
                            effective_robust_score_v5_config
                        ),
                    )
                )

                if temporal_fold_scores is None:
                    score = None
                else:
                    robust_score_v5 = (
                        calculate_robust_score_v5(
                            robust_score_v5_quality.score,
                            temporal_fold_scores,
                            effective_robust_score_v5_config,
                        )
                    )
                    score = (
                        robust_score_v5.score
                    )
        else:
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
                robust_score_v4=(
                    robust_score_v4
                ),
                robust_score_v5_quality=(
                    robust_score_v5_quality
                ),
                robust_score_v5=(
                    robust_score_v5
                ),
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

        robust_v4_values: dict[
            str,
            object,
        ] = {}

        if result.robust_score_v4 is not None:
            robust_v4_values = {
                "robust_v4_score": (
                    result.robust_score_v4.score
                ),
                "robust_v4_base_score": (
                    result.robust_score_v4
                    .base_score
                ),
                "robust_v4_temporal_downside_score": (
                    result.robust_score_v4
                    .temporal_downside_score
                ),
                "robust_v4_downside_fold_count": (
                    result.robust_score_v4
                    .downside_fold_count
                ),
                **{
                    (
                        "robust_v4_temporal_fold_"
                        f"{fold_number}_score"
                    ): fold_score
                    for (
                        fold_number,
                        fold_score,
                    ) in enumerate(
                        result.robust_score_v4
                        .temporal_fold_scores,
                        start=1,
                    )
                },
            }

        robust_v5_values: dict[
            str,
            object,
        ] = {}

        if (
            result.robust_score_v5 is not None
            and result.robust_score_v5_quality
            is not None
        ):
            robust_v5_values = {
                "robust_v5_score": (
                    result.robust_score_v5.score
                ),
                "robust_v5_base_score": (
                    result.robust_score_v5
                    .base_score
                ),
                "robust_v5_base_sharpe": (
                    result.robust_score_v5_quality
                    .sharpe
                ),
                "robust_v5_base_cagr": (
                    result.robust_score_v5_quality
                    .cagr
                ),
                "robust_v5_base_max_drawdown": (
                    result.robust_score_v5_quality
                    .max_drawdown
                ),
                "robust_v5_base_completed_trades": (
                    result.robust_score_v5_quality
                    .completed_trades
                ),
                "robust_v5_base_trade_shortfall": (
                    result.robust_score_v5_quality
                    .trade_shortfall
                ),
                "robust_v5_temporal_downside_score": (
                    result.robust_score_v5
                    .temporal_downside_score
                ),
                "robust_v5_downside_fold_count": (
                    result.robust_score_v5
                    .downside_fold_count
                ),
                **{
                    (
                        "robust_v5_temporal_fold_"
                        f"{fold_number}_score"
                    ): fold_score
                    for (
                        fold_number,
                        fold_score,
                    ) in enumerate(
                        result.robust_score_v5
                        .temporal_fold_scores,
                        start=1,
                    )
                },
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
                **robust_v4_values,
                **robust_v5_values,
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
    robust_score_v4_config: (
        RobustScoreV4Config | None
    ) = None,
    robust_score_v5_config: (
        RobustScoreV5Config | None
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

    effective_robust_score_v4_config = (
        _resolve_robust_score_v4_config(
            score_metric,
            robust_score_v4_config,
        )
    )

    effective_robust_score_v5_config = (
        _resolve_robust_score_v5_config(
            score_metric,
            robust_score_v5_config,
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
                robust_score_v4_config=(
                    effective_robust_score_v4_config
                ),
                robust_score_v5_config=(
                    effective_robust_score_v5_config
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
                robust_score_v4=(
                    selected_result
                    .robust_score_v4
                ),
                robust_score_v5_quality=(
                    selected_result
                    .robust_score_v5_quality
                ),
                robust_score_v5=(
                    selected_result
                    .robust_score_v5
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
        robust_score_v4_config=(
            effective_robust_score_v4_config
        ),
        robust_score_v5_config=(
            effective_robust_score_v5_config
        ),
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
                "selected_robust_score_v4": (
                    asdict(
                        window.robust_score_v4
                    )
                    if window.robust_score_v4
                    is not None
                    else None
                ),
                "selected_robust_score_v5_quality": (
                    asdict(
                        window.robust_score_v5_quality
                    )
                    if window.robust_score_v5_quality
                    is not None
                    else None
                ),
                "selected_robust_score_v5": (
                    asdict(
                        window.robust_score_v5
                    )
                    if window.robust_score_v5
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
        "robust_score_v4_config": (
            asdict(
                result.robust_score_v4_config
            )
            if result.robust_score_v4_config
            is not None
            else None
        ),
        "robust_score_v5_config": (
            asdict(
                result.robust_score_v5_config
            )
            if result.robust_score_v5_config
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
                "selected_robust_v4_score": (
                    window.robust_score_v4.score
                    if window.robust_score_v4
                    is not None
                    else None
                ),
                "selected_robust_v4_base_score": (
                    window.robust_score_v4
                    .base_score
                    if window.robust_score_v4
                    is not None
                    else None
                ),
                "selected_robust_v4_temporal_downside_score": (
                    window.robust_score_v4
                    .temporal_downside_score
                    if window.robust_score_v4
                    is not None
                    else None
                ),
                "selected_robust_v4_downside_fold_count": (
                    window.robust_score_v4
                    .downside_fold_count
                    if window.robust_score_v4
                    is not None
                    else None
                ),
                **(
                    {
                        (
                            "selected_robust_v4_temporal_fold_"
                            f"{fold_number}_score"
                        ): fold_score
                        for (
                            fold_number,
                            fold_score,
                        ) in enumerate(
                            window.robust_score_v4
                            .temporal_fold_scores,
                            start=1,
                        )
                    }
                    if window.robust_score_v4
                    is not None
                    else {}
                ),
                "selected_robust_v5_score": (
                    window.robust_score_v5.score
                    if window.robust_score_v5
                    is not None
                    else None
                ),
                "selected_robust_v5_base_score": (
                    window.robust_score_v5
                    .base_score
                    if window.robust_score_v5
                    is not None
                    else None
                ),
                "selected_robust_v5_base_sharpe": (
                    window.robust_score_v5_quality
                    .sharpe
                    if window.robust_score_v5_quality
                    is not None
                    else None
                ),
                "selected_robust_v5_base_cagr": (
                    window.robust_score_v5_quality
                    .cagr
                    if window.robust_score_v5_quality
                    is not None
                    else None
                ),
                "selected_robust_v5_base_max_drawdown": (
                    window.robust_score_v5_quality
                    .max_drawdown
                    if window.robust_score_v5_quality
                    is not None
                    else None
                ),
                "selected_robust_v5_base_completed_trades": (
                    window.robust_score_v5_quality
                    .completed_trades
                    if window.robust_score_v5_quality
                    is not None
                    else None
                ),
                "selected_robust_v5_base_trade_shortfall": (
                    window.robust_score_v5_quality
                    .trade_shortfall
                    if window.robust_score_v5_quality
                    is not None
                    else None
                ),
                "selected_robust_v5_temporal_downside_score": (
                    window.robust_score_v5
                    .temporal_downside_score
                    if window.robust_score_v5
                    is not None
                    else None
                ),
                "selected_robust_v5_downside_fold_count": (
                    window.robust_score_v5
                    .downside_fold_count
                    if window.robust_score_v5
                    is not None
                    else None
                ),
                **(
                    {
                        (
                            "selected_robust_v5_temporal_fold_"
                            f"{fold_number}_score"
                        ): fold_score
                        for (
                            fold_number,
                            fold_score,
                        ) in enumerate(
                            window.robust_score_v5
                            .temporal_fold_scores,
                            start=1,
                        )
                    }
                    if window.robust_score_v5
                    is not None
                    else {}
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