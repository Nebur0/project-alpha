from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from numbers import Real
from pathlib import Path

import pandas as pd

from project_alpha.backtest import run_long_only_backtest
from project_alpha.config import StrategyParameters
from project_alpha.optimization import (
    GridValue,
    _prepare_out_of_sample_frame,
    _select_best_training_result,
    run_parameter_grid_search,
)
from project_alpha.reporting import sanitize_metric_values
from project_alpha.scoring import RobustScoreConfig
from project_alpha.validation import generate_walk_forward_splits

MetricValue = float | int | None

EXACT_RETURN_TOLERANCE = 1e-12
ONE_PERCENTAGE_POINT = 0.01
TWO_PERCENTAGE_POINTS = 0.02

ANALYZED_PARAMETER_COLUMNS = (
    "fast_ema_period",
    "slow_ema_period",
    "trend_ema_period",
)


@dataclass(frozen=True)
class GridFlatnessWindow:
    symbol: str
    window_number: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    candidate_count: int
    selected_fast_ema_period: int
    selected_slow_ema_period: int
    selected_trend_ema_period: int
    selected_oos_total_return: float
    best_oos_total_return: float
    mean_oos_total_return: float
    median_oos_total_return: float
    std_oos_total_return: float
    iqr_oos_total_return: float
    range_oos_total_return: float
    unique_oos_return_count: int
    selected_equivalent_count: int
    selected_equivalent_rate: float
    within_1pp_best_count: int
    within_1pp_best_rate: float
    within_2pp_best_count: int
    within_2pp_best_rate: float
    selected_regret: float
    selected_within_1pp_best: bool
    selected_within_2pp_best: bool
    selected_neighbor_count: int
    selected_neighbor_mean_return: float | None
    selected_neighbor_best_return: float | None
    selected_neighbor_worst_return: float | None
    selected_neighbor_return_range: float | None
    selected_minus_neighbor_mean_return: float | None
    selected_mean_abs_neighbor_delta: float | None


@dataclass(frozen=True)
class GridFlatnessSummary:
    symbol_count: int
    window_count: int
    mean_unique_oos_return_count: float
    median_unique_oos_return_count: float
    mean_selected_equivalent_rate: float
    median_selected_equivalent_rate: float
    mean_within_1pp_best_rate: float
    median_within_1pp_best_rate: float
    mean_within_2pp_best_rate: float
    median_within_2pp_best_rate: float
    selected_within_1pp_best_count: int
    selected_within_1pp_best_rate: float
    selected_within_2pp_best_count: int
    selected_within_2pp_best_rate: float
    mean_std_oos_total_return: float
    median_std_oos_total_return: float
    mean_iqr_oos_total_return: float
    median_iqr_oos_total_return: float
    mean_range_oos_total_return: float
    median_range_oos_total_return: float
    mean_selected_regret: float
    median_selected_regret: float
    mean_selected_minus_neighbor_mean_return: float | None
    median_selected_minus_neighbor_mean_return: float | None
    mean_selected_mean_abs_neighbor_delta: float | None
    median_selected_mean_abs_neighbor_delta: float | None


def _required_numeric(value: object, *, name: str) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} deve essere numerico.")

    numeric_value = float(value)

    if not isfinite(numeric_value):
        raise ValueError(f"{name} deve essere finito.")

    return numeric_value


def _required_metric(
    metrics: Mapping[str, MetricValue],
    metric_name: str,
) -> float:
    if metric_name not in metrics:
        raise ValueError(f"Metrica '{metric_name}' mancante.")

    return _required_numeric(
        metrics[metric_name],
        name=metric_name,
    )


def _normalized_grid_axes(
    grid: Mapping[str, Sequence[GridValue]],
) -> dict[str, tuple[float, ...]]:
    missing = [
        name
        for name in ANALYZED_PARAMETER_COLUMNS
        if name not in grid
    ]

    if missing:
        raise ValueError(
            "La grid deve contenere i parametri "
            f"{list(ANALYZED_PARAMETER_COLUMNS)}. "
            f"Mancanti: {missing}."
        )

    axes: dict[str, tuple[float, ...]] = {}

    for name in ANALYZED_PARAMETER_COLUMNS:
        values = tuple(
            sorted(
                {
                    _required_numeric(
                        value,
                        name=name,
                    )
                    for value in grid[name]
                }
            )
        )

        if not values:
            raise ValueError(f"La grid di '{name}' è vuota.")

        axes[name] = values

    return axes


def _parameter_value(
    parameters: StrategyParameters,
    parameter_name: str,
) -> float:
    return _required_numeric(
        getattr(parameters, parameter_name),
        name=parameter_name,
    )


def _are_immediate_neighbors(
    first: StrategyParameters,
    second: StrategyParameters,
    axes: Mapping[str, Sequence[float]],
) -> bool:
    differing_parameter_count = 0

    for parameter_name in ANALYZED_PARAMETER_COLUMNS:
        first_value = _parameter_value(first, parameter_name)
        second_value = _parameter_value(second, parameter_name)

        if first_value == second_value:
            continue

        axis = tuple(axes[parameter_name])

        try:
            first_index = axis.index(first_value)
            second_index = axis.index(second_value)
        except ValueError as exc:
            raise ValueError(
                "Parametro selezionato fuori dalla grid: "
                f"{parameter_name}."
            ) from exc

        if abs(first_index - second_index) != 1:
            return False

        differing_parameter_count += 1

        if differing_parameter_count > 1:
            return False

    return differing_parameter_count == 1


def _unique_return_count(values: Sequence[float]) -> int:
    representatives: list[float] = []

    for value in sorted(values):
        if not any(
            isclose(
                value,
                representative,
                rel_tol=0.0,
                abs_tol=EXACT_RETURN_TOLERANCE,
            )
            for representative in representatives
        ):
            representatives.append(value)

    return len(representatives)


def analyze_grid_flatness(
    frame: pd.DataFrame,
    grid: Mapping[str, Sequence[GridValue]],
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
    score_metric: str = "robust_score",
    minimum_completed_trades: int = 0,
    robust_score_config: RobustScoreConfig | None = None,
) -> tuple[GridFlatnessWindow, ...]:
    if frame.empty:
        raise ValueError("Il dataset è vuoto.")

    if "symbol" not in frame.columns:
        raise ValueError("La colonna 'symbol' è obbligatoria.")

    symbols = (
        frame["symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
        .unique()
    )

    if len(symbols) != 1:
        raise ValueError(
            "Grid Flatness deve ricevere un solo simbolo alla volta."
        )

    symbol = str(symbols[0])
    axes = _normalized_grid_axes(grid)

    splits = generate_walk_forward_splits(
        frame=frame,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )

    windows: list[GridFlatnessWindow] = []

    for split in splits:
        training_results = run_parameter_grid_search(
            split.in_sample,
            grid,
            initial_capital=initial_capital,
            cost_per_side=cost_per_side,
            score_metric=score_metric,
            robust_score_config=robust_score_config,
        )

        selected_result = _select_best_training_result(
            training_results,
            minimum_completed_trades=minimum_completed_trades,
        )

        candidate_rows: list[tuple[StrategyParameters, float]] = []
        selected_return: float | None = None

        for candidate in training_results:
            positioned_oos = _prepare_out_of_sample_frame(
                split.in_sample,
                split.out_of_sample,
                candidate.parameters,
            )

            backtest_result = run_long_only_backtest(
                positioned_oos,
                initial_capital=initial_capital,
                cost_per_side=cost_per_side,
            )

            metrics = sanitize_metric_values(backtest_result.metrics)

            oos_total_return = _required_metric(
                metrics,
                "total_return",
            )

            candidate_rows.append(
                (
                    candidate.parameters,
                    oos_total_return,
                )
            )

            if candidate.parameters == selected_result.parameters:
                selected_return = oos_total_return

        if selected_return is None:
            raise RuntimeError(
                "La combinazione selezionata non è presente nella grid OOS."
            )

        returns = [row[1] for row in candidate_rows]
        return_series = pd.Series(returns, dtype=float)
        candidate_count = len(returns)

        best_return = max(returns)
        mean_return = float(return_series.mean())
        median_return = float(return_series.median())
        std_return = float(return_series.std(ddof=0))
        q1 = float(return_series.quantile(0.25))
        q3 = float(return_series.quantile(0.75))
        iqr_return = q3 - q1
        return_range = max(returns) - min(returns)

        equivalent_count = sum(
            isclose(
                value,
                selected_return,
                rel_tol=0.0,
                abs_tol=EXACT_RETURN_TOLERANCE,
            )
            for value in returns
        )

        within_1pp_count = sum(
            (best_return - value)
            <= (
                ONE_PERCENTAGE_POINT
                + EXACT_RETURN_TOLERANCE
            )
            for value in returns
        )

        within_2pp_count = sum(
            (best_return - value)
            <= (
                TWO_PERCENTAGE_POINTS
                + EXACT_RETURN_TOLERANCE
            )
            for value in returns
        )

        selected_regret = best_return - selected_return

        neighbor_returns = [
            candidate_return
            for (
                candidate_parameters,
                candidate_return,
            ) in candidate_rows
            if _are_immediate_neighbors(
                selected_result.parameters,
                candidate_parameters,
                axes,
            )
        ]

        neighbor_count = len(neighbor_returns)

        if neighbor_returns:
            neighbor_series = pd.Series(
                neighbor_returns,
                dtype=float,
            )
            neighbor_mean = float(neighbor_series.mean())
            neighbor_best = max(neighbor_returns)
            neighbor_worst = min(neighbor_returns)
            neighbor_range = neighbor_best - neighbor_worst
            selected_minus_neighbor_mean = (
                selected_return - neighbor_mean
            )
            mean_abs_neighbor_delta = float(
                sum(
                    abs(selected_return - value)
                    for value in neighbor_returns
                )
                / neighbor_count
            )
        else:
            neighbor_mean = None
            neighbor_best = None
            neighbor_worst = None
            neighbor_range = None
            selected_minus_neighbor_mean = None
            mean_abs_neighbor_delta = None

        windows.append(
            GridFlatnessWindow(
                symbol=symbol,
                window_number=split.window_number,
                test_start=split.test_start,
                test_end=split.test_end,
                candidate_count=candidate_count,
                selected_fast_ema_period=(
                    selected_result.parameters.fast_ema_period
                ),
                selected_slow_ema_period=(
                    selected_result.parameters.slow_ema_period
                ),
                selected_trend_ema_period=(
                    selected_result.parameters.trend_ema_period
                ),
                selected_oos_total_return=selected_return,
                best_oos_total_return=best_return,
                mean_oos_total_return=mean_return,
                median_oos_total_return=median_return,
                std_oos_total_return=std_return,
                iqr_oos_total_return=iqr_return,
                range_oos_total_return=return_range,
                unique_oos_return_count=_unique_return_count(returns),
                selected_equivalent_count=equivalent_count,
                selected_equivalent_rate=(
                    equivalent_count / candidate_count
                ),
                within_1pp_best_count=within_1pp_count,
                within_1pp_best_rate=(
                    within_1pp_count / candidate_count
                ),
                within_2pp_best_count=within_2pp_count,
                within_2pp_best_rate=(
                    within_2pp_count / candidate_count
                ),
                selected_regret=selected_regret,
                selected_within_1pp_best=(
                    selected_regret
                    <= (
                        ONE_PERCENTAGE_POINT
                        + EXACT_RETURN_TOLERANCE
                    )
                ),
                selected_within_2pp_best=(
                    selected_regret
                    <= (
                        TWO_PERCENTAGE_POINTS
                        + EXACT_RETURN_TOLERANCE
                    )
                ),
                selected_neighbor_count=neighbor_count,
                selected_neighbor_mean_return=neighbor_mean,
                selected_neighbor_best_return=neighbor_best,
                selected_neighbor_worst_return=neighbor_worst,
                selected_neighbor_return_range=neighbor_range,
                selected_minus_neighbor_mean_return=(
                    selected_minus_neighbor_mean
                ),
                selected_mean_abs_neighbor_delta=(
                    mean_abs_neighbor_delta
                ),
            )
        )

    return tuple(windows)


def grid_flatness_to_frame(
    windows: Sequence[GridFlatnessWindow],
) -> pd.DataFrame:
    return pd.DataFrame(
        asdict(window)
        for window in windows
    )


def summarize_grid_flatness(
    windows: Sequence[GridFlatnessWindow],
) -> GridFlatnessSummary:
    if not windows:
        raise ValueError(
            "Servono almeno una finestra Grid Flatness."
        )

    frame = grid_flatness_to_frame(windows)

    neighbor_mean_delta = pd.to_numeric(
        frame["selected_minus_neighbor_mean_return"],
        errors="coerce",
    ).dropna()

    neighbor_abs_delta = pd.to_numeric(
        frame["selected_mean_abs_neighbor_delta"],
        errors="coerce",
    ).dropna()

    selected_within_1pp_count = int(
        frame["selected_within_1pp_best"].sum()
    )
    selected_within_2pp_count = int(
        frame["selected_within_2pp_best"].sum()
    )

    return GridFlatnessSummary(
        symbol_count=int(frame["symbol"].nunique()),
        window_count=len(frame),
        mean_unique_oos_return_count=float(
            frame["unique_oos_return_count"].mean()
        ),
        median_unique_oos_return_count=float(
            frame["unique_oos_return_count"].median()
        ),
        mean_selected_equivalent_rate=float(
            frame["selected_equivalent_rate"].mean()
        ),
        median_selected_equivalent_rate=float(
            frame["selected_equivalent_rate"].median()
        ),
        mean_within_1pp_best_rate=float(
            frame["within_1pp_best_rate"].mean()
        ),
        median_within_1pp_best_rate=float(
            frame["within_1pp_best_rate"].median()
        ),
        mean_within_2pp_best_rate=float(
            frame["within_2pp_best_rate"].mean()
        ),
        median_within_2pp_best_rate=float(
            frame["within_2pp_best_rate"].median()
        ),
        selected_within_1pp_best_count=(
            selected_within_1pp_count
        ),
        selected_within_1pp_best_rate=(
            selected_within_1pp_count / len(frame)
        ),
        selected_within_2pp_best_count=(
            selected_within_2pp_count
        ),
        selected_within_2pp_best_rate=(
            selected_within_2pp_count / len(frame)
        ),
        mean_std_oos_total_return=float(
            frame["std_oos_total_return"].mean()
        ),
        median_std_oos_total_return=float(
            frame["std_oos_total_return"].median()
        ),
        mean_iqr_oos_total_return=float(
            frame["iqr_oos_total_return"].mean()
        ),
        median_iqr_oos_total_return=float(
            frame["iqr_oos_total_return"].median()
        ),
        mean_range_oos_total_return=float(
            frame["range_oos_total_return"].mean()
        ),
        median_range_oos_total_return=float(
            frame["range_oos_total_return"].median()
        ),
        mean_selected_regret=float(
            frame["selected_regret"].mean()
        ),
        median_selected_regret=float(
            frame["selected_regret"].median()
        ),
        mean_selected_minus_neighbor_mean_return=(
            float(neighbor_mean_delta.mean())
            if not neighbor_mean_delta.empty
            else None
        ),
        median_selected_minus_neighbor_mean_return=(
            float(neighbor_mean_delta.median())
            if not neighbor_mean_delta.empty
            else None
        ),
        mean_selected_mean_abs_neighbor_delta=(
            float(neighbor_abs_delta.mean())
            if not neighbor_abs_delta.empty
            else None
        ),
        median_selected_mean_abs_neighbor_delta=(
            float(neighbor_abs_delta.median())
            if not neighbor_abs_delta.empty
            else None
        ),
    )


def grid_flatness_summary_to_frame(
    summary: GridFlatnessSummary,
) -> pd.DataFrame:
    return pd.DataFrame([asdict(summary)])


def run_grid_flatness_batch(
    frame: pd.DataFrame,
    symbols: Sequence[str],
    grid: Mapping[str, Sequence[GridValue]],
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
    score_metric: str = "robust_score",
    minimum_completed_trades: int = 0,
    robust_score_config: RobustScoreConfig | None = None,
) -> tuple[GridFlatnessWindow, ...]:
    if not symbols:
        raise ValueError("Servono almeno un simbolo.")

    normalized_symbols = list(
        dict.fromkeys(
            str(symbol).upper().strip()
            for symbol in symbols
        )
    )

    if any(
        not symbol
        for symbol in normalized_symbols
    ):
        raise ValueError(
            "I simboli non possono essere vuoti."
        )

    windows: list[GridFlatnessWindow] = []

    for symbol in normalized_symbols:
        selected = frame.loc[
            frame["symbol"]
            .astype(str)
            .str.upper()
            .eq(symbol)
        ].copy()

        if selected.empty:
            raise ValueError(
                f"Simbolo {symbol} non trovato."
            )

        windows.extend(
            analyze_grid_flatness(
                selected,
                grid,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
                initial_capital=initial_capital,
                cost_per_side=cost_per_side,
                score_metric=score_metric,
                minimum_completed_trades=minimum_completed_trades,
                robust_score_config=robust_score_config,
            )
        )

    return tuple(windows)


def run_grid_flatness_from_csv(
    market_path: Path,
    symbols: Sequence[str],
    grid: Mapping[str, Sequence[GridValue]],
    output_dir: Path,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
    score_metric: str = "robust_score",
    minimum_completed_trades: int = 0,
    robust_score_config: RobustScoreConfig | None = None,
) -> dict[str, object]:
    if not market_path.exists():
        raise FileNotFoundError(
            "Market dataset non trovato: "
            f"{market_path}"
        )

    frame = pd.read_csv(
        market_path,
        parse_dates=["timestamp"],
    )

    windows = run_grid_flatness_batch(
        frame,
        symbols,
        grid,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        initial_capital=initial_capital,
        cost_per_side=cost_per_side,
        score_metric=score_metric,
        minimum_completed_trades=minimum_completed_trades,
        robust_score_config=robust_score_config,
    )

    summary = summarize_grid_flatness(windows)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows_path = (
        output_dir
        / "grid_flatness_windows.csv"
    )
    summary_path = (
        output_dir
        / "grid_flatness_summary.csv"
    )

    grid_flatness_to_frame(windows).to_csv(
        windows_path,
        index=False,
    )
    grid_flatness_summary_to_frame(summary).to_csv(
        summary_path,
        index=False,
    )

    return {
        "windows": windows,
        "summary": summary,
        "windows_path": windows_path,
        "summary_path": summary_path,
    }
