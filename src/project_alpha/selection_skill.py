from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import ceil, isfinite
from numbers import Real
from pathlib import Path

import pandas as pd

from project_alpha.backtest import (
    run_long_only_backtest,
)
from project_alpha.optimization import (
    GridValue,
    _prepare_out_of_sample_frame,
    _select_best_training_result,
    run_parameter_grid_search,
)
from project_alpha.reporting import (
    sanitize_metric_values,
)
from project_alpha.scoring import (
    RobustScoreConfig,
)
from project_alpha.validation import (
    generate_walk_forward_splits,
)


MetricValue = float | int | None


@dataclass(frozen=True)
class SelectionSkillWindow:
    symbol: str
    window_number: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    candidate_count: int
    valid_training_score_count: int
    selected_fast_ema_period: int
    selected_slow_ema_period: int
    selected_trend_ema_period: int
    selected_training_score: float
    selected_oos_total_return: float
    selected_oos_rank: int
    selected_oos_percentile: float
    selected_is_top_quartile: bool
    selected_is_top_half: bool
    best_oos_total_return: float
    mean_oos_total_return: float
    median_oos_total_return: float
    selected_minus_mean_oos_return: float
    selected_minus_median_oos_return: float
    oracle_regret: float
    training_score_oos_return_spearman: float | None


@dataclass(frozen=True)
class SelectionSkillSummary:
    symbol_count: int
    window_count: int
    mean_selected_oos_rank: float
    median_selected_oos_rank: float
    mean_selected_oos_percentile: float
    median_selected_oos_percentile: float
    top_quartile_count: int
    top_quartile_rate: float
    top_half_count: int
    top_half_rate: float
    mean_selected_minus_mean_oos_return: float
    median_selected_minus_mean_oos_return: float
    mean_selected_minus_median_oos_return: float
    median_selected_minus_median_oos_return: float
    mean_oracle_regret: float
    median_oracle_regret: float
    positive_training_oos_correlation_count: int
    valid_training_oos_correlation_count: int
    positive_training_oos_correlation_rate: float | None
    mean_training_score_oos_return_spearman: float | None
    median_training_score_oos_return_spearman: float | None


def _required_numeric(
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


def _metric_as_required_float(
    metrics: Mapping[
        str,
        MetricValue,
    ],
    metric_name: str,
) -> float:
    if metric_name not in metrics:
        raise ValueError(
            f"Metrica '{metric_name}' mancante."
        )

    return _required_numeric(
        metrics[metric_name],
        name=metric_name,
    )


def _spearman_rank_correlation(
    training_scores: Sequence[
        float,
    ],
    oos_returns: Sequence[
        float,
    ],
) -> float | None:
    if len(training_scores) != len(
        oos_returns
    ):
        raise ValueError(
            "training_scores e oos_returns "
            "devono avere la stessa lunghezza."
        )

    if len(training_scores) < 2:
        return None

    x = pd.Series(
        training_scores,
        dtype=float,
    ).rank(
        method="average"
    )

    y = pd.Series(
        oos_returns,
        dtype=float,
    ).rank(
        method="average"
    )

    if (
        x.nunique() <= 1
        or y.nunique() <= 1
    ):
        return None

    correlation = x.corr(
        y,
        method="pearson",
    )

    if pd.isna(correlation):
        return None

    return float(
        correlation
    )


def _oos_rank(
    values: Sequence[
        float,
    ],
    selected_index: int,
) -> int:
    if not values:
        raise ValueError(
            "Servono almeno un valore OOS."
        )

    if (
        selected_index < 0
        or selected_index >= len(
            values
        )
    ):
        raise ValueError(
            "selected_index fuori intervallo."
        )

    selected_value = values[
        selected_index
    ]

    better_count = sum(
        value > selected_value
        for value in values
    )

    return (
        better_count + 1
    )


def _oos_percentile(
    rank: int,
    candidate_count: int,
) -> float:
    if candidate_count <= 0:
        raise ValueError(
            "candidate_count deve essere "
            "positivo."
        )

    if (
        rank <= 0
        or rank > candidate_count
    ):
        raise ValueError(
            "rank deve essere compreso tra "
            "1 e candidate_count."
        )

    if candidate_count == 1:
        return 1.0

    return (
        candidate_count - rank
    ) / (
        candidate_count - 1
    )


def analyze_selection_skill(
    frame: pd.DataFrame,
    grid: Mapping[
        str,
        Sequence[
            GridValue,
        ],
    ],
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
    score_metric: str = "robust_score",
    minimum_completed_trades: int = 0,
    robust_score_config: (
        RobustScoreConfig | None
    ) = None,
) -> tuple[
    SelectionSkillWindow,
    ...,
]:
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
        .str.strip()
        .unique()
    )

    if len(symbols) != 1:
        raise ValueError(
            "Selection Skill deve ricevere "
            "un solo simbolo alla volta."
        )

    symbol = str(
        symbols[0]
    )

    splits = generate_walk_forward_splits(
        frame=frame,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )

    results: list[
        SelectionSkillWindow
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
                score_metric=(
                    score_metric
                ),
                robust_score_config=(
                    robust_score_config
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

        candidate_oos_returns: list[
            float
        ] = []
        valid_training_scores: list[
            float
        ] = []
        correlated_oos_returns: list[
            float
        ] = []

        selected_index: int | None = None

        for index, candidate in enumerate(
            training_results
        ):
            positioned_oos = (
                _prepare_out_of_sample_frame(
                    split.in_sample,
                    split.out_of_sample,
                    candidate.parameters,
                )
            )

            backtest_result = (
                run_long_only_backtest(
                    positioned_oos,
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

            oos_total_return = (
                _metric_as_required_float(
                    metrics,
                    "total_return",
                )
            )

            candidate_oos_returns.append(
                oos_total_return
            )

            if candidate.score is not None:
                score = _required_numeric(
                    candidate.score,
                    name="training_score",
                )

                valid_training_scores.append(
                    score
                )
                correlated_oos_returns.append(
                    oos_total_return
                )

            if (
                candidate.parameters
                == selected_result.parameters
            ):
                selected_index = index

        if selected_index is None:
            raise RuntimeError(
                "La combinazione selezionata "
                "non è presente nei candidati."
            )

        candidate_count = len(
            candidate_oos_returns
        )

        selected_oos_total_return = (
            candidate_oos_returns[
                selected_index
            ]
        )

        selected_rank = _oos_rank(
            candidate_oos_returns,
            selected_index,
        )

        selected_percentile = (
            _oos_percentile(
                selected_rank,
                candidate_count,
            )
        )

        mean_oos_return = float(
            pd.Series(
                candidate_oos_returns,
                dtype=float,
            ).mean()
        )

        median_oos_return = float(
            pd.Series(
                candidate_oos_returns,
                dtype=float,
            ).median()
        )

        best_oos_return = max(
            candidate_oos_returns
        )

        top_quartile_cutoff = max(
            1,
            ceil(
                candidate_count
                * 0.25
            ),
        )

        top_half_cutoff = max(
            1,
            ceil(
                candidate_count
                * 0.50
            ),
        )

        correlation = (
            _spearman_rank_correlation(
                valid_training_scores,
                correlated_oos_returns,
            )
        )

        results.append(
            SelectionSkillWindow(
                symbol=symbol,
                window_number=(
                    split.window_number
                ),
                test_start=(
                    split.test_start
                ),
                test_end=(
                    split.test_end
                ),
                candidate_count=(
                    candidate_count
                ),
                valid_training_score_count=(
                    len(
                        valid_training_scores
                    )
                ),
                selected_fast_ema_period=(
                    selected_result
                    .parameters
                    .fast_ema_period
                ),
                selected_slow_ema_period=(
                    selected_result
                    .parameters
                    .slow_ema_period
                ),
                selected_trend_ema_period=(
                    selected_result
                    .parameters
                    .trend_ema_period
                ),
                selected_training_score=(
                    _required_numeric(
                        selected_result.score,
                        name=(
                            "selected_training_score"
                        ),
                    )
                ),
                selected_oos_total_return=(
                    selected_oos_total_return
                ),
                selected_oos_rank=(
                    selected_rank
                ),
                selected_oos_percentile=(
                    selected_percentile
                ),
                selected_is_top_quartile=(
                    selected_rank
                    <= top_quartile_cutoff
                ),
                selected_is_top_half=(
                    selected_rank
                    <= top_half_cutoff
                ),
                best_oos_total_return=(
                    best_oos_return
                ),
                mean_oos_total_return=(
                    mean_oos_return
                ),
                median_oos_total_return=(
                    median_oos_return
                ),
                selected_minus_mean_oos_return=(
                    selected_oos_total_return
                    - mean_oos_return
                ),
                selected_minus_median_oos_return=(
                    selected_oos_total_return
                    - median_oos_return
                ),
                oracle_regret=(
                    best_oos_return
                    - selected_oos_total_return
                ),
                training_score_oos_return_spearman=(
                    correlation
                ),
            )
        )

    return tuple(
        results
    )


def summarize_selection_skill(
    windows: Sequence[
        SelectionSkillWindow,
    ],
) -> SelectionSkillSummary:
    if not windows:
        raise ValueError(
            "Servono almeno una finestra "
            "Selection Skill."
        )

    frame = selection_skill_to_frame(
        windows
    )

    correlations = pd.to_numeric(
        frame[
            "training_score_oos_return_spearman"
        ],
        errors="coerce",
    ).dropna()

    positive_correlation_count = int(
        (
            correlations > 0.0
        ).sum()
    )

    valid_correlation_count = len(
        correlations
    )

    return SelectionSkillSummary(
        symbol_count=int(
            frame[
                "symbol"
            ].nunique()
        ),
        window_count=len(
            frame
        ),
        mean_selected_oos_rank=float(
            frame[
                "selected_oos_rank"
            ].mean()
        ),
        median_selected_oos_rank=float(
            frame[
                "selected_oos_rank"
            ].median()
        ),
        mean_selected_oos_percentile=float(
            frame[
                "selected_oos_percentile"
            ].mean()
        ),
        median_selected_oos_percentile=float(
            frame[
                "selected_oos_percentile"
            ].median()
        ),
        top_quartile_count=int(
            frame[
                "selected_is_top_quartile"
            ].sum()
        ),
        top_quartile_rate=float(
            frame[
                "selected_is_top_quartile"
            ].mean()
        ),
        top_half_count=int(
            frame[
                "selected_is_top_half"
            ].sum()
        ),
        top_half_rate=float(
            frame[
                "selected_is_top_half"
            ].mean()
        ),
        mean_selected_minus_mean_oos_return=float(
            frame[
                "selected_minus_mean_oos_return"
            ].mean()
        ),
        median_selected_minus_mean_oos_return=float(
            frame[
                "selected_minus_mean_oos_return"
            ].median()
        ),
        mean_selected_minus_median_oos_return=float(
            frame[
                "selected_minus_median_oos_return"
            ].mean()
        ),
        median_selected_minus_median_oos_return=float(
            frame[
                "selected_minus_median_oos_return"
            ].median()
        ),
        mean_oracle_regret=float(
            frame[
                "oracle_regret"
            ].mean()
        ),
        median_oracle_regret=float(
            frame[
                "oracle_regret"
            ].median()
        ),
        positive_training_oos_correlation_count=(
            positive_correlation_count
        ),
        valid_training_oos_correlation_count=(
            valid_correlation_count
        ),
        positive_training_oos_correlation_rate=(
            positive_correlation_count
            / valid_correlation_count
            if valid_correlation_count > 0
            else None
        ),
        mean_training_score_oos_return_spearman=(
            float(
                correlations.mean()
            )
            if valid_correlation_count > 0
            else None
        ),
        median_training_score_oos_return_spearman=(
            float(
                correlations.median()
            )
            if valid_correlation_count > 0
            else None
        ),
    )


def selection_skill_to_frame(
    windows: Sequence[
        SelectionSkillWindow,
    ],
) -> pd.DataFrame:
    return pd.DataFrame(
        asdict(
            window
        )
        for window in windows
    )


def selection_skill_summary_to_frame(
    summary: SelectionSkillSummary,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            asdict(
                summary
            )
        ]
    )


def run_selection_skill_batch(
    frame: pd.DataFrame,
    symbols: Sequence[
        str,
    ],
    grid: Mapping[
        str,
        Sequence[
            GridValue,
        ],
    ],
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
    score_metric: str = "robust_score",
    minimum_completed_trades: int = 0,
    robust_score_config: (
        RobustScoreConfig | None
    ) = None,
) -> tuple[
    SelectionSkillWindow,
    ...,
]:
    if not symbols:
        raise ValueError(
            "Servono almeno un simbolo."
        )

    normalized_symbols = list(
        dict.fromkeys(
            str(symbol)
            .upper()
            .strip()
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

    all_results: list[
        SelectionSkillWindow
    ] = []

    for symbol in normalized_symbols:
        selected = frame.loc[
            frame[
                "symbol"
            ]
            .astype(str)
            .str.upper()
            .eq(
                symbol
            )
        ].copy()

        if selected.empty:
            raise ValueError(
                f"Simbolo {symbol} non trovato."
            )

        symbol_results = (
            analyze_selection_skill(
                selected,
                grid,
                train_size=(
                    train_size
                ),
                test_size=(
                    test_size
                ),
                step_size=(
                    step_size
                ),
                initial_capital=(
                    initial_capital
                ),
                cost_per_side=(
                    cost_per_side
                ),
                score_metric=(
                    score_metric
                ),
                minimum_completed_trades=(
                    minimum_completed_trades
                ),
                robust_score_config=(
                    robust_score_config
                ),
            )
        )

        all_results.extend(
            symbol_results
        )

    return tuple(
        all_results
    )


def run_selection_skill_from_csv(
    market_path: Path,
    symbols: Sequence[
        str,
    ],
    grid: Mapping[
        str,
        Sequence[
            GridValue,
        ],
    ],
    output_dir: Path,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
    score_metric: str = "robust_score",
    minimum_completed_trades: int = 0,
    robust_score_config: (
        RobustScoreConfig | None
    ) = None,
) -> dict[str, object]:
    if not market_path.exists():
        raise FileNotFoundError(
            "Market dataset non trovato: "
            f"{market_path}"
        )

    frame = pd.read_csv(
        market_path,
        parse_dates=[
            "timestamp",
        ],
    )

    windows = run_selection_skill_batch(
        frame,
        symbols,
        grid,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        initial_capital=initial_capital,
        cost_per_side=cost_per_side,
        score_metric=score_metric,
        minimum_completed_trades=(
            minimum_completed_trades
        ),
        robust_score_config=(
            robust_score_config
        ),
    )

    summary = summarize_selection_skill(
        windows
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows_path = (
        output_dir
        / "selection_skill_windows.csv"
    )

    summary_path = (
        output_dir
        / "selection_skill_summary.csv"
    )

    selection_skill_to_frame(
        windows
    ).to_csv(
        windows_path,
        index=False,
    )

    selection_skill_summary_to_frame(
        summary
    ).to_csv(
        summary_path,
        index=False,
    )

    return {
        "windows": windows,
        "summary": summary,
        "windows_path": (
            windows_path
        ),
        "summary_path": (
            summary_path
        ),
    }
