from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import isfinite, log
from numbers import Real
from pathlib import Path

import pandas as pd

PARAMETER_COLUMNS = (
    "fast_ema_period",
    "slow_ema_period",
    "trend_ema_period",
)


@dataclass(frozen=True)
class ParameterCombinationFrequency:
    fast_ema_period: int
    slow_ema_period: int
    trend_ema_period: int
    selection_count: int
    selection_rate: float


@dataclass(frozen=True)
class SymbolParameterStability:
    symbol: str
    window_count: int
    unique_combination_count: int
    modal_fast_ema_period: int
    modal_slow_ema_period: int
    modal_trend_ema_period: int
    modal_combination_count: int
    modal_combination_rate: float
    normalized_combination_entropy: float
    adjacent_transition_count: int
    adjacent_change_count: int
    adjacent_change_rate: float
    fast_change_count: int
    fast_change_rate: float
    slow_change_count: int
    slow_change_rate: float
    trend_change_count: int
    trend_change_rate: float


@dataclass(frozen=True)
class ParameterStabilitySummary:
    symbol_count: int
    total_window_count: int
    unique_combination_count: int
    global_modal_fast_ema_period: int
    global_modal_slow_ema_period: int
    global_modal_trend_ema_period: int
    global_modal_combination_count: int
    global_modal_combination_rate: float
    normalized_combination_entropy: float
    total_adjacent_transition_count: int
    total_adjacent_change_count: int
    overall_adjacent_change_rate: float
    mean_symbol_adjacent_change_rate: float
    median_symbol_adjacent_change_rate: float
    mean_symbol_modal_combination_rate: float
    median_symbol_modal_combination_rate: float


def _required_positive_int(
    value: object,
    *,
    name: str,
) -> int:
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} deve essere numerico.")

    numeric_value = float(value)

    if not isfinite(numeric_value) or numeric_value <= 0.0 or not numeric_value.is_integer():
        raise ValueError(f"{name} deve essere un intero positivo.")

    return int(numeric_value)


def _validate_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "symbol",
        "window_number",
        *PARAMETER_COLUMNS,
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(f"Colonne mancanti nel report walk-forward: {sorted(missing_columns)}")

    if frame.empty:
        raise ValueError("Il report walk-forward è vuoto.")

    result = frame.copy()

    result["symbol"] = result["symbol"].astype(str).str.upper().str.strip()

    if result["symbol"].eq("").any():
        raise ValueError("symbol non può essere vuoto.")

    normalized_windows: list[int] = []
    normalized_parameters: dict[
        str,
        list[int],
    ] = {column: [] for column in PARAMETER_COLUMNS}

    for row in result.itertuples(index=False):
        normalized_windows.append(
            _required_positive_int(
                row.window_number,
                name="window_number",
            )
        )

        for column in PARAMETER_COLUMNS:
            normalized_parameters[column].append(
                _required_positive_int(
                    getattr(
                        row,
                        column,
                    ),
                    name=column,
                )
            )

    result["window_number"] = normalized_windows

    for (
        column,
        values,
    ) in normalized_parameters.items():
        result[column] = values

    duplicate_mask = result.duplicated(
        subset=[
            "symbol",
            "window_number",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = result.loc[
            duplicate_mask,
            [
                "symbol",
                "window_number",
            ],
        ]

        raise ValueError(
            f"Sono presenti finestre duplicate per simbolo: {duplicates.to_dict('records')}"
        )

    return result.sort_values(
        [
            "symbol",
            "window_number",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _combination_tuple(
    row: pd.Series,
) -> tuple[int, int, int]:
    return (
        int(row["fast_ema_period"]),
        int(row["slow_ema_period"]),
        int(row["trend_ema_period"]),
    )


def _normalized_entropy(
    counts: pd.Series,
) -> float:
    if counts.empty:
        raise ValueError("Servono almeno una frequenza.")

    total = int(counts.sum())

    if total <= 0:
        raise ValueError("La somma delle frequenze deve essere positiva.")

    category_count = len(counts)

    if category_count <= 1:
        return 0.0

    probabilities = counts.astype(float) / float(total)

    entropy = -sum(
        probability * log(probability) for probability in probabilities if probability > 0.0
    )

    maximum_entropy = log(float(category_count))

    return float(entropy / maximum_entropy)


def parameter_combination_frequencies(
    frame: pd.DataFrame,
) -> tuple[
    ParameterCombinationFrequency,
    ...,
]:
    validated = _validate_frame(frame)

    grouped = (
        validated.groupby(
            list(PARAMETER_COLUMNS),
            sort=True,
        )
        .size()
        .reset_index(name="selection_count")
    )

    total_count = len(validated)

    grouped = grouped.sort_values(
        [
            "selection_count",
            "fast_ema_period",
            "slow_ema_period",
            "trend_ema_period",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    results: list[ParameterCombinationFrequency] = []

    for row in grouped.itertuples(index=False):
        selection_count = int(row.selection_count)

        results.append(
            ParameterCombinationFrequency(
                fast_ema_period=int(row.fast_ema_period),
                slow_ema_period=int(row.slow_ema_period),
                trend_ema_period=int(row.trend_ema_period),
                selection_count=(selection_count),
                selection_rate=(selection_count / total_count),
            )
        )

    return tuple(results)


def _symbol_stability(
    symbol_frame: pd.DataFrame,
) -> SymbolParameterStability:
    symbol = str(symbol_frame["symbol"].iloc[0])

    combinations = symbol_frame.loc[
        :,
        list(PARAMETER_COLUMNS),
    ].apply(
        tuple,
        axis=1,
    )

    counts = combinations.value_counts(sort=True)

    modal_combination = tuple(int(value) for value in counts.index[0])
    modal_count = int(counts.iloc[0])
    window_count = len(symbol_frame)

    adjacent_transition_count = max(
        window_count - 1,
        0,
    )

    adjacent_change_count = 0
    fast_change_count = 0
    slow_change_count = 0
    trend_change_count = 0

    if adjacent_transition_count > 0:
        ordered = symbol_frame.sort_values(
            "window_number",
            kind="stable",
        ).reset_index(drop=True)

        previous = ordered.iloc[0]

        for index in range(
            1,
            len(ordered),
        ):
            current = ordered.iloc[index]

            fast_changed = int(current["fast_ema_period"]) != int(previous["fast_ema_period"])
            slow_changed = int(current["slow_ema_period"]) != int(previous["slow_ema_period"])
            trend_changed = int(current["trend_ema_period"]) != int(previous["trend_ema_period"])

            if fast_changed or slow_changed or trend_changed:
                adjacent_change_count += 1

            if fast_changed:
                fast_change_count += 1

            if slow_changed:
                slow_change_count += 1

            if trend_changed:
                trend_change_count += 1

            previous = current

    denominator = adjacent_transition_count

    return SymbolParameterStability(
        symbol=symbol,
        window_count=window_count,
        unique_combination_count=(len(counts)),
        modal_fast_ema_period=(modal_combination[0]),
        modal_slow_ema_period=(modal_combination[1]),
        modal_trend_ema_period=(modal_combination[2]),
        modal_combination_count=(modal_count),
        modal_combination_rate=(modal_count / window_count),
        normalized_combination_entropy=(_normalized_entropy(counts)),
        adjacent_transition_count=(adjacent_transition_count),
        adjacent_change_count=(adjacent_change_count),
        adjacent_change_rate=(adjacent_change_count / denominator if denominator > 0 else 0.0),
        fast_change_count=(fast_change_count),
        fast_change_rate=(fast_change_count / denominator if denominator > 0 else 0.0),
        slow_change_count=(slow_change_count),
        slow_change_rate=(slow_change_count / denominator if denominator > 0 else 0.0),
        trend_change_count=(trend_change_count),
        trend_change_rate=(trend_change_count / denominator if denominator > 0 else 0.0),
    )


def analyze_symbol_parameter_stability(
    frame: pd.DataFrame,
) -> tuple[
    SymbolParameterStability,
    ...,
]:
    validated = _validate_frame(frame)

    results = [
        _symbol_stability(symbol_frame)
        for _, symbol_frame in (
            validated.groupby(
                "symbol",
                sort=True,
            )
        )
    ]

    return tuple(results)


def summarize_parameter_stability(
    frame: pd.DataFrame,
    symbol_results: Iterable[SymbolParameterStability] | None = None,
) -> ParameterStabilitySummary:
    validated = _validate_frame(frame)

    effective_symbol_results = (
        tuple(symbol_results)
        if symbol_results is not None
        else analyze_symbol_parameter_stability(validated)
    )

    if not effective_symbol_results:
        raise ValueError("Servono risultati per almeno un simbolo.")

    combinations = validated.loc[
        :,
        list(PARAMETER_COLUMNS),
    ].apply(
        tuple,
        axis=1,
    )

    counts = combinations.value_counts(sort=True)

    modal_combination = tuple(int(value) for value in counts.index[0])

    modal_count = int(counts.iloc[0])

    total_window_count = len(validated)

    total_transition_count = sum(
        result.adjacent_transition_count for result in effective_symbol_results
    )

    total_change_count = sum(result.adjacent_change_count for result in effective_symbol_results)

    symbol_change_rates = pd.Series(
        [result.adjacent_change_rate for result in effective_symbol_results],
        dtype=float,
    )

    symbol_modal_rates = pd.Series(
        [result.modal_combination_rate for result in effective_symbol_results],
        dtype=float,
    )

    return ParameterStabilitySummary(
        symbol_count=len(effective_symbol_results),
        total_window_count=(total_window_count),
        unique_combination_count=(len(counts)),
        global_modal_fast_ema_period=(modal_combination[0]),
        global_modal_slow_ema_period=(modal_combination[1]),
        global_modal_trend_ema_period=(modal_combination[2]),
        global_modal_combination_count=(modal_count),
        global_modal_combination_rate=(modal_count / total_window_count),
        normalized_combination_entropy=(_normalized_entropy(counts)),
        total_adjacent_transition_count=(total_transition_count),
        total_adjacent_change_count=(total_change_count),
        overall_adjacent_change_rate=(
            total_change_count / total_transition_count if total_transition_count > 0 else 0.0
        ),
        mean_symbol_adjacent_change_rate=(float(symbol_change_rates.mean())),
        median_symbol_adjacent_change_rate=(float(symbol_change_rates.median())),
        mean_symbol_modal_combination_rate=(float(symbol_modal_rates.mean())),
        median_symbol_modal_combination_rate=(float(symbol_modal_rates.median())),
    )


def combination_frequencies_to_frame(
    frequencies: Iterable[ParameterCombinationFrequency],
) -> pd.DataFrame:
    return pd.DataFrame(asdict(frequency) for frequency in frequencies)


def symbol_stability_to_frame(
    results: Iterable[SymbolParameterStability],
) -> pd.DataFrame:
    return pd.DataFrame(asdict(result) for result in results)


def parameter_stability_summary_to_frame(
    summary: ParameterStabilitySummary,
) -> pd.DataFrame:
    return pd.DataFrame([asdict(summary)])


def run_parameter_stability_from_csv(
    optimization_report_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if not optimization_report_path.exists():
        raise FileNotFoundError(f"Report walk-forward non trovato: {optimization_report_path}")

    frame = pd.read_csv(optimization_report_path)

    frequencies = parameter_combination_frequencies(frame)

    symbol_results = analyze_symbol_parameter_stability(frame)

    summary = summarize_parameter_stability(
        frame,
        symbol_results,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frequencies_path = output_dir / "parameter_combination_frequencies.csv"

    symbol_path = output_dir / "parameter_stability_by_symbol.csv"

    summary_path = output_dir / "parameter_stability_summary.csv"

    combination_frequencies_to_frame(frequencies).to_csv(
        frequencies_path,
        index=False,
    )

    symbol_stability_to_frame(symbol_results).to_csv(
        symbol_path,
        index=False,
    )

    parameter_stability_summary_to_frame(summary).to_csv(
        summary_path,
        index=False,
    )

    return {
        "frequencies": frequencies,
        "symbol_results": (symbol_results),
        "summary": summary,
        "frequencies_path": (frequencies_path),
        "symbol_path": (symbol_path),
        "summary_path": (summary_path),
    }
