from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from project_alpha.backtest import (
    ExecutionConfig,
    run_long_only_backtest,
)
from project_alpha.batch_optimization import (
    BatchWalkForwardOptimizationResult,
)
from project_alpha.optimization import (
    WalkForwardOptimizationResult,
    _prepare_out_of_sample_frame,
    prepare_flat_out_of_sample_frame,
)
from project_alpha.portfolio import (
    PortfolioResult,
    run_fixed_weight_portfolio_backtest,
)
from project_alpha.validation import (
    WalkForwardSplit,
    generate_walk_forward_splits,
)


@dataclass(frozen=True)
class WalkForwardPortfolioResult:
    portfolio: PortfolioResult
    symbol_curves: dict[str, pd.DataFrame]


def _normalize_symbol_frame(
    frame: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Il dataset walk-forward è vuoto.")

    required = {
        "symbol",
        "timestamp",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"Colonne mancanti nel dataset: {sorted(missing)}")

    normalized_symbol = symbol.strip().upper()

    selected = frame.loc[frame["symbol"].astype(str).str.upper().eq(normalized_symbol)].copy()

    if selected.empty:
        raise ValueError(f"Simbolo {normalized_symbol} non trovato nel dataset.")

    selected["timestamp"] = pd.to_datetime(
        selected["timestamp"],
        utc=True,
        errors="raise",
    )

    return selected.sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)


def _validate_split_against_window(
    split: WalkForwardSplit,
    result: WalkForwardOptimizationResult,
    window_index: int,
) -> None:
    window = result.windows[window_index]

    expected_values = (
        split.window_number,
        split.train_start,
        split.train_end,
        split.test_start,
        split.test_end,
    )

    actual_values = (
        window.window_number,
        window.train_start,
        window.train_end,
        window.test_start,
        window.test_end,
    )

    if expected_values != actual_values:
        raise ValueError(
            "Le finestre walk-forward del "
            f"simbolo {result.symbol} non "
            "corrispondono al dataset fornito."
        )


def reconstruct_walk_forward_oos_curve(
    frame: pd.DataFrame,
    result: WalkForwardOptimizationResult,
    *,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
    execution: ExecutionConfig | None = None,
) -> pd.DataFrame:
    if result.windows_overlap:
        raise ValueError(
            "La ricostruzione della curva OOS non supporta finestre walk-forward sovrapposte."
        )

    selected = _normalize_symbol_frame(
        frame,
        result.symbol,
    )

    splits = generate_walk_forward_splits(
        frame=selected,
        train_size=result.train_size,
        test_size=result.test_size,
        step_size=result.step_size,
    )

    if len(splits) != len(result.windows):
        raise ValueError(
            "Il numero di finestre generate non corrisponde al risultato walk-forward."
        )

    curves: list[pd.DataFrame] = []

    for window_index, split in enumerate(splits):
        _validate_split_against_window(
            split,
            result,
            window_index,
        )

        window = result.windows[window_index]

        if window.selected_parameters is None:
            positioned = prepare_flat_out_of_sample_frame(split.out_of_sample)
        else:
            positioned = _prepare_out_of_sample_frame(
                split.in_sample,
                split.out_of_sample,
                window.selected_parameters,
            )

        # Le selezioni dei parametri restano quelle del walk-forward
        # (congelate): qui cambia solo come vengono eseguite.
        backtest = run_long_only_backtest(
            positioned,
            initial_capital=initial_capital,
            cost_per_side=cost_per_side,
            execution=(execution if execution is not None else result.execution),
        )

        curve = backtest.equity_curve.copy()
        curve["symbol"] = result.symbol.strip().upper()
        curve["window_number"] = window.window_number

        curves.append(curve)

    if not curves:
        raise ValueError("Il risultato walk-forward non contiene finestre OOS.")

    combined = pd.concat(
        curves,
        ignore_index=True,
    )

    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"],
        utc=True,
        errors="raise",
    )

    combined = combined.sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)

    if combined["timestamp"].duplicated().any():
        raise ValueError("La curva OOS ricostruita contiene timestamp duplicati.")

    combined["walk_forward_equity"] = (
        initial_capital * (1.0 + combined["strategy_return"]).cumprod()
    )

    combined["walk_forward_benchmark_equity"] = (
        initial_capital * (1.0 + combined["asset_return"]).cumprod()
    )

    return combined


def run_walk_forward_portfolio_backtest(
    frame: pd.DataFrame,
    batch_result: BatchWalkForwardOptimizationResult,
    *,
    weights: Mapping[str, float] | None = None,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
    execution: ExecutionConfig | None = None,
) -> WalkForwardPortfolioResult:
    if not batch_result.symbols:
        raise ValueError("Il risultato batch non contiene simboli.")

    symbol_curves: dict[
        str,
        pd.DataFrame,
    ] = {}

    for symbol in batch_result.symbols:
        if symbol not in batch_result.results:
            raise ValueError(f"Risultato walk-forward mancante per il simbolo {symbol}.")

        symbol_curves[symbol] = reconstruct_walk_forward_oos_curve(
            frame,
            batch_result.results[symbol],
            initial_capital=initial_capital,
            cost_per_side=cost_per_side,
            execution=execution,
        )

    portfolio = run_fixed_weight_portfolio_backtest(
        symbol_curves,
        weights=weights,
        initial_capital=initial_capital,
    )

    return WalkForwardPortfolioResult(
        portfolio=portfolio,
        symbol_curves=symbol_curves,
    )
