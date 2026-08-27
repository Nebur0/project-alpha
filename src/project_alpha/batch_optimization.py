from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from project_alpha.optimization import (
    GridValue,
    WalkForwardOptimizationResult,
    run_walk_forward_optimization,
    walk_forward_optimization_to_dict,
    walk_forward_optimization_to_frame,
)


@dataclass(frozen=True)
class BatchWalkForwardOptimizationResult:
    symbols: tuple[str, ...]
    score_metric: str
    train_size: int
    test_size: int
    step_size: int
    minimum_completed_trades: int
    results: dict[
        str,
        WalkForwardOptimizationResult,
    ]


def _normalize_symbols(
    symbols: Sequence[str],
) -> tuple[str, ...]:
    if not symbols:
        raise ValueError(
            "È necessario specificare almeno "
            "un simbolo."
        )

    normalized = tuple(
        dict.fromkeys(
            symbol.strip().upper()
            for symbol in symbols
            if symbol.strip()
        )
    )

    if not normalized:
        raise ValueError(
            "È necessario specificare almeno "
            "un simbolo valido."
        )

    return normalized


def _validate_batch_frame(
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        raise ValueError(
            "Il dataset batch è vuoto."
        )

    if "symbol" not in frame.columns:
        raise ValueError(
            "La colonna 'symbol' è obbligatoria."
        )


def run_batch_walk_forward_optimization(
    frame: pd.DataFrame,
    symbols: Sequence[str],
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
) -> BatchWalkForwardOptimizationResult:
    _validate_batch_frame(frame)

    normalized_symbols = _normalize_symbols(
        symbols
    )

    available_symbols = set(
        frame["symbol"]
        .astype(str)
        .str.upper()
        .unique()
    )

    missing_symbols = [
        symbol
        for symbol in normalized_symbols
        if symbol not in available_symbols
    ]

    if missing_symbols:
        raise ValueError(
            "Simboli non trovati nel dataset: "
            f"{missing_symbols}"
        )

    results: dict[
        str,
        WalkForwardOptimizationResult,
    ] = {}

    for symbol in normalized_symbols:
        selected = frame.loc[
            frame["symbol"]
            .astype(str)
            .str.upper()
            .eq(symbol)
        ].copy()

        result = run_walk_forward_optimization(
            selected,
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
        )

        results[symbol] = result

    effective_step_size = (
        test_size
        if step_size is None
        else step_size
    )

    return BatchWalkForwardOptimizationResult(
        symbols=normalized_symbols,
        score_metric=score_metric,
        train_size=train_size,
        test_size=test_size,
        step_size=effective_step_size,
        minimum_completed_trades=(
            minimum_completed_trades
        ),
        results=results,
    )


def batch_walk_forward_optimization_to_dict(
    result: BatchWalkForwardOptimizationResult,
) -> dict[str, object]:
    return {
        "symbols": list(result.symbols),
        "score_metric": result.score_metric,
        "train_size": result.train_size,
        "test_size": result.test_size,
        "step_size": result.step_size,
        "minimum_completed_trades": (
            result.minimum_completed_trades
        ),
        "symbol_count": len(
            result.symbols
        ),
        "results": {
            symbol: (
                walk_forward_optimization_to_dict(
                    symbol_result
                )
            )
            for symbol, symbol_result
            in result.results.items()
        },
    }


def batch_walk_forward_optimization_to_frame(
    result: BatchWalkForwardOptimizationResult,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for symbol in result.symbols:
        symbol_frame = (
            walk_forward_optimization_to_frame(
                result.results[symbol]
            )
        )

        frames.append(
            symbol_frame
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def batch_walk_forward_comparison_to_frame(
    result: BatchWalkForwardOptimizationResult,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for symbol in result.symbols:
        symbol_result = (
            result.results[symbol]
        )

        rows.append(
            {
                "symbol": symbol,
                "score_metric": (
                    result.score_metric
                ),
                "train_size": (
                    result.train_size
                ),
                "test_size": (
                    result.test_size
                ),
                "step_size": (
                    result.step_size
                ),
                "minimum_completed_trades": (
                    result
                    .minimum_completed_trades
                ),
                "window_count": len(
                    symbol_result.windows
                ),
                "windows_overlap": (
                    symbol_result
                    .windows_overlap
                ),
                "compounding_enabled": (
                    symbol_result
                    .compounding_enabled
                ),
                **symbol_result.summary,
            }
        )

    return pd.DataFrame(rows)