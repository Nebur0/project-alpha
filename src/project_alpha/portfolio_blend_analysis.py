from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from project_alpha.portfolio_walk_forward import (
    WalkForwardPortfolioResult,
)


DEFAULT_DEFENSIVE_WEIGHTS = (
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
)


@dataclass(frozen=True)
class PortfolioBlendAnalysisResult:
    overall: pd.DataFrame
    by_window: pd.DataFrame
    leave_one_window_out: pd.DataFrame


def _max_drawdown(
    equity: pd.Series,
) -> float:
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def _normalize_defensive_weights(
    defensive_weights: Sequence[float],
) -> tuple[float, ...]:
    if not defensive_weights:
        raise ValueError(
            "È necessario specificare almeno "
            "un peso difensivo."
        )

    normalized: list[float] = []

    for raw_weight in defensive_weights:
        weight = float(raw_weight)

        if not np.isfinite(weight):
            raise ValueError(
                "I pesi difensivi devono essere "
                "finiti."
            )

        if weight < 0.0 or weight > 1.0:
            raise ValueError(
                "I pesi difensivi devono essere "
                "compresi tra 0.0 e 1.0."
            )

        if weight not in normalized:
            normalized.append(weight)

    return tuple(normalized)


def _calculate_blend_metrics(
    defensive_returns: pd.Series,
    passive_returns: pd.Series,
    *,
    defensive_weight: float,
    initial_capital: float,
) -> dict[str, float | int]:
    if len(defensive_returns) != len(
        passive_returns
    ):
        raise ValueError(
            "Le serie defensive e passive devono "
            "avere la stessa lunghezza."
        )

    if defensive_returns.empty:
        raise ValueError(
            "La serie dei rendimenti del blend "
            "è vuota."
        )

    passive_weight = 1.0 - defensive_weight

    returns = (
        defensive_weight * defensive_returns
        + passive_weight * passive_returns
    )

    equity = (
        initial_capital
        * (1.0 + returns).cumprod()
    )

    observations = max(
        len(returns) - 1,
        1,
    )

    years = observations / 252.0

    final_equity = float(
        equity.iloc[-1]
    )

    total_return = (
        final_equity
        / initial_capital
        - 1.0
    )

    cagr = (
        (
            final_equity
            / initial_capital
        )
        ** (1.0 / years)
        - 1.0
        if years > 0
        and final_equity > 0
        else float("nan")
    )

    daily_mean = returns.mean()
    daily_std = returns.std(ddof=1)

    annualized_sharpe = (
        float(
            daily_mean
            / daily_std
            * sqrt(252.0)
        )
        if daily_std
        and not np.isnan(daily_std)
        else float("nan")
    )

    annualized_volatility = (
        float(
            daily_std * sqrt(252.0)
        )
        if not np.isnan(daily_std)
        else float("nan")
    )

    return {
        "defensive_weight": float(
            defensive_weight
        ),
        "passive_weight": float(
            passive_weight
        ),
        "observation_count": len(
            returns
        ),
        "final_equity": final_equity,
        "total_return": float(
            total_return
        ),
        "cagr": float(cagr),
        "annualized_sharpe": (
            annualized_sharpe
        ),
        "annualized_volatility": (
            annualized_volatility
        ),
        "max_drawdown": _max_drawdown(
            equity
        ),
    }


def _portfolio_curve_with_windows(
    result: WalkForwardPortfolioResult,
) -> pd.DataFrame:
    portfolio_curve = (
        result.portfolio.equity_curve.copy()
    )

    required_portfolio_columns = {
        "timestamp",
        "portfolio_return",
        "benchmark_return",
    }

    missing_portfolio_columns = (
        required_portfolio_columns.difference(
            portfolio_curve.columns
        )
    )

    if missing_portfolio_columns:
        raise ValueError(
            "Colonne mancanti nella curva del "
            "portfolio: "
            f"{sorted(missing_portfolio_columns)}"
        )

    if not result.symbol_curves:
        raise ValueError(
            "Il risultato walk-forward non "
            "contiene curve per simbolo."
        )

    portfolio_curve["timestamp"] = pd.to_datetime(
        portfolio_curve["timestamp"],
        utc=True,
        errors="raise",
    )

    reference_mapping: pd.DataFrame | None = None

    for symbol, symbol_curve in (
        result.symbol_curves.items()
    ):
        required_symbol_columns = {
            "timestamp",
            "window_number",
        }

        missing_symbol_columns = (
            required_symbol_columns.difference(
                symbol_curve.columns
            )
        )

        if missing_symbol_columns:
            raise ValueError(
                "Colonne mancanti nella curva "
                f"di {symbol}: "
                f"{sorted(missing_symbol_columns)}"
            )

        mapping = symbol_curve.loc[
            :,
            [
                "timestamp",
                "window_number",
            ],
        ].copy()

        mapping["timestamp"] = pd.to_datetime(
            mapping["timestamp"],
            utc=True,
            errors="raise",
        )

        mapping = (
            mapping.sort_values(
                "timestamp",
                kind="stable",
            )
            .reset_index(drop=True)
        )

        if mapping[
            "timestamp"
        ].duplicated().any():
            raise ValueError(
                "Timestamp duplicati nella "
                f"mappatura finestre di {symbol}."
            )

        if reference_mapping is None:
            reference_mapping = mapping
            continue

        if not reference_mapping.equals(
            mapping
        ):
            raise ValueError(
                "Le mappature delle finestre "
                "walk-forward non coincidono tra "
                "i simboli."
            )

    if reference_mapping is None:
        raise ValueError(
            "Impossibile costruire la mappatura "
            "delle finestre walk-forward."
        )

    merged = portfolio_curve.merge(
        reference_mapping,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    if merged["window_number"].isna().any():
        raise ValueError(
            "Alcuni timestamp del portfolio non "
            "hanno una finestra walk-forward."
        )

    merged["window_number"] = (
        merged["window_number"].astype(int)
    )

    return (
        merged.sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
    )


def analyze_walk_forward_portfolio_blends(
    result: WalkForwardPortfolioResult,
    *,
    defensive_weights: Sequence[float] = (
        DEFAULT_DEFENSIVE_WEIGHTS
    ),
) -> PortfolioBlendAnalysisResult:
    weights = _normalize_defensive_weights(
        defensive_weights
    )

    initial_capital_value = (
        result.portfolio.metrics.get(
            "initial_capital"
        )
    )

    if initial_capital_value is None:
        raise ValueError(
            "Il capitale iniziale non è presente "
            "nelle metriche del portfolio."
        )

    initial_capital = float(
        initial_capital_value
    )

    if (
        not np.isfinite(initial_capital)
        or initial_capital <= 0
    ):
        raise ValueError(
            "Il capitale iniziale del portfolio "
            "deve essere positivo e finito."
        )

    curve = _portfolio_curve_with_windows(
        result
    )

    overall_rows: list[
        dict[str, object]
    ] = []

    for defensive_weight in weights:
        overall_rows.append(
            _calculate_blend_metrics(
                curve["portfolio_return"],
                curve["benchmark_return"],
                defensive_weight=(
                    defensive_weight
                ),
                initial_capital=(
                    initial_capital
                ),
            )
        )

    by_window_rows: list[
        dict[str, object]
    ] = []

    window_numbers = tuple(
        sorted(
            curve[
                "window_number"
            ].unique()
        )
    )

    for window_number in window_numbers:
        selected = curve.loc[
            curve["window_number"].eq(
                window_number
            )
        ].copy()

        window_start = selected[
            "timestamp"
        ].min()

        window_end = selected[
            "timestamp"
        ].max()

        for defensive_weight in weights:
            metrics = _calculate_blend_metrics(
                selected[
                    "portfolio_return"
                ],
                selected[
                    "benchmark_return"
                ],
                defensive_weight=(
                    defensive_weight
                ),
                initial_capital=(
                    initial_capital
                ),
            )

            by_window_rows.append(
                {
                    "window_number": int(
                        window_number
                    ),
                    "window_start": (
                        window_start
                    ),
                    "window_end": window_end,
                    **metrics,
                }
            )

    leave_one_out_rows: list[
        dict[str, object]
    ] = []

    for omitted_window_number in (
        window_numbers
    ):
        selected = curve.loc[
            ~curve["window_number"].eq(
                omitted_window_number
            )
        ].copy()

        if selected.empty:
            continue

        for defensive_weight in weights:
            metrics = _calculate_blend_metrics(
                selected[
                    "portfolio_return"
                ],
                selected[
                    "benchmark_return"
                ],
                defensive_weight=(
                    defensive_weight
                ),
                initial_capital=(
                    initial_capital
                ),
            )

            leave_one_out_rows.append(
                {
                    "omitted_window_number": (
                        int(
                            omitted_window_number
                        )
                    ),
                    **metrics,
                }
            )

    return PortfolioBlendAnalysisResult(
        overall=pd.DataFrame(
            overall_rows
        ),
        by_window=pd.DataFrame(
            by_window_rows
        ),
        leave_one_window_out=pd.DataFrame(
            leave_one_out_rows
        ),
    )
