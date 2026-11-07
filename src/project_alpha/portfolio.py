from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

MetricValue = float | int | None


@dataclass(frozen=True)
class PortfolioResult:
    equity_curve: pd.DataFrame
    metrics: dict[str, MetricValue]
    weights: dict[str, float]


def _max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def _annualized_sharpe(
    returns: pd.Series,
) -> float:
    daily_mean = returns.mean()
    daily_std = returns.std(ddof=1)

    return (
        float(daily_mean / daily_std * sqrt(252.0))
        if daily_std and not np.isnan(daily_std)
        else float("nan")
    )


def _annualized_volatility(
    returns: pd.Series,
) -> float:
    daily_std = returns.std(ddof=1)

    return float(daily_std * sqrt(252.0)) if not np.isnan(daily_std) else float("nan")


def _cagr(
    final_equity: float,
    initial_capital: float,
    years: float,
) -> float:
    return (
        float((final_equity / initial_capital) ** (1.0 / years) - 1.0)
        if years > 0 and final_equity > 0
        else float("nan")
    )


def _normalize_curves(
    curves: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if not curves:
        raise ValueError("È necessario fornire almeno una curva strategica.")

    normalized: dict[str, pd.DataFrame] = {}

    required_columns = {
        "timestamp",
        "asset_return",
        "executed_position",
        "strategy_return",
    }

    for raw_symbol, frame in curves.items():
        symbol = str(raw_symbol).strip().upper()

        if not symbol:
            raise ValueError("I simboli del portfolio non possono essere vuoti.")

        if symbol in normalized:
            raise ValueError(f"Simbolo duplicato nel portfolio: {symbol}.")

        if frame.empty:
            raise ValueError(f"La curva di {symbol} è vuota.")

        missing = required_columns.difference(frame.columns)

        if missing:
            raise ValueError(f"Colonne mancanti per {symbol}: {sorted(missing)}")

        selected = frame.loc[
            :,
            [
                "timestamp",
                "asset_return",
                "executed_position",
                "strategy_return",
            ],
        ].copy()

        selected["timestamp"] = pd.to_datetime(
            selected["timestamp"],
            utc=True,
            errors="raise",
        )

        selected = selected.sort_values(
            "timestamp",
            kind="stable",
        ).reset_index(drop=True)

        if selected["timestamp"].duplicated().any():
            raise ValueError(f"Timestamp duplicati nella curva di {symbol}.")

        numeric_columns = [
            "asset_return",
            "executed_position",
            "strategy_return",
        ]

        for column in numeric_columns:
            selected[column] = pd.to_numeric(
                selected[column],
                errors="raise",
            )

            values = selected[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(f"La colonna '{column}' di {symbol} contiene valori non finiti.")

        normalized[symbol] = selected

    return normalized


def _resolve_weights(
    symbols: tuple[str, ...],
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if weights is None:
        equal_weight = 1.0 / len(symbols)

        return {symbol: equal_weight for symbol in symbols}

    normalized_weights: dict[str, float] = {}

    for raw_symbol, raw_weight in weights.items():
        symbol = str(raw_symbol).strip().upper()

        if symbol in normalized_weights:
            raise ValueError(f"Peso duplicato per il simbolo {symbol}.")

        weight = float(raw_weight)

        if not np.isfinite(weight):
            raise ValueError(f"Il peso di {symbol} non è finito.")

        if weight < 0:
            raise ValueError("I pesi del portfolio long-only non possono essere negativi.")

        normalized_weights[symbol] = weight

    expected_symbols = set(symbols)
    provided_symbols = set(normalized_weights)

    if provided_symbols != expected_symbols:
        missing = sorted(expected_symbols - provided_symbols)
        extra = sorted(provided_symbols - expected_symbols)

        raise ValueError(
            "I pesi devono contenere esattamente "
            "i simboli del portfolio. "
            f"Mancanti: {missing}. Extra: {extra}."
        )

    total_weight = sum(normalized_weights.values())

    if not np.isclose(
        total_weight,
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"La somma dei pesi deve essere 1.0, ricevuto {total_weight}.")

    return normalized_weights


def _validate_common_timestamps(
    curves: Mapping[str, pd.DataFrame],
    symbols: tuple[str, ...],
) -> pd.Series:
    reference_symbol = symbols[0]
    reference = curves[reference_symbol]["timestamp"].reset_index(drop=True)

    for symbol in symbols[1:]:
        candidate = curves[symbol]["timestamp"].reset_index(drop=True)

        if not reference.equals(candidate):
            raise ValueError(
                "Tutte le curve del portfolio "
                "devono avere gli stessi timestamp. "
                f"Mismatch rilevato per {symbol}."
            )

    return reference


def run_fixed_weight_portfolio_backtest(
    curves: Mapping[str, pd.DataFrame],
    *,
    weights: Mapping[str, float] | None = None,
    initial_capital: float = 10_000.0,
) -> PortfolioResult:
    if initial_capital <= 0:
        raise ValueError("Il capitale iniziale deve essere positivo.")

    normalized_curves = _normalize_curves(curves)
    symbols = tuple(normalized_curves)

    resolved_weights = _resolve_weights(
        symbols,
        weights,
    )

    timestamps = _validate_common_timestamps(
        normalized_curves,
        symbols,
    )

    portfolio = pd.DataFrame(
        {
            "timestamp": timestamps.copy(),
        }
    )

    portfolio_return = np.zeros(
        len(portfolio),
        dtype=float,
    )

    benchmark_return = np.zeros(
        len(portfolio),
        dtype=float,
    )

    gross_exposure = np.zeros(
        len(portfolio),
        dtype=float,
    )

    for symbol in symbols:
        weight = resolved_weights[symbol]
        curve = normalized_curves[symbol]

        strategy_return = curve["strategy_return"].to_numpy(dtype=float)

        asset_return = curve["asset_return"].to_numpy(dtype=float)

        executed_position = curve["executed_position"].to_numpy(dtype=float)

        strategy_contribution_column = f"{symbol}_strategy_contribution"
        benchmark_contribution_column = f"{symbol}_benchmark_contribution"

        portfolio[strategy_contribution_column] = weight * strategy_return

        portfolio[benchmark_contribution_column] = weight * asset_return

        portfolio_return += weight * strategy_return

        benchmark_return += weight * asset_return

        gross_exposure += weight * executed_position

    portfolio["portfolio_return"] = portfolio_return

    portfolio["benchmark_return"] = benchmark_return

    portfolio["gross_exposure"] = gross_exposure

    portfolio["equity"] = initial_capital * (1.0 + portfolio["portfolio_return"]).cumprod()

    portfolio["benchmark_equity"] = (
        initial_capital * (1.0 + portfolio["benchmark_return"]).cumprod()
    )

    observations = max(
        len(portfolio) - 1,
        1,
    )

    years = observations / 252.0

    final_equity = float(portfolio["equity"].iloc[-1])

    benchmark_final_equity = float(portfolio["benchmark_equity"].iloc[-1])

    total_return = final_equity / initial_capital - 1.0

    benchmark_total_return = benchmark_final_equity / initial_capital - 1.0

    cagr = _cagr(
        final_equity,
        initial_capital,
        years,
    )

    benchmark_cagr = _cagr(
        benchmark_final_equity,
        initial_capital,
        years,
    )

    annualized_sharpe = _annualized_sharpe(portfolio["portfolio_return"])

    benchmark_annualized_sharpe = _annualized_sharpe(portfolio["benchmark_return"])

    annualized_volatility = _annualized_volatility(portfolio["portfolio_return"])

    benchmark_annualized_volatility = _annualized_volatility(portfolio["benchmark_return"])

    max_drawdown = _max_drawdown(portfolio["equity"])

    benchmark_max_drawdown = _max_drawdown(portfolio["benchmark_equity"])

    metrics: dict[str, MetricValue] = {
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return": float(total_return),
        "cagr": cagr,
        "annualized_sharpe": (annualized_sharpe),
        "annualized_volatility": (annualized_volatility),
        "max_drawdown": max_drawdown,
        "benchmark_final_equity": (benchmark_final_equity),
        "benchmark_total_return": float(benchmark_total_return),
        "benchmark_cagr": (benchmark_cagr),
        "benchmark_annualized_sharpe": (benchmark_annualized_sharpe),
        "benchmark_annualized_volatility": (benchmark_annualized_volatility),
        "benchmark_max_drawdown": (benchmark_max_drawdown),
        "mean_gross_exposure": float(portfolio["gross_exposure"].mean()),
        "symbol_count": len(symbols),
    }

    return PortfolioResult(
        equity_curve=portfolio,
        metrics=metrics,
        weights=resolved_weights,
    )
