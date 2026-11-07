"""Stress test dell'esecuzione a selezione congelata.

Come ``cost_stress_analysis``, ma invece del costo per lato cambia *come*
vengono eseguiti gli stessi segnali: chiusura del giorno del segnale,
apertura del giorno dopo, con eventuali giorni di ritardo. Le selezioni
dei parametri del walk-forward restano identiche in tutti gli scenari, così
la differenza misura solo la sensibilità all'esecuzione.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from project_alpha.backtest import ExecutionConfig
from project_alpha.batch_optimization import (
    BatchWalkForwardOptimizationResult,
)
from project_alpha.portfolio_walk_forward import (
    run_walk_forward_portfolio_backtest,
)

DEFAULT_EXECUTION_SCENARIOS: tuple[tuple[str, ExecutionConfig], ...] = (
    ("close", ExecutionConfig(mode="close")),
    ("next_open", ExecutionConfig(mode="next_open")),
    ("next_open_delay_1", ExecutionConfig(mode="next_open", delay_days=1)),
    ("close_delay_1", ExecutionConfig(mode="close", delay_days=1)),
)


@dataclass(frozen=True)
class ExecutionStressAnalysisResult:
    cost_per_side: float
    scenarios: pd.DataFrame


def _normalize_scenarios(
    scenarios: Sequence[tuple[str, ExecutionConfig]],
) -> tuple[tuple[str, ExecutionConfig], ...]:
    if not scenarios:
        raise ValueError("È necessario specificare almeno uno scenario di esecuzione.")

    names: set[str] = set()
    normalized: list[tuple[str, ExecutionConfig]] = []

    for name, execution in scenarios:
        label = str(name).strip()

        if not label:
            raise ValueError("Il nome dello scenario non può essere vuoto.")

        if label in names:
            raise ValueError(f"Scenario di esecuzione duplicato: {label}.")

        if not isinstance(execution, ExecutionConfig):
            raise ValueError(f"Lo scenario {label} deve usare un ExecutionConfig.")

        names.add(label)
        normalized.append((label, execution))

    return tuple(normalized)


def analyze_frozen_selection_execution_stress(
    frame: pd.DataFrame,
    baseline_batch: BatchWalkForwardOptimizationResult,
    *,
    scenarios: Sequence[tuple[str, ExecutionConfig]] = DEFAULT_EXECUTION_SCENARIOS,
    cost_per_side: float = 0.001,
    initial_capital: float = 10_000.0,
) -> ExecutionStressAnalysisResult:
    normalized = _normalize_scenarios(scenarios)

    cost = float(cost_per_side)

    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError("Il costo per lato deve essere finito e non negativo.")

    if initial_capital <= 0:
        raise ValueError("Il capitale iniziale deve essere positivo.")

    rows: list[dict[str, float | int | str]] = []

    for name, execution in normalized:
        result = run_walk_forward_portfolio_backtest(
            frame,
            baseline_batch,
            initial_capital=initial_capital,
            cost_per_side=cost,
            execution=execution,
        )

        metrics = result.portfolio.metrics

        rows.append(
            {
                "scenario": name,
                "execution_mode": execution.mode,
                "delay_days": execution.delay_days,
                "cash_rate_annual": execution.cash_rate_annual,
                "cost_bps_per_side": cost * 10_000.0,
                "total_return": float(metrics["total_return"]),
                "cagr": float(metrics["cagr"]),
                "annualized_sharpe": float(metrics["annualized_sharpe"]),
                "annualized_volatility": float(metrics["annualized_volatility"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "mean_gross_exposure": float(metrics["mean_gross_exposure"]),
                "symbol_count": int(metrics["symbol_count"]),
            }
        )

    return ExecutionStressAnalysisResult(
        cost_per_side=cost,
        scenarios=pd.DataFrame(rows),
    )
