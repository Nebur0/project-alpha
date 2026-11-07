from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from project_alpha.batch_optimization import (
    BatchWalkForwardOptimizationResult,
)
from project_alpha.portfolio_walk_forward import (
    run_walk_forward_portfolio_backtest,
)

DEFAULT_COSTS_PER_SIDE = (
    0.0000,
    0.0005,
    0.0010,
    0.0020,
    0.0030,
)


@dataclass(frozen=True)
class CostStressAnalysisResult:
    baseline_cost_per_side: float
    scenarios: pd.DataFrame


def _normalize_costs(
    costs_per_side: Sequence[float],
) -> tuple[float, ...]:
    if not costs_per_side:
        raise ValueError("È necessario specificare almeno un livello di costo.")

    normalized: list[float] = []

    for raw_cost in costs_per_side:
        cost = float(raw_cost)

        if not np.isfinite(cost):
            raise ValueError("I costi per lato devono essere finiti.")

        if cost < 0.0:
            raise ValueError("I costi per lato non possono essere negativi.")

        if cost not in normalized:
            normalized.append(cost)

    return tuple(normalized)


def analyze_frozen_selection_cost_stress(
    frame: pd.DataFrame,
    baseline_batch: BatchWalkForwardOptimizationResult,
    *,
    costs_per_side: Sequence[float] = DEFAULT_COSTS_PER_SIDE,
    baseline_cost_per_side: float = 0.001,
    initial_capital: float = 10_000.0,
) -> CostStressAnalysisResult:
    costs = _normalize_costs(costs_per_side)

    baseline_cost = float(baseline_cost_per_side)

    if not np.isfinite(baseline_cost) or baseline_cost < 0.0:
        raise ValueError("Il costo baseline deve essere finito e non negativo.")

    if initial_capital <= 0:
        raise ValueError("Il capitale iniziale deve essere positivo.")

    rows: list[dict[str, float | int]] = []
    reference_exposure: float | None = None

    for cost_per_side in costs:
        result = run_walk_forward_portfolio_backtest(
            frame,
            baseline_batch,
            initial_capital=initial_capital,
            cost_per_side=cost_per_side,
        )

        metrics = result.portfolio.metrics
        exposure = float(metrics["mean_gross_exposure"])

        if reference_exposure is None:
            reference_exposure = exposure
        elif not np.isclose(
            exposure,
            reference_exposure,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "L'esposizione è cambiata tra gli scenari di costo. "
                "Le selezioni walk-forward devono rimanere congelate."
            )

        rows.append(
            {
                "cost_per_side": float(cost_per_side),
                "cost_bps_per_side": float(cost_per_side * 10_000.0),
                "total_return": float(metrics["total_return"]),
                "cagr": float(metrics["cagr"]),
                "annualized_sharpe": float(metrics["annualized_sharpe"]),
                "annualized_volatility": float(metrics["annualized_volatility"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "mean_gross_exposure": exposure,
                "symbol_count": int(metrics["symbol_count"]),
            }
        )

    return CostStressAnalysisResult(
        baseline_cost_per_side=baseline_cost,
        scenarios=pd.DataFrame(rows),
    )
