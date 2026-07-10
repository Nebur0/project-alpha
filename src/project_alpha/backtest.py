from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    metrics: dict[str, float | int]


def _max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def run_long_only_backtest(
    frame: pd.DataFrame,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
) -> BacktestResult:
    if initial_capital <= 0:
        raise ValueError("Il capitale iniziale deve essere positivo.")
    if cost_per_side < 0:
        raise ValueError("Il costo per lato non può essere negativo.")

    required = {"timestamp", "close", "target_position"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonne mancanti per il backtest: {sorted(missing)}")

    data = frame.sort_values("timestamp").copy()
    if data["timestamp"].duplicated().any():
        raise ValueError("Timestamp duplicati nel dataset del simbolo.")

    data["asset_return"] = data["close"].pct_change().fillna(0.0)

    # Il target calcolato alla chiusura è applicato dalla seduta successiva.
    data["executed_position"] = data["target_position"].shift(1).fillna(0.0)
    data["turnover"] = data["executed_position"].diff().abs().fillna(
        data["executed_position"].abs()
    )
    data["trading_cost"] = data["turnover"] * cost_per_side
    data["strategy_return"] = (
        data["executed_position"] * data["asset_return"] - data["trading_cost"]
    )

    data["equity"] = initial_capital * (1.0 + data["strategy_return"]).cumprod()
    data["benchmark_equity"] = initial_capital * (1.0 + data["asset_return"]).cumprod()

    observations = max(len(data) - 1, 1)
    years = observations / 252.0
    total_return = data["equity"].iloc[-1] / initial_capital - 1.0
    benchmark_return = data["benchmark_equity"].iloc[-1] / initial_capital - 1.0

    cagr = (
        (data["equity"].iloc[-1] / initial_capital) ** (1.0 / years) - 1.0
        if years > 0 and data["equity"].iloc[-1] > 0
        else float("nan")
    )

    daily_mean = data["strategy_return"].mean()
    daily_std = data["strategy_return"].std(ddof=1)
    sharpe = (
        float(daily_mean / daily_std * sqrt(252.0))
        if daily_std and not np.isnan(daily_std)
        else float("nan")
    )

    trade_count = int((data["turnover"] > 0).sum())

    metrics: dict[str, float | int] = {
        "initial_capital": float(initial_capital),
        "final_equity": float(data["equity"].iloc[-1]),
        "total_return": float(total_return),
        "benchmark_total_return": float(benchmark_return),
        "cagr": float(cagr),
        "annualized_sharpe": sharpe,
        "max_drawdown": _max_drawdown(data["equity"]),
        "trade_events": trade_count,
        "market_exposure": float(data["executed_position"].mean()),
        "cost_per_side": float(cost_per_side),
    }

    return BacktestResult(equity_curve=data, metrics=metrics)
