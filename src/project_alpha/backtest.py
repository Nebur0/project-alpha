from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

EXECUTION_MODES = (
    "close",
    "next_open",
)


@dataclass(frozen=True)
class ExecutionConfig:
    """Come e quando un segnale calcolato alla chiusura viene eseguito.

    - ``mode="close"`` (default storico): il target deciso alla chiusura
      del giorno t è eseguito allo stesso prezzo di chiusura e vale per
      il rendimento close-to-close del giorno t+1. È un'approssimazione
      ottimistica: il segnale dipende da close e volume finali del giorno.
    - ``mode="next_open"``: il target deciso alla chiusura del giorno t è
      eseguito all'apertura del giorno t+1. Il gap notturno (close t ->
      open t+1) resta sulla posizione precedente; la nuova posizione vale
      dall'apertura alla chiusura.
    - ``delay_days``: giorni di ritardo aggiuntivi prima dell'esecuzione.
    - ``cash_rate_annual``: rendimento annuo della quota non investita
      (capitalizzato su 252 giorni). Default 0, come in passato.
    """

    mode: str = "close"
    delay_days: int = 0
    cash_rate_annual: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in EXECUTION_MODES:
            raise ValueError(
                "La modalità di esecuzione deve essere una tra "
                f"{list(EXECUTION_MODES)}."
            )

        if (
            isinstance(self.delay_days, bool)
            or not isinstance(self.delay_days, int)
            or self.delay_days < 0
        ):
            raise ValueError(
                "delay_days deve essere un intero non negativo."
            )

        rate = self.cash_rate_annual

        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not np.isfinite(rate)
            or rate <= -1.0
        ):
            raise ValueError(
                "cash_rate_annual deve essere un numero finito maggiore di -1."
            )

    @property
    def is_default(self) -> bool:
        return self == ExecutionConfig()


DEFAULT_EXECUTION = ExecutionConfig()


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    metrics: dict[str, float | int | None]


def _max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def _completed_trade_returns(data: pd.DataFrame) -> list[float]:
    trade_returns: list[float] = []
    entry_index: int | None = None

    position_changes = (
        data["executed_position"]
        .diff()
        .fillna(data["executed_position"])
    )

    for index, change in position_changes.items():
        if change == 1:
            entry_index = int(index)

        elif change == -1 and entry_index is not None:
            trade_slice = data.loc[
                entry_index:index,
                "strategy_return",
            ]

            trade_return = float(
                (1.0 + trade_slice).prod() - 1.0
            )

            trade_returns.append(trade_return)
            entry_index = None

    return trade_returns


def _completed_holding_lengths(
    position: pd.Series,
) -> list[int]:
    """Durata (in sedute) di ogni posizione aperta e poi chiusa."""
    lengths: list[int] = []
    current = 0

    for value in position.to_numpy(dtype=float):
        if value > 0.0:
            current += 1
        elif current > 0:
            lengths.append(current)
            current = 0

    return lengths


def _apply_execution(
    data: pd.DataFrame,
    execution: ExecutionConfig,
    cost_per_side: float,
) -> None:
    """Calcola posizioni eseguite, turnover, costi e rendimenti in place."""
    lag = 1 + execution.delay_days

    # Posizione detenuta durante la seduta (fino alla chiusura).
    data["executed_position"] = (
        data["target_position"]
        .shift(lag)
        .fillna(0.0)
    )

    if execution.mode == "close":
        data["overnight_position"] = data["executed_position"]

        data["turnover"] = (
            data["executed_position"]
            .diff()
            .abs()
            .fillna(data["executed_position"].abs())
        )

        data["trading_cost"] = (
            data["turnover"] * cost_per_side
        )

        data["strategy_return"] = (
            data["executed_position"] * data["asset_return"]
            - data["trading_cost"]
        )
    else:
        # Posizione portata nella notte: quella della seduta precedente.
        data["overnight_position"] = (
            data["executed_position"]
            .shift(1)
            .fillna(0.0)
        )

        overnight_return = (
            data["open"] / data["close"].shift(1) - 1.0
        ).fillna(0.0)

        intraday_return = (
            data["close"] / data["open"] - 1.0
        )

        data["turnover"] = (
            data["executed_position"]
            - data["overnight_position"]
        ).abs()

        data["trading_cost"] = (
            data["turnover"] * cost_per_side
        )

        data["strategy_return"] = (
            (1.0 + data["overnight_position"] * overnight_return)
            * (1.0 + data["executed_position"] * intraday_return)
            - 1.0
            - data["trading_cost"]
        )

    if execution.cash_rate_annual != 0.0:
        daily_cash_rate = (
            (1.0 + execution.cash_rate_annual) ** (1.0 / 252.0)
            - 1.0
        )

        data["cash_return"] = (
            (1.0 - data["executed_position"].clip(0.0, 1.0))
            * daily_cash_rate
        )

        data["strategy_return"] = (
            data["strategy_return"] + data["cash_return"]
        )


def run_long_only_backtest(
    frame: pd.DataFrame,
    initial_capital: float = 10_000.0,
    cost_per_side: float = 0.001,
    *,
    execution: ExecutionConfig | None = None,
) -> BacktestResult:
    effective_execution = (
        DEFAULT_EXECUTION
        if execution is None
        else execution
    )

    if initial_capital <= 0:
        raise ValueError(
            "Il capitale iniziale deve essere positivo."
        )

    if cost_per_side < 0:
        raise ValueError(
            "Il costo per lato non può essere negativo."
        )

    required = {
        "timestamp",
        "close",
        "target_position",
    }

    if effective_execution.mode == "next_open":
        required.add("open")

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Colonne mancanti per il backtest: {sorted(missing)}"
        )

    data = (
        frame.sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    if data.empty:
        raise ValueError(
            "Il dataset del backtest è vuoto."
        )

    if data["timestamp"].duplicated().any():
        raise ValueError(
            "Timestamp duplicati nel dataset del simbolo."
        )

    data["asset_return"] = (
        data["close"]
        .pct_change()
        .fillna(0.0)
    )

    if effective_execution.mode == "next_open":
        open_values = pd.to_numeric(
            data["open"],
            errors="coerce",
        ).to_numpy(dtype=float)

        if (
            not np.isfinite(open_values).all()
            or (open_values <= 0.0).any()
        ):
            raise ValueError(
                "La colonna 'open' deve contenere prezzi "
                "positivi e finiti per l'esecuzione next_open."
            )

    # Il target calcolato alla chiusura viene applicato
    # dalla seduta successiva (o più tardi, se c'è ritardo).
    _apply_execution(
        data,
        effective_execution,
        cost_per_side,
    )

    data["equity"] = (
        initial_capital
        * (1.0 + data["strategy_return"]).cumprod()
    )

    data["benchmark_equity"] = (
        initial_capital
        * (1.0 + data["asset_return"]).cumprod()
    )

    observations = max(len(data) - 1, 1)
    years = observations / 252.0

    total_return = (
        data["equity"].iloc[-1]
        / initial_capital
        - 1.0
    )

    benchmark_return = (
        data["benchmark_equity"].iloc[-1]
        / initial_capital
        - 1.0
    )

    cagr = (
        (
            data["equity"].iloc[-1]
            / initial_capital
        )
        ** (1.0 / years)
        - 1.0
        if years > 0 and data["equity"].iloc[-1] > 0
        else float("nan")
    )

    daily_mean = data["strategy_return"].mean()
    daily_std = data["strategy_return"].std(ddof=1)

    sharpe = (
        float(
            daily_mean
            / daily_std
            * sqrt(252.0)
        )
        if daily_std
        and not np.isnan(daily_std)
        else float("nan")
    )

    trade_events = int(
        (data["turnover"] > 0).sum()
    )

    completed_trade_returns = _completed_trade_returns(
        data
    )

    winning_trade_returns = [
        trade_return
        for trade_return in completed_trade_returns
        if trade_return > 0
    ]

    losing_trade_returns = [
        trade_return
        for trade_return in completed_trade_returns
        if trade_return < 0
    ]

    completed_trades = len(
        completed_trade_returns
    )

    winning_trades = len(
        winning_trade_returns
    )

    losing_trades = len(
        losing_trade_returns
    )

    win_rate = (
        winning_trades / completed_trades
        if completed_trades > 0
        else None
    )

    average_trade_return = (
        float(np.mean(completed_trade_returns))
        if completed_trade_returns
        else None
    )

    average_win = (
        float(np.mean(winning_trade_returns))
        if winning_trade_returns
        else None
    )

    average_loss = (
        float(np.mean(losing_trade_returns))
        if losing_trade_returns
        else None
    )

    gross_profit = float(
        sum(winning_trade_returns)
    )

    gross_loss = abs(
        float(sum(losing_trade_returns))
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else None
    )

    holding_lengths = _completed_holding_lengths(
        data["executed_position"]
    )

    average_holding_days = (
        float(np.mean(holding_lengths))
        if holding_lengths
        else None
    )

    annual_turnover = float(
        data["turnover"].sum() / years
    ) if years > 0 else float("nan")

    metrics: dict[str, float | int | None] = {
        "initial_capital": float(initial_capital),
        "final_equity": float(data["equity"].iloc[-1]),
        "total_return": float(total_return),
        "benchmark_total_return": float(
            benchmark_return
        ),
        "cagr": float(cagr),
        "annualized_sharpe": sharpe,
        "max_drawdown": _max_drawdown(
            data["equity"]
        ),
        "trade_events": trade_events,
        "completed_trades": completed_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "average_trade_return": average_trade_return,
        "average_win": average_win,
        "average_loss": average_loss,
        "profit_factor": profit_factor,
        "market_exposure": float(
            data["executed_position"].mean()
        ),
        "cost_per_side": float(cost_per_side),
        "annual_turnover": annual_turnover,
        "average_holding_days": average_holding_days,
        "execution_delay_days": int(
            effective_execution.delay_days
        ),
        "execution_next_open": int(
            effective_execution.mode == "next_open"
        ),
        "cash_rate_annual": float(
            effective_execution.cash_rate_annual
        ),
    }

    return BacktestResult(
        equity_curve=data,
        metrics=metrics,
    )