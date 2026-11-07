from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd

from project_alpha.backtest import (
    BacktestResult,
    ExecutionConfig,
    run_long_only_backtest,
)
from project_alpha.config import (
    Settings,
    StrategyParameters,
)
from project_alpha.data.alpaca import (
    fetch_daily_bars,
)
from project_alpha.indicators import (
    add_indicators,
)
from project_alpha.io_utils import (
    write_csv_atomic,
    write_json_atomic,
)
from project_alpha.strategy import (
    add_baseline_position,
)


def ensure_directories() -> None:
    Path("data/raw").mkdir(
        parents=True,
        exist_ok=True,
    )
    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True,
    )
    Path("reports").mkdir(
        parents=True,
        exist_ok=True,
    )


def fetch_and_save(
    symbols: Iterable[str],
    start: date,
    end: date | None = None,
    output: Path = Path("data/raw/bars.csv"),
) -> pd.DataFrame:
    ensure_directories()

    settings = Settings.from_env()

    frame = fetch_daily_bars(
        settings,
        symbols,
        start,
        end,
    )

    write_csv_atomic(
        frame,
        output,
    )

    return frame


def analyze_and_save(
    input_path: Path = Path("data/raw/bars.csv"),
    output: Path = Path("data/processed/features.csv"),
    parameters: StrategyParameters | None = None,
) -> pd.DataFrame:
    ensure_directories()

    parameters = parameters if parameters is not None else StrategyParameters()

    frame = pd.read_csv(
        input_path,
        parse_dates=["timestamp"],
    )

    enriched = add_indicators(
        frame,
        parameters,
    )

    enriched = add_baseline_position(
        enriched,
        parameters,
    )

    write_csv_atomic(
        enriched,
        output,
    )

    return enriched


def backtest_and_save(
    input_path: Path,
    symbol: str,
    initial_capital: float,
    cost_per_side: float,
    execution: ExecutionConfig | None = None,
) -> BacktestResult:
    ensure_directories()

    frame = pd.read_csv(
        input_path,
        parse_dates=["timestamp"],
    )

    symbol = symbol.upper()

    selected = frame.loc[frame["symbol"].astype(str).str.upper() == symbol].copy()

    if selected.empty:
        raise ValueError(f"Simbolo {symbol} non trovato in {input_path}.")

    result = run_long_only_backtest(
        selected,
        initial_capital=initial_capital,
        cost_per_side=cost_per_side,
        execution=execution,
    )

    equity_path = Path("reports") / f"{symbol}_equity.csv"

    metrics_path = Path("reports") / f"{symbol}_metrics.json"

    # Le metriche vengono serializzate prima di scrivere qualsiasi
    # file: metriche non finite (es. Sharpe senza trade) diventano
    # null invece di lasciare un JSON troncato.
    write_json_atomic(
        metrics_path,
        result.metrics,
    )

    write_csv_atomic(
        result.equity_curve,
        equity_path,
    )

    return result
