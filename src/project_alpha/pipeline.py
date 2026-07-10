from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from project_alpha.backtest import BacktestResult, run_long_only_backtest
from project_alpha.config import Settings
from project_alpha.data.alpaca import fetch_daily_bars
from project_alpha.indicators import add_indicators
from project_alpha.strategy import add_baseline_position


def ensure_directories() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)


def fetch_and_save(
    symbols: Iterable[str],
    start: date,
    end: date | None = None,
    output: Path = Path("data/raw/bars.csv"),
) -> pd.DataFrame:
    ensure_directories()
    settings = Settings.from_env()
    frame = fetch_daily_bars(settings, symbols, start, end)
    frame.to_csv(output, index=False)
    return frame


def analyze_and_save(
    input_path: Path = Path("data/raw/bars.csv"),
    output: Path = Path("data/processed/features.csv"),
) -> pd.DataFrame:
    ensure_directories()
    frame = pd.read_csv(input_path, parse_dates=["timestamp"])
    enriched = add_baseline_position(add_indicators(frame))
    enriched.to_csv(output, index=False)
    return enriched


def backtest_and_save(
    input_path: Path,
    symbol: str,
    initial_capital: float,
    cost_per_side: float,
) -> BacktestResult:
    ensure_directories()
    frame = pd.read_csv(input_path, parse_dates=["timestamp"])
    symbol = symbol.upper()
    selected = frame.loc[frame["symbol"].str.upper() == symbol].copy()
    if selected.empty:
        raise ValueError(f"Simbolo {symbol} non trovato in {input_path}.")

    result = run_long_only_backtest(
        selected,
        initial_capital=initial_capital,
        cost_per_side=cost_per_side,
    )

    result.equity_curve.to_csv(Path("reports") / f"{symbol}_equity.csv", index=False)
    with (Path("reports") / f"{symbol}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2, allow_nan=False)

    return result
