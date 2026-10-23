from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from numbers import Real
from typing import Iterable

import pandas as pd


REGIME_UP_LOW_VOL = "UP_LOW_VOL"
REGIME_UP_HIGH_VOL = "UP_HIGH_VOL"
REGIME_DOWN_LOW_VOL = "DOWN_LOW_VOL"
REGIME_DOWN_HIGH_VOL = "DOWN_HIGH_VOL"


@dataclass(frozen=True)
class RegimeAnalysisConfig:
    annualization_factor: int = 252
    high_volatility_threshold: float = 0.25

    def __post_init__(self) -> None:
        if (
            isinstance(self.annualization_factor, bool)
            or not isinstance(self.annualization_factor, int)
            or self.annualization_factor <= 0
        ):
            raise ValueError(
                "annualization_factor deve essere un intero positivo."
            )

        threshold = self.high_volatility_threshold

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not isfinite(float(threshold))
            or float(threshold) < 0.0
        ):
            raise ValueError(
                "high_volatility_threshold deve essere un numero finito non negativo."
            )


@dataclass(frozen=True)
class RegimeWindowResult:
    symbol: str
    window_number: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    regime: str
    market_direction: str
    volatility_regime: str
    benchmark_total_return: float
    annualized_realized_volatility: float
    strategy_total_return: float
    excess_return: float
    annualized_sharpe: float | None
    max_drawdown: float | None
    completed_trades: int | None


@dataclass(frozen=True)
class RegimeSummary:
    regime: str
    window_count: int
    positive_window_count: int
    positive_window_rate: float
    benchmark_beat_count: int
    benchmark_beat_rate: float
    mean_strategy_total_return: float
    median_strategy_total_return: float
    mean_benchmark_total_return: float
    mean_excess_return: float
    mean_annualized_sharpe: float | None
    worst_max_drawdown: float | None
    total_completed_trades: int


def _required_numeric(
    value: object,
    *,
    name: str,
) -> float:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} deve essere numerico."
        )

    numeric_value = float(value)

    if not isfinite(numeric_value):
        raise ValueError(
            f"{name} deve essere finito."
        )

    return numeric_value


def _optional_numeric(
    value: object,
    *,
    name: str,
) -> float | None:
    if value is None or pd.isna(value):
        return None

    return _required_numeric(
        value,
        name=name,
    )


def _optional_non_negative_int(
    value: object,
    *,
    name: str,
) -> int | None:
    if value is None or pd.isna(value):
        return None

    numeric_value = _required_numeric(
        value,
        name=name,
    )

    if (
        numeric_value < 0.0
        or not numeric_value.is_integer()
    ):
        raise ValueError(
            f"{name} deve essere un intero non negativo."
        )

    return int(numeric_value)


def calculate_annualized_realized_volatility(
    frame: pd.DataFrame,
    *,
    close_column: str = "close",
    annualization_factor: int = 252,
) -> float:
    if frame.empty:
        raise ValueError(
            "La finestra di mercato è vuota."
        )

    if close_column not in frame.columns:
        raise ValueError(
            f"Colonna '{close_column}' mancante."
        )

    if (
        isinstance(annualization_factor, bool)
        or not isinstance(annualization_factor, int)
        or annualization_factor <= 0
    ):
        raise ValueError(
            "annualization_factor deve essere un intero positivo."
        )

    close = pd.to_numeric(
        frame[close_column],
        errors="coerce",
    )

    if close.isna().any():
        raise ValueError(
            "La serie close contiene valori mancanti o non numerici."
        )

    if (close <= 0.0).any():
        raise ValueError(
            "La serie close deve contenere solo valori positivi."
        )

    returns = close.pct_change().dropna()

    if len(returns) < 2:
        return 0.0

    volatility = (
        float(returns.std(ddof=1))
        * sqrt(float(annualization_factor))
    )

    if not isfinite(volatility):
        raise ValueError(
            "Volatilità annualizzata non finita."
        )

    return volatility


def classify_market_regime(
    benchmark_total_return: Real,
    annualized_realized_volatility: Real,
    config: RegimeAnalysisConfig | None = None,
) -> tuple[str, str, str]:
    effective_config = (
        config
        if config is not None
        else RegimeAnalysisConfig()
    )

    benchmark_return = _required_numeric(
        benchmark_total_return,
        name="benchmark_total_return",
    )
    realized_volatility = _required_numeric(
        annualized_realized_volatility,
        name="annualized_realized_volatility",
    )

    if realized_volatility < 0.0:
        raise ValueError(
            "annualized_realized_volatility non può essere negativa."
        )

    market_direction = (
        "UP"
        if benchmark_return >= 0.0
        else "DOWN"
    )

    volatility_regime = (
        "HIGH_VOL"
        if realized_volatility
        >= effective_config.high_volatility_threshold
        else "LOW_VOL"
    )

    regime = (
        f"{market_direction}_{volatility_regime}"
    )

    return (
        regime,
        market_direction,
        volatility_regime,
    )


def _validate_market_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "symbol",
        "timestamp",
        "close",
    }

    missing_columns = (
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colonne mancanti nel market frame: "
            f"{sorted(missing_columns)}"
        )

    if frame.empty:
        raise ValueError(
            "Il market frame è vuoto."
        )

    result = frame.copy()
    result["symbol"] = (
        result["symbol"].astype(str).str.upper()
    )
    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
    )

    return result.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)


def _validate_optimization_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "symbol",
        "window_number",
        "test_start",
        "test_end",
        "out_of_sample_total_return",
        "out_of_sample_benchmark_total_return",
        "out_of_sample_annualized_sharpe",
        "out_of_sample_max_drawdown",
        "out_of_sample_completed_trades",
    }

    missing_columns = (
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colonne mancanti nel report walk-forward: "
            f"{sorted(missing_columns)}"
        )

    if frame.empty:
        raise ValueError(
            "Il report walk-forward è vuoto."
        )

    result = frame.copy()
    result["symbol"] = (
        result["symbol"].astype(str).str.upper()
    )
    result["test_start"] = pd.to_datetime(
        result["test_start"],
        utc=True,
    )
    result["test_end"] = pd.to_datetime(
        result["test_end"],
        utc=True,
    )

    return result.sort_values(
        ["symbol", "window_number"],
        kind="stable",
    ).reset_index(drop=True)


def analyze_walk_forward_regimes(
    market_frame: pd.DataFrame,
    optimization_frame: pd.DataFrame,
    config: RegimeAnalysisConfig | None = None,
) -> tuple[RegimeWindowResult, ...]:
    effective_config = (
        config
        if config is not None
        else RegimeAnalysisConfig()
    )

    market = _validate_market_frame(market_frame)
    optimization = _validate_optimization_frame(
        optimization_frame
    )

    results: list[RegimeWindowResult] = []

    available_symbols = set(
        market["symbol"].unique()
    )

    for row in optimization.itertuples(index=False):
        symbol = str(row.symbol).upper()

        if symbol not in available_symbols:
            raise ValueError(
                f"Simbolo {symbol} assente dal market frame."
            )

        window_number = _optional_non_negative_int(
            row.window_number,
            name="window_number",
        )

        if (
            window_number is None
            or window_number <= 0
        ):
            raise ValueError(
                "window_number deve essere un intero positivo."
            )

        test_start = pd.Timestamp(row.test_start)
        test_end = pd.Timestamp(row.test_end)

        if test_end < test_start:
            raise ValueError(
                "test_end non può precedere test_start."
            )

        window_market = market.loc[
            market["symbol"].eq(symbol)
            & market["timestamp"].ge(test_start)
            & market["timestamp"].le(test_end)
        ]

        if window_market.empty:
            raise ValueError(
                "Nessun dato di mercato per "
                f"{symbol}, window {window_number}."
            )

        realized_volatility = (
            calculate_annualized_realized_volatility(
                window_market,
                annualization_factor=(
                    effective_config.annualization_factor
                ),
            )
        )

        benchmark_total_return = _required_numeric(
            row.out_of_sample_benchmark_total_return,
            name="out_of_sample_benchmark_total_return",
        )
        strategy_total_return = _required_numeric(
            row.out_of_sample_total_return,
            name="out_of_sample_total_return",
        )

        (
            regime,
            market_direction,
            volatility_regime,
        ) = classify_market_regime(
            benchmark_total_return,
            realized_volatility,
            effective_config,
        )

        results.append(
            RegimeWindowResult(
                symbol=symbol,
                window_number=window_number,
                test_start=test_start,
                test_end=test_end,
                regime=regime,
                market_direction=market_direction,
                volatility_regime=volatility_regime,
                benchmark_total_return=benchmark_total_return,
                annualized_realized_volatility=(
                    realized_volatility
                ),
                strategy_total_return=strategy_total_return,
                excess_return=(
                    strategy_total_return
                    - benchmark_total_return
                ),
                annualized_sharpe=_optional_numeric(
                    row.out_of_sample_annualized_sharpe,
                    name="out_of_sample_annualized_sharpe",
                ),
                max_drawdown=_optional_numeric(
                    row.out_of_sample_max_drawdown,
                    name="out_of_sample_max_drawdown",
                ),
                completed_trades=_optional_non_negative_int(
                    row.out_of_sample_completed_trades,
                    name="out_of_sample_completed_trades",
                ),
            )
        )

    return tuple(results)


def regime_window_results_to_frame(
    results: Iterable[RegimeWindowResult],
) -> pd.DataFrame:
    return pd.DataFrame(
        asdict(result)
        for result in results
    )


def summarize_regime_results(
    results: Iterable[RegimeWindowResult],
) -> tuple[RegimeSummary, ...]:
    items = tuple(results)

    if not items:
        return ()

    frame = regime_window_results_to_frame(items)

    summaries: list[RegimeSummary] = []

    for regime, group in frame.groupby(
        "regime",
        sort=True,
    ):
        strategy_returns = pd.to_numeric(
            group["strategy_total_return"],
            errors="raise",
        )
        benchmark_returns = pd.to_numeric(
            group["benchmark_total_return"],
            errors="raise",
        )
        excess_returns = pd.to_numeric(
            group["excess_return"],
            errors="raise",
        )

        positive_window_count = int(
            (strategy_returns > 0.0).sum()
        )
        benchmark_beat_count = int(
            (excess_returns > 0.0).sum()
        )
        window_count = int(len(group))

        sharpe_values = pd.to_numeric(
            group["annualized_sharpe"],
            errors="coerce",
        ).dropna()

        drawdown_values = pd.to_numeric(
            group["max_drawdown"],
            errors="coerce",
        ).dropna()

        trade_values = pd.to_numeric(
            group["completed_trades"],
            errors="coerce",
        ).dropna()

        summaries.append(
            RegimeSummary(
                regime=str(regime),
                window_count=window_count,
                positive_window_count=positive_window_count,
                positive_window_rate=(
                    positive_window_count / window_count
                ),
                benchmark_beat_count=benchmark_beat_count,
                benchmark_beat_rate=(
                    benchmark_beat_count / window_count
                ),
                mean_strategy_total_return=float(
                    strategy_returns.mean()
                ),
                median_strategy_total_return=float(
                    strategy_returns.median()
                ),
                mean_benchmark_total_return=float(
                    benchmark_returns.mean()
                ),
                mean_excess_return=float(
                    excess_returns.mean()
                ),
                mean_annualized_sharpe=(
                    float(sharpe_values.mean())
                    if not sharpe_values.empty
                    else None
                ),
                worst_max_drawdown=(
                    float(drawdown_values.min())
                    if not drawdown_values.empty
                    else None
                ),
                total_completed_trades=(
                    int(trade_values.sum())
                    if not trade_values.empty
                    else 0
                ),
            )
        )

    return tuple(summaries)


def regime_summaries_to_frame(
    summaries: Iterable[RegimeSummary],
) -> pd.DataFrame:
    return pd.DataFrame(
        asdict(summary)
        for summary in summaries
    )
