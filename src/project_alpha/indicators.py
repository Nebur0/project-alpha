from __future__ import annotations

import numpy as np
import pandas as pd

from project_alpha.config import StrategyParameters


def ema(
    series: pd.Series,
    span: int,
) -> pd.Series:
    if span <= 0:
        raise ValueError(
            "Lo span EMA deve essere positivo."
        )

    return series.ewm(
        span=span,
        adjust=False,
        min_periods=span,
    ).mean()


def rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:
    if period <= 0:
        raise ValueError(
            "Il periodo RSI deve essere positivo."
        )

    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = (
        avg_gain
        / avg_loss.replace(0.0, np.nan)
    )

    output = (
        100.0
        - (
            100.0
            / (1.0 + relative_strength)
        )
    )

    output = output.where(
        avg_loss.ne(0.0),
        100.0,
    )

    output = output.where(
        avg_gain.ne(0.0),
        0.0,
    )

    return output


def atr(
    frame: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    if period <= 0:
        raise ValueError(
            "Il periodo ATR deve essere positivo."
        )

    previous_close = frame["close"].shift(1)

    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (
                frame["high"]
                - previous_close
            ).abs(),
            (
                frame["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def add_indicators(
    frame: pd.DataFrame,
    parameters: StrategyParameters | None = None,
) -> pd.DataFrame:
    required = {
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"Colonne mancanti: {sorted(missing)}"
        )

    parameters = (
        parameters
        if parameters is not None
        else StrategyParameters()
    )

    fast_ema_column = (
        f"ema_{parameters.fast_ema_period}"
    )
    slow_ema_column = (
        f"ema_{parameters.slow_ema_period}"
    )
    trend_ema_column = (
        f"ema_{parameters.trend_ema_period}"
    )
    rsi_column = (
        f"rsi_{parameters.rsi_period}"
    )
    atr_column = (
        f"atr_{parameters.atr_period}"
    )
    volume_column = (
        f"volume_sma_{parameters.volume_period}"
    )
    breakout_column = (
        f"previous_high_"
        f"{parameters.breakout_period}"
    )

    parts: list[pd.DataFrame] = []

    for _, group in frame.groupby(
        "symbol",
        sort=False,
    ):
        group = (
            group.sort_values("timestamp")
            .copy()
        )

        group[fast_ema_column] = ema(
            group["close"],
            parameters.fast_ema_period,
        )

        group[slow_ema_column] = ema(
            group["close"],
            parameters.slow_ema_period,
        )

        group[trend_ema_column] = ema(
            group["close"],
            parameters.trend_ema_period,
        )

        group[rsi_column] = rsi(
            group["close"],
            parameters.rsi_period,
        )

        group[atr_column] = atr(
            group,
            parameters.atr_period,
        )

        group[volume_column] = (
            group["volume"]
            .rolling(
                parameters.volume_period,
                min_periods=(
                    parameters.volume_period
                ),
            )
            .mean()
        )

        group[breakout_column] = (
            group["high"]
            .rolling(
                parameters.breakout_period,
                min_periods=(
                    parameters.breakout_period
                ),
            )
            .max()
            .shift(1)
        )

        parts.append(group)

    return (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .sort_values(
            [
                "symbol",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )