from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    if span <= 0:
        raise ValueError("Lo span EMA deve essere positivo.")
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if period <= 0:
        raise ValueError("Il periodo RSI deve essere positivo.")

    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    output = 100.0 - (100.0 / (1.0 + relative_strength))

    output = output.where(avg_loss.ne(0.0), 100.0)
    output = output.where(avg_gain.ne(0.0), 0.0)
    return output


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    if period <= 0:
        raise ValueError("Il periodo ATR deve essere positivo.")

    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonne mancanti: {sorted(missing)}")

    parts: list[pd.DataFrame] = []

    for _, group in frame.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").copy()
        group["ema_20"] = ema(group["close"], 20)
        group["ema_50"] = ema(group["close"], 50)
        group["ema_200"] = ema(group["close"], 200)
        group["rsi_14"] = rsi(group["close"], 14)
        group["atr_14"] = atr(group, 14)
        group["volume_sma_20"] = group["volume"].rolling(20, min_periods=20).mean()
        group["previous_high_20"] = (
            group["high"].rolling(20, min_periods=20).max().shift(1)
        )
        parts.append(group)

    return pd.concat(parts, ignore_index=True).sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)
