from __future__ import annotations

import pandas as pd

from project_alpha.config import StrategyParameters


def add_baseline_position(
    frame: pd.DataFrame,
    parameters: StrategyParameters | None = None,
) -> pd.DataFrame:
    parameters = parameters if parameters is not None else StrategyParameters()

    fast_ema_column = f"ema_{parameters.fast_ema_period}"
    slow_ema_column = f"ema_{parameters.slow_ema_period}"
    trend_ema_column = f"ema_{parameters.trend_ema_period}"
    rsi_column = f"rsi_{parameters.rsi_period}"
    volume_column = f"volume_sma_{parameters.volume_period}"
    breakout_column = f"previous_high_{parameters.breakout_period}"

    required = {
        "symbol",
        "timestamp",
        "close",
        "volume",
        fast_ema_column,
        slow_ema_column,
        trend_ema_column,
        rsi_column,
        volume_column,
        breakout_column,
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"Colonne mancanti per la strategia: {sorted(missing)}")

    parts: list[pd.DataFrame] = []

    for _, group in frame.groupby(
        "symbol",
        sort=False,
    ):
        group = group.sort_values("timestamp").copy()

        entry = (
            (group["close"] > group[trend_ema_column])
            & (group[fast_ema_column] > group[slow_ema_column])
            & group[rsi_column].between(
                parameters.rsi_lower,
                parameters.rsi_upper,
                inclusive="both",
            )
            & (group["close"] > group[breakout_column])
            & (group["volume"] > group[volume_column])
        )

        exit_signal = (group["close"] < group[fast_ema_column]) | (
            group[fast_ema_column] < group[slow_ema_column]
        )

        state = 0
        positions: list[int] = []
        entries: list[bool] = []
        exits: list[bool] = []

        for enter_now, exit_now in zip(
            entry.fillna(False),
            exit_signal.fillna(False),
            strict=True,
        ):
            did_enter = False
            did_exit = False

            if state == 0 and bool(enter_now):
                state = 1
                did_enter = True

            elif state == 1 and bool(exit_now):
                state = 0
                did_exit = True

            positions.append(state)
            entries.append(did_enter)
            exits.append(did_exit)

        group["entry_signal"] = entries
        group["exit_signal"] = exits
        group["target_position"] = positions

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
