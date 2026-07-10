from __future__ import annotations

import pandas as pd


def add_baseline_position(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "symbol",
        "timestamp",
        "close",
        "volume",
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "volume_sma_20",
        "previous_high_20",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonne mancanti per la strategia: {sorted(missing)}")

    parts: list[pd.DataFrame] = []

    for _, group in frame.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").copy()

        entry = (
            (group["close"] > group["ema_200"])
            & (group["ema_20"] > group["ema_50"])
            & group["rsi_14"].between(50.0, 70.0, inclusive="both")
            & (group["close"] > group["previous_high_20"])
            & (group["volume"] > group["volume_sma_20"])
        )

        exit_signal = (group["close"] < group["ema_20"]) | (
            group["ema_20"] < group["ema_50"]
        )

        state = 0
        positions: list[int] = []
        entries: list[bool] = []
        exits: list[bool] = []

        for enter_now, exit_now in zip(entry.fillna(False), exit_signal.fillna(False)):
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

    return pd.concat(parts, ignore_index=True).sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)
