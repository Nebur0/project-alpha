from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


@dataclass(frozen=True)
class ValidationSplit:
    in_sample: pd.DataFrame
    out_of_sample: pd.DataFrame
    split_timestamp: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardSplit:
    window_number: int
    in_sample: pd.DataFrame
    out_of_sample: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _to_utc_timestamp(
    value: str | date | datetime | pd.Timestamp,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def split_by_date(
    frame: pd.DataFrame,
    split_date: str | date | datetime | pd.Timestamp,
) -> ValidationSplit:
    if frame.empty:
        raise ValueError("Il dataset da dividere è vuoto.")

    if "timestamp" not in frame.columns:
        raise ValueError("La colonna 'timestamp' è obbligatoria.")

    data = frame.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="raise",
    )

    data = data.sort_values("timestamp").reset_index(drop=True)

    split_timestamp = _to_utc_timestamp(split_date)

    in_sample = data.loc[data["timestamp"] < split_timestamp].copy().reset_index(drop=True)

    out_of_sample = data.loc[data["timestamp"] >= split_timestamp].copy().reset_index(drop=True)

    if in_sample.empty:
        raise ValueError("La porzione in-sample è vuota.")

    if out_of_sample.empty:
        raise ValueError("La porzione out-of-sample è vuota.")

    return ValidationSplit(
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        split_timestamp=split_timestamp,
    )


def generate_walk_forward_splits(
    frame: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[WalkForwardSplit]:
    if frame.empty:
        raise ValueError("Il dataset walk-forward è vuoto.")

    if "timestamp" not in frame.columns:
        raise ValueError("La colonna 'timestamp' è obbligatoria.")

    if train_size <= 0:
        raise ValueError("train_size deve essere positivo.")

    if test_size <= 0:
        raise ValueError("test_size deve essere positivo.")

    if step_size is None:
        step_size = test_size

    if step_size <= 0:
        raise ValueError("step_size deve essere positivo.")

    data = frame.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="raise",
    )

    data = data.sort_values("timestamp").reset_index(drop=True)

    minimum_required_rows = train_size + test_size

    if len(data) < minimum_required_rows:
        raise ValueError(
            "Il dataset non contiene abbastanza righe per creare una finestra walk-forward."
        )

    splits: list[WalkForwardSplit] = []

    train_end_index = train_size
    window_number = 1

    while train_end_index + test_size <= len(data):
        test_end_index = train_end_index + test_size

        in_sample = data.iloc[:train_end_index].copy().reset_index(drop=True)

        out_of_sample = data.iloc[train_end_index:test_end_index].copy().reset_index(drop=True)

        splits.append(
            WalkForwardSplit(
                window_number=window_number,
                in_sample=in_sample,
                out_of_sample=out_of_sample,
                train_start=(in_sample["timestamp"].iloc[0]),
                train_end=(in_sample["timestamp"].iloc[-1]),
                test_start=(out_of_sample["timestamp"].iloc[0]),
                test_end=(out_of_sample["timestamp"].iloc[-1]),
            )
        )

        train_end_index += step_size
        window_number += 1

    return splits
