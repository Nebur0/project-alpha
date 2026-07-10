from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


@dataclass(frozen=True)
class ValidationSplit:
    in_sample: pd.DataFrame
    out_of_sample: pd.DataFrame
    split_timestamp: pd.Timestamp


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
        raise ValueError(
            "Il dataset da dividere è vuoto."
        )

    if "timestamp" not in frame.columns:
        raise ValueError(
            "La colonna 'timestamp' è obbligatoria."
        )

    data = frame.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="raise",
    )

    data = (
        data.sort_values("timestamp")
        .reset_index(drop=True)
    )

    split_timestamp = _to_utc_timestamp(split_date)

    in_sample = (
        data.loc[
            data["timestamp"] < split_timestamp
        ]
        .copy()
        .reset_index(drop=True)
    )

    out_of_sample = (
        data.loc[
            data["timestamp"] >= split_timestamp
        ]
        .copy()
        .reset_index(drop=True)
    )

    if in_sample.empty:
        raise ValueError(
            "La porzione in-sample è vuota."
        )

    if out_of_sample.empty:
        raise ValueError(
            "La porzione out-of-sample è vuota."
        )

    return ValidationSplit(
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        split_timestamp=split_timestamp,
    )