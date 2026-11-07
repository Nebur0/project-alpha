"""Serializzazione JSON sicura e scritture atomiche dei report.

Due problemi risolti qui, in un unico punto:

1. Metriche non finite (NaN, +/-inf), ad esempio lo Sharpe di una
   strategia che non entra mai a mercato, rendevano impossibile
   ``json.dumps(..., allow_nan=False)``. Ora diventano ``null``.
2. I report venivano scritti direttamente sul file di destinazione:
   un errore a metà serializzazione lasciava un JSON troncato.
   Ora si scrive su un file temporaneo nella stessa cartella e lo si
   rinomina solo a scrittura completata (``os.replace`` è atomico).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_json_safe(value: Any) -> Any:
    """Converte ricorsivamente ``value`` in strutture serializzabili.

    - float non finiti -> ``None``
    - scalari numpy -> tipi Python nativi
    - timestamp/date -> stringa ISO 8601
    - tuple/set -> liste
    """
    if value is None or isinstance(value, (bool, str)):
        return value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return [to_json_safe(item) for item in value.tolist()]

    raise TypeError(
        f"Tipo non serializzabile in JSON: {type(value).__name__}."
    )


def dumps_json(
    value: Any,
    *,
    indent: int | None = 2,
    allow_nan: bool = False,
) -> str:
    """``json.dumps`` che non fallisce su metriche non finite.

    ``allow_nan`` resta ``False``: dopo la conversione non possono
    esserci NaN, quindi l'output è sempre JSON standard.
    """
    return json.dumps(
        to_json_safe(value),
        indent=indent,
        allow_nan=allow_nan,
    )


def write_text_atomic(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Scrive ``content`` in ``path`` in modo atomico."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )

    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    return target


def write_json_atomic(
    path: Path | str,
    value: Any,
    *,
    indent: int | None = 2,
) -> Path:
    """Serializza ``value`` (prima di toccare il disco) e lo scrive atomicamente."""
    content = dumps_json(value, indent=indent)
    return write_text_atomic(path, content)


def write_csv_atomic(
    frame: pd.DataFrame,
    path: Path | str,
    *,
    index: bool = False,
) -> Path:
    """Scrive un DataFrame in CSV in modo atomico."""
    return write_text_atomic(path, frame.to_csv(index=index))
