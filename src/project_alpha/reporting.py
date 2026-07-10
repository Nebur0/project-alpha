from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd


MetricValue = float | int


def save_batch_summary_csv(
    results: Mapping[str, Mapping[str, MetricValue]],
    output_path: Path = Path("reports/batch_metrics.csv"),
) -> Path:
    if not results:
        raise ValueError("Nessun risultato disponibile per creare il report CSV.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.DataFrame.from_dict(
        results,
        orient="index",
    )

    frame.index.name = "symbol"
    frame = frame.reset_index()

    frame.to_csv(
        output_path,
        index=False,
    )

    return output_path