from pathlib import Path

import pandas as pd
import pytest

from project_alpha.reporting import save_batch_summary_csv


def test_save_batch_summary_csv(tmp_path: Path) -> None:
    results = {
        "SPY": {
            "initial_capital": 10_000.0,
            "total_return": 0.10,
        },
        "QQQ": {
            "initial_capital": 10_000.0,
            "total_return": 0.25,
        },
    }

    output_path = tmp_path / "batch_metrics.csv"

    returned_path = save_batch_summary_csv(
        results,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()

    frame = pd.read_csv(output_path)

    assert frame["symbol"].tolist() == ["SPY", "QQQ"]
    assert frame.loc[0, "total_return"] == pytest.approx(0.10)
    assert frame.loc[1, "total_return"] == pytest.approx(0.25)


def test_save_batch_summary_csv_rejects_empty_results(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "batch_metrics.csv"

    with pytest.raises(
        ValueError,
        match="Nessun risultato",
    ):
        save_batch_summary_csv({}, output_path)