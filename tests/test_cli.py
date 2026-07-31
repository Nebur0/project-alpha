from datetime import date
from pathlib import Path

import pytest

from project_alpha.cli import build_parser


def test_batch_validate_parser_reads_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "batch-validate",
            "--input",
            "data/processed/features.csv",
            "--symbols",
            "SPY",
            "QQQ",
            "AAPL",
            "--split-date",
            "2023-01-01",
            "--initial-capital",
            "15000",
            "--cost-per-side",
            "0.002",
        ]
    )

    assert args.command == "batch-validate"
    assert args.input == Path(
        "data/processed/features.csv"
    )
    assert args.symbols == [
        "SPY",
        "QQQ",
        "AAPL",
    ]
    assert args.split_date == date(
        2023,
        1,
        1,
    )
    assert args.initial_capital == pytest.approx(
        15_000.0
    )
    assert args.cost_per_side == pytest.approx(
        0.002
    )


def test_batch_validate_requires_symbols() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "batch-validate",
                "--split-date",
                "2023-01-01",
            ]
        )