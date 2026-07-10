from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from project_alpha.reporting import save_batch_summary_csv

from project_alpha.pipeline import (
    analyze_and_save,
    backtest_and_save,
    fetch_and_save,
)


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Data non valida: {value}. Usare YYYY-MM-DD."
        ) from exc


def _add_backtest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--cost-per-side", type=float, default=0.001)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="project-alpha",
        description="Project Alpha quantitative research MVP",
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    fetch = commands.add_parser(
        "fetch",
        help="Scarica candele giornaliere.",
    )
    fetch.add_argument("--symbols", nargs="+", required=True)
    fetch.add_argument("--start", type=_iso_date, required=True)
    fetch.add_argument("--end", type=_iso_date)
    fetch.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/bars.csv"),
    )

    analyze = commands.add_parser(
        "analyze",
        help="Calcola indicatori e segnali.",
    )
    analyze.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/bars.csv"),
    )
    analyze.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features.csv"),
    )

    backtest = commands.add_parser(
        "backtest",
        help="Esegue il backtest.",
    )
    backtest.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features.csv"),
    )
    _add_backtest_arguments(backtest)

    batch = commands.add_parser(
        "batch-backtest",
        help="Esegue il backtest su più simboli.",
    )
    batch.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features.csv"),
    )
    batch.add_argument("--symbols", nargs="+", required=True)
    batch.add_argument("--initial-capital", type=float, default=10_000.0)
    batch.add_argument("--cost-per-side", type=float, default=0.001)

    run = commands.add_parser(
        "run",
        help="Esegue fetch, analisi e backtest.",
    )
    run.add_argument("--symbols", nargs="+", required=True)
    run.add_argument("--start", type=_iso_date, required=True)
    run.add_argument("--end", type=_iso_date)
    run.add_argument(
        "--symbol",
        help="Simbolo da testare; predefinito: il primo in --symbols.",
    )
    run.add_argument("--initial-capital", type=float, default=10_000.0)
    run.add_argument("--cost-per-side", type=float, default=0.001)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "fetch":
        frame = fetch_and_save(
            args.symbols,
            args.start,
            args.end,
            args.output,
        )

        print(f"Salvate {len(frame):,} righe in {args.output}")
        return

    if args.command == "analyze":
        frame = analyze_and_save(
            args.input,
            args.output,
        )

        print(f"Salvate {len(frame):,} righe in {args.output}")
        return

    if args.command == "backtest":
        result = backtest_and_save(
            args.input,
            args.symbol,
            args.initial_capital,
            args.cost_per_side,
        )

        print(
            json.dumps(
                result.metrics,
                indent=2,
                allow_nan=False,
            )
        )
        return

    if args.command == "batch-backtest":
        results = {}

        for symbol in args.symbols:
            normalized_symbol = symbol.upper()

            result = backtest_and_save(
                args.input,
                normalized_symbol,
                args.initial_capital,
                args.cost_per_side,
            )

            results[normalized_symbol] = result.metrics

        report_path = Path("reports/batch_metrics.json")
        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_content = json.dumps(
            results,
            indent=2,
            allow_nan=False,
        )

        report_path.write_text(
        report_content,
            encoding="utf-8",
)

        csv_report_path = save_batch_summary_csv(results)

        print(report_content)
        print(f"\nReport JSON salvato in {report_path}")
        print(f"Report CSV salvato in {csv_report_path}")
        return  

    if args.command == "run":
        raw_path = Path("data/raw/bars.csv")
        processed_path = Path("data/processed/features.csv")

        fetch_and_save(
            args.symbols,
            args.start,
            args.end,
            raw_path,
        )

        analyze_and_save(
            raw_path,
            processed_path,
        )

        symbol = (args.symbol or args.symbols[0]).upper()

        result = backtest_and_save(
            processed_path,
            symbol,
            args.initial_capital,
            args.cost_per_side,
        )

        print(
            json.dumps(
                result.metrics,
                indent=2,
                allow_nan=False,
            )
        )
        return

    raise RuntimeError(
        f"Comando non gestito: {args.command}"
    )