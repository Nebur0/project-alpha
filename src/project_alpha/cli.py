from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from project_alpha.backtest import run_long_only_backtest
from project_alpha.config import StrategyParameters
from project_alpha.pipeline import (
    analyze_and_save,
    backtest_and_save,
    fetch_and_save,
)

from project_alpha.reporting import (
    sanitize_metric_values,
    save_batch_summary_csv,
    summarize_walk_forward_metrics,
)

from project_alpha.validation import (
    generate_walk_forward_splits,
    split_by_date,
)


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Data non valida: {value}. Usare YYYY-MM-DD."
        ) from exc


def _add_backtest_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--symbol",
        required=True,
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
    )
    parser.add_argument(
        "--cost-per-side",
        type=float,
        default=0.001,
    )


def _add_strategy_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--fast-ema-period",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--slow-ema-period",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--trend-ema-period",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--rsi-period",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--rsi-lower",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--rsi-upper",
        type=float,
        default=70.0,
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--volume-period",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--breakout-period",
        type=int,
        default=20,
    )


def _strategy_parameters_from_args(
    args: argparse.Namespace,
) -> StrategyParameters:
    return StrategyParameters(
        fast_ema_period=args.fast_ema_period,
        slow_ema_period=args.slow_ema_period,
        trend_ema_period=args.trend_ema_period,
        rsi_period=args.rsi_period,
        rsi_lower=args.rsi_lower,
        rsi_upper=args.rsi_upper,
        atr_period=args.atr_period,
        volume_period=args.volume_period,
        breakout_period=args.breakout_period,
    )


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
    fetch.add_argument(
        "--symbols",
        nargs="+",
        required=True,
    )
    fetch.add_argument(
        "--start",
        type=_iso_date,
        required=True,
    )
    fetch.add_argument(
        "--end",
        type=_iso_date,
    )
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
    _add_strategy_arguments(analyze)

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
    batch.add_argument(
        "--symbols",
        nargs="+",
        required=True,
    )
    batch.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
    )
    batch.add_argument(
        "--cost-per-side",
        type=float,
        default=0.001,
    )

    validate = commands.add_parser(
        "validate",
        help="Confronta risultati in-sample e out-of-sample.",
    )
    validate.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features.csv"),
    )
    validate.add_argument(
        "--symbol",
        required=True,
    )
    validate.add_argument(
        "--split-date",
        type=_iso_date,
        required=True,
    )
    validate.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
    )
    validate.add_argument(
        "--cost-per-side",
        type=float,
        default=0.001,
    )

    batch_validate = commands.add_parser(
        "batch-validate",
        help="Valida più simboli in-sample e out-of-sample.",
    )
    batch_validate.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features.csv"),
    )
    batch_validate.add_argument(
        "--symbols",
        nargs="+",
        required=True,
    )
    batch_validate.add_argument(
        "--split-date",
        type=_iso_date,
        required=True,
    )
    batch_validate.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
    )
    batch_validate.add_argument(
        "--cost-per-side",
        type=float,
        default=0.001,
    )

    batch_walk_forward = commands.add_parser(
        "batch-walk-forward",
        help="Esegue la validazione walk-forward su più simboli.",
    )
    batch_walk_forward.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features.csv"),
    )
    batch_walk_forward.add_argument(
        "--symbols",
        nargs="+",
        required=True,
    )
    batch_walk_forward.add_argument(
        "--train-size",
        type=int,
        required=True,
        help="Numero di righe nella prima finestra in-sample.",
    )
    batch_walk_forward.add_argument(
        "--test-size",
        type=int,
        required=True,
        help="Numero di righe in ogni finestra out-of-sample.",
    )
    batch_walk_forward.add_argument(
        "--step-size",
        type=int,
        help=(
            "Avanzamento tra le finestre; "
            "predefinito: test-size."
        ),
    )
    batch_walk_forward.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
    )
    batch_walk_forward.add_argument(
        "--cost-per-side",
        type=float,
        default=0.001,
    )

    walk_forward = commands.add_parser(
        "walk-forward",
        help="Esegue una validazione walk-forward su un simbolo.",
    )
    walk_forward.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features.csv"),
    )
    walk_forward.add_argument(
        "--symbol",
        required=True,
    )
    walk_forward.add_argument(
        "--train-size",
        type=int,
        required=True,
        help="Numero di righe nella prima finestra in-sample.",
    )
    walk_forward.add_argument(
        "--test-size",
        type=int,
        required=True,
        help="Numero di righe in ogni finestra out-of-sample.",
    )
    walk_forward.add_argument(
        "--step-size",
        type=int,
        help=(
            "Avanzamento tra le finestre; "
            "predefinito: test-size."
        ),
    )
    walk_forward.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
    )
    walk_forward.add_argument(
        "--cost-per-side",
        type=float,
        default=0.001,
    )

    run = commands.add_parser(
        "run",
        help="Esegue fetch, analisi e backtest.",
    )
    run.add_argument(
        "--symbols",
        nargs="+",
        required=True,
    )
    run.add_argument(
        "--start",
        type=_iso_date,
        required=True,
    )
    run.add_argument(
        "--end",
        type=_iso_date,
    )
    run.add_argument(
        "--symbol",
        help="Simbolo da testare; predefinito: il primo in --symbols.",
    )
    run.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
    )
    run.add_argument(
        "--cost-per-side",
        type=float,
        default=0.001,
    )
    _add_strategy_arguments(run)

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
        parameters = (
            _strategy_parameters_from_args(args)
        )

        frame = analyze_and_save(
            args.input,
            args.output,
            parameters=parameters,
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

        report_path = Path(
            "reports/batch_metrics.json"
        )
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

        csv_report_path = save_batch_summary_csv(
            results
        )

        print(report_content)
        print(
            f"\nReport JSON salvato in {report_path}"
        )
        print(
            f"Report CSV salvato in {csv_report_path}"
        )
        return

    if args.command == "validate":
        frame = pd.read_csv(
            args.input,
            parse_dates=["timestamp"],
        )

        symbol = args.symbol.upper()

        selected = frame.loc[
            frame["symbol"].astype(str).str.upper()
            == symbol
        ].copy()

        if selected.empty:
            raise ValueError(
                f"Simbolo {symbol} non trovato in "
                f"{args.input}."
            )

        validation_split = split_by_date(
            selected,
            args.split_date,
        )

        in_sample_result = run_long_only_backtest(
            validation_split.in_sample,
            initial_capital=args.initial_capital,
            cost_per_side=args.cost_per_side,
        )

        out_of_sample_result = run_long_only_backtest(
            validation_split.out_of_sample,
            initial_capital=args.initial_capital,
            cost_per_side=args.cost_per_side,
        )

        report = {
            "symbol": symbol,
            "split_date": (
                validation_split
                .split_timestamp
                .isoformat()
            ),
            "in_sample": in_sample_result.metrics,
            "out_of_sample": (
                out_of_sample_result.metrics
            ),
        }

        report_path = Path(
            f"reports/{symbol}_validation.json"
        )
        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_content = json.dumps(
            report,
            indent=2,
            allow_nan=False,
        )

        report_path.write_text(
            report_content,
            encoding="utf-8",
        )

        print(report_content)
        print(
            f"\nReport salvato in {report_path}"
        )
        return

    if args.command == "batch-validate":
        frame = pd.read_csv(
            args.input,
            parse_dates=["timestamp"],
        )

        normalized_symbols = list(
            dict.fromkeys(
                symbol.upper()
                for symbol in args.symbols
            )
        )

        results = {}
        summary_rows = []

        for symbol in normalized_symbols:
            selected = frame.loc[
                frame["symbol"].astype(str).str.upper()
                == symbol
            ].copy()

            if selected.empty:
                raise ValueError(
                    f"Simbolo {symbol} non trovato in "
                    f"{args.input}."
                )

            validation_split = split_by_date(
                selected,
                args.split_date,
            )

            in_sample_result = (
                run_long_only_backtest(
                    validation_split.in_sample,
                    initial_capital=(
                        args.initial_capital
                    ),
                    cost_per_side=(
                        args.cost_per_side
                    ),
                )
            )

            out_of_sample_result = (
                run_long_only_backtest(
                    validation_split.out_of_sample,
                    initial_capital=(
                        args.initial_capital
                    ),
                    cost_per_side=(
                        args.cost_per_side
                    ),
                )
            )

            split_timestamp = (
                validation_split
                .split_timestamp
                .isoformat()
            )

            results[symbol] = {
                "split_date": split_timestamp,
                "in_sample": (
                    in_sample_result.metrics
                ),
                "out_of_sample": (
                    out_of_sample_result.metrics
                ),
            }

            summary_rows.append(
                {
                    "symbol": symbol,
                    "sample": "in_sample",
                    "split_date": split_timestamp,
                    **in_sample_result.metrics,
                }
            )

            summary_rows.append(
                {
                    "symbol": symbol,
                    "sample": "out_of_sample",
                    "split_date": split_timestamp,
                    **out_of_sample_result.metrics,
                }
            )

        json_report_path = Path(
            "reports/batch_validation.json"
        )
        csv_report_path = Path(
            "reports/batch_validation.csv"
        )

        json_report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_content = json.dumps(
            results,
            indent=2,
            allow_nan=False,
        )

        json_report_path.write_text(
            report_content,
            encoding="utf-8",
        )

        pd.DataFrame(
            summary_rows
        ).to_csv(
            csv_report_path,
            index=False,
        )

        print(report_content)
        print(
            "\nReport JSON salvato in "
            f"{json_report_path}"
        )
        print(
            "Report CSV salvato in "
            f"{csv_report_path}"
        )
        return

    if args.command == "batch-walk-forward":
        frame = pd.read_csv(
            args.input,
            parse_dates=["timestamp"],
        )

        normalized_symbols = list(
            dict.fromkeys(
                symbol.upper()
                for symbol in args.symbols
            )
        )

        effective_step_size = (
            args.test_size
            if args.step_size is None
            else args.step_size
        )

        windows_overlap = (
            effective_step_size
            < args.test_size
        )

        results = {}
        comparison_rows = []

        for symbol in normalized_symbols:
            selected = frame.loc[
                frame["symbol"].astype(str).str.upper()
                == symbol
            ].copy()

            if selected.empty:
                raise ValueError(
                    f"Simbolo {symbol} non trovato in "
                    f"{args.input}."
                )

            splits = generate_walk_forward_splits(
                frame=selected,
                train_size=args.train_size,
                test_size=args.test_size,
                step_size=args.step_size,
            )

            windows = []
            out_of_sample_metrics = []

            for split in splits:
                in_sample_result = (
                    run_long_only_backtest(
                        split.in_sample,
                        initial_capital=(
                            args.initial_capital
                        ),
                        cost_per_side=(
                            args.cost_per_side
                        ),
                    )
                )

                out_of_sample_result = (
                    run_long_only_backtest(
                        split.out_of_sample,
                        initial_capital=(
                            args.initial_capital
                        ),
                        cost_per_side=(
                            args.cost_per_side
                        ),
                    )
                )

                in_sample_metrics = (
                    sanitize_metric_values(
                        in_sample_result.metrics
                    )
                )
                sanitized_out_of_sample_metrics = (
                    sanitize_metric_values(
                        out_of_sample_result.metrics
                    )
                )

                out_of_sample_metrics.append(
                    sanitized_out_of_sample_metrics
                )

                windows.append(
                    {
                        "window_number": (
                            split.window_number
                        ),
                        "train_start": (
                            split.train_start
                            .isoformat()
                        ),
                        "train_end": (
                            split.train_end
                            .isoformat()
                        ),
                        "test_start": (
                            split.test_start
                            .isoformat()
                        ),
                        "test_end": (
                            split.test_end
                            .isoformat()
                        ),
                        "in_sample": (
                            in_sample_metrics
                        ),
                        "out_of_sample": (
                            sanitized_out_of_sample_metrics
                        ),
                    }
                )

            summary = (
                summarize_walk_forward_metrics(
                    out_of_sample_metrics,
                    allow_compounding=(
                        not windows_overlap
                    ),
                )
            )

            results[symbol] = {
                "train_size": args.train_size,
                "test_size": args.test_size,
                "step_size": effective_step_size,
                "window_count": len(windows),
                "windows_overlap": (
                    windows_overlap
                ),
                "compounding_enabled": (
                    not windows_overlap
                ),
                "summary": summary,
                "windows": windows,
            }

            comparison_rows.append(
                {
                    "symbol": symbol,
                    "train_size": args.train_size,
                    "test_size": args.test_size,
                    "step_size": (
                        effective_step_size
                    ),
                    "windows_overlap": (
                        windows_overlap
                    ),
                    "compounding_enabled": (
                        not windows_overlap
                    ),
                    **summary,
                }
            )

        json_report_path = Path(
            "reports/batch_walk_forward.json"
        )
        csv_report_path = Path(
            "reports/batch_walk_forward.csv"
        )

        json_report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = {
            "symbols": normalized_symbols,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "step_size": effective_step_size,
            "windows_overlap": windows_overlap,
            "compounding_enabled": (
                not windows_overlap
            ),
            "results": results,
        }

        report_content = json.dumps(
            report,
            indent=2,
            allow_nan=False,
        )

        json_report_path.write_text(
            report_content,
            encoding="utf-8",
        )

        pd.DataFrame(
            comparison_rows
        ).to_csv(
            csv_report_path,
            index=False,
        )

        print(report_content)
        print(
            "\nReport JSON salvato in "
            f"{json_report_path}"
        )
        print(
            "Report CSV comparativo salvato in "
            f"{csv_report_path}"
        )
        return

    if args.command == "walk-forward":
        frame = pd.read_csv(
            args.input,
            parse_dates=["timestamp"],
        )

        symbol = args.symbol.upper()

        selected = frame.loc[
            frame["symbol"].astype(str).str.upper()
            == symbol
        ].copy()

        if selected.empty:
            raise ValueError(
                f"Simbolo {symbol} non trovato in "
                f"{args.input}."
            )

        splits = generate_walk_forward_splits(
            frame=selected,
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
        )

        effective_step_size = (
            args.test_size
            if args.step_size is None
            else args.step_size
        )

        windows = []
        summary_rows = []
        out_of_sample_metrics = []

        for split in splits:
            in_sample_result = (
                run_long_only_backtest(
                    split.in_sample,
                    initial_capital=(
                        args.initial_capital
                    ),
                    cost_per_side=(
                        args.cost_per_side
                    ),
                )
            )

            out_of_sample_result = (
                run_long_only_backtest(
                    split.out_of_sample,
                    initial_capital=(
                        args.initial_capital
                    ),
                    cost_per_side=(
                        args.cost_per_side
                    ),
                )
            )

            in_sample_metrics = (
                sanitize_metric_values(
                    in_sample_result.metrics
                )
            )
            sanitized_out_of_sample_metrics = (
                sanitize_metric_values(
                    out_of_sample_result.metrics
                )
            )

            out_of_sample_metrics.append(
                sanitized_out_of_sample_metrics
            )

            metadata = {
                "window_number": split.window_number,
                "train_start": (
                    split.train_start.isoformat()
                ),
                "train_end": (
                    split.train_end.isoformat()
                ),
                "test_start": (
                    split.test_start.isoformat()
                ),
                "test_end": (
                    split.test_end.isoformat()
                ),
            }

            windows.append(
                {
                    **metadata,
                    "in_sample": (
                        in_sample_metrics
                    ),
                    "out_of_sample": (
                        sanitized_out_of_sample_metrics
                    ),
                }
            )

            summary_rows.append(
                {
                    "symbol": symbol,
                    "sample": "in_sample",
                    **metadata,
                    **in_sample_result.metrics,
                }
            )

            summary_rows.append(
                {
                    "symbol": symbol,
                    "sample": "out_of_sample",
                    **metadata,
                    **out_of_sample_result.metrics,
                }
            )

        windows_overlap = (
            effective_step_size
            < args.test_size
        )

        summary = summarize_walk_forward_metrics(
            out_of_sample_metrics,
            allow_compounding=(
                not windows_overlap
            ),
        )

        report = {
            "symbol": symbol,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "step_size": effective_step_size,
            "window_count": len(windows),
            "windows_overlap": windows_overlap,
            "compounding_enabled": (
                not windows_overlap
            ),
            "summary": summary,
            "windows": windows,
        }

        json_report_path = Path(
            f"reports/{symbol}_walk_forward.json"
        )
        csv_report_path = Path(
            f"reports/{symbol}_walk_forward.csv"
        )

        json_report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_content = json.dumps(
            report,
            indent=2,
            allow_nan=False,
        )

        json_report_path.write_text(
            report_content,
            encoding="utf-8",
        )

        pd.DataFrame(
            summary_rows
        ).to_csv(
            csv_report_path,
            index=False,
        )

        print(report_content)
        print(
            "\nReport JSON salvato in "
            f"{json_report_path}"
        )
        print(
            "Report CSV salvato in "
            f"{csv_report_path}"
        )
        return

    if args.command == "run":
        raw_path = Path(
            "data/raw/bars.csv"
        )
        processed_path = Path(
            "data/processed/features.csv"
        )

        fetch_and_save(
            args.symbols,
            args.start,
            args.end,
            raw_path,
        )

        parameters = (
            _strategy_parameters_from_args(args)
        )

        analyze_and_save(
            raw_path,
            processed_path,
            parameters=parameters,
        )

        symbol = (
            args.symbol or args.symbols[0]
        ).upper()

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