from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from project_alpha.backtest import (
    EXECUTION_MODES,
    ExecutionConfig,
    run_long_only_backtest,
)
from project_alpha.batch_optimization import (
    batch_walk_forward_comparison_to_frame,
    batch_walk_forward_optimization_to_dict,
    batch_walk_forward_optimization_to_frame,
    run_batch_walk_forward_optimization,
)
from project_alpha.config import StrategyParameters
from project_alpha.io_utils import (
    dumps_json,
    write_text_atomic,
)
from project_alpha.optimization import (
    grid_search_results_to_frame,
    run_parameter_grid_search,
    run_walk_forward_optimization,
    walk_forward_optimization_to_dict,
    walk_forward_optimization_to_frame,
)
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
from project_alpha.scoring import RobustScoreConfig
from project_alpha.validation import (
    generate_walk_forward_splits,
    split_by_date,
)


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Data non valida: {value}. Usare YYYY-MM-DD.") from exc


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


def _add_parameter_grid_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--fast-ema-periods",
        nargs="+",
        type=int,
        default=[20],
    )
    parser.add_argument(
        "--slow-ema-periods",
        nargs="+",
        type=int,
        default=[50],
    )
    parser.add_argument(
        "--trend-ema-periods",
        nargs="+",
        type=int,
        default=[200],
    )
    parser.add_argument(
        "--rsi-periods",
        nargs="+",
        type=int,
        default=[14],
    )
    parser.add_argument(
        "--rsi-lowers",
        nargs="+",
        type=float,
        default=[50.0],
    )
    parser.add_argument(
        "--rsi-uppers",
        nargs="+",
        type=float,
        default=[70.0],
    )
    parser.add_argument(
        "--atr-periods",
        nargs="+",
        type=int,
        default=[14],
    )
    parser.add_argument(
        "--volume-periods",
        nargs="+",
        type=int,
        default=[20],
    )
    parser.add_argument(
        "--breakout-periods",
        nargs="+",
        type=int,
        default=[20],
    )
    parser.add_argument(
        "--score-metric",
        choices=[
            "annualized_sharpe",
            "total_return",
            "cagr",
            "profit_factor",
            "win_rate",
            "max_drawdown",
            "robust_score",
        ],
        default="annualized_sharpe",
    )
    parser.add_argument(
        "--robust-sharpe-weight",
        type=float,
        default=1.0,
        help=("Peso dello Sharpe nel robust score."),
    )
    parser.add_argument(
        "--robust-excess-return-weight",
        type=float,
        default=1.0,
        help=("Peso dell'excess return rispetto al benchmark nel robust score."),
    )
    parser.add_argument(
        "--robust-drawdown-weight",
        type=float,
        default=1.0,
        help=("Peso del max drawdown nel robust score."),
    )
    parser.add_argument(
        "--robust-trade-shortfall-weight",
        type=float,
        default=0.5,
        help=("Peso della penalità per carenza di trade nel robust score."),
    )
    parser.add_argument(
        "--robust-target-completed-trades",
        type=int,
        default=20,
        help=("Numero target di trade completati usato dal robust score."),
    )


def _parameter_grid_from_args(
    args: argparse.Namespace,
) -> dict[str, list[int | float]]:
    return {
        "fast_ema_period": (args.fast_ema_periods),
        "slow_ema_period": (args.slow_ema_periods),
        "trend_ema_period": (args.trend_ema_periods),
        "rsi_period": args.rsi_periods,
        "rsi_lower": args.rsi_lowers,
        "rsi_upper": args.rsi_uppers,
        "atr_period": args.atr_periods,
        "volume_period": (args.volume_periods),
        "breakout_period": (args.breakout_periods),
    }


def _robust_score_config_from_args(
    args: argparse.Namespace,
) -> RobustScoreConfig | None:
    if args.score_metric != "robust_score":
        return None

    return RobustScoreConfig(
        sharpe_weight=(args.robust_sharpe_weight),
        excess_return_weight=(args.robust_excess_return_weight),
        drawdown_weight=(args.robust_drawdown_weight),
        trade_shortfall_weight=(args.robust_trade_shortfall_weight),
        target_completed_trades=(args.robust_target_completed_trades),
    )


EXECUTION_COMMANDS = (
    "backtest",
    "batch-backtest",
    "run",
    "grid-search",
    "optimize-walk-forward",
    "batch-optimize-walk-forward",
)


def _add_execution_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--execution-mode",
        choices=list(EXECUTION_MODES),
        default="close",
        help=(
            "Quando eseguire il segnale calcolato alla chiusura: "
            "'close' (stessa chiusura, default storico) o "
            "'next_open' (apertura del giorno dopo)."
        ),
    )
    parser.add_argument(
        "--execution-delay-days",
        type=int,
        default=0,
        help="Giorni di ritardo aggiuntivi prima dell'esecuzione.",
    )
    parser.add_argument(
        "--cash-rate",
        type=float,
        default=0.0,
        help="Rendimento annuo della quota in cash (es. 0.03 = 3%%).",
    )


def _execution_from_args(
    args: argparse.Namespace,
) -> ExecutionConfig | None:
    """None quando le opzioni sono quelle di default (comportamento storico)."""
    execution = ExecutionConfig(
        mode=getattr(args, "execution_mode", "close"),
        delay_days=getattr(args, "execution_delay_days", 0),
        cash_rate_annual=getattr(args, "cash_rate", 0.0),
    )

    return None if execution.is_default else execution


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
    fetch.add_argument(
        "--feed",
        choices=["iex", "sip"],
        default=None,
        help="Fonte dati Alpaca (default: quella del piano dati).",
    )

    check_data = commands.add_parser(
        "check-data",
        help="Controlla la qualità di un file di barre (duplicati, buchi, disallineamenti).",
    )
    check_data.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    check_data.add_argument(
        "--symbols",
        nargs="+",
        help="Simboli attesi (default: tutti quelli presenti).",
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
        help=("Avanzamento tra le finestre; predefinito: test-size."),
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
        help=("Avanzamento tra le finestre; predefinito: test-size."),
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

    grid_search = commands.add_parser(
        "grid-search",
        help=("Ricerca e ordina combinazioni di parametri strategici."),
    )
    grid_search.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/bars.csv"),
        help=("Dataset OHLCV grezzo usato per rigenerare indicatori e segnali."),
    )
    grid_search.add_argument(
        "--symbol",
        required=True,
    )
    _add_parameter_grid_arguments(grid_search)
    grid_search.add_argument(
        "--top",
        type=int,
        default=10,
        help=("Numero di combinazioni migliori da mostrare nel terminale."),
    )
    grid_search.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
    )
    grid_search.add_argument(
        "--cost-per-side",
        type=float,
        default=0.001,
    )

    optimize_walk_forward = commands.add_parser(
        "optimize-walk-forward",
        help=("Ottimizza i parametri in-sample e li valuta out-of-sample."),
    )
    optimize_walk_forward.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/bars.csv"),
        help=("Dataset OHLCV grezzo usato per rigenerare indicatori e segnali."),
    )
    optimize_walk_forward.add_argument(
        "--symbol",
        required=True,
    )
    optimize_walk_forward.add_argument(
        "--train-size",
        type=int,
        required=True,
        help=("Numero di righe nella prima finestra in-sample."),
    )
    optimize_walk_forward.add_argument(
        "--test-size",
        type=int,
        required=True,
        help=("Numero di righe in ogni finestra out-of-sample."),
    )
    optimize_walk_forward.add_argument(
        "--step-size",
        type=int,
        help=("Avanzamento tra le finestre; predefinito: test-size."),
    )
    _add_parameter_grid_arguments(optimize_walk_forward)
    optimize_walk_forward.add_argument(
        "--minimum-completed-trades",
        type=int,
        default=1,
        help=("Numero minimo di trade completati richiesto nella finestra in-sample."),
    )
    optimize_walk_forward.add_argument(
        "--no-candidate-policy",
        choices=["raise", "flat"],
        default="raise",
        help=(
            "Se nessuna combinazione supera i requisiti "
            "in-sample: 'raise' interrompe, 'flat' tiene "
            "la finestra OOS in cash."
        ),
    )
    optimize_walk_forward.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
    )
    optimize_walk_forward.add_argument(
        "--cost-per-side",
        type=float,
        default=0.001,
    )

    batch_optimize_walk_forward = commands.add_parser(
        "batch-optimize-walk-forward",
        help=("Ottimizza i parametri in-sample e li valuta out-of-sample su più simboli."),
    )
    batch_optimize_walk_forward.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/bars.csv"),
        help=("Dataset OHLCV grezzo usato per rigenerare indicatori e segnali."),
    )
    batch_optimize_walk_forward.add_argument(
        "--symbols",
        nargs="+",
        required=True,
    )
    batch_optimize_walk_forward.add_argument(
        "--train-size",
        type=int,
        required=True,
        help=("Numero di righe nella prima finestra in-sample."),
    )
    batch_optimize_walk_forward.add_argument(
        "--test-size",
        type=int,
        required=True,
        help=("Numero di righe in ogni finestra out-of-sample."),
    )
    batch_optimize_walk_forward.add_argument(
        "--step-size",
        type=int,
        help=("Avanzamento tra le finestre; predefinito: test-size."),
    )
    _add_parameter_grid_arguments(batch_optimize_walk_forward)
    batch_optimize_walk_forward.add_argument(
        "--minimum-completed-trades",
        type=int,
        default=1,
        help=("Numero minimo di trade completati richiesto nella finestra in-sample."),
    )
    batch_optimize_walk_forward.add_argument(
        "--no-candidate-policy",
        choices=["raise", "flat"],
        default="raise",
        help=(
            "Se nessuna combinazione supera i requisiti "
            "in-sample: 'raise' interrompe, 'flat' tiene "
            "la finestra OOS in cash."
        ),
    )
    batch_optimize_walk_forward.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
    )
    batch_optimize_walk_forward.add_argument(
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

    research = commands.add_parser(
        "research",
        help=(
            "Esegue la pipeline di ricerca completa da un file TOML "
            "(walk-forward, portafoglio, valutazione, stress, blend)."
        ),
    )
    research.add_argument(
        "--config",
        type=Path,
        required=True,
        help="File di configurazione TOML (es. configs/dev15_v1.toml).",
    )

    signals = commands.add_parser(
        "signals",
        help=(
            "Calcola i segnali sull'ultima barra e li aggiunge al registro forward (nessun ordine)."
        ),
    )
    signals.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Configurazione TOML della strategia congelata.",
    )
    signals.add_argument(
        "--log",
        type=Path,
        default=Path("reports/signals/signal_log.csv"),
        help="Registro forward dei segnali (CSV, append-only).",
    )

    paper = commands.add_parser(
        "paper-rebalance",
        help=(
            "Paper trading del core (conto Alpaca paper, soldi finti). "
            "Senza --submit calcola e registra ma non invia ordini."
        ),
    )
    paper.add_argument("--config", type=Path, required=True, help="Configurazione TOML paper.")
    paper.add_argument(
        "--submit",
        action="store_true",
        help="Invia davvero gli ordini al conto paper (default: solo prova).",
    )

    resume = commands.add_parser(
        "paper-resume",
        help="Sblocca il paper trading dopo uno stop di drawdown (decisione manuale).",
    )
    resume.add_argument("--config", type=Path, required=True, help="Configurazione TOML paper.")

    dashboard = commands.add_parser(
        "dashboard",
        help="Dashboard locale in sola lettura (http://127.0.0.1:8050).",
    )
    dashboard.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_core_v1.toml"),
        help="Configurazione paper da mostrare.",
    )
    dashboard.add_argument("--port", type=int, default=8050, help="Porta locale (default 8050).")

    for command_name in EXECUTION_COMMANDS:
        _add_execution_arguments(commands.choices[command_name])

    return parser


def main() -> None:
    args = build_parser().parse_args()
    execution = _execution_from_args(args)

    if args.command == "fetch":
        from project_alpha.data_quality import check_bars

        frame = fetch_and_save(
            args.symbols,
            args.start,
            args.end,
            args.output,
            feed=args.feed,
        )

        print(f"Salvate {len(frame):,} righe in {args.output}")

        report = check_bars(frame, args.symbols)
        for issue in report.issues:
            print(f"[{issue.severity.upper()}] {issue.code} {issue.symbol or ''}: {issue.detail}")
        if not report.ok:
            raise SystemExit(
                "Dati scaricati con errori di qualità: controllare prima di usarli "
                f"(project-alpha check-data --input {args.output})."
            )
        return

    if args.command == "check-data":
        from project_alpha.data_quality import check_bars

        report = check_bars(pd.read_csv(args.input), args.symbols)
        print(dumps_json(report.summary(), indent=2))
        if not report.ok:
            raise SystemExit(1)
        return

    if args.command == "analyze":
        parameters = _strategy_parameters_from_args(args)

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
            execution,
        )

        print(
            dumps_json(
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
                execution,
            )

            results[normalized_symbol] = result.metrics

        report_path = Path("reports/batch_metrics.json")
        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_content = dumps_json(
            results,
            indent=2,
            allow_nan=False,
        )

        write_text_atomic(
            report_path,
            report_content,
        )

        csv_report_path = save_batch_summary_csv(results)

        print(report_content)
        print(f"\nReport JSON salvato in {report_path}")
        print(f"Report CSV salvato in {csv_report_path}")
        return

    if args.command == "validate":
        frame = pd.read_csv(
            args.input,
            parse_dates=["timestamp"],
        )

        symbol = args.symbol.upper()

        selected = frame.loc[frame["symbol"].astype(str).str.upper() == symbol].copy()

        if selected.empty:
            raise ValueError(f"Simbolo {symbol} non trovato in {args.input}.")

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
            "split_date": (validation_split.split_timestamp.isoformat()),
            "in_sample": in_sample_result.metrics,
            "out_of_sample": (out_of_sample_result.metrics),
        }

        report_path = Path(f"reports/{symbol}_validation.json")
        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_content = dumps_json(
            report,
            indent=2,
            allow_nan=False,
        )

        write_text_atomic(
            report_path,
            report_content,
        )

        print(report_content)
        print(f"\nReport salvato in {report_path}")
        return

    if args.command == "batch-validate":
        frame = pd.read_csv(
            args.input,
            parse_dates=["timestamp"],
        )

        normalized_symbols = list(dict.fromkeys(symbol.upper() for symbol in args.symbols))

        results = {}
        summary_rows = []

        for symbol in normalized_symbols:
            selected = frame.loc[frame["symbol"].astype(str).str.upper() == symbol].copy()

            if selected.empty:
                raise ValueError(f"Simbolo {symbol} non trovato in {args.input}.")

            validation_split = split_by_date(
                selected,
                args.split_date,
            )

            in_sample_result = run_long_only_backtest(
                validation_split.in_sample,
                initial_capital=(args.initial_capital),
                cost_per_side=(args.cost_per_side),
            )

            out_of_sample_result = run_long_only_backtest(
                validation_split.out_of_sample,
                initial_capital=(args.initial_capital),
                cost_per_side=(args.cost_per_side),
            )

            split_timestamp = validation_split.split_timestamp.isoformat()

            results[symbol] = {
                "split_date": split_timestamp,
                "in_sample": (in_sample_result.metrics),
                "out_of_sample": (out_of_sample_result.metrics),
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

        json_report_path = Path("reports/batch_validation.json")
        csv_report_path = Path("reports/batch_validation.csv")

        json_report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_content = dumps_json(
            results,
            indent=2,
            allow_nan=False,
        )

        write_text_atomic(
            json_report_path,
            report_content,
        )

        pd.DataFrame(summary_rows).to_csv(
            csv_report_path,
            index=False,
        )

        print(report_content)
        print(f"\nReport JSON salvato in {json_report_path}")
        print(f"Report CSV salvato in {csv_report_path}")
        return

    if args.command == "batch-walk-forward":
        frame = pd.read_csv(
            args.input,
            parse_dates=["timestamp"],
        )

        normalized_symbols = list(dict.fromkeys(symbol.upper() for symbol in args.symbols))

        effective_step_size = args.test_size if args.step_size is None else args.step_size

        windows_overlap = effective_step_size < args.test_size

        results = {}
        comparison_rows = []

        for symbol in normalized_symbols:
            selected = frame.loc[frame["symbol"].astype(str).str.upper() == symbol].copy()

            if selected.empty:
                raise ValueError(f"Simbolo {symbol} non trovato in {args.input}.")

            splits = generate_walk_forward_splits(
                frame=selected,
                train_size=args.train_size,
                test_size=args.test_size,
                step_size=args.step_size,
            )

            windows = []
            out_of_sample_metrics = []

            for split in splits:
                in_sample_result = run_long_only_backtest(
                    split.in_sample,
                    initial_capital=(args.initial_capital),
                    cost_per_side=(args.cost_per_side),
                )

                out_of_sample_result = run_long_only_backtest(
                    split.out_of_sample,
                    initial_capital=(args.initial_capital),
                    cost_per_side=(args.cost_per_side),
                )

                in_sample_metrics = sanitize_metric_values(in_sample_result.metrics)
                sanitized_out_of_sample_metrics = sanitize_metric_values(
                    out_of_sample_result.metrics
                )

                out_of_sample_metrics.append(sanitized_out_of_sample_metrics)

                windows.append(
                    {
                        "window_number": (split.window_number),
                        "train_start": (split.train_start.isoformat()),
                        "train_end": (split.train_end.isoformat()),
                        "test_start": (split.test_start.isoformat()),
                        "test_end": (split.test_end.isoformat()),
                        "in_sample": (in_sample_metrics),
                        "out_of_sample": (sanitized_out_of_sample_metrics),
                    }
                )

            summary = summarize_walk_forward_metrics(
                out_of_sample_metrics,
                allow_compounding=(not windows_overlap),
            )

            results[symbol] = {
                "train_size": args.train_size,
                "test_size": args.test_size,
                "step_size": effective_step_size,
                "window_count": len(windows),
                "windows_overlap": (windows_overlap),
                "compounding_enabled": (not windows_overlap),
                "summary": summary,
                "windows": windows,
            }

            comparison_rows.append(
                {
                    "symbol": symbol,
                    "train_size": args.train_size,
                    "test_size": args.test_size,
                    "step_size": (effective_step_size),
                    "windows_overlap": (windows_overlap),
                    "compounding_enabled": (not windows_overlap),
                    **summary,
                }
            )

        json_report_path = Path("reports/batch_walk_forward.json")
        csv_report_path = Path("reports/batch_walk_forward.csv")

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
            "compounding_enabled": (not windows_overlap),
            "results": results,
        }

        report_content = dumps_json(
            report,
            indent=2,
            allow_nan=False,
        )

        write_text_atomic(
            json_report_path,
            report_content,
        )

        pd.DataFrame(comparison_rows).to_csv(
            csv_report_path,
            index=False,
        )

        print(report_content)
        print(f"\nReport JSON salvato in {json_report_path}")
        print(f"Report CSV comparativo salvato in {csv_report_path}")
        return

    if args.command == "walk-forward":
        frame = pd.read_csv(
            args.input,
            parse_dates=["timestamp"],
        )

        symbol = args.symbol.upper()

        selected = frame.loc[frame["symbol"].astype(str).str.upper() == symbol].copy()

        if selected.empty:
            raise ValueError(f"Simbolo {symbol} non trovato in {args.input}.")

        splits = generate_walk_forward_splits(
            frame=selected,
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
        )

        effective_step_size = args.test_size if args.step_size is None else args.step_size

        windows = []
        summary_rows = []
        out_of_sample_metrics = []

        for split in splits:
            in_sample_result = run_long_only_backtest(
                split.in_sample,
                initial_capital=(args.initial_capital),
                cost_per_side=(args.cost_per_side),
            )

            out_of_sample_result = run_long_only_backtest(
                split.out_of_sample,
                initial_capital=(args.initial_capital),
                cost_per_side=(args.cost_per_side),
            )

            in_sample_metrics = sanitize_metric_values(in_sample_result.metrics)
            sanitized_out_of_sample_metrics = sanitize_metric_values(out_of_sample_result.metrics)

            out_of_sample_metrics.append(sanitized_out_of_sample_metrics)

            metadata = {
                "window_number": split.window_number,
                "train_start": (split.train_start.isoformat()),
                "train_end": (split.train_end.isoformat()),
                "test_start": (split.test_start.isoformat()),
                "test_end": (split.test_end.isoformat()),
            }

            windows.append(
                {
                    **metadata,
                    "in_sample": (in_sample_metrics),
                    "out_of_sample": (sanitized_out_of_sample_metrics),
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

        windows_overlap = effective_step_size < args.test_size

        summary = summarize_walk_forward_metrics(
            out_of_sample_metrics,
            allow_compounding=(not windows_overlap),
        )

        report = {
            "symbol": symbol,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "step_size": effective_step_size,
            "window_count": len(windows),
            "windows_overlap": windows_overlap,
            "compounding_enabled": (not windows_overlap),
            "summary": summary,
            "windows": windows,
        }

        json_report_path = Path(f"reports/{symbol}_walk_forward.json")
        csv_report_path = Path(f"reports/{symbol}_walk_forward.csv")

        json_report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_content = dumps_json(
            report,
            indent=2,
            allow_nan=False,
        )

        write_text_atomic(
            json_report_path,
            report_content,
        )

        pd.DataFrame(summary_rows).to_csv(
            csv_report_path,
            index=False,
        )

        print(report_content)
        print(f"\nReport JSON salvato in {json_report_path}")
        print(f"Report CSV salvato in {csv_report_path}")
        return

    if args.command == "grid-search":
        if args.top <= 0:
            raise ValueError("top deve essere positivo.")

        frame = pd.read_csv(
            args.input,
            parse_dates=["timestamp"],
        )

        symbol = args.symbol.upper()

        selected = frame.loc[frame["symbol"].astype(str).str.upper() == symbol].copy()

        if selected.empty:
            raise ValueError(f"Simbolo {symbol} non trovato in {args.input}.")

        grid = _parameter_grid_from_args(args)

        robust_score_config = _robust_score_config_from_args(args)

        results = run_parameter_grid_search(
            selected,
            grid,
            initial_capital=(args.initial_capital),
            cost_per_side=(args.cost_per_side),
            score_metric=args.score_metric,
            robust_score_config=(robust_score_config),
            execution=execution,
        )

        result_frame = grid_search_results_to_frame(results)

        top_count = min(
            args.top,
            len(results),
        )

        report_rows = [
            {
                "rank": rank,
                "parameters": asdict(result.parameters),
                "score_metric": (result.score_metric),
                "score": result.score,
                "robust_score": (
                    asdict(result.robust_score) if result.robust_score is not None else None
                ),
                "metrics": result.metrics,
            }
            for rank, result in enumerate(
                results,
                start=1,
            )
        ]

        report = {
            "symbol": symbol,
            "score_metric": (args.score_metric),
            "robust_score_config": (
                asdict(robust_score_config) if robust_score_config is not None else None
            ),
            "combination_count": (len(results)),
            "displayed_top_count": (top_count),
            "results": report_rows,
        }

        json_report_path = Path(f"reports/{symbol}_grid_search.json")
        csv_report_path = Path(f"reports/{symbol}_grid_search.csv")

        json_report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_text_atomic(
            json_report_path,
            dumps_json(
                report,
                indent=2,
                allow_nan=False,
            ),
        )

        result_frame.to_csv(
            csv_report_path,
            index=False,
        )

        print(result_frame.head(top_count).to_string(index=False))
        print(f"\nCombinazioni valutate: {len(results)}")
        print(f"Report JSON salvato in {json_report_path}")
        print(f"Report CSV salvato in {csv_report_path}")
        return

    if args.command == "optimize-walk-forward":
        frame = pd.read_csv(
            args.input,
            parse_dates=["timestamp"],
        )

        symbol = args.symbol.upper()

        selected = frame.loc[frame["symbol"].astype(str).str.upper() == symbol].copy()

        if selected.empty:
            raise ValueError(f"Simbolo {symbol} non trovato in {args.input}.")

        grid = _parameter_grid_from_args(args)

        robust_score_config = _robust_score_config_from_args(args)

        result = run_walk_forward_optimization(
            selected,
            grid,
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            initial_capital=(args.initial_capital),
            cost_per_side=(args.cost_per_side),
            score_metric=args.score_metric,
            minimum_completed_trades=(args.minimum_completed_trades),
            robust_score_config=(robust_score_config),
            no_candidate_policy=(args.no_candidate_policy),
            execution=execution,
        )

        report = walk_forward_optimization_to_dict(result)

        result_frame = walk_forward_optimization_to_frame(result)

        json_report_path = Path(f"reports/{symbol}_walk_forward_optimization.json")
        csv_report_path = Path(f"reports/{symbol}_walk_forward_optimization.csv")

        json_report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_text_atomic(
            json_report_path,
            dumps_json(
                report,
                indent=2,
                allow_nan=False,
            ),
        )

        result_frame.to_csv(
            csv_report_path,
            index=False,
        )

        print(
            dumps_json(
                report["summary"],
                indent=2,
                allow_nan=False,
            )
        )
        print(f"\nFinestre ottimizzate: {len(result.windows)}")
        print(f"Report JSON salvato in {json_report_path}")
        print(f"Report CSV salvato in {csv_report_path}")
        return

    if args.command == "batch-optimize-walk-forward":
        frame = pd.read_csv(
            args.input,
            parse_dates=["timestamp"],
        )

        grid = _parameter_grid_from_args(args)

        robust_score_config = _robust_score_config_from_args(args)

        result = run_batch_walk_forward_optimization(
            frame,
            args.symbols,
            grid,
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            initial_capital=(args.initial_capital),
            cost_per_side=(args.cost_per_side),
            score_metric=args.score_metric,
            minimum_completed_trades=(args.minimum_completed_trades),
            robust_score_config=(robust_score_config),
            no_candidate_policy=(args.no_candidate_policy),
            execution=execution,
        )

        report = batch_walk_forward_optimization_to_dict(result)

        detailed_frame = batch_walk_forward_optimization_to_frame(result)

        comparison_frame = batch_walk_forward_comparison_to_frame(result)

        json_report_path = Path("reports/batch_walk_forward_optimization.json")
        detailed_csv_report_path = Path("reports/batch_walk_forward_optimization.csv")
        comparison_csv_report_path = Path("reports/batch_walk_forward_optimization_comparison.csv")

        json_report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_text_atomic(
            json_report_path,
            dumps_json(
                report,
                indent=2,
                allow_nan=False,
            ),
        )

        detailed_frame.to_csv(
            detailed_csv_report_path,
            index=False,
        )

        comparison_frame.to_csv(
            comparison_csv_report_path,
            index=False,
        )

        print(comparison_frame.to_string(index=False))
        print(f"\nSimboli ottimizzati: {len(result.symbols)}")
        print(f"Report JSON salvato in {json_report_path}")
        print(f"Report CSV dettagliato salvato in {detailed_csv_report_path}")
        print(f"Report CSV comparativo salvato in {comparison_csv_report_path}")
        return

    if args.command == "research":
        from project_alpha.research import (
            load_research_config,
            run_research,
        )

        config = load_research_config(args.config)
        outcome = run_research(config)
        summary = outcome.summary

        print(
            dumps_json(
                {
                    "run_id": outcome.run_id,
                    "output_dir": str(outcome.output_dir),
                    "summary": summary,
                },
                indent=2,
            )
        )
        return

    if args.command == "signals":
        from project_alpha.research import load_research_config
        from project_alpha.risk_engine import RiskPolicyV2
        from project_alpha.signals import record_forward_signals

        config = load_research_config(args.config)
        # Nel registro il portafoglio si costruisce da zero: il limite di
        # turnover per ribilanciamento non ha senso sul primo giorno.
        snapshot = record_forward_signals(
            config,
            log_path=args.log,
            policy=RiskPolicyV2(max_turnover=None),
        )
        active = snapshot.rows.loc[snapshot.rows["target_position"] > 0, "symbol"].tolist()

        print(
            dumps_json(
                {
                    "data_timestamp": snapshot.data_timestamp.isoformat(),
                    "appended": snapshot.appended,
                    "risk_allowed": snapshot.risk_allowed,
                    "risk_mode": snapshot.risk_mode,
                    "risk_violations": list(snapshot.risk_violations),
                    "long_symbols": active,
                    "log": str(args.log),
                },
                indent=2,
            )
        )
        return

    if args.command == "paper-rebalance":
        from project_alpha.config import Settings
        from project_alpha.execution.alpaca_paper import AlpacaPaperBroker
        from project_alpha.paper import load_paper_config, run_paper

        config = load_paper_config(args.config)
        broker = AlpacaPaperBroker(Settings.from_env())
        try:
            outcome = run_paper(config, broker, submit=args.submit)
        except ValueError as exc:
            raise SystemExit(f"Paper trading fermato: {exc}") from exc
        print(dumps_json(outcome.summary(), indent=2))

        if outcome.status in {"PARTIAL", "REJECTED_BY_RISK"}:
            raise SystemExit(1)
        return

    if args.command == "paper-resume":
        from project_alpha.paper import load_paper_config, resume_paper

        state = resume_paper(load_paper_config(args.config))
        print(dumps_json({"halted": state.halted, "peak_equity": state.peak_equity}, indent=2))
        return

    if args.command == "dashboard":
        from project_alpha.dashboard.server import serve

        serve(Path.cwd(), args.config, port=args.port)
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

        parameters = _strategy_parameters_from_args(args)

        analyze_and_save(
            raw_path,
            processed_path,
            parameters=parameters,
        )

        symbol = (args.symbol or args.symbols[0]).upper()

        result = backtest_and_save(
            processed_path,
            symbol,
            args.initial_capital,
            args.cost_per_side,
            execution,
        )

        print(
            dumps_json(
                result.metrics,
                indent=2,
                allow_nan=False,
            )
        )
        return

    raise RuntimeError(f"Comando non gestito: {args.command}")
