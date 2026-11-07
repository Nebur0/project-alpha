from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from project_alpha.robustness import (
    RegimeAnalysisConfig,
    analyze_walk_forward_regimes,
    regime_summaries_to_frame,
    regime_window_results_to_frame,
    summarize_regime_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Analizza le finestre walk-forward out-of-sample per regime di mercato.")
    )

    parser.add_argument(
        "--market",
        type=Path,
        required=True,
        help=("CSV OHLCV contenente almeno symbol, timestamp e close."),
    )
    parser.add_argument(
        "--optimization-report",
        type=Path,
        required=True,
        help=("CSV dettagliato prodotto dal walk-forward optimization."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/advanced_robustness"),
        help=("Directory in cui salvare i report della regime analysis."),
    )
    parser.add_argument(
        "--annualization-factor",
        type=int,
        default=252,
        help=("Fattore di annualizzazione della volatilità. Default congelato: 252."),
    )
    parser.add_argument(
        "--high-volatility-threshold",
        type=float,
        default=0.25,
        help=("Soglia di volatilità annualizzata per HIGH_VOL. Default congelato: 0.25."),
    )

    return parser


def run_regime_analysis(
    market_path: Path,
    optimization_report_path: Path,
    output_dir: Path,
    *,
    annualization_factor: int = 252,
    high_volatility_threshold: float = 0.25,
) -> dict[str, object]:
    if not market_path.exists():
        raise FileNotFoundError(f"Market dataset non trovato: {market_path}")

    if not optimization_report_path.exists():
        raise FileNotFoundError(f"Report walk-forward non trovato: {optimization_report_path}")

    config = RegimeAnalysisConfig(
        annualization_factor=(annualization_factor),
        high_volatility_threshold=(high_volatility_threshold),
    )

    market_frame = pd.read_csv(
        market_path,
        parse_dates=[
            "timestamp",
        ],
    )

    optimization_frame = pd.read_csv(
        optimization_report_path,
        parse_dates=[
            "test_start",
            "test_end",
        ],
    )

    window_results = analyze_walk_forward_regimes(
        market_frame,
        optimization_frame,
        config,
    )

    summaries = summarize_regime_results(window_results)

    window_frame = regime_window_results_to_frame(window_results)

    summary_frame = regime_summaries_to_frame(summaries)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_csv_path = output_dir / "regime_windows.csv"
    summary_csv_path = output_dir / "regime_summary.csv"
    json_path = output_dir / "regime_analysis.json"

    window_frame.to_csv(
        window_csv_path,
        index=False,
    )

    summary_frame.to_csv(
        summary_csv_path,
        index=False,
    )

    report = {
        "config": asdict(config),
        "market_path": str(market_path),
        "optimization_report_path": str(optimization_report_path),
        "window_count": len(window_results),
        "regime_count": len(summaries),
        "regimes": [asdict(summary) for summary in summaries],
    }

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return {
        "config": config,
        "window_results": (window_results),
        "summaries": summaries,
        "window_csv_path": (window_csv_path),
        "summary_csv_path": (summary_csv_path),
        "json_path": json_path,
    }


def main() -> None:
    args = build_parser().parse_args()

    result = run_regime_analysis(
        args.market,
        args.optimization_report,
        args.output_dir,
        annualization_factor=(args.annualization_factor),
        high_volatility_threshold=(args.high_volatility_threshold),
    )

    print(f"Finestre analizzate: {len(result['window_results'])}")
    print(f"Regimi osservati: {len(result['summaries'])}")
    print(f"Report finestre: {result['window_csv_path']}")
    print(f"Report riepilogo: {result['summary_csv_path']}")
    print(f"Report JSON: {result['json_path']}")


if __name__ == "__main__":
    main()
