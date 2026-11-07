"""Pipeline di ricerca riproducibile.

Un solo comando rigenera l'intera analisi di una strategia a partire da un
file di configurazione TOML:

    project-alpha research --config configs/dev15_v1.toml

Passi: walk-forward batch → portafoglio OOS → valutazione standard →
stress dei costi e dell'esecuzione (a selezione congelata) → analisi blend.

Ogni esecuzione scrive un ``manifest.json`` (configurazione, commit git,
hash SHA-256 dei dati, versioni) e aggiunge una riga al registro delle
prove. Il registro serve a contare quante prove sono state fatte sugli
stessi dati, così il Deflated Sharpe Ratio tiene conto della ricerca fatta.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project_alpha import __version__
from project_alpha.backtest import ExecutionConfig
from project_alpha.batch_optimization import (
    batch_walk_forward_comparison_to_frame,
    batch_walk_forward_optimization_to_frame,
    run_batch_walk_forward_optimization,
)
from project_alpha.cost_stress_analysis import (
    analyze_frozen_selection_cost_stress,
)
from project_alpha.evaluation import (
    TRADING_DAYS,
    deflated_sharpe_ratio,
    evaluate_walk_forward_portfolio,
)
from project_alpha.execution_stress_analysis import (
    DEFAULT_EXECUTION_SCENARIOS,
    analyze_frozen_selection_execution_stress,
)
from project_alpha.io_utils import (
    write_csv_atomic,
    write_json_atomic,
)
from project_alpha.portfolio_blend_analysis import (
    analyze_walk_forward_portfolio_blends,
)
from project_alpha.portfolio_walk_forward import (
    run_walk_forward_portfolio_backtest,
)
from project_alpha.scoring import RobustScoreConfig

REGISTRY_COLUMNS = (
    "run_id",
    "timestamp_utc",
    "name",
    "git_commit",
    "git_dirty",
    "data_sha256",
    "symbol_count",
    "grid_size",
    "score_metric",
    "execution_mode",
    "execution_delay_days",
    "daily_sharpe",
    "annualized_sharpe",
    "cagr",
    "max_drawdown",
    "mean_exposure",
    "cagr_vs_static_exposure",
    "permutation_p_value",
    "output_dir",
)


# --------------------------------------------------------------------------
# Configurazione
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardSettings:
    train_size: int
    test_size: int
    step_size: int | None = None
    score_metric: str = "robust_score"
    minimum_completed_trades: int = 1
    no_candidate_policy: str = "raise"
    initial_capital: float = 10_000.0
    cost_per_side: float = 0.001


@dataclass(frozen=True)
class AnalysisSettings:
    costs_bps_per_side: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0)
    execution_stress: bool = True
    blend_defensive_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    permutations: int = 5_000
    seed: int = 42
    hac_lags: int = 10


@dataclass(frozen=True)
class ResearchConfig:
    name: str
    data: Path
    symbols: tuple[str, ...]
    output_dir: Path
    walk_forward: WalkForwardSettings
    grid: dict[str, list[int | float]]
    robust_score: RobustScoreConfig | None = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    registry: Path | None = None
    source: Path | None = None


def _require(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Configurazione: manca '{key}' in [{section}].")
    return mapping[key]


def _build(cls, values: Mapping[str, Any], section: str):
    """Istanzia una sezione segnalando chiavi sconosciute in modo leggibile."""
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed

    if unknown:
        raise ValueError(
            f"Configurazione: chiavi sconosciute in [{section}]: {sorted(unknown)}. "
            f"Ammesse: {sorted(allowed)}."
        )

    return cls(**values)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def load_research_config(
    path: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> ResearchConfig:
    """Legge un file TOML di ricerca; i percorsi relativi partono da ``base_dir``.

    ``base_dir`` di default è la cartella di lavoro corrente (la radice del
    repository quando si usa la CLI).
    """
    source = Path(path)

    with source.open("rb") as handle:
        raw = tomllib.load(handle)

    base = Path(base_dir) if base_dir is not None else Path.cwd()

    known = {
        "name",
        "data",
        "symbols",
        "output_dir",
        "registry",
        "walk_forward",
        "grid",
        "robust_score",
        "execution",
        "analysis",
    }
    unknown = set(raw) - known

    if unknown:
        raise ValueError(f"Configurazione: chiavi sconosciute {sorted(unknown)}.")

    walk_forward_raw = _require(raw, "walk_forward", "root")
    walk_forward = _build(WalkForwardSettings, walk_forward_raw, "walk_forward")

    symbols = tuple(str(symbol).strip().upper() for symbol in _require(raw, "symbols", "root"))

    if not symbols:
        raise ValueError("Configurazione: 'symbols' è vuoto.")

    grid = {str(key): list(values) for key, values in _require(raw, "grid", "root").items()}

    robust_raw = raw.get("robust_score")
    robust_score = _build(RobustScoreConfig, robust_raw, "robust_score") if robust_raw else None

    execution = _build(ExecutionConfig, raw.get("execution", {}), "execution")

    analysis_raw = dict(raw.get("analysis", {}))
    for key in ("costs_bps_per_side", "blend_defensive_weights"):
        if key in analysis_raw:
            analysis_raw[key] = tuple(float(value) for value in analysis_raw[key])
    analysis = _build(AnalysisSettings, analysis_raw, "analysis")

    registry = raw.get("registry", "reports/research/experiment_registry.csv")

    return ResearchConfig(
        name=str(_require(raw, "name", "root")),
        data=_resolve(base, _require(raw, "data", "root")),
        symbols=symbols,
        output_dir=_resolve(base, _require(raw, "output_dir", "root")),
        walk_forward=walk_forward,
        grid=grid,
        robust_score=robust_score,
        execution=execution,
        analysis=analysis,
        registry=_resolve(base, registry) if registry else None,
        source=source,
    )


# --------------------------------------------------------------------------
# Manifest e registro delle prove
# --------------------------------------------------------------------------


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_state(directory: Path | str) -> dict[str, str | bool | None]:
    """Commit corrente e presenza di modifiche non committate (se git c'è)."""

    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")

    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def _grid_size(grid: Mapping[str, Sequence[Any]]) -> int:
    size = 1
    for values in grid.values():
        size *= len(set(values))
    return size


def read_registry(path: Path | str) -> pd.DataFrame:
    registry = Path(path)

    if not registry.exists():
        return pd.DataFrame(columns=list(REGISTRY_COLUMNS))

    return pd.read_csv(registry)


def append_to_registry(
    path: Path | str,
    row: Mapping[str, Any],
) -> pd.DataFrame:
    registry = read_registry(path)
    entry = pd.DataFrame([{column: row.get(column) for column in REGISTRY_COLUMNS}])
    updated = entry if registry.empty else pd.concat([registry, entry], ignore_index=True)
    write_csv_atomic(updated, path)
    return updated


def registry_trials(
    registry: pd.DataFrame,
    data_sha256: str,
) -> tuple[int, float | None]:
    """Numero di prove sugli stessi dati e dispersione dei loro Sharpe giornalieri."""
    if registry.empty:
        return 0, None

    same_data = registry.loc[registry["data_sha256"] == data_sha256]
    sharpes = pd.to_numeric(same_data["daily_sharpe"], errors="coerce").dropna()

    if len(sharpes) < 2:
        return len(same_data), None

    return len(same_data), float(sharpes.std(ddof=1))


# --------------------------------------------------------------------------
# Esecuzione
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchRunResult:
    run_id: str
    output_dir: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]


def _load_frame(config: ResearchConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.data, parse_dates=["timestamp"])
    available = set(frame["symbol"].astype(str).str.upper())
    missing = [symbol for symbol in config.symbols if symbol not in available]

    if missing:
        raise ValueError(f"Simboli assenti nei dati {config.data}: {missing}")

    return frame.loc[frame["symbol"].astype(str).str.upper().isin(config.symbols)].copy()


def run_research(
    config: ResearchConfig,
    *,
    repo_dir: Path | str | None = None,
) -> ResearchRunResult:
    run_id = uuid.uuid4().hex[:12]
    started = datetime.now(UTC)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    frame = _load_frame(config)
    data_hash = file_sha256(config.data)
    wf = config.walk_forward
    execution = None if config.execution.is_default else config.execution

    batch = run_batch_walk_forward_optimization(
        frame,
        list(config.symbols),
        config.grid,
        train_size=wf.train_size,
        test_size=wf.test_size,
        step_size=wf.step_size,
        initial_capital=wf.initial_capital,
        cost_per_side=wf.cost_per_side,
        score_metric=wf.score_metric,
        minimum_completed_trades=wf.minimum_completed_trades,
        robust_score_config=config.robust_score,
        no_candidate_policy=wf.no_candidate_policy,
        execution=execution,
    )

    write_csv_atomic(batch_walk_forward_optimization_to_frame(batch), output / "walk_forward.csv")
    write_csv_atomic(
        batch_walk_forward_comparison_to_frame(batch),
        output / "walk_forward_comparison.csv",
    )

    portfolio = run_walk_forward_portfolio_backtest(
        frame,
        batch,
        initial_capital=wf.initial_capital,
        cost_per_side=wf.cost_per_side,
    )
    write_json_atomic(output / "portfolio_metrics.json", portfolio.portfolio.metrics)
    write_csv_atomic(portfolio.portfolio.equity_curve, output / "portfolio_equity.csv")

    analysis = config.analysis
    evaluation = evaluate_walk_forward_portfolio(
        portfolio,
        permutations=analysis.permutations,
        seed=analysis.seed,
        hac_lags=analysis.hac_lags,
    )

    if analysis.costs_bps_per_side:
        cost_stress = analyze_frozen_selection_cost_stress(
            frame,
            batch,
            costs_per_side=[value / 10_000.0 for value in analysis.costs_bps_per_side],
            baseline_cost_per_side=wf.cost_per_side,
            initial_capital=wf.initial_capital,
        )
        write_csv_atomic(cost_stress.scenarios, output / "cost_stress.csv")

    if analysis.execution_stress:
        execution_stress = analyze_frozen_selection_execution_stress(
            frame,
            batch,
            scenarios=DEFAULT_EXECUTION_SCENARIOS,
            cost_per_side=wf.cost_per_side,
            initial_capital=wf.initial_capital,
        )
        write_csv_atomic(execution_stress.scenarios, output / "execution_stress.csv")

    if analysis.blend_defensive_weights:
        blends = analyze_walk_forward_portfolio_blends(
            portfolio,
            defensive_weights=analysis.blend_defensive_weights,
        )
        write_csv_atomic(blends.overall, output / "blend_overall.csv")
        write_csv_atomic(blends.by_window, output / "blend_by_window.csv")
        write_csv_atomic(
            blends.leave_one_window_out,
            output / "blend_leave_one_window_out.csv",
        )

    strategy_returns = portfolio.portfolio.equity_curve["portfolio_return"].to_numpy(dtype=float)
    returns_std = strategy_returns.std(ddof=1)
    daily_sharpe = float(strategy_returns.mean() / returns_std) if returns_std > 0 else None

    # Registro: questa prova si aggiunge a quelle già fatte sugli stessi dati.
    trials = 1
    trials_std: float | None = None
    deflated: float | None = evaluation.probabilistic_sharpe

    if config.registry is not None:
        previous = read_registry(config.registry)
        previous_count, _ = registry_trials(previous, data_hash)
        sharpes = pd.to_numeric(
            previous.loc[previous["data_sha256"] == data_hash, "daily_sharpe"]
            if not previous.empty
            else pd.Series(dtype=float),
            errors="coerce",
        ).dropna().tolist()

        if daily_sharpe is not None:
            sharpes.append(daily_sharpe)

        trials = previous_count + 1
        trials_std = float(np.std(sharpes, ddof=1)) if len(sharpes) >= 2 else None

        if trials > 1:
            deflated = (
                deflated_sharpe_ratio(
                    strategy_returns,
                    trials=trials,
                    trials_sharpe_std=trials_std,
                )
                if trials_std is not None and daily_sharpe is not None
                else None
            )

    summary = {
        **evaluation.summary(),
        "trials": trials,
        "trials_daily_sharpe_std": trials_std,
        "deflated_sharpe": deflated,
    }

    evaluation_payload = evaluation.to_dict()
    evaluation_payload["summary"] = summary
    write_json_atomic(output / "evaluation.json", evaluation_payload)
    write_csv_atomic(evaluation.by_window, output / "evaluation_by_window.csv")
    write_csv_atomic(evaluation.on_off, output / "evaluation_on_off.csv")

    repo = Path(repo_dir) if repo_dir is not None else Path.cwd()
    git = git_state(repo)

    manifest = {
        "run_id": run_id,
        "name": config.name,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "config_file": str(config.source) if config.source else None,
        "config": {
            "data": str(config.data),
            "symbols": list(config.symbols),
            "walk_forward": asdict(wf),
            "grid": config.grid,
            "grid_size": _grid_size(config.grid),
            "robust_score": asdict(config.robust_score) if config.robust_score else None,
            "execution": asdict(config.execution),
            "analysis": asdict(analysis),
        },
        "data": {
            "path": str(config.data),
            "sha256": data_hash,
            "rows": len(frame),
            "start": frame["timestamp"].min().isoformat(),
            "end": frame["timestamp"].max().isoformat(),
        },
        "git": git,
        "versions": {
            "project_alpha": __version__,
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "trials_on_same_data": trials,
    }
    write_json_atomic(output / "manifest.json", manifest)

    if config.registry is not None:
        metrics = portfolio.portfolio.metrics
        append_to_registry(
            config.registry,
            {
                "run_id": run_id,
                "timestamp_utc": started.isoformat(),
                "name": config.name,
                "git_commit": git["commit"],
                "git_dirty": git["dirty"],
                "data_sha256": data_hash,
                "symbol_count": len(config.symbols),
                "grid_size": _grid_size(config.grid),
                "score_metric": wf.score_metric,
                "execution_mode": config.execution.mode,
                "execution_delay_days": config.execution.delay_days,
                "daily_sharpe": daily_sharpe,
                "annualized_sharpe": (
                    daily_sharpe * np.sqrt(TRADING_DAYS) if daily_sharpe is not None else None
                ),
                "cagr": metrics["cagr"],
                "max_drawdown": metrics["max_drawdown"],
                "mean_exposure": metrics["mean_gross_exposure"],
                "cagr_vs_static_exposure": summary["cagr_vs_static_exposure"],
                "permutation_p_value": summary["permutation_p_value_return"],
                "output_dir": str(output),
            },
        )

    return ResearchRunResult(
        run_id=run_id,
        output_dir=output,
        manifest=manifest,
        summary=summary,
    )
