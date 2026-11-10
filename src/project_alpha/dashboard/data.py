"""Dati della dashboard: legge solo file del repository, mai il broker né ``.env``.

Ogni sezione è indipendente: se un file manca (es. paper trading non ancora
avviato) la sezione dice cosa manca invece di far fallire tutta la pagina.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from project_alpha.monitor import check_paper, load_inputs
from project_alpha.paper import PaperConfig, PaperState, load_paper_config

HISTORY_LIMIT = 2000


def _clean(value: Any) -> Any:
    """Valori pandas/numpy → JSON (NaN → None)."""
    if value is None:
        return None

    if isinstance(value, float):
        return value if isfinite(value) else None

    if hasattr(value, "item"):
        return _clean(value.item())

    return value


def _json_cell(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    frame = pd.read_csv(path)
    return None if frame.empty else frame


# ----------------------------------------------------------------------
# Sezioni
# ----------------------------------------------------------------------


def ips_section(config: PaperConfig) -> dict[str, Any]:
    groups: dict[str, float] = {}
    for symbol, weight in config.weights.items():
        group = config.groups.get(symbol, "altro")
        groups[group] = groups.get(group, 0.0) + weight

    vol = config.vol_target
    risk = config.risk

    return {
        "name": config.name,
        "budget": config.budget,
        "weights": dict(sorted(config.weights.items(), key=lambda item: -item[1])),
        "groups": config.groups,
        "group_weights": groups,
        "vol_target": None if vol is None else vol.target_volatility,
        "limits": {
            "reduce_only_drawdown": -risk.max_drawdown,
            "stop_drawdown": -config.stop_drawdown,
            "ips_max_drawdown": (
                None if config.ips_max_drawdown is None else -config.ips_max_drawdown
            ),
            "max_daily_loss": -risk.max_daily_loss,
            "max_position_weight": risk.max_position_weight,
            "max_group_weight": risk.max_group_weight,
        },
        "bands": {
            "drift_threshold": config.drift_threshold,
            "exposure_threshold": config.exposure_threshold,
        },
    }


def paper_section(config: PaperConfig) -> dict[str, Any]:
    state = PaperState.load(config.state)
    log = _read_csv(config.log)
    base: dict[str, Any] = {
        "log": str(config.log),
        "state": {
            "halted": state.halted,
            "halted_at": state.halted_at,
            "halt_reason": state.halt_reason,
            "peak_equity": state.peak_equity,
            "last_run_utc": state.last_run_utc,
        },
    }

    if log is None:
        return {**base, "available": False, "runs": 0, "history": [], "last": None}

    log = log.tail(HISTORY_LIMIT)
    history = [
        {
            "run": _clean(row.run_timestamp_utc),
            "data": _clean(row.data_timestamp),
            "mode": _clean(row.mode),
            "status": _clean(row.status),
            "equity": _clean(row.equity),
            "drawdown": _clean(row.drawdown),
            "target_exposure": _clean(row.target_exposure),
            "current_exposure": _clean(row.current_exposure),
        }
        for row in log.itertuples()
    ]
    last = log.iloc[-1]
    last_row = {key: _clean(last.get(key)) for key in log.columns}

    for key in ("orders", "target_weights", "current_weights"):
        last_row[key] = _json_cell(last.get(key)) if key in log.columns else None

    submitted = log.loc[log["mode"] == "SUBMIT"]

    return {
        **base,
        "available": True,
        "runs": int(len(log)),
        "submitted_runs": int(len(submitted)),
        "first_run": _clean(log["run_timestamp_utc"].iloc[0]),
        "history": history,
        "last": last_row,
    }


def signals_section(log_path: Path) -> dict[str, Any]:
    log = _read_csv(log_path)

    if log is None:
        return {"available": False, "log": str(log_path), "rows": []}

    last_date = log["data_timestamp"].max()
    latest = log.loc[log["data_timestamp"] == last_date]
    columns = ["symbol", "close", "target_position", "target_weight", "risk_allowed", "strategy"]
    rows = [
        {column: _clean(row[column]) for column in columns if column in latest.columns}
        for _, row in latest.iterrows()
    ]

    return {
        "available": True,
        "log": str(log_path),
        "days": int(log["data_timestamp"].nunique()),
        "first_date": _clean(log["data_timestamp"].min()),
        "last_date": _clean(last_date),
        "long_count": int((latest["target_position"] > 0).sum()),
        "rows": rows,
    }


def research_section(root: Path) -> dict[str, Any]:
    registry = _read_csv(root / "reports" / "research" / "experiment_registry.csv")
    runs: list[dict[str, Any]] = []
    baselines: dict[str, list[dict[str, Any]]] = {}

    if registry is not None:
        columns = [
            "run_id",
            "timestamp_utc",
            "name",
            "git_dirty",
            "execution_mode",
            "annualized_sharpe",
            "cagr",
            "max_drawdown",
            "mean_exposure",
            "cagr_vs_static_exposure",
            "permutation_p_value",
        ]
        latest = registry.sort_values("timestamp_utc").groupby("name").tail(1)

        for _, row in latest.iterrows():
            runs.append({column: _clean(row.get(column)) for column in columns})
            table = _read_csv(
                root / "reports" / "research" / str(row["name"]) / "policy_baselines.csv"
            )
            if table is not None:
                baselines[str(row["name"])] = [
                    {key: _clean(value) for key, value in record.items()}
                    for record in table.to_dict(orient="records")
                ]

    return {
        "available": bool(runs),
        "trials": 0 if registry is None else int(len(registry)),
        "runs": runs,
        "baselines": baselines,
    }


def overview(root: Path | str, config_path: Path | str) -> dict[str, Any]:
    """Tutto ciò che mostra la dashboard, in un'unica struttura JSON."""
    repo = Path(root)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repo / config_file

    config = load_paper_config(config_file, root=repo)
    log, state, bars = load_inputs(config)
    monitor = check_paper(config, log, state, bars=bars).to_dict()

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "ips": ips_section(config),
        "paper": paper_section(config),
        "monitor": monitor,
        "signals": signals_section(repo / "reports" / "signals" / "signal_log.csv"),
        "research": research_section(repo),
    }
