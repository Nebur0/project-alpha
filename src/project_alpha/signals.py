"""Registro forward dei segnali (Fase 3, senza ordini).

Ogni giorno, sui dati più recenti, calcola il target che la strategia
congelata vorrebbe detenere domani e lo salva con timestamp in un registro
append-only. Nessun ordine viene inviato: il registro serve a misurare la
strategia **in avanti**, su dati che non esistevano quando è stata
progettata, e a confrontare più avanti paper trading e backtest.

Per ogni simbolo i parametri sono scelti come farebbe la prossima finestra
walk-forward: griglia valutata su tutta la storia disponibile, stesso
score e stesso minimo di trade. La proposta di pesi passa dal Risk Engine
V2 e la decisione viene registrata insieme ai segnali.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from project_alpha.indicators import add_indicators
from project_alpha.io_utils import write_csv_atomic
from project_alpha.optimization import (
    _execution_kwargs,
    _select_best_training_result,
    run_parameter_grid_search,
)
from project_alpha.research import ResearchConfig, file_sha256
from project_alpha.risk_engine import (
    PortfolioState,
    RiskPolicyV2,
    evaluate_proposal,
)
from project_alpha.strategy import add_baseline_position

SIGNAL_LOG_COLUMNS = (
    "run_timestamp_utc",
    "data_timestamp",
    "symbol",
    "close",
    "target_position",
    "entry_signal",
    "exit_signal",
    "target_weight",
    "fast_ema_period",
    "slow_ema_period",
    "trend_ema_period",
    "selection_score",
    "risk_allowed",
    "risk_mode",
    "risk_violations",
    "strategy",
    "data_sha256",
)


@dataclass(frozen=True)
class SignalSnapshot:
    data_timestamp: pd.Timestamp
    rows: pd.DataFrame
    risk_allowed: bool
    risk_mode: str
    risk_violations: tuple[str, ...]
    appended: bool


def _symbol_signal(
    frame: pd.DataFrame,
    symbol: str,
    config: ResearchConfig,
) -> dict[str, object]:
    selected = frame.loc[frame["symbol"].astype(str).str.upper() == symbol].copy()
    selected = selected.sort_values("timestamp").reset_index(drop=True)
    wf = config.walk_forward
    execution = None if config.execution.is_default else config.execution

    results = run_parameter_grid_search(
        selected,
        config.grid,
        initial_capital=wf.initial_capital,
        cost_per_side=wf.cost_per_side,
        score_metric=wf.score_metric,
        robust_score_config=config.robust_score,
        **_execution_kwargs(execution),
    )

    try:
        best = _select_best_training_result(
            results,
            minimum_completed_trades=wf.minimum_completed_trades,
        )
    except ValueError:
        best = None

    last = selected.iloc[-1]

    if best is None:
        return {
            "data_timestamp": last["timestamp"],
            "symbol": symbol,
            "close": float(last["close"]),
            "target_position": 0,
            "entry_signal": False,
            "exit_signal": False,
            "fast_ema_period": None,
            "slow_ema_period": None,
            "trend_ema_period": None,
            "selection_score": None,
        }

    positioned = add_baseline_position(add_indicators(selected, best.parameters), best.parameters)
    final = positioned.iloc[-1]
    parameters = asdict(best.parameters)

    return {
        "data_timestamp": final["timestamp"],
        "symbol": symbol,
        "close": float(final["close"]),
        "target_position": int(final["target_position"]),
        "entry_signal": bool(final["entry_signal"]),
        "exit_signal": bool(final["exit_signal"]),
        "fast_ema_period": parameters["fast_ema_period"],
        "slow_ema_period": parameters["slow_ema_period"],
        "trend_ema_period": parameters["trend_ema_period"],
        "selection_score": best.score,
    }


def read_signal_log(path: Path | str) -> pd.DataFrame:
    log = Path(path)

    if not log.exists():
        return pd.DataFrame(columns=list(SIGNAL_LOG_COLUMNS))

    return pd.read_csv(log)


def _previous_weights(log: pd.DataFrame, strategy: str) -> dict[str, float]:
    if log.empty:
        return {}

    same = log.loc[log["strategy"] == strategy]

    if same.empty:
        return {}

    last_timestamp = same["data_timestamp"].max()
    latest = same.loc[same["data_timestamp"] == last_timestamp]

    return {
        str(row.symbol): float(row.target_weight)
        for row in latest.itertuples()
        if float(row.target_weight) != 0.0
    }


def record_forward_signals(
    config: ResearchConfig,
    *,
    log_path: Path | str,
    policy: RiskPolicyV2 | None = None,
    now: datetime | None = None,
) -> SignalSnapshot:
    """Calcola i segnali sull'ultima barra e li aggiunge al registro.

    Idempotente: se il registro contiene già la stessa data dei dati per la
    stessa strategia, non aggiunge righe duplicate.
    """
    run_time = now or datetime.now(UTC)
    frame = pd.read_csv(config.data, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    available = set(frame["symbol"].astype(str).str.upper())
    missing = [symbol for symbol in config.symbols if symbol not in available]

    if missing:
        raise ValueError(f"Simboli assenti nei dati {config.data}: {missing}")

    rows = [_symbol_signal(frame, symbol, config) for symbol in config.symbols]
    table = pd.DataFrame(rows)
    data_timestamp = pd.Timestamp(table["data_timestamp"].max())

    stale_symbols = table.loc[table["data_timestamp"] < data_timestamp, "symbol"].tolist()
    if stale_symbols:
        raise ValueError(
            f"Dati non allineati: {stale_symbols} non arrivano a {data_timestamp.date()}."
        )

    weight = 1.0 / len(config.symbols)
    table["target_weight"] = table["target_position"] * weight

    log = read_signal_log(log_path)
    proposed = {
        str(row.symbol): float(row.target_weight) for row in table.itertuples() if row.target_weight
    }
    state = PortfolioState(
        equity=1.0,
        peak_equity=1.0,
        daily_return=0.0,
        current_weights=_previous_weights(log, config.name),
        data_timestamp=data_timestamp,
        as_of=pd.Timestamp(run_time),
    )
    decision = evaluate_proposal(proposed, state, policy)

    table["run_timestamp_utc"] = pd.Timestamp(run_time).isoformat()
    table["data_timestamp"] = table["data_timestamp"].map(lambda value: value.isoformat())
    table["risk_allowed"] = decision.allowed
    table["risk_mode"] = str(decision.mode)
    table["risk_violations"] = ";".join(decision.violations)
    table["strategy"] = config.name
    table["data_sha256"] = file_sha256(config.data)
    table = table[list(SIGNAL_LOG_COLUMNS)]

    already = (
        not log.empty
        and (
            (log["strategy"] == config.name) & (log["data_timestamp"] == data_timestamp.isoformat())
        ).any()
    )

    if not already:
        updated = table if log.empty else pd.concat([log, table], ignore_index=True)
        write_csv_atomic(updated, log_path)

    return SignalSnapshot(
        data_timestamp=data_timestamp,
        rows=table,
        risk_allowed=decision.allowed,
        risk_mode=str(decision.mode),
        risk_violations=decision.violations,
        appended=not already,
    )
