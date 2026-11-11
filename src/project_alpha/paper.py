"""Paper trading del portafoglio core (Fase 3, soldi finti).

Ogni sera, dopo la chiusura USA:

1. legge le barre aggiornate e calcola i pesi target del core bilanciato
   (pesi fissi decisi nell'IPS, scalati dal volatility targeting con dati
   fino a oggi);
2. legge conto e posizioni dal broker (fonte di verità) e lo stato locale
   (massimo storico dell'equity, eventuale blocco);
3. applica lo **stop di drawdown**: oltre la soglia (es. −12%) il target
   diventa tutto cash e il sistema resta fermo finché il proprietario non
   lo sblocca con ``paper-resume``;
4. ribilancia solo se i pesi sono usciti dalle bande, passando sempre dal
   Risk Engine V2; con ``submit=False`` (default) non invia nulla;
5. scrive una riga nel registro ``paper_log.csv``.

Gli ordini inviati la sera restano in coda e vengono eseguiti all'apertura
del giorno dopo (come l'esecuzione ``next_open`` dei backtest).
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

import pandas as pd

from project_alpha.allocation import VolTargetConfig
from project_alpha.evaluation import TRADING_DAYS
from project_alpha.execution.engine import RebalanceResult, rebalance
from project_alpha.execution.orders import OrderPlanSettings
from project_alpha.io_utils import dumps_json, write_csv_atomic, write_json_atomic
from project_alpha.risk_engine import RiskPolicyV2

PAPER_LOG_COLUMNS = (
    "run_timestamp_utc",
    "strategy",
    "mode",
    "status",
    "data_timestamp",
    "equity",
    "peak_equity",
    "drawdown",
    "daily_return",
    "realized_volatility",
    "target_exposure",
    "current_exposure",
    "halted",
    "risk_allowed",
    "risk_mode",
    "risk_violations",
    "order_count",
    "orders",
    "max_weight_error",
    "note",
    "target_weights",
    "current_weights",
    "account_equity",
)


@dataclass(frozen=True)
class PaperConfig:
    name: str
    data: Path
    # Capitale massimo gestito: con un conto più grande si usa solo questa quota.
    budget: float
    weights: Mapping[str, float]
    vol_target: VolTargetConfig | None
    # Drawdown oltre cui si va tutto in cash e ci si ferma (es. 0.12 = −12%).
    stop_drawdown: float
    # Ribilancia se un peso si scosta dal target di almeno questa quantità…
    drift_threshold: float
    # …o se l'esposizione totale si scosta di almeno questa quantità.
    exposure_threshold: float
    risk: RiskPolicyV2
    orders: OrderPlanSettings
    log: Path
    state: Path
    groups: Mapping[str, str] = field(default_factory=dict)
    # Perdita massima dichiarata nell'IPS (solo informativa: la fanno rispettare
    # risk.max_drawdown e stop_drawdown, che devono stare sotto).
    ips_max_drawdown: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.budget) or self.budget <= 0:
            raise ValueError("budget deve essere positivo.")

        if not self.weights:
            raise ValueError("Servono i pesi del portafoglio core.")

        total = 0.0
        for symbol, weight in self.weights.items():
            if not isfinite(weight) or weight < 0:
                raise ValueError(f"Peso non valido per {symbol}.")
            total += weight

        if total > 1.0 + 1e-9:
            raise ValueError("La somma dei pesi non può superare 1 (niente leva).")

        if not 0 < self.stop_drawdown < 1:
            raise ValueError("stop_drawdown deve essere in (0, 1).")

        if self.stop_drawdown < self.risk.max_drawdown - 1e-12:
            raise ValueError(
                "stop_drawdown deve essere almeno pari al drawdown della modalità "
                "solo-riduzione del Risk Engine (prima si smette di aumentare, poi si esce)."
            )

        if self.ips_max_drawdown is not None and self.stop_drawdown >= self.ips_max_drawdown:
            raise ValueError("stop_drawdown deve essere inferiore alla perdita massima dell'IPS.")

        for name in ("drift_threshold", "exposure_threshold"):
            value = getattr(self, name)
            if not 0 < value < 1:
                raise ValueError(f"{name} deve essere in (0, 1).")

    @property
    def symbols(self) -> list[str]:
        return sorted(self.weights)


@dataclass
class PaperState:
    # Valori del capitale gestito (sotto-conto virtuale), non del conto intero.
    peak_equity: float | None = None
    last_equity: float | None = None
    # Equity del conto e capitale gestito al primo invio: il capitale gestito è
    # capital_baseline + (equity del conto − account_baseline).
    account_baseline: float | None = None
    capital_baseline: float | None = None
    halted: bool = False
    halted_at: str | None = None
    halt_reason: str | None = None
    last_run_utc: str | None = None

    @classmethod
    def load(cls, path: Path) -> PaperState:
        if not path.exists():
            return cls()

        payload = json.loads(path.read_text(encoding="utf-8"))
        known = {key: payload.get(key) for key in cls.__dataclass_fields__}
        known["halted"] = bool(known.get("halted"))
        return cls(**known)

    def save(self, path: Path) -> None:
        write_json_atomic(path, dict(self.__dict__))


@dataclass(frozen=True)
class CoreTargets:
    data_timestamp: pd.Timestamp
    closes: dict[str, float]
    realized_volatility: float | None
    exposure: float
    weights: dict[str, float]


@dataclass(frozen=True)
class PaperOutcome:
    status: str
    mode: str
    targets: CoreTargets
    equity: float
    peak_equity: float
    drawdown: float
    daily_return: float
    halted: bool
    rebalance: RebalanceResult | None
    note: str

    def summary(self) -> dict[str, Any]:
        result = self.rebalance
        return {
            "status": self.status,
            "mode": self.mode,
            "data_timestamp": self.targets.data_timestamp.isoformat(),
            "equity": round(self.equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "drawdown": round(self.drawdown, 4),
            "daily_return": round(self.daily_return, 4),
            "realized_volatility": self.targets.realized_volatility,
            "target_exposure": round(self.targets.exposure, 4),
            "halted": self.halted,
            "risk_allowed": None if result is None else result.risk.allowed,
            "risk_violations": [] if result is None else list(result.risk.violations),
            "risk_details": [] if result is None else list(result.risk.details),
            "orders": [] if result is None else _orders_summary(result),
            "note": self.note,
        }


# ----------------------------------------------------------------------
# Configurazione
# ----------------------------------------------------------------------


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def load_paper_config(path: Path | str, *, root: Path | str | None = None) -> PaperConfig:
    """Legge la configurazione TOML; i percorsi sono relativi a ``root``
    (default: la cartella superiore a quella del file, cioè la radice del repo)."""
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    base = Path(root) if root is not None else config_path.resolve().parent.parent

    allowed = {
        "name",
        "data",
        "budget",
        "log",
        "state",
        "stop_drawdown",
        "ips_max_drawdown",
        "drift_threshold",
        "exposure_threshold",
        "weights",
        "groups",
        "vol_target",
        "risk",
        "orders",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Chiavi sconosciute nella configurazione paper: {sorted(unknown)}")

    def section(name: str, cls) -> Any:
        values = dict(raw.get(name) or {})
        fields = set(cls.__dataclass_fields__)
        extra = set(values) - fields
        if extra:
            raise ValueError(f"Chiavi sconosciute nella sezione [{name}]: {sorted(extra)}")
        return values

    weights = {str(symbol).upper(): float(weight) for symbol, weight in raw["weights"].items()}
    groups = {
        str(symbol).upper(): str(group) for symbol, group in (raw.get("groups") or {}).items()
    }

    vol_values = raw.get("vol_target")
    vol_target = None
    if vol_values is not None and vol_values.get("enabled", True):
        vol_values = {key: value for key, value in vol_values.items() if key != "enabled"}
        extra = set(vol_values) - set(VolTargetConfig.__dataclass_fields__)
        if extra:
            raise ValueError(f"Chiavi sconosciute nella sezione [vol_target]: {sorted(extra)}")
        vol_target = VolTargetConfig(**vol_values)

    risk_values = section("risk", RiskPolicyV2)
    risk_values.pop("groups", None)
    risk_values.pop("allowed_symbols", None)
    risk = RiskPolicyV2(**risk_values, groups=groups, allowed_symbols=frozenset(weights))

    return PaperConfig(
        name=str(raw["name"]),
        data=_resolve(base, raw["data"]),
        budget=float(raw["budget"]),
        weights=weights,
        groups=groups,
        vol_target=vol_target,
        stop_drawdown=float(raw["stop_drawdown"]),
        ips_max_drawdown=(
            None if raw.get("ips_max_drawdown") is None else float(raw["ips_max_drawdown"])
        ),
        drift_threshold=float(raw.get("drift_threshold", 0.03)),
        exposure_threshold=float(raw.get("exposure_threshold", 0.05)),
        risk=risk,
        orders=OrderPlanSettings(**section("orders", OrderPlanSettings)),
        log=_resolve(base, raw.get("log", "reports/paper/paper_log.csv")),
        state=_resolve(base, raw.get("state", "reports/paper/state.json")),
    )


# ----------------------------------------------------------------------
# Target
# ----------------------------------------------------------------------


MARKET_TZ = "America/New_York"
# Chiusura USA (16:00) più un margine perché la barra giornaliera sia definitiva.
SESSION_FINAL_HOUR = 16
SESSION_FINAL_MINUTE = 15


def us_market_open(now: datetime) -> bool:
    """Seduta USA in corso (lun–ven, 9:30–16:15 di New York; festività ignorate)."""
    local = pd.Timestamp(now).tz_convert(MARKET_TZ)
    if local.weekday() >= 5:
        return False
    minutes = local.hour * 60 + local.minute
    return 9 * 60 + 30 <= minutes < SESSION_FINAL_HOUR * 60 + SESSION_FINAL_MINUTE


def complete_sessions(frame: pd.DataFrame, now: datetime) -> tuple[pd.DataFrame, bool]:
    """Toglie la barra della seduta in corso, se il programma gira a borsa aperta.

    Durante la seduta Alpaca restituisce per oggi una barra parziale: usarla
    come chiusura farebbe decidere su un prezzo provvisorio.
    """
    data = frame.copy()
    stamps = pd.to_datetime(data["timestamp"], utc=True)
    session = stamps.dt.tz_convert(MARKET_TZ).dt.normalize()
    final = session + pd.Timedelta(hours=SESSION_FINAL_HOUR, minutes=SESSION_FINAL_MINUTE)
    keep = final <= pd.Timestamp(now).tz_convert(MARKET_TZ)
    return data.loc[keep.to_numpy()], bool((~keep).any())


def core_targets(frame: pd.DataFrame, config: PaperConfig) -> CoreTargets:
    """Pesi target del core sull'ultima barra (nessuna informazione futura).

    L'esposizione usa la volatilità dei rendimenti del core negli ultimi
    ``lookback_days`` giorni fino all'ultima chiusura inclusa: è la decisione
    per l'apertura successiva.
    """
    data = frame.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    missing = [symbol for symbol in config.symbols if symbol not in set(data["symbol"])]

    if missing:
        raise ValueError(f"Simboli assenti nei dati {config.data}: {missing}")

    closes = (
        data.loc[data["symbol"].isin(config.symbols)]
        .pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        .sort_index()
    )
    last_timestamp = closes.index.max()
    last_row = closes.loc[last_timestamp]
    stale = sorted(symbol for symbol in config.symbols if pd.isna(last_row.get(symbol)))

    if stale:
        raise ValueError(f"Dati non allineati: {stale} non arrivano a {last_timestamp.date()}.")

    exposure = 1.0
    realized: float | None = None

    if config.vol_target is not None:
        vt = config.vol_target
        returns = closes[config.symbols].pct_change().dropna()
        core = sum(returns[symbol] * weight for symbol, weight in config.weights.items())
        window = core.tail(vt.lookback_days)

        if len(window) < vt.lookback_days:
            exposure = vt.warmup_exposure
        else:
            realized = float(window.std(ddof=1) * sqrt(TRADING_DAYS))
            if realized <= 0:
                exposure = vt.max_exposure
            else:
                exposure = min(
                    max(vt.target_volatility / realized, vt.min_exposure), vt.max_exposure
                )

    weights = {symbol: weight * exposure for symbol, weight in config.weights.items()}

    return CoreTargets(
        data_timestamp=pd.Timestamp(last_timestamp),
        closes={symbol: float(last_row[symbol]) for symbol in config.symbols},
        realized_volatility=realized,
        exposure=exposure,
        weights=weights,
    )


# ----------------------------------------------------------------------
# Esecuzione
# ----------------------------------------------------------------------


def _orders_summary(result: RebalanceResult) -> list[dict[str, Any]]:
    statuses = {report.client_order_id: report for report in result.reports}
    rows = []

    for order in result.orders:
        report = statuses.get(order.client_order_id)
        rows.append(
            {
                "symbol": order.symbol,
                "side": str(order.side),
                "quantity": order.quantity,
                "reference_price": order.reference_price,
                "status": None if report is None else str(report.status),
                "reason": None if report is None else report.reason,
            }
        )

    return rows


def _append_log(path: Path, row: Mapping[str, Any]) -> None:
    if path.exists():
        log = pd.read_csv(path)
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row], columns=list(PAPER_LOG_COLUMNS))

    # reindex: un registro scritto da una versione precedente acquista le nuove colonne.
    write_csv_atomic(log.reindex(columns=list(PAPER_LOG_COLUMNS)), path)


def run_paper(
    config: PaperConfig,
    broker,
    *,
    submit: bool = False,
    now: datetime | None = None,
    frame: pd.DataFrame | None = None,
) -> PaperOutcome:
    """Un ciclo di paper trading. ``submit=False``: calcola e registra, non invia."""
    run_time = now or datetime.now(UTC)
    mode = "SUBMIT" if submit else "DRY_RUN"
    bars = frame if frame is not None else pd.read_csv(config.data)
    bars, dropped_partial = complete_sessions(bars, run_time)
    targets = core_targets(bars, config)
    state = PaperState.load(config.state)

    prices = dict(targets.closes)
    if hasattr(broker, "market_prices"):
        prices = {**broker.market_prices(), **prices}

    account = broker.account(prices)
    account_equity = account.equity

    if account_equity <= 0:
        raise ValueError("Equity del conto non positiva: nessuna operazione.")

    # Sotto-conto virtuale: il sistema gestisce il budget più i propri guadagni
    # e perdite, anche se il conto è più grande. Presuppone che il conto sia
    # usato solo da Project Alpha (ogni variazione di valore è sua).
    account_baseline = (
        state.account_baseline if state.account_baseline is not None else account_equity
    )
    capital_baseline = (
        state.capital_baseline
        if state.capital_baseline is not None
        else min(config.budget, account_baseline)
    )
    equity = capital_baseline + (account_equity - account_baseline)

    if equity <= 0:
        raise ValueError("Capitale gestito non positivo: nessuna operazione.")

    peak = max(state.peak_equity or equity, equity)
    drawdown = equity / peak - 1.0
    daily_return = equity / state.last_equity - 1.0 if state.last_equity else 0.0
    halted = state.halted
    note = ""

    if dropped_partial:
        note = "Seduta USA ancora aperta: ignorata la barra di oggi (incompleta). "

    if account_equity > equity * 1.01:
        note += (
            f"Sotto-conto virtuale: gestiti {equity:,.0f} su un conto da "
            f"{account_equity:,.0f}; il resto non viene toccato."
        )

    if not halted and drawdown <= -config.stop_drawdown:
        halted = True
        note = (
            f"Stop di drawdown: {drawdown:.2%} oltre −{config.stop_drawdown:.0%}. "
            "Tutto in cash; sblocco manuale con paper-resume."
        )

    # Pesi relativi al capitale gestito.
    target_weights = {} if halted else dict(targets.weights)

    current = {
        symbol: quantity * prices[symbol] / equity for symbol, quantity in account.positions.items()
    }
    symbols = set(current) | set(target_weights)
    max_drift = max(
        (abs(target_weights.get(s, 0.0) - current.get(s, 0.0)) for s in symbols), default=0.0
    )
    gross_gap = abs(sum(target_weights.values()) - sum(current.values()))
    open_orders = broker.open_orders() if hasattr(broker, "open_orders") else []

    result: RebalanceResult | None = None

    if submit and us_market_open(run_time) and not (halted and current):
        # Esecuzione di recupero a borsa aperta: gli ordini a mercato verrebbero
        # eseguiti subito a prezzi del momento. Si aspetta l'esecuzione serale.
        # Eccezione: la liquidazione dopo lo stop non aspetta.
        status = "MARKET_OPEN"
        note = (note + " " if note else "") + (
            "Borsa USA aperta: nessun ordine, se ne occupa l'esecuzione delle 23:00."
        )
    elif open_orders:
        status = "PENDING_ORDERS"
        note = note or f"{len(open_orders)} ordini ancora aperti: nessun nuovo ordine."
    elif halted and not current:
        status = "HALTED"
        note = note or "Sistema fermo dopo lo stop di drawdown (paper-resume per ripartire)."
    elif (
        not halted and max_drift < config.drift_threshold and gross_gap < config.exposure_threshold
    ):
        status = "WITHIN_BANDS"
    else:
        result = rebalance(
            broker,
            target_weights,
            prices,
            strategy=config.name,
            as_of=run_time,
            data_timestamp=targets.data_timestamp,
            peak_equity=peak,
            daily_return=daily_return,
            policy=config.risk,
            settings=config.orders,
            dry_run=not submit,
            capital=equity,
        )
        status = result.status

    if submit:
        state.account_baseline = account_baseline
        state.capital_baseline = capital_baseline
        state.peak_equity = peak
        state.last_equity = equity
        state.last_run_utc = run_time.isoformat()
        if halted and not state.halted:
            state.halted = True
            state.halted_at = run_time.isoformat()
            state.halt_reason = note
        config.state.parent.mkdir(parents=True, exist_ok=True)
        state.save(config.state)

    outcome = PaperOutcome(
        status=status,
        mode=mode,
        targets=targets,
        equity=equity,
        peak_equity=peak,
        drawdown=drawdown,
        daily_return=daily_return,
        halted=halted,
        rebalance=result,
        note=note,
    )

    config.log.parent.mkdir(parents=True, exist_ok=True)
    _append_log(
        config.log,
        {
            "run_timestamp_utc": run_time.isoformat(),
            "strategy": config.name,
            "mode": mode,
            "status": status,
            "data_timestamp": targets.data_timestamp.isoformat(),
            "equity": round(equity, 2),
            "peak_equity": round(peak, 2),
            "drawdown": round(drawdown, 6),
            "daily_return": round(daily_return, 6),
            "realized_volatility": targets.realized_volatility,
            "target_exposure": round(sum(target_weights.values()), 6),
            "current_exposure": round(sum(current.values()), 6),
            "halted": halted,
            "risk_allowed": None if result is None else result.risk.allowed,
            "risk_mode": None if result is None else str(result.risk.mode),
            "risk_violations": "" if result is None else ";".join(result.risk.violations),
            "order_count": 0 if result is None else len(result.orders),
            "orders": "" if result is None else dumps_json(_orders_summary(result)),
            "max_weight_error": None if result is None else result.max_weight_error,
            "note": note,
            "target_weights": dumps_json({k: round(v, 6) for k, v in target_weights.items()}),
            "current_weights": dumps_json({k: round(v, 6) for k, v in current.items()}),
            "account_equity": round(account_equity, 2),
        },
    )

    return outcome


def resume_paper(config: PaperConfig, *, now: datetime | None = None) -> PaperState:
    """Sblocco manuale dopo lo stop: il massimo storico riparte dall'equity attuale."""
    state = PaperState.load(config.state)
    state = replace(
        state,
        halted=False,
        halted_at=None,
        halt_reason=None,
        peak_equity=None,
        last_run_utc=(now or datetime.now(UTC)).isoformat(),
    )
    config.state.parent.mkdir(parents=True, exist_ok=True)
    state.save(config.state)
    return state
