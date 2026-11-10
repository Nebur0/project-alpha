"""Assistente di monitoraggio (Fase 5, primo gradino: sola lettura).

Ogni sera, dopo il paper trading, controlla che il sistema si sia comportato
come previsto e scrive un riepilogo in italiano. **Non decide e non
opera**: legge il registro, lo stato e i dati di mercato e segnala.

I controlli sono deterministici (soglie fisse, nessun modello): un avviso
deve essere riproducibile e spiegabile. Un eventuale assistente basato su
LLM, più avanti, partirà da questi stessi fatti.

Livelli: ``info`` (da sapere), ``warning`` (da guardare), ``critical``
(intervenire).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any

import pandas as pd

from project_alpha.paper import PaperConfig, PaperState

SEVERITIES = ("critical", "warning", "info")

# Soglie fisse (non ottimizzate): cambiano solo con una decisione esplicita.
MAX_DAYS_WITHOUT_RUN = 4.0  # copre il fine settimana
CRITICAL_DAYS_WITHOUT_RUN = 7.0
MAX_DATA_AGE_DAYS = 4.0
DRIFT_ALERT_MULTIPLIER = 2.0  # scostamento oltre 2× la banda di ribilanciamento
TRACKING_WINDOW = 20  # intervalli tra esecuzioni
TRACKING_ALERT = 0.02  # 2 punti di scostamento cumulato dal modello
FILLED_STATUSES = {"EXECUTED", "SUBMITTED", "PARTIAL"}


@dataclass(frozen=True)
class Alert:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class TrackingResult:
    intervals: int
    actual: float
    expected: float

    @property
    def difference(self) -> float:
        return self.actual - self.expected


@dataclass(frozen=True)
class MonitorResult:
    generated_at: str
    alerts: tuple[Alert, ...]
    last_run: dict[str, Any] | None
    tracking: TrackingResult | None

    @property
    def worst(self) -> str | None:
        for severity in SEVERITIES:
            if any(alert.severity == severity for alert in self.alerts):
                return severity
        return None

    def to_dict(self) -> dict[str, Any]:
        tracking = None
        if self.tracking is not None:
            tracking = {**asdict(self.tracking), "difference": self.tracking.difference}
        return {
            "generated_at": self.generated_at,
            "worst": self.worst,
            "alerts": [asdict(alert) for alert in self.alerts],
            "tracking": tracking,
        }


def _weights(cell: Any) -> dict[str, float]:
    if not isinstance(cell, str) or not cell.strip():
        return {}
    try:
        payload = json.loads(cell)
    except json.JSONDecodeError:
        return {}
    return {str(key): float(value) for key, value in payload.items()}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _closes(bars: pd.DataFrame) -> pd.DataFrame:
    data = bars.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")


def tracking_vs_model(
    log: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    window: int = TRACKING_WINDOW,
) -> TrackingResult | None:
    """Rendimento del capitale gestito contro quello dei pesi che avrebbe dovuto detenere.

    Per ogni coppia di esecuzioni reali consecutive: rendimento atteso = pesi
    detenuti dopo la prima esecuzione × variazione delle chiusure fino alla
    seconda. Approssimazione: gli ordini si eseguono all'apertura, non alla
    chiusura; su molti giorni lo scostamento deve restare piccolo.
    """
    real = log.loc[log["mode"] == "SUBMIT"].copy()
    if len(real) < 2:
        return None

    real["data_timestamp"] = pd.to_datetime(real["data_timestamp"], utc=True)
    real = real.sort_values("run_timestamp_utc").drop_duplicates("data_timestamp", keep="last")
    closes = _closes(bars)
    actual_growth = 1.0
    expected_growth = 1.0
    intervals = 0

    rows = list(real.itertuples())[-(window + 1) :]
    for before, after in zip(rows, rows[1:], strict=False):
        equity_before = _number(before.equity)
        equity_after = _number(after.equity)
        if not equity_before or equity_after is None:
            continue
        if before.data_timestamp not in closes.index or after.data_timestamp not in closes.index:
            continue

        status = str(before.status)
        weights = _weights(
            before.target_weights if status in FILLED_STATUSES else before.current_weights
        )
        start = closes.loc[before.data_timestamp]
        end = closes.loc[after.data_timestamp]
        expected = 0.0
        for symbol, weight in weights.items():
            if symbol in start and symbol in end and start[symbol] > 0:
                expected += weight * (end[symbol] / start[symbol] - 1.0)

        actual_growth *= equity_after / equity_before
        expected_growth *= 1.0 + expected
        intervals += 1

    if intervals == 0:
        return None

    return TrackingResult(
        intervals=intervals, actual=actual_growth - 1.0, expected=expected_growth - 1.0
    )


def check_paper(
    config: PaperConfig,
    log: pd.DataFrame | None,
    state: PaperState,
    *,
    bars: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> MonitorResult:
    """Controlli serali sul paper trading. Non modifica nulla."""
    clock = now or datetime.now(UTC)
    alerts: list[Alert] = []

    def add(severity: str, code: str, message: str) -> None:
        alerts.append(Alert(severity, code, message))

    if log is None or log.empty:
        add(
            "info",
            "NOT_STARTED",
            "Paper trading non ancora avviato: nessuna esecuzione registrata.",
        )
        return MonitorResult(clock.isoformat(), tuple(alerts), None, None)

    last = log.iloc[-1].to_dict()
    real = log.loc[log["mode"] == "SUBMIT"]

    if real.empty:
        add(
            "info",
            "DRY_RUN_ONLY",
            "Finora solo prove (dry-run): nessun ordine inviato al conto paper.",
        )
    else:
        last_real = pd.Timestamp(real["run_timestamp_utc"].iloc[-1])
        days = (pd.Timestamp(clock) - last_real).total_seconds() / 86_400
        if days > CRITICAL_DAYS_WITHOUT_RUN:
            add(
                "critical",
                "NO_RECENT_RUN",
                f"Nessuna esecuzione da {days:.0f} giorni: l'attività pianificata è ferma?",
            )
        elif days > MAX_DAYS_WITHOUT_RUN:
            add(
                "warning",
                "NO_RECENT_RUN",
                f"Ultima esecuzione {days:.1f} giorni fa: controllare l'attività pianificata.",
            )

    run_time = pd.Timestamp(last["run_timestamp_utc"])
    data_time = pd.Timestamp(last["data_timestamp"])
    data_age = (run_time - data_time).total_seconds() / 86_400
    if data_age > MAX_DATA_AGE_DAYS:
        add(
            "warning",
            "STALE_DATA",
            f"I dati dell'ultima esecuzione erano vecchi di {data_age:.1f} giorni.",
        )

    if state.halted or bool(last.get("halted")):
        add(
            "critical",
            "HALTED",
            "Sistema fermo dopo lo stop di drawdown: tutto in cash finché non lo sblocchi "
            "(paper-resume), dopo averci ragionato.",
        )

    drawdown = _number(last.get("drawdown"))
    reduce_only = -config.risk.max_drawdown
    if drawdown is not None and not state.halted:
        if drawdown <= reduce_only:
            add(
                "warning",
                "REDUCE_ONLY",
                f"Calo dal massimo {drawdown:.1%}: oltre {reduce_only:.0%} il sistema può solo "
                f"ridurre; lo stop è a −{config.stop_drawdown:.0%}.",
            )
        elif drawdown <= reduce_only / 2:
            add(
                "info",
                "DRAWDOWN",
                f"Calo dal massimo {drawdown:.1%} (soglia solo-riduzione {reduce_only:.0%}).",
            )

    status = str(last.get("status"))
    if status == "REJECTED_BY_RISK":
        add(
            "warning",
            "RISK_REJECTED",
            f"Il Risk Engine ha bloccato il ribilanciamento: {last.get('risk_violations') or '—'}.",
        )
    if status == "PARTIAL":
        add(
            "critical",
            "PARTIAL",
            "Esecuzione parziale: almeno un ordine è stato rifiutato dal broker.",
        )

    orders = last.get("orders")
    if isinstance(orders, str) and orders.strip():
        try:
            rejected = [o for o in json.loads(orders) if o.get("status") == "REJECTED"]
        except json.JSONDecodeError:
            rejected = []
        for order in rejected:
            add(
                "critical",
                "ORDER_REJECTED",
                f"Ordine {order.get('side')} {order.get('symbol')} rifiutato: {order.get('reason')}.",
            )

    statuses = log["status"].astype(str).tolist()
    if len(statuses) >= 2 and statuses[-1] == statuses[-2] == "PENDING_ORDERS":
        add(
            "warning",
            "PENDING_ORDERS",
            "Ordini ancora aperti per due esecuzioni di fila: verificarli sul conto paper.",
        )

    if status in {"WITHIN_BANDS", "PENDING_ORDERS", "HALTED"}:
        target = _weights(last.get("target_weights"))
        current = _weights(last.get("current_weights"))
        symbols = set(target) | set(current)
        drift = max((abs(target.get(s, 0.0) - current.get(s, 0.0)) for s in symbols), default=0.0)
        limit = DRIFT_ALERT_MULTIPLIER * config.drift_threshold
        if drift > limit and not state.halted:
            add(
                "warning",
                "DRIFT",
                f"Pesi lontani dal target ({drift:.1%} > {limit:.0%}) senza ribilanciamento.",
            )

    tracking = None
    if bars is not None and not real.empty:
        tracking = tracking_vs_model(log, bars)
        if tracking is not None and abs(tracking.difference) > TRACKING_ALERT:
            add(
                "warning",
                "TRACKING",
                f"Ultimi {tracking.intervals} intervalli: conto {tracking.actual:+.2%} contro "
                f"modello {tracking.expected:+.2%} ({tracking.difference:+.2%}).",
            )

    order = {severity: index for index, severity in enumerate(SEVERITIES)}
    alerts.sort(key=lambda alert: order[alert.severity])
    return MonitorResult(clock.isoformat(), tuple(alerts), last, tracking)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

STATUS_TEXT = {
    "EXECUTED": "ordini eseguiti",
    "SUBMITTED": "ordini inviati, in coda per l'apertura",
    "DRY_RUN": "prova senza ordini",
    "WITHIN_BANDS": "portafoglio in banda, nessun ordine",
    "NOTHING_TO_DO": "niente da fare",
    "PENDING_ORDERS": "ordini precedenti ancora aperti",
    "HALTED": "fermo dopo lo stop",
    "REJECTED_BY_RISK": "bloccato dal Risk Engine",
    "PARTIAL": "esecuzione parziale",
}
ICONS = {"critical": "⛔", "warning": "⚠️", "info": "ℹ️"}


def _money(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(value: Any, digits: int = 1) -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"{number * 100:.{digits}f}%".replace(".", ",")


def render_report(config: PaperConfig, result: MonitorResult) -> str:
    """Riepilogo serale in italiano (Markdown)."""
    lines = [f"# Project Alpha — riepilogo {result.generated_at[:10]}", ""]
    last = result.last_run

    headline = {
        "critical": "⛔ Serve un tuo intervento.",
        "warning": "⚠️ Qualcosa da guardare.",
        "info": "✅ Tutto regolare.",
        None: "✅ Tutto regolare.",
    }[result.worst]
    lines += [f"**{headline}**", ""]

    if last is not None:
        status = str(last.get("status"))
        lines += [
            f"- Esecuzione: {str(last.get('run_timestamp_utc'))[:16].replace('T', ' ')} UTC "
            f"({'reale' if last.get('mode') == 'SUBMIT' else 'prova'}), "
            f"esito: {STATUS_TEXT.get(status, status)}",
            f"- Dati di mercato fino al {str(last.get('data_timestamp'))[:10]}",
            f"- Capitale gestito: {_money(last.get('equity'))} "
            f"(budget {_money(config.budget)}, dall'ultima esecuzione "
            f"{_pct(last.get('daily_return'), 2)})",
            f"- Calo dal massimo: {_pct(last.get('drawdown'))} "
            f"(solo riduzioni a −{config.risk.max_drawdown:.0%}, stop a "
            f"−{config.stop_drawdown:.0%})",
            f"- Investito: {_pct(last.get('current_exposure'), 0)} · target "
            f"{_pct(last.get('target_exposure'), 0)} · volatilità del core "
            f"{_pct(last.get('realized_volatility'))}",
            "",
        ]

        orders = last.get("orders")
        parsed = []
        if isinstance(orders, str) and orders.strip():
            try:
                parsed = json.loads(orders)
            except json.JSONDecodeError:
                parsed = []
        if parsed:
            lines += [
                "## Ordini",
                "",
                "| ETF | Lato | Quantità | Controvalore | Esito |",
                "|---|---|---|---|---|",
            ]
            for order in parsed:
                value = (order.get("quantity") or 0) * (order.get("reference_price") or 0)
                outcome = {
                    "FILLED": "eseguito",
                    "SUBMITTED": "in coda",
                    "REJECTED": "rifiutato",
                }.get(order.get("status"), "prova")
                side = "acquisto" if order.get("side") == "BUY" else "vendita"
                lines.append(
                    f"| {order.get('symbol')} | {side} | "
                    f"{str(order.get('quantity')).replace('.', ',')} | "
                    f"{_money(value)} | {outcome} |"
                )
            lines.append("")

    lines += ["## Avvisi", ""]
    if result.alerts:
        lines += [f"- {ICONS[a.severity]} {a.message}" for a in result.alerts]
    else:
        lines.append("- Nessun avviso.")
    lines.append("")

    if result.tracking is not None:
        t = result.tracking
        lines += [
            "## Conto contro modello",
            "",
            f"Ultimi {t.intervals} intervalli: conto {_pct(t.actual, 2)}, modello "
            f"{_pct(t.expected, 2)}, scostamento {_pct(t.difference, 2)} "
            f"(avviso oltre ±{TRACKING_ALERT:.0%}).",
            "",
        ]

    lines += [
        "---",
        "Questo riepilogo è generato in sola lettura: non invia ordini e non cambia nulla.",
        "",
    ]
    return "\n".join(lines)


def load_inputs(config: PaperConfig) -> tuple[pd.DataFrame | None, PaperState, pd.DataFrame | None]:
    log = None
    if config.log.exists() and config.log.stat().st_size > 0:
        log = pd.read_csv(config.log)
    bars = pd.read_csv(config.data) if config.data.exists() else None
    return log, PaperState.load(config.state), bars
