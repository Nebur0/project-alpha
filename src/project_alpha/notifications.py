"""Notifica serale sul telefono: esito dell'esecuzione e avvisi del monitoraggio.

Un solo messaggio per esecuzione, breve (deve stare in una notifica). Il
dettaglio resta nel riepilogo e nella dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from project_alpha.monitor import MonitorResult, check_paper, load_inputs
from project_alpha.paper import PaperConfig
from project_alpha.push import (
    DEFAULT_CONTACT,
    DeliveryResult,
    SubscriptionStore,
    VapidKeys,
    send_notification,
)

STATUS_SHORT = {
    "EXECUTED": "Ordini eseguiti",
    "SUBMITTED": "Ordini inviati, eseguiti all'apertura",
    "DRY_RUN": "Prova senza ordini",
    "WITHIN_BANDS": "In banda, nessun ordine",
    "NOTHING_TO_DO": "Niente da fare",
    "PENDING_ORDERS": "Ordini ancora aperti",
    "HALTED": "Sistema fermo dopo lo stop",
    "REJECTED_BY_RISK": "Bloccato dal Risk Engine",
    "PARTIAL": "Esecuzione parziale",
    "MARKET_OPEN": "Borsa aperta, ordini rimandati alla sera",
}
TITLES = {
    "critical": "⛔ Project Alpha: intervenire",
    "warning": "⚠️ Project Alpha: da guardare",
    "info": "✅ Project Alpha",
    None: "✅ Project Alpha",
}


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _it(value: float, digits: int) -> str:
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def build_message(result: MonitorResult) -> dict[str, Any]:
    """Titolo, testo e urgenza della notifica a partire dal monitoraggio."""
    last = result.last_run or {}
    parts: list[str] = []

    if last:
        status = str(last.get("status"))
        orders = int(_number(last.get("order_count")) or 0)
        line = STATUS_SHORT.get(status, status)
        if orders and status in {"EXECUTED", "SUBMITTED", "DRY_RUN", "PARTIAL"}:
            line += f" ({orders})"
        parts.append(line + ".")

        equity = _number(last.get("equity"))
        change = _number(last.get("daily_return"))
        drawdown = _number(last.get("drawdown"))
        if equity is not None:
            text = f"Capitale {_it(equity, 0)} $"
            if change is not None:
                text += f" ({'+' if change >= 0 else ''}{_it(change * 100, 2)}%)"
            parts.append(text + ".")
        if drawdown is not None and drawdown < 0:
            parts.append(f"Dal massimo {_it(drawdown * 100, 1)}%.")

    important = [a for a in result.alerts if a.severity in ("critical", "warning")]
    if important:
        parts.append(important[0].message)
        if len(important) > 1:
            parts.append(f"(+{len(important) - 1} altri avvisi)")
    elif not last:
        parts.append(result.alerts[0].message if result.alerts else "Nessuna esecuzione.")

    return {
        "title": TITLES[result.worst],
        "body": " ".join(parts),
        "tag": "project-alpha-daily",
        "url": "/",
        "urgency": "high" if result.worst == "critical" else "normal",
    }


def push_directory(root: Path | str) -> Path:
    return Path(root) / "data" / "push"


def notify_latest_run(
    config: PaperConfig,
    root: Path | str,
    *,
    contact: str = DEFAULT_CONTACT,
    post=None,
) -> tuple[dict[str, Any], DeliveryResult]:
    """Invia ai dispositivi iscritti la notifica dell'ultima esecuzione."""
    log, state, bars = load_inputs(config)
    result = check_paper(config, log, state, bars=bars)
    message = build_message(result)
    directory = push_directory(root)
    delivery = send_notification(
        SubscriptionStore(directory),
        VapidKeys.load_or_create(directory),
        {key: message[key] for key in ("title", "body", "tag", "url")},
        contact=contact,
        urgency=message["urgency"],
        post=post,
    )
    return message, delivery
