"""Sentinella durante la seduta: controlla il conto e avvisa, **senza operare**.

Gira ogni ora (attività pianificata). Con la borsa USA aperta legge il conto
paper ai prezzi correnti e calcola il capitale gestito, la variazione dalla
chiusura di ieri e la distanza dal massimo. Se supera una soglia manda una
notifica; non invia ordini e non modifica lo stato del paper trading (le
decisioni restano all'esecuzione serale).

Per non ripetere lo stesso avviso ogni ora, si notifica solo quando il
livello peggiora rispetto a quanto già segnalato nella stessa giornata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from project_alpha.io_utils import write_text_atomic
from project_alpha.notifications import push_directory
from project_alpha.paper import MARKET_TZ, PaperConfig, PaperState, us_market_open
from project_alpha.push import SubscriptionStore, VapidKeys, send_notification

DAILY_DROP_ALERT = 0.03  # calo nella giornata che merita un avviso
LEVELS = {None: 0, "warning": 1, "critical": 2}


@dataclass(frozen=True)
class WatchResult:
    status: str
    capital: float | None = None
    drawdown: float | None = None
    day_change: float | None = None
    level: str | None = None
    notified: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _it(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%".replace(".", ",")


def _assess(config: PaperConfig, drawdown: float, day_change: float) -> tuple[str | None, str]:
    if drawdown <= -config.stop_drawdown:
        return "critical", (
            f"Calo dal massimo {_it(drawdown)}: oltre lo stop (−{config.stop_drawdown:.0%}). "
            "Il sistema venderà tutto all'esecuzione delle 23:00."
        )
    if drawdown <= -config.risk.max_drawdown:
        return "warning", (
            f"Calo dal massimo {_it(drawdown)}: oltre −{config.risk.max_drawdown:.0%} il "
            f"sistema può solo ridurre. Stop a −{config.stop_drawdown:.0%}."
        )
    if day_change <= -DAILY_DROP_ALERT:
        return (
            "warning",
            f"Giornata pesante: {_it(day_change)} da ieri sera (calo dal massimo {_it(drawdown)}).",
        )
    return None, ""


def watch_paper(
    config: PaperConfig,
    broker,
    root: Path | str,
    *,
    now: datetime | None = None,
    post=None,
) -> WatchResult:
    clock = now or datetime.now(UTC)
    if not us_market_open(clock):
        return WatchResult(status="MARKET_CLOSED")

    state = PaperState.load(config.state)
    if state.account_baseline is None or state.capital_baseline is None:
        return WatchResult(status="NOT_STARTED")

    account = broker.account(broker.market_prices())
    capital = state.capital_baseline + (account.equity - state.account_baseline)
    peak = max(state.peak_equity or capital, capital)
    drawdown = capital / peak - 1.0
    day_change = capital / state.last_equity - 1.0 if state.last_equity else 0.0
    level, message = _assess(config, drawdown, day_change)

    notified = False
    if level is not None:
        path = push_directory(root) / "watch_state.json"
        today = pd.Timestamp(clock).tz_convert(MARKET_TZ).date().isoformat()
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        already = previous.get("level") if previous.get("date") == today else None
        if LEVELS[level] > LEVELS[already]:
            directory = push_directory(root)
            send_notification(
                SubscriptionStore(directory),
                VapidKeys.load_or_create(directory),
                {
                    "title": ("⛔" if level == "critical" else "⚠️") + " Project Alpha: sentinella",
                    "body": f"Capitale {capital:,.0f} $. {message}".replace(",", "."),
                    "tag": "project-alpha-watch",
                    "url": "/",
                },
                urgency="high" if level == "critical" else "normal",
                post=post,
            )
            write_text_atomic(path, json.dumps({"date": today, "level": level}))
            notified = True

    return WatchResult(
        status="CHECKED",
        capital=capital,
        drawdown=drawdown,
        day_change=day_change,
        level=level,
        notified=notified,
        message=message,
    )
