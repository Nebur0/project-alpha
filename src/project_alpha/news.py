"""Notizie finanziarie per la dashboard (Alpaca News API, fonte Benzinga).

Solo informazione: le notizie **non** entrano nelle decisioni del sistema.
Due flussi:

- ``portfolio``: notizie sugli ETF del portafoglio;
- ``market``: flusso generale di mercato (le più recenti, con un limite).

Le notizie si salvano in ``data/news/news.json`` (ignorato da git), senza
duplicati e solo per gli ultimi ``RETENTION_DAYS`` giorni.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from project_alpha.config import Settings
from project_alpha.data.alpaca import _get_with_retries
from project_alpha.io_utils import write_text_atomic

NEWS_URL = "/v1beta1/news"
PAGE_LIMIT = 50
RETENTION_DAYS = 30
MARKET_MAX_ITEMS = 300


def _clean(item: dict[str, Any], feed: str) -> dict[str, Any]:
    return {
        "id": int(item["id"]),
        "headline": str(item.get("headline") or "").strip(),
        "summary": str(item.get("summary") or "").strip()[:600],
        "source": str(item.get("source") or ""),
        "author": str(item.get("author") or ""),
        "url": item.get("url") or None,
        "symbols": [str(s).upper() for s in item.get("symbols") or []],
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "feeds": [feed],
    }


def fetch_news(
    settings: Settings,
    *,
    symbols: Iterable[str] | None,
    start: datetime,
    end: datetime | None = None,
    max_items: int = 1000,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Scarica le notizie (le più recenti per prime), al massimo ``max_items``."""
    http = session if session is not None else requests.Session()
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
        "accept": "application/json",
    }
    params: dict[str, str | int] = {
        "start": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "limit": PAGE_LIMIT,
        "sort": "desc",
    }
    if end is not None:
        params["end"] = end.astimezone(UTC).isoformat().replace("+00:00", "Z")
    clean_symbols = sorted({s.strip().upper() for s in symbols or [] if s.strip()})
    if clean_symbols:
        params["symbols"] = ",".join(clean_symbols)

    items: list[dict[str, Any]] = []
    while len(items) < max_items:
        response = _get_with_retries(
            http,
            f"{settings.alpaca_data_base_url}{NEWS_URL}",
            headers=headers,
            params=params,
            max_retries=3,
            backoff_seconds=1.0,
            sleep=sleep,
        )
        payload = response.json()
        items.extend(payload.get("news") or [])
        token = payload.get("next_page_token")
        if not token:
            break
        params["page_token"] = token

    return items[:max_items]


class NewsStore:
    """Archivio locale delle notizie, senza duplicati."""

    def __init__(self, directory: Path | str) -> None:
        self.path = Path(directory) / "news.json"

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return list(json.loads(self.path.read_text(encoding="utf-8")))

    def merge(
        self,
        items: Iterable[dict[str, Any]],
        *,
        feed: str,
        now: datetime | None = None,
    ) -> int:
        """Aggiunge le notizie nuove; restituisce quante erano nuove."""
        cutoff = (now or datetime.now(UTC)) - timedelta(days=RETENTION_DAYS)

        def recent(item: dict[str, Any]) -> bool:
            stamp = item.get("created_at")
            return bool(stamp) and datetime.fromisoformat(stamp.replace("Z", "+00:00")) >= cutoff

        current = {item["id"]: item for item in self.load()}
        added = 0
        for raw in items:
            if not raw.get("headline") or not recent(raw):
                continue
            item = _clean(raw, feed)
            existing = current.get(item["id"])
            if existing is None:
                current[item["id"]] = item
                added += 1
            elif feed not in existing["feeds"]:
                existing["feeds"].append(feed)

        kept = [item for item in current.values() if recent(item)]
        kept.sort(key=lambda item: item["created_at"], reverse=True)
        write_text_atomic(self.path, json.dumps(kept, ensure_ascii=False, indent=1))
        return added


def news_directory(root: Path | str) -> Path:
    return Path(root) / "data" / "news"


def update_news(
    settings: Settings,
    root: Path | str,
    symbols: Iterable[str],
    *,
    days: int = 3,
    now: datetime | None = None,
    session: requests.Session | None = None,
) -> dict[str, int]:
    """Aggiorna entrambi i flussi; restituisce quante notizie nuove per flusso."""
    clock = now or datetime.now(UTC)
    start = clock - timedelta(days=days)
    store = NewsStore(news_directory(root))
    portfolio = fetch_news(settings, symbols=list(symbols), start=start, session=session)
    market = fetch_news(
        settings, symbols=None, start=start, max_items=MARKET_MAX_ITEMS, session=session
    )
    return {
        "portfolio": store.merge(portfolio, feed="portfolio", now=clock),
        "market": store.merge(market, feed="market", now=clock),
    }
