from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from datetime import date

import pandas as pd
import requests

from project_alpha.config import Settings

_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
]

# Errori temporanei per cui ha senso riprovare.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
FEEDS = ("iex", "sip")


def _get_with_retries(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, str | int],
    max_retries: int,
    backoff_seconds: float,
    sleep: Callable[[float], None],
) -> requests.Response:
    """GET con retry esponenziale su errori temporanei (429, 5xx, rete)."""
    attempt = 0

    while True:
        try:
            response = session.get(url, headers=headers, params=params, timeout=45)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Alpaca non raggiungibile dopo {attempt + 1} tentativi: {exc}"
                ) from exc
            sleep(backoff_seconds * 2**attempt)
            attempt += 1
            continue

        if response.status_code in RETRYABLE_STATUS and attempt < max_retries:
            retry_after = response.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after is not None else None
            except ValueError:
                wait = None
            sleep(wait if wait is not None else backoff_seconds * 2**attempt)
            attempt += 1
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:500]
            raise RuntimeError(f"Errore Alpaca HTTP {response.status_code}: {detail}") from exc

        return response


def fetch_daily_bars(
    settings: Settings,
    symbols: Iterable[str],
    start: date,
    end: date | None = None,
    *,
    feed: str | None = None,
    session: requests.Session | None = None,
    max_retries: int = 4,
    backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Scarica barre giornaliere aggiustate (split e dividendi) da Alpaca.

    ``feed`` sceglie la fonte (``"iex"`` o ``"sip"``); se ``None`` decide
    Alpaca in base al piano dati. Registrare il feed usato rende i dati
    confrontabili tra un download e l'altro.
    """
    clean_symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    if not clean_symbols:
        raise ValueError("Fornire almeno un simbolo.")

    if feed is not None and feed not in FEEDS:
        raise ValueError(f"feed deve essere uno tra {list(FEEDS)}.")

    if max_retries < 0:
        raise ValueError("max_retries deve essere non negativo.")

    url = f"{settings.alpaca_data_base_url}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
        "accept": "application/json",
    }
    params: dict[str, str | int] = {
        "symbols": ",".join(clean_symbols),
        "timeframe": "1Day",
        "start": start.isoformat(),
        "limit": 10000,
        "adjustment": "all",
        "sort": "asc",
    }
    if end is not None:
        params["end"] = end.isoformat()
    if feed is not None:
        params["feed"] = feed

    http = session if session is not None else requests.Session()
    records: list[dict[str, object]] = []
    next_page_token: str | None = None

    while True:
        if next_page_token:
            params["page_token"] = next_page_token
        elif "page_token" in params:
            del params["page_token"]

        response = _get_with_retries(
            http,
            url,
            headers=headers,
            params=params,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
        )

        payload = response.json()
        bars_by_symbol = payload.get("bars") or {}

        for symbol, bars in bars_by_symbol.items():
            for bar in bars:
                records.append(
                    {
                        "symbol": symbol,
                        "timestamp": bar["t"],
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar["v"],
                        "trade_count": bar.get("n"),
                        "vwap": bar.get("vw"),
                    }
                )

        next_page_token = payload.get("next_page_token")
        if not next_page_token:
            break

    if not records:
        raise RuntimeError(
            "Nessuna candela ricevuta. Controllare simboli, intervallo, credenziali e piano dati."
        )

    frame = pd.DataFrame.from_records(records, columns=_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
