from __future__ import annotations

from datetime import date
from typing import Iterable

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


def fetch_daily_bars(
    settings: Settings,
    symbols: Iterable[str],
    start: date,
    end: date | None = None,
) -> pd.DataFrame:
    clean_symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    if not clean_symbols:
        raise ValueError("Fornire almeno un simbolo.")

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

    records: list[dict[str, object]] = []
    next_page_token: str | None = None

    while True:
        if next_page_token:
            params["page_token"] = next_page_token
        elif "page_token" in params:
            del params["page_token"]

        response = requests.get(url, headers=headers, params=params, timeout=45)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:500]
            raise RuntimeError(
                f"Errore Alpaca HTTP {response.status_code}: {detail}"
            ) from exc

        payload = response.json()
        bars_by_symbol = payload.get("bars", {})

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
            "Nessuna candela ricevuta. Controllare simboli, intervallo, credenziali "
            "e piano dati."
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
