from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    alpaca_api_key: str
    alpaca_api_secret: str
    alpaca_data_base_url: str = "https://data.alpaca.markets"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        key = os.getenv("ALPACA_API_KEY", "").strip()
        secret = os.getenv("ALPACA_API_SECRET", "").strip()
        base_url = os.getenv(
            "ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"
        ).rstrip("/")

        if not key or not secret or key == "replace_me" or secret == "replace_me":
            raise RuntimeError(
                "Credenziali Alpaca mancanti. Copia .env.example in .env e inserisci "
                "ALPACA_API_KEY e ALPACA_API_SECRET."
            )

        return cls(
            alpaca_api_key=key,
            alpaca_api_secret=secret,
            alpaca_data_base_url=base_url,
        )
