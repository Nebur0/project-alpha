from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class StrategyParameters:
    fast_ema_period: int = 20
    slow_ema_period: int = 50
    trend_ema_period: int = 200
    rsi_period: int = 14
    rsi_lower: float = 50.0
    rsi_upper: float = 70.0
    atr_period: int = 14
    volume_period: int = 20
    breakout_period: int = 20

    def __post_init__(self) -> None:
        periods = {
            "fast_ema_period": self.fast_ema_period,
            "slow_ema_period": self.slow_ema_period,
            "trend_ema_period": self.trend_ema_period,
            "rsi_period": self.rsi_period,
            "atr_period": self.atr_period,
            "volume_period": self.volume_period,
            "breakout_period": self.breakout_period,
        }

        for name, value in periods.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} deve essere un intero positivo.")

        if not (self.fast_ema_period < self.slow_ema_period < self.trend_ema_period):
            raise ValueError("I periodi EMA devono rispettare: fast < slow < trend.")

        if not (0.0 <= self.rsi_lower < self.rsi_upper <= 100.0):
            raise ValueError("Le soglie RSI devono rispettare: 0 <= lower < upper <= 100.")


@dataclass(frozen=True)
class Settings:
    alpaca_api_key: str
    alpaca_api_secret: str
    alpaca_data_base_url: str = "https://data.alpaca.markets"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()

        key = os.getenv(
            "ALPACA_API_KEY",
            "",
        ).strip()

        secret = os.getenv(
            "ALPACA_API_SECRET",
            "",
        ).strip()

        base_url = os.getenv(
            "ALPACA_DATA_BASE_URL",
            "https://data.alpaca.markets",
        ).rstrip("/")

        if not key or not secret or key == "replace_me" or secret == "replace_me":
            raise RuntimeError(
                "Credenziali Alpaca mancanti. "
                "Copia .env.example in .env e inserisci "
                "ALPACA_API_KEY e ALPACA_API_SECRET."
            )

        return cls(
            alpaca_api_key=key,
            alpaca_api_secret=secret,
            alpaca_data_base_url=base_url,
        )
