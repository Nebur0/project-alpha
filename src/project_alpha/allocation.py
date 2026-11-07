"""Portafoglio di policy: sleeve con pesi decisi a priori + volatility targeting.

Il portafoglio è composto da "sleeve" (es. un core passivo e un overlay
di trend), ciascuna descritta dalla propria serie di rendimenti giornalieri.
I pesi delle sleeve sono **decisi a priori** (Investment Policy Statement),
non ottimizzati sugli stessi dati; la quota non allocata resta in cash.

Opzionalmente l'intero portafoglio viene scalato per puntare a una
volatilità annua obiettivo, usando solo la volatilità realizzata fino al
giorno precedente (nessuna informazione futura) e senza mai usare leva.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
import pandas as pd

from project_alpha.evaluation import TRADING_DAYS, performance_summary


@dataclass(frozen=True)
class VolTargetConfig:
    target_volatility: float = 0.10
    lookback_days: int = 63
    max_exposure: float = 1.0
    min_exposure: float = 0.0
    # Esposizione usata finché non ci sono abbastanza dati.
    warmup_exposure: float = 1.0
    # Ribilancia solo se l'esposizione cambia almeno di questa quantità.
    rebalance_threshold: float = 0.05

    def __post_init__(self) -> None:
        if not (isfinite(self.target_volatility) and self.target_volatility > 0):
            raise ValueError("target_volatility deve essere positiva.")

        if isinstance(self.lookback_days, bool) or self.lookback_days < 2:
            raise ValueError("lookback_days deve essere almeno 2.")

        if not 0.0 <= self.min_exposure <= self.max_exposure <= 1.0:
            raise ValueError("Serve 0 <= min_exposure <= max_exposure <= 1 (niente leva).")

        if not self.min_exposure <= self.warmup_exposure <= self.max_exposure:
            raise ValueError("warmup_exposure deve stare tra min_exposure e max_exposure.")

        if not 0.0 <= self.rebalance_threshold < 1.0:
            raise ValueError("rebalance_threshold deve essere in [0, 1).")


def volatility_target_exposure(
    returns: pd.Series,
    config: VolTargetConfig,
) -> pd.Series:
    """Esposizione giornaliera per il vol targeting, senza look-ahead.

    L'esposizione del giorno t usa la volatilità dei rendimenti fino a t-1.
    """
    realized = returns.rolling(config.lookback_days, min_periods=config.lookback_days).std(ddof=1)
    realized = realized * sqrt(TRADING_DAYS)
    raw = (config.target_volatility / realized).clip(config.min_exposure, config.max_exposure)
    raw = raw.shift(1)
    # Volatilità nulla: nessun rischio misurato, esposizione massima ammessa.
    raw = raw.mask(realized.shift(1) == 0, config.max_exposure)
    raw = raw.fillna(config.warmup_exposure)

    if config.rebalance_threshold <= 0:
        return raw

    # Isteresi: si cambia esposizione solo se lo scostamento è significativo.
    values = raw.to_numpy(dtype=float)
    applied = np.empty_like(values)
    current = values[0]

    for index, target in enumerate(values):
        if abs(target - current) >= config.rebalance_threshold:
            current = target
        applied[index] = current

    return pd.Series(applied, index=raw.index)


@dataclass(frozen=True)
class PolicyPortfolioResult:
    returns: pd.Series
    exposure: pd.Series
    sleeve_weights: dict[str, float]
    metrics: dict[str, float | None]


def run_policy_portfolio(
    sleeves: Mapping[str, pd.Series],
    weights: Mapping[str, float],
    *,
    vol_target: VolTargetConfig | None = None,
    cost_per_side: float = 0.001,
    cash_rate_annual: float = 0.0,
) -> PolicyPortfolioResult:
    """Combina le sleeve con pesi fissi, ribilanciati giornalmente.

    ``sleeves`` sono rendimenti giornalieri allineati sullo stesso indice;
    la somma di ``weights`` non può superare 1 (il resto è cash).
    Con ``vol_target`` il portafoglio viene scalato; il costo del cambio di
    esposizione è ``|delta esposizione| * cost_per_side``.
    """
    if not sleeves:
        raise ValueError("Serve almeno una sleeve.")

    if set(sleeves) != set(weights):
        raise ValueError("Sleeve e pesi devono avere gli stessi nomi.")

    total_weight = 0.0
    for name, weight in weights.items():
        if not isfinite(weight) or weight < 0:
            raise ValueError(f"Peso non valido per la sleeve {name}.")
        total_weight += weight

    if total_weight > 1.0 + 1e-12:
        raise ValueError("La somma dei pesi delle sleeve non può superare 1 (niente leva).")

    if cost_per_side < 0 or not isfinite(cost_per_side):
        raise ValueError("cost_per_side deve essere finito e non negativo.")

    frame = pd.DataFrame(dict(sleeves))

    if frame.isna().any().any():
        raise ValueError("Le sleeve devono essere allineate e senza valori mancanti.")

    daily_cash = (1.0 + cash_rate_annual) ** (1.0 / TRADING_DAYS) - 1.0
    risky = sum(frame[name] * weight for name, weight in weights.items())

    if vol_target is None:
        exposure = pd.Series(1.0, index=frame.index)
    else:
        exposure = volatility_target_exposure(risky, vol_target)

    turnover = exposure.diff().abs().fillna(0.0)
    invested = exposure * total_weight
    returns = (
        exposure * risky + (1.0 - invested) * daily_cash - turnover * total_weight * cost_per_side
    )

    metrics: dict[str, float | None] = dict(performance_summary(returns))
    metrics["mean_sleeve_allocation"] = float(invested.mean())
    metrics["min_sleeve_allocation"] = float(invested.min())

    return PolicyPortfolioResult(
        returns=returns,
        exposure=invested,
        sleeve_weights=dict(weights),
        metrics=metrics,
    )
