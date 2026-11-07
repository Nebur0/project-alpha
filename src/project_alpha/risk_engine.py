"""Risk Engine V2: ultimo filtro deterministico prima di ogni ordine.

Qualunque proposta di allocazione (da una strategia, da un ottimizzatore o
in futuro da un agente AI) passa da ``evaluate_proposal``. Il modulo non
decide *cosa* comprare: decide soltanto se una proposta rispetta la policy.

Principi:

- **Fail-closed.** Dati vecchi, stato incoerente o kill switch attivo:
  nessun aumento di rischio.
- **Ridurre il rischio non è mai vietato.** Un peso che scende in valore
  assoluto non viola i limiti superiori, anche se resta sopra il limite
  (es. dopo un movimento di mercato) e anche se il turnover è alto.
- **Modalità solo-riduzione.** Dopo una perdita giornaliera o un drawdown
  oltre soglia, sono ammesse solo proposte che riducono ogni posizione.
- **Long-only di default.** Pesi negativi rifiutati se lo short non è
  esplicitamente consentito.
- La policy è dati immutabili: nessuna proposta può modificarla.

``risk.evaluate_risk_policy`` (Risk Gate V1) resta disponibile invariato.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite

import pandas as pd

WEIGHT_TOLERANCE = 1e-9


class RiskMode(StrEnum):
    NORMAL = "NORMAL"
    REDUCE_ONLY = "REDUCE_ONLY"
    HALTED = "HALTED"


class Violation(StrEnum):
    UNKNOWN_SYMBOL = "UNKNOWN_SYMBOL"
    SHORT_NOT_ALLOWED = "SHORT_NOT_ALLOWED"
    MAX_POSITION_WEIGHT = "MAX_POSITION_WEIGHT"
    MAX_GROSS_EXPOSURE = "MAX_GROSS_EXPOSURE"
    MAX_GROUP_WEIGHT = "MAX_GROUP_WEIGHT"
    MIN_CASH_WEIGHT = "MIN_CASH_WEIGHT"
    MAX_TURNOVER = "MAX_TURNOVER"
    MAX_ORDER_NOTIONAL = "MAX_ORDER_NOTIONAL"
    STALE_DATA = "STALE_DATA"
    REDUCE_ONLY_MODE = "REDUCE_ONLY_MODE"
    KILL_SWITCH = "KILL_SWITCH"


def _positive_finite(name: str, value: float | None, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} deve essere un numero finito.")

    if value <= 0:
        raise ValueError(f"{name} deve essere positivo.")


@dataclass(frozen=True)
class RiskPolicyV2:
    """Limiti decisi dal proprietario. Valori di default prudenti, non ottimizzati."""

    max_position_weight: float = 0.20
    max_gross_exposure: float = 1.00
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.15
    allow_short: bool = False
    # Peso massimo per gruppo (settore, asset class, cluster di correlazione).
    max_group_weight: float | None = 0.40
    groups: Mapping[str, str] = field(default_factory=dict)
    min_cash_weight: float = 0.0
    # Somma dei |delta peso| ammessa in un singolo ribilanciamento.
    max_turnover: float | None = 0.50
    # Notional massimo per singolo ordine, in valuta del conto.
    max_order_notional: float | None = None
    # Età massima dei dati di mercato, in giorni di calendario.
    max_data_age_days: float = 4.0
    # Simboli ammessi; None = qualunque simbolo.
    allowed_symbols: frozenset[str] | None = None
    kill_switch: bool = False

    def __post_init__(self) -> None:
        _positive_finite("max_position_weight", self.max_position_weight)
        _positive_finite("max_gross_exposure", self.max_gross_exposure)
        _positive_finite("max_daily_loss", self.max_daily_loss)
        _positive_finite("max_drawdown", self.max_drawdown)
        _positive_finite("max_group_weight", self.max_group_weight, allow_none=True)
        _positive_finite("max_turnover", self.max_turnover, allow_none=True)
        _positive_finite("max_order_notional", self.max_order_notional, allow_none=True)
        _positive_finite("max_data_age_days", self.max_data_age_days)

        if not 0.0 <= self.min_cash_weight < 1.0:
            raise ValueError("min_cash_weight deve essere compreso in [0, 1).")

        if self.max_gross_exposure > 1.0 + WEIGHT_TOLERANCE and not self.allow_short:
            raise ValueError(
                "Leva non ammessa: max_gross_exposure > 1 richiede una policy esplicita "
                "e questo motore è long-only senza leva."
            )

        if self.allowed_symbols is not None:
            object.__setattr__(
                self,
                "allowed_symbols",
                frozenset(_normalize_symbol(symbol) for symbol in self.allowed_symbols),
            )

        object.__setattr__(
            self,
            "groups",
            {_normalize_symbol(symbol): str(group) for symbol, group in self.groups.items()},
        )


@dataclass(frozen=True)
class PortfolioState:
    """Stato del conto al momento della decisione."""

    equity: float
    peak_equity: float
    daily_return: float
    current_weights: Mapping[str, float]
    data_timestamp: datetime | pd.Timestamp
    as_of: datetime | pd.Timestamp

    def __post_init__(self) -> None:
        _positive_finite("equity", self.equity)
        _positive_finite("peak_equity", self.peak_equity)

        if self.equity > self.peak_equity * (1 + WEIGHT_TOLERANCE):
            raise ValueError("equity non può superare peak_equity.")

        if not isfinite(self.daily_return):
            raise ValueError("daily_return deve essere finito.")

        object.__setattr__(self, "current_weights", _normalize_weights(self.current_weights))

    @property
    def drawdown(self) -> float:
        return self.equity / self.peak_equity - 1.0

    @property
    def data_age_days(self) -> float:
        as_of = pd.Timestamp(self.as_of)
        data = pd.Timestamp(self.data_timestamp)

        if (as_of.tzinfo is None) != (data.tzinfo is None):
            raise ValueError("as_of e data_timestamp devono avere entrambi (o nessuno) il fuso.")

        return (as_of - data).total_seconds() / 86_400.0


@dataclass(frozen=True)
class RiskDecisionV2:
    allowed: bool
    mode: RiskMode
    violations: tuple[str, ...]
    details: tuple[str, ...]
    gross_exposure: float
    turnover: float
    proposed_weights: Mapping[str, float]


def _normalize_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().upper()

    if not normalized:
        raise ValueError("Simbolo vuoto.")

    return normalized


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}

    for raw_symbol, raw_weight in weights.items():
        symbol = _normalize_symbol(raw_symbol)

        if symbol in normalized:
            raise ValueError(f"Simbolo duplicato: {symbol}.")

        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(f"Il peso di {symbol} deve essere numerico.")

        weight = float(raw_weight)

        if not isfinite(weight):
            raise ValueError(f"Il peso di {symbol} deve essere finito.")

        normalized[symbol] = weight

    return normalized


def _increases(new: float, old: float) -> bool:
    return abs(new) > abs(old) + WEIGHT_TOLERANCE


def _is_pure_reduction(
    proposed: Mapping[str, float],
    current: Mapping[str, float],
) -> bool:
    """Ogni posizione resta nello stesso verso e non cresce in valore assoluto."""
    for symbol in set(proposed) | set(current):
        new = proposed.get(symbol, 0.0)
        old = current.get(symbol, 0.0)

        if _increases(new, old):
            return False

        if new * old < -WEIGHT_TOLERANCE:
            return False

    return True


def _group_weights(weights: Mapping[str, float], groups: Mapping[str, str]) -> dict[str, float]:
    totals: dict[str, float] = {}

    for symbol, weight in weights.items():
        group = groups.get(symbol)

        if group is not None:
            totals[group] = totals.get(group, 0.0) + abs(weight)

    return totals


def evaluate_proposal(
    proposed_weights: Mapping[str, float],
    state: PortfolioState,
    policy: RiskPolicyV2 | None = None,
) -> RiskDecisionV2:
    """Valuta una proposta di pesi target contro la policy e lo stato del conto.

    ``proposed_weights`` sono pesi target sull'equity; i simboli assenti
    sono considerati a peso zero (cioè da vendere se oggi presenti).
    """
    effective = RiskPolicyV2() if policy is None else policy
    proposed = _normalize_weights(proposed_weights)
    current = dict(state.current_weights)
    symbols = set(proposed) | set(current)

    violations: list[Violation] = []
    details: list[str] = []

    def flag(violation: Violation, detail: str) -> None:
        if violation not in violations:
            violations.append(violation)
        details.append(detail)

    gross = sum(abs(weight) for weight in proposed.values())
    current_gross = sum(abs(weight) for weight in current.values())
    turnover = sum(abs(proposed.get(symbol, 0.0) - current.get(symbol, 0.0)) for symbol in symbols)
    reduction = _is_pure_reduction(proposed, current)

    # --- Modalità -----------------------------------------------------
    stale = state.data_age_days > effective.max_data_age_days
    loss_breach = (
        state.daily_return < -effective.max_daily_loss or state.drawdown < -effective.max_drawdown
    )

    if effective.kill_switch:
        mode = RiskMode.HALTED
    elif loss_breach:
        mode = RiskMode.REDUCE_ONLY
    else:
        mode = RiskMode.NORMAL

    # --- Stato: dati, kill switch, solo-riduzione ----------------------
    # Con dati vecchi non si aumenta il rischio; ridurre resta possibile.
    if stale and not reduction:
        flag(
            Violation.STALE_DATA,
            f"Dati vecchi di {state.data_age_days:.1f} giorni "
            f"(massimo {effective.max_data_age_days}).",
        )

    if mode is RiskMode.HALTED and not reduction:
        flag(Violation.KILL_SWITCH, "Kill switch attivo: ammesse solo riduzioni.")

    if mode is RiskMode.REDUCE_ONLY and not reduction:
        flag(
            Violation.REDUCE_ONLY_MODE,
            f"Modalità solo-riduzione (perdita giornaliera {state.daily_return:.2%}, "
            f"drawdown {state.drawdown:.2%}): la proposta aumenta il rischio.",
        )

    # --- Struttura della proposta --------------------------------------
    for symbol in sorted(proposed):
        weight = proposed[symbol]
        old = current.get(symbol, 0.0)

        if effective.allowed_symbols is not None and symbol not in effective.allowed_symbols:
            if abs(weight) > WEIGHT_TOLERANCE and _increases(weight, old):
                flag(Violation.UNKNOWN_SYMBOL, f"{symbol} non è tra i simboli ammessi.")

        if weight < -WEIGHT_TOLERANCE and not effective.allow_short:
            flag(Violation.SHORT_NOT_ALLOWED, f"{symbol}: peso negativo {weight:.2%}.")

        if abs(weight) > effective.max_position_weight + WEIGHT_TOLERANCE and _increases(
            weight, old
        ):
            flag(
                Violation.MAX_POSITION_WEIGHT,
                f"{symbol}: {abs(weight):.2%} > {effective.max_position_weight:.2%}.",
            )

    if gross > effective.max_gross_exposure + WEIGHT_TOLERANCE and gross > current_gross:
        flag(
            Violation.MAX_GROSS_EXPOSURE,
            f"Esposizione lorda {gross:.2%} > {effective.max_gross_exposure:.2%}.",
        )

    cash_weight = 1.0 - sum(proposed.values())
    if cash_weight < effective.min_cash_weight - WEIGHT_TOLERANCE and not reduction:
        flag(
            Violation.MIN_CASH_WEIGHT,
            f"Cash {cash_weight:.2%} < minimo {effective.min_cash_weight:.2%}.",
        )

    if effective.max_group_weight is not None and effective.groups:
        new_groups = _group_weights(proposed, effective.groups)
        old_groups = _group_weights(current, effective.groups)

        for group, total in sorted(new_groups.items()):
            if (
                total > effective.max_group_weight + WEIGHT_TOLERANCE
                and total > old_groups.get(group, 0.0) + WEIGHT_TOLERANCE
            ):
                flag(
                    Violation.MAX_GROUP_WEIGHT,
                    f"Gruppo {group}: {total:.2%} > {effective.max_group_weight:.2%}.",
                )

    # Turnover e notional limitano i ribilanciamenti, mai le riduzioni.
    if not reduction:
        if (
            effective.max_turnover is not None
            and turnover > effective.max_turnover + WEIGHT_TOLERANCE
        ):
            flag(
                Violation.MAX_TURNOVER,
                f"Turnover {turnover:.2%} > {effective.max_turnover:.2%}.",
            )

        if effective.max_order_notional is not None:
            for symbol in sorted(symbols):
                delta = abs(proposed.get(symbol, 0.0) - current.get(symbol, 0.0))
                notional = delta * state.equity

                if notional > effective.max_order_notional + 1e-6:
                    flag(
                        Violation.MAX_ORDER_NOTIONAL,
                        f"{symbol}: ordine da {notional:,.2f} > {effective.max_order_notional:,.2f}.",
                    )

    return RiskDecisionV2(
        allowed=not violations,
        mode=mode,
        violations=tuple(str(violation) for violation in violations),
        details=tuple(details),
        gross_exposure=gross,
        turnover=turnover,
        proposed_weights=proposed,
    )
