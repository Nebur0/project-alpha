from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np


class RiskDecisionStatus(str, Enum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"


@dataclass(frozen=True)
class RiskPolicy:
    max_position_weight: float = 0.20
    max_gross_exposure: float = 1.00
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.15

    def __post_init__(self) -> None:
        limits = {
            "max_position_weight": self.max_position_weight,
            "max_gross_exposure": self.max_gross_exposure,
            "max_daily_loss": self.max_daily_loss,
            "max_drawdown": self.max_drawdown,
        }

        for name, value in limits.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"{name} deve essere un numero positivo e finito."
                )


@dataclass(frozen=True)
class RiskDecision:
    status: RiskDecisionStatus
    violations: tuple[str, ...]
    gross_exposure: float
    max_absolute_position_weight: float
    current_drawdown: float
    daily_return: float

    @property
    def allowed(self) -> bool:
        return self.status is RiskDecisionStatus.ALLOW


def _normalize_weights(
    proposed_weights: Mapping[str, float],
) -> dict[str, float]:
    if not proposed_weights:
        raise ValueError(
            "È necessario specificare almeno un peso proposto."
        )

    normalized: dict[str, float] = {}

    for raw_symbol, raw_weight in proposed_weights.items():
        symbol = str(raw_symbol).strip().upper()

        if not symbol:
            raise ValueError(
                "I simboli della proposta non possono essere vuoti."
            )

        if symbol in normalized:
            raise ValueError(
                f"Simbolo duplicato nella proposta: {symbol}."
            )

        if (
            isinstance(raw_weight, bool)
            or not isinstance(raw_weight, (int, float))
        ):
            raise ValueError(
                f"Il peso di {symbol} deve essere numerico."
            )

        weight = float(raw_weight)

        if not np.isfinite(weight):
            raise ValueError(
                f"Il peso di {symbol} deve essere finito."
            )

        normalized[symbol] = weight

    return normalized


def _validate_equity_state(
    current_equity: float,
    peak_equity: float,
    daily_return: float,
) -> tuple[float, float, float]:
    values = {
        "current_equity": current_equity,
        "peak_equity": peak_equity,
        "daily_return": daily_return,
    }

    normalized: dict[str, float] = {}

    for name, raw_value in values.items():
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
        ):
            raise ValueError(
                f"{name} deve essere numerico."
            )

        value = float(raw_value)

        if not np.isfinite(value):
            raise ValueError(
                f"{name} deve essere finito."
            )

        normalized[name] = value

    if normalized["current_equity"] <= 0.0:
        raise ValueError(
            "current_equity deve essere positivo."
        )

    if normalized["peak_equity"] <= 0.0:
        raise ValueError(
            "peak_equity deve essere positivo."
        )

    if normalized["current_equity"] > normalized["peak_equity"]:
        raise ValueError(
            "current_equity non può essere maggiore di peak_equity."
        )

    return (
        normalized["current_equity"],
        normalized["peak_equity"],
        normalized["daily_return"],
    )


def evaluate_risk_policy(
    proposed_weights: Mapping[str, float],
    *,
    current_equity: float,
    peak_equity: float,
    daily_return: float,
    policy: RiskPolicy | None = None,
) -> RiskDecision:
    effective_policy = RiskPolicy() if policy is None else policy

    weights = _normalize_weights(proposed_weights)

    (
        normalized_current_equity,
        normalized_peak_equity,
        normalized_daily_return,
    ) = _validate_equity_state(
        current_equity,
        peak_equity,
        daily_return,
    )

    gross_exposure = float(
        sum(abs(weight) for weight in weights.values())
    )

    max_absolute_position_weight = float(
        max(abs(weight) for weight in weights.values())
    )

    current_drawdown = float(
        normalized_current_equity / normalized_peak_equity - 1.0
    )

    violations: list[str] = []

    if (
        max_absolute_position_weight
        > effective_policy.max_position_weight
    ):
        violations.append("MAX_POSITION_WEIGHT")

    if (
        gross_exposure
        > effective_policy.max_gross_exposure
    ):
        violations.append("MAX_GROSS_EXPOSURE")

    if (
        normalized_daily_return
        < -effective_policy.max_daily_loss
    ):
        violations.append("MAX_DAILY_LOSS")

    if (
        current_drawdown
        < -effective_policy.max_drawdown
    ):
        violations.append("MAX_DRAWDOWN")

    status = (
        RiskDecisionStatus.ALLOW
        if not violations
        else RiskDecisionStatus.REJECT
    )

    return RiskDecision(
        status=status,
        violations=tuple(violations),
        gross_exposure=gross_exposure,
        max_absolute_position_weight=max_absolute_position_weight,
        current_drawdown=current_drawdown,
        daily_return=normalized_daily_return,
    )
