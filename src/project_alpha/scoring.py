from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from statistics import median


MetricValue = float | int | None


@dataclass(frozen=True)
class RobustScoreConfig:
    sharpe_weight: float = 1.0
    excess_return_weight: float = 1.0
    drawdown_weight: float = 1.0
    trade_shortfall_weight: float = 0.5
    target_completed_trades: int = 20

    def __post_init__(self) -> None:
        weights = {
            "sharpe_weight": self.sharpe_weight,
            "excess_return_weight": (
                self.excess_return_weight
            ),
            "drawdown_weight": (
                self.drawdown_weight
            ),
            "trade_shortfall_weight": (
                self.trade_shortfall_weight
            ),
        }

        for name, value in weights.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"{name} deve essere un numero "
                    "finito non negativo."
                )

        if (
            isinstance(
                self.target_completed_trades,
                bool,
            )
            or not isinstance(
                self.target_completed_trades,
                int,
            )
            or self.target_completed_trades <= 0
        ):
            raise ValueError(
                "target_completed_trades deve "
                "essere un intero positivo."
            )


@dataclass(frozen=True)
class RobustScoreResult:
    score: float
    sharpe: float
    total_return: float
    benchmark_total_return: float
    excess_return: float
    max_drawdown: float
    completed_trades: int
    trade_shortfall: float


@dataclass(frozen=True)
class RobustScoreV3Config:
    base_score_weight: float = 0.5
    neighborhood_score_weight: float = 0.5

    def __post_init__(self) -> None:
        weights = {
            "base_score_weight": (
                self.base_score_weight
            ),
            "neighborhood_score_weight": (
                self.neighborhood_score_weight
            ),
        }

        for name, value in weights.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"{name} deve essere un numero "
                    "finito non negativo."
                )

        if (
            float(self.base_score_weight)
            + float(
                self.neighborhood_score_weight
            )
            <= 0.0
        ):
            raise ValueError(
                "Almeno uno dei pesi del robust "
                "score v3 deve essere positivo."
            )


@dataclass(frozen=True)
class RobustScoreV3Result:
    score: float
    base_score: float
    neighborhood_score: float
    neighbor_count: int


def _required_float_metric(
    metrics: Mapping[
        str,
        MetricValue,
    ],
    name: str,
) -> float:
    value = metrics.get(name)

    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"Metrica '{name}' mancante "
            "o non numerica."
        )

    numeric_value = float(value)

    if not isfinite(numeric_value):
        raise ValueError(
            f"Metrica '{name}' non finita."
        )

    return numeric_value


def _required_trade_count(
    metrics: Mapping[
        str,
        MetricValue,
    ],
) -> int:
    value = metrics.get(
        "completed_trades"
    )

    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise ValueError(
            "Metrica 'completed_trades' "
            "mancante o non numerica."
        )

    numeric_value = float(value)

    if (
        not isfinite(numeric_value)
        or numeric_value < 0.0
        or not numeric_value.is_integer()
    ):
        raise ValueError(
            "Metrica 'completed_trades' "
            "deve essere un intero non negativo."
        )

    return int(numeric_value)


def _required_score_value(
    value: Real,
    *,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} deve essere un numero finito."
        )

    numeric_value = float(value)

    if not isfinite(numeric_value):
        raise ValueError(
            f"{name} deve essere un numero finito."
        )

    return numeric_value


def calculate_robust_score(
    metrics: Mapping[
        str,
        MetricValue,
    ],
    config: RobustScoreConfig | None = None,
) -> RobustScoreResult:
    effective_config = (
        config or RobustScoreConfig()
    )

    sharpe = _required_float_metric(
        metrics,
        "annualized_sharpe",
    )

    total_return = _required_float_metric(
        metrics,
        "total_return",
    )

    benchmark_total_return = (
        _required_float_metric(
            metrics,
            "benchmark_total_return",
        )
    )

    max_drawdown = _required_float_metric(
        metrics,
        "max_drawdown",
    )

    completed_trades = (
        _required_trade_count(
            metrics
        )
    )

    excess_return = (
        total_return
        - benchmark_total_return
    )

    trade_shortfall = max(
        0.0,
        (
            effective_config
            .target_completed_trades
            - completed_trades
        )
        / effective_config
        .target_completed_trades,
    )

    score = (
        effective_config.sharpe_weight
        * sharpe
        + effective_config.excess_return_weight
        * excess_return
        + effective_config.drawdown_weight
        * max_drawdown
        - effective_config.trade_shortfall_weight
        * trade_shortfall
    )

    return RobustScoreResult(
        score=score,
        sharpe=sharpe,
        total_return=total_return,
        benchmark_total_return=(
            benchmark_total_return
        ),
        excess_return=excess_return,
        max_drawdown=max_drawdown,
        completed_trades=completed_trades,
        trade_shortfall=trade_shortfall,
    )


def calculate_robust_score_v3(
    base_score: Real,
    neighbor_scores: Sequence[Real],
    config: RobustScoreV3Config | None = None,
) -> RobustScoreV3Result:
    effective_config = (
        config or RobustScoreV3Config()
    )

    numeric_base_score = (
        _required_score_value(
            base_score,
            name="base_score",
        )
    )

    numeric_neighbor_scores = [
        _required_score_value(
            value,
            name="neighbor_score",
        )
        for value in neighbor_scores
    ]

    if numeric_neighbor_scores:
        neighborhood_score = float(
            median(
                numeric_neighbor_scores
            )
        )
    else:
        neighborhood_score = (
            numeric_base_score
        )

    score = (
        effective_config.base_score_weight
        * numeric_base_score
        + effective_config
        .neighborhood_score_weight
        * neighborhood_score
    )

    return RobustScoreV3Result(
        score=score,
        base_score=numeric_base_score,
        neighborhood_score=(
            neighborhood_score
        ),
        neighbor_count=len(
            numeric_neighbor_scores
        ),
    )
