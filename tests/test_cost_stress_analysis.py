import numpy as np
import pandas as pd
import pytest

from project_alpha.batch_optimization import (
    run_batch_walk_forward_optimization,
)
from project_alpha.cost_stress_analysis import (
    analyze_frozen_selection_cost_stress,
)


def _sample_market_frame(
    symbol: str,
    periods: int = 100,
    phase: float = 0.0,
) -> pd.DataFrame:
    index = np.arange(periods)

    close = 100.0 + index * 0.30 + np.sin(index / 3.0 + phase) * 2.0

    return pd.DataFrame(
        {
            "symbol": [symbol] * periods,
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=periods,
                freq="D",
                tz="UTC",
            ),
            "open": close - 0.50,
            "high": close + 1.00,
            "low": close - 1.00,
            "close": close,
            "volume": 1_000.0 + (index % 10) * 100.0,
        }
    )


def _small_parameter_grid() -> dict[str, list[int | float]]:
    return {
        "fast_ema_period": [3, 4],
        "slow_ema_period": [6],
        "trend_ema_period": [12],
        "rsi_period": [5],
        "rsi_lower": [40.0],
        "rsi_upper": [80.0],
        "atr_period": [5],
        "volume_period": [5],
        "breakout_period": [5],
    }


def _baseline() -> tuple[pd.DataFrame, object]:
    frame = pd.concat(
        [
            _sample_market_frame("AAA", phase=0.0),
            _sample_market_frame("BBB", phase=0.7),
        ],
        ignore_index=True,
    )

    batch = run_batch_walk_forward_optimization(
        frame,
        ["AAA", "BBB"],
        _small_parameter_grid(),
        train_size=40,
        test_size=20,
        step_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
        cost_per_side=0.001,
    )

    return frame, batch


def test_cost_stress_reports_requested_scenarios() -> None:
    frame, batch = _baseline()

    result = analyze_frozen_selection_cost_stress(
        frame,
        batch,
        costs_per_side=[0.0, 0.001, 0.003],
    )

    assert result.scenarios["cost_bps_per_side"].tolist() == [0.0, 10.0, 30.0]


def test_cost_stress_keeps_exposure_constant() -> None:
    frame, batch = _baseline()

    result = analyze_frozen_selection_cost_stress(
        frame,
        batch,
        costs_per_side=[0.0, 0.001, 0.003],
    )

    assert result.scenarios["mean_gross_exposure"].nunique() == 1


def test_higher_cost_reduces_final_return() -> None:
    frame, batch = _baseline()

    result = analyze_frozen_selection_cost_stress(
        frame,
        batch,
        costs_per_side=[0.0, 0.003],
    )

    low = result.scenarios.loc[
        result.scenarios["cost_per_side"].eq(0.0),
        "total_return",
    ].iloc[0]

    high = result.scenarios.loc[
        result.scenarios["cost_per_side"].eq(0.003),
        "total_return",
    ].iloc[0]

    assert high <= low


@pytest.mark.parametrize(
    "costs",
    [
        [],
        [-0.001],
        [float("nan")],
        [float("inf")],
    ],
)
def test_cost_stress_rejects_invalid_costs(
    costs: list[float],
) -> None:
    frame, batch = _baseline()

    with pytest.raises(ValueError):
        analyze_frozen_selection_cost_stress(
            frame,
            batch,
            costs_per_side=costs,
        )


def test_cost_stress_records_baseline_cost() -> None:
    frame, batch = _baseline()

    result = analyze_frozen_selection_cost_stress(
        frame,
        batch,
        baseline_cost_per_side=0.001,
    )

    assert result.baseline_cost_per_side == pytest.approx(0.001)
