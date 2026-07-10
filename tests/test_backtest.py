import pandas as pd
import pytest

from project_alpha.backtest import run_long_only_backtest


def test_signal_is_applied_next_day() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC"),
            "close": [100.0, 110.0, 121.0, 133.1],
            "target_position": [0, 1, 1, 0],
        }
    )

    result = run_long_only_backtest(frame, initial_capital=1000.0, cost_per_side=0.0)
    curve = result.equity_curve

    assert curve.loc[1, "executed_position"] == 0
    assert curve.loc[2, "executed_position"] == 1
    assert curve.loc[2, "strategy_return"] == pytest.approx(0.10)


def test_cost_is_charged_on_position_change() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC"),
            "close": [100.0, 100.0, 100.0, 100.0],
            "target_position": [1, 1, 0, 0],
        }
    )

    result = run_long_only_backtest(frame, initial_capital=1000.0, cost_per_side=0.001)
    assert result.equity_curve["trading_cost"].sum() == pytest.approx(0.002)
