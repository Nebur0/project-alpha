import numpy as np
import pandas as pd

from project_alpha.config import StrategyParameters
from project_alpha.indicators import (
    add_indicators,
    atr,
    ema,
    rsi,
)


def test_ema_preserves_length() -> None:
    series = pd.Series(np.arange(1.0, 101.0))

    result = ema(
        series,
        20,
    )

    assert len(result) == len(series)
    assert result.iloc[:19].isna().all()
    assert result.iloc[19:].notna().all()


def test_rsi_uptrend_reaches_high_value() -> None:
    series = pd.Series(np.arange(1.0, 101.0))

    result = rsi(
        series,
        14,
    )

    assert result.iloc[-1] == 100.0


def test_atr_constant_range() -> None:
    frame = pd.DataFrame(
        {
            "high": np.arange(
                11.0,
                51.0,
            ),
            "low": np.arange(
                9.0,
                49.0,
            ),
            "close": np.arange(
                10.0,
                50.0,
            ),
        }
    )

    result = atr(
        frame,
        14,
    )

    assert result.iloc[-1] == 2.0


def test_add_indicators_uses_custom_strategy_parameters() -> None:
    periods = 30

    frame = pd.DataFrame(
        {
            "symbol": ["TEST"] * periods,
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=periods,
                freq="D",
                tz="UTC",
            ),
            "open": np.arange(
                100.0,
                100.0 + periods,
            ),
            "high": np.arange(
                101.0,
                101.0 + periods,
            ),
            "low": np.arange(
                99.0,
                99.0 + periods,
            ),
            "close": np.arange(
                100.5,
                100.5 + periods,
            ),
            "volume": np.arange(
                1_000.0,
                1_000.0 + periods,
            ),
        }
    )

    parameters = StrategyParameters(
        fast_ema_period=3,
        slow_ema_period=5,
        trend_ema_period=10,
        rsi_period=4,
        rsi_lower=45.0,
        rsi_upper=75.0,
        atr_period=3,
        volume_period=4,
        breakout_period=5,
    )

    result = add_indicators(
        frame,
        parameters,
    )

    expected_columns = {
        "ema_3",
        "ema_5",
        "ema_10",
        "rsi_4",
        "atr_3",
        "volume_sma_4",
        "previous_high_5",
    }

    assert expected_columns.issubset(result.columns)

    assert result["ema_3"].iloc[:2].isna().all()
    assert result["ema_3"].iloc[2:].notna().all()

    assert result["volume_sma_4"].iloc[:3].isna().all()

    assert result["previous_high_5"].iloc[:5].isna().all()

    assert result["previous_high_5"].iloc[5:].notna().all()
