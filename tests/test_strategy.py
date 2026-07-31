import pandas as pd
import pytest

from project_alpha.config import StrategyParameters
from project_alpha.strategy import add_baseline_position


def test_add_baseline_position_uses_default_parameters() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["TEST"] * 3,
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=3,
                freq="D",
                tz="UTC",
            ),
            "close": [
                110.0,
                112.0,
                95.0,
            ],
            "volume": [
                200.0,
                200.0,
                200.0,
            ],
            "ema_20": [
                105.0,
                106.0,
                100.0,
            ],
            "ema_50": [
                100.0,
                101.0,
                99.0,
            ],
            "ema_200": [
                90.0,
                90.0,
                90.0,
            ],
            "rsi_14": [
                60.0,
                60.0,
                60.0,
            ],
            "volume_sma_20": [
                100.0,
                100.0,
                100.0,
            ],
            "previous_high_20": [
                109.0,
                111.0,
                94.0,
            ],
        }
    )

    result = add_baseline_position(frame)

    assert result["entry_signal"].tolist() == [
        True,
        False,
        False,
    ]

    assert result["exit_signal"].tolist() == [
        False,
        False,
        True,
    ]

    assert result["target_position"].tolist() == [
        1,
        1,
        0,
    ]


def test_add_baseline_position_uses_custom_parameters() -> None:
    parameters = StrategyParameters(
        fast_ema_period=3,
        slow_ema_period=5,
        trend_ema_period=10,
        rsi_period=4,
        rsi_lower=60.0,
        rsi_upper=80.0,
        atr_period=3,
        volume_period=4,
        breakout_period=5,
    )

    frame = pd.DataFrame(
        {
            "symbol": ["TEST"] * 3,
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=3,
                freq="D",
                tz="UTC",
            ),
            "close": [
                110.0,
                112.0,
                95.0,
            ],
            "volume": [
                200.0,
                200.0,
                200.0,
            ],
            "ema_3": [
                105.0,
                106.0,
                100.0,
            ],
            "ema_5": [
                100.0,
                101.0,
                99.0,
            ],
            "ema_10": [
                90.0,
                90.0,
                90.0,
            ],
            "rsi_4": [
                55.0,
                65.0,
                65.0,
            ],
            "volume_sma_4": [
                100.0,
                100.0,
                100.0,
            ],
            "previous_high_5": [
                109.0,
                111.0,
                94.0,
            ],
        }
    )

    result = add_baseline_position(
        frame,
        parameters,
    )

    assert result["entry_signal"].tolist() == [
        False,
        True,
        False,
    ]

    assert result["exit_signal"].tolist() == [
        False,
        False,
        True,
    ]

    assert result["target_position"].tolist() == [
        0,
        1,
        0,
    ]


def test_add_baseline_position_requires_dynamic_columns() -> None:
    parameters = StrategyParameters(
        fast_ema_period=3,
        slow_ema_period=5,
        trend_ema_period=10,
        rsi_period=4,
        atr_period=3,
        volume_period=4,
        breakout_period=5,
    )

    frame = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "timestamp": [
                pd.Timestamp(
                    "2024-01-01",
                    tz="UTC",
                )
            ],
            "close": [100.0],
            "volume": [1_000.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Colonne mancanti per la strategia",
    ):
        add_baseline_position(
            frame,
            parameters,
        )