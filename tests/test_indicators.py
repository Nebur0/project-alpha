import numpy as np
import pandas as pd

from project_alpha.indicators import atr, ema, rsi


def test_ema_preserves_length() -> None:
    series = pd.Series(np.arange(1.0, 101.0))
    result = ema(series, 20)
    assert len(result) == len(series)
    assert result.iloc[:19].isna().all()
    assert result.iloc[19:].notna().all()


def test_rsi_uptrend_reaches_high_value() -> None:
    series = pd.Series(np.arange(1.0, 101.0))
    result = rsi(series, 14)
    assert result.iloc[-1] == 100.0


def test_atr_constant_range() -> None:
    frame = pd.DataFrame(
        {
            "high": np.arange(11.0, 51.0),
            "low": np.arange(9.0, 49.0),
            "close": np.arange(10.0, 50.0),
        }
    )
    result = atr(frame, 14)
    assert result.iloc[-1] == 2.0
