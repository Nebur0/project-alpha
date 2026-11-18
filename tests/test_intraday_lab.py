"""Test del laboratorio intraday su candele sintetiche (nessuna rete)."""

import numpy as np
import pandas as pd
import pytest

from project_alpha import intraday_lab as lab


def _bars(days=520, *, persistence=0.0, seed=0, symbol="SPY", start="2021-01-04"):
    """Candele da 15 minuti; ``persistence`` > 0 rende la direzione prevedibile."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    rows = []
    price = 100.0
    previous = 0.0
    for date in dates:
        for bar in range(26):
            stamp = pd.Timestamp(date).tz_localize(lab.MARKET_TZ) + pd.Timedelta(
                minutes=570 + 15 * bar
            )
            ret = persistence * np.sign(previous) * 0.001 + rng.normal(0, 0.001)
            opening = price
            price = price * (1 + ret)
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": stamp.tz_convert("UTC"),
                    "open": opening,
                    "high": max(opening, price) * 1.0002,
                    "low": min(opening, price) * 0.9998,
                    "close": price,
                    "volume": 1000 + rng.integers(0, 100),
                }
            )
            previous = ret
        # una candela di pre-mercato da ignorare
        rows.append(
            {
                **rows[-1],
                "timestamp": (
                    pd.Timestamp(date).tz_localize(lab.MARKET_TZ) + pd.Timedelta(hours=8)
                ).tz_convert("UTC"),
            }
        )
    return pd.DataFrame(rows)


def test_sessions_keep_regular_hours_only() -> None:
    days = lab.load_sessions(_bars(days=3), "SPY")

    assert len(days) == 3
    assert all(len(d.close) == 26 for d in days)


def test_trades_hold_two_bars_never_overnight_and_pay_costs() -> None:
    day = lab.load_sessions(_bars(days=1), "SPY")[0]
    signal = np.zeros(26)
    signal[0] = 1
    signal[24] = -1  # a 1 candela dalla fine: esce all'ultima

    trades = lab.simulate_day(day, signal, cost=0.0001)

    assert len(trades) == 2
    assert trades[0] == pytest.approx(day.close[2] / day.close[0] - 1 - 0.0002)
    assert trades[1] == pytest.approx(-(day.close[25] / day.close[24] - 1) - 0.0002)


def test_one_position_at_a_time() -> None:
    day = lab.load_sessions(_bars(days=1), "SPY")[0]

    trades = lab.simulate_day(day, np.ones(26), cost=0.0)

    assert len(trades) == 13  # 0→2, 2→4, ..., 24→25


def test_opening_range_breakout_one_trade_until_close() -> None:
    day = lab.load_sessions(_bars(days=1, persistence=0.0, seed=3), "SPY")[0]

    trades = lab.opening_range_trades(day, cost=0.0)

    assert len(trades) <= 1


def test_engulfing_detection() -> None:
    day = lab.Day(
        date=pd.Timestamp("2024-01-02"),
        open=np.array(
            [10.0, 9.4, 10.4, 10.6]
        ),  # rossa, verde che ingloba, verde, rossa che ingloba
        high=np.array([10.1, 10.6, 10.6, 10.7]),
        low=np.array([9.4, 9.3, 10.3, 9.2]),
        close=np.array([9.5, 10.5, 10.5, 9.3]),
        volume=np.ones(4),
    )

    assert list(lab.engulfing(day)) == [0, 1, 0, -1]


def test_random_walk_passes_nothing() -> None:
    table = lab.evaluate(_bars(days=1300, seed=7), symbols=("SPY",))

    assert (table["esito"] == "scartata").all()
    assert table.attrs["alpha"] == pytest.approx(0.05 / 5)


def test_planted_momentum_is_found_and_reversal_loses() -> None:
    table = lab.evaluate(_bars(days=1300, persistence=0.8, seed=1), symbols=("SPY",)).set_index(
        "regola"
    )

    assert table.loc["L1_momentum_1", "esito"] == "FUNZIONA"
    assert table.loc["L2_inversione_1", "media_giornaliera_bps"] < 0
    assert table.loc["L5_logistica", "media_giornaliera_bps"] > 0
