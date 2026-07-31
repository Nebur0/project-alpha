import pytest

from project_alpha.config import StrategyParameters


def test_strategy_parameters_defaults_match_baseline() -> None:
    parameters = StrategyParameters()

    assert parameters.fast_ema_period == 20
    assert parameters.slow_ema_period == 50
    assert parameters.trend_ema_period == 200
    assert parameters.rsi_period == 14
    assert parameters.rsi_lower == pytest.approx(50.0)
    assert parameters.rsi_upper == pytest.approx(70.0)
    assert parameters.atr_period == 14
    assert parameters.volume_period == 20
    assert parameters.breakout_period == 20


def test_strategy_parameters_accept_custom_values() -> None:
    parameters = StrategyParameters(
        fast_ema_period=10,
        slow_ema_period=30,
        trend_ema_period=100,
        rsi_period=7,
        rsi_lower=40.0,
        rsi_upper=80.0,
        atr_period=10,
        volume_period=15,
        breakout_period=25,
    )

    assert parameters.fast_ema_period == 10
    assert parameters.slow_ema_period == 30
    assert parameters.trend_ema_period == 100
    assert parameters.rsi_period == 7
    assert parameters.rsi_lower == pytest.approx(40.0)
    assert parameters.rsi_upper == pytest.approx(80.0)
    assert parameters.atr_period == 10
    assert parameters.volume_period == 15
    assert parameters.breakout_period == 25


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("fast_ema_period", 0),
        ("slow_ema_period", -1),
        ("trend_ema_period", 0),
        ("rsi_period", -5),
        ("atr_period", 0),
        ("volume_period", -10),
        ("breakout_period", 0),
    ],
)
def test_strategy_parameters_reject_invalid_periods(
    field_name: str,
    invalid_value: int,
) -> None:
    values = {
        field_name: invalid_value,
    }

    with pytest.raises(
        ValueError,
        match=f"{field_name} deve essere un intero positivo",
    ):
        StrategyParameters(**values)


def test_strategy_parameters_reject_invalid_ema_order() -> None:
    with pytest.raises(
        ValueError,
        match="fast < slow < trend",
    ):
        StrategyParameters(
            fast_ema_period=50,
            slow_ema_period=20,
            trend_ema_period=200,
        )


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (-1.0, 70.0),
        (50.0, 50.0),
        (80.0, 70.0),
        (50.0, 101.0),
    ],
)
def test_strategy_parameters_reject_invalid_rsi_thresholds(
    lower: float,
    upper: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="0 <= lower < upper <= 100",
    ):
        StrategyParameters(
            rsi_lower=lower,
            rsi_upper=upper,
        )