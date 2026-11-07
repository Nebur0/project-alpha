import pytest

from project_alpha.risk import (
    RiskDecisionStatus,
    RiskPolicy,
    evaluate_risk_policy,
)


def test_default_policy_values() -> None:
    policy = RiskPolicy()

    assert policy.max_position_weight == pytest.approx(0.20)
    assert policy.max_gross_exposure == pytest.approx(1.00)
    assert policy.max_daily_loss == pytest.approx(0.03)
    assert policy.max_drawdown == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("max_position_weight", 0.0),
        ("max_gross_exposure", -1.0),
        ("max_daily_loss", float("nan")),
        ("max_drawdown", float("inf")),
    ],
)
def test_policy_rejects_invalid_limits(
    field_name: str,
    invalid_value: float,
) -> None:
    with pytest.raises(ValueError):
        RiskPolicy(**{field_name: invalid_value})


def test_policy_allows_valid_proposal() -> None:
    decision = evaluate_risk_policy(
        {
            "AAA": 0.20,
            "BBB": 0.20,
            "CCC": 0.20,
            "DDD": 0.20,
            "EEE": 0.20,
        },
        current_equity=10_000.0,
        peak_equity=10_000.0,
        daily_return=0.01,
    )

    assert decision.status is RiskDecisionStatus.ALLOW
    assert decision.allowed is True
    assert decision.violations == ()
    assert decision.gross_exposure == pytest.approx(1.0)


def test_policy_rejects_position_limit_breach() -> None:
    decision = evaluate_risk_policy(
        {
            "AAA": 0.30,
            "BBB": 0.20,
        },
        current_equity=10_000.0,
        peak_equity=10_000.0,
        daily_return=0.0,
    )

    assert decision.allowed is False
    assert "MAX_POSITION_WEIGHT" in decision.violations


def test_policy_rejects_gross_exposure_breach() -> None:
    decision = evaluate_risk_policy(
        {
            "AAA": 0.20,
            "BBB": 0.20,
            "CCC": 0.20,
            "DDD": 0.20,
            "EEE": 0.20,
            "FFF": 0.20,
        },
        current_equity=10_000.0,
        peak_equity=10_000.0,
        daily_return=0.0,
    )

    assert decision.allowed is False
    assert "MAX_GROSS_EXPOSURE" in decision.violations


def test_policy_rejects_daily_loss_breach() -> None:
    decision = evaluate_risk_policy(
        {"AAA": 0.10},
        current_equity=9_900.0,
        peak_equity=10_000.0,
        daily_return=-0.04,
    )

    assert decision.allowed is False
    assert "MAX_DAILY_LOSS" in decision.violations


def test_policy_rejects_drawdown_breach() -> None:
    decision = evaluate_risk_policy(
        {"AAA": 0.10},
        current_equity=8_400.0,
        peak_equity=10_000.0,
        daily_return=0.0,
    )

    assert decision.allowed is False
    assert "MAX_DRAWDOWN" in decision.violations
    assert decision.current_drawdown == pytest.approx(-0.16)


def test_policy_reports_multiple_violations() -> None:
    decision = evaluate_risk_policy(
        {
            "AAA": 0.70,
            "BBB": 0.70,
        },
        current_equity=8_000.0,
        peak_equity=10_000.0,
        daily_return=-0.05,
    )

    assert decision.status is RiskDecisionStatus.REJECT
    assert decision.violations == (
        "MAX_POSITION_WEIGHT",
        "MAX_GROSS_EXPOSURE",
        "MAX_DAILY_LOSS",
        "MAX_DRAWDOWN",
    )


def test_policy_supports_short_weights_in_gross_exposure() -> None:
    policy = RiskPolicy(
        max_position_weight=0.60,
        max_gross_exposure=1.00,
    )

    decision = evaluate_risk_policy(
        {
            "AAA": 0.50,
            "BBB": -0.50,
        },
        current_equity=10_000.0,
        peak_equity=10_000.0,
        daily_return=0.0,
        policy=policy,
    )

    assert decision.gross_exposure == pytest.approx(1.0)
    assert decision.allowed is True


def test_policy_rejects_invalid_equity_state() -> None:
    with pytest.raises(
        ValueError,
        match="maggiore di peak_equity",
    ):
        evaluate_risk_policy(
            {"AAA": 0.10},
            current_equity=10_100.0,
            peak_equity=10_000.0,
            daily_return=0.0,
        )


def test_policy_rejects_empty_proposal() -> None:
    with pytest.raises(
        ValueError,
        match="almeno un peso",
    ):
        evaluate_risk_policy(
            {},
            current_equity=10_000.0,
            peak_equity=10_000.0,
            daily_return=0.0,
        )
