"""Test del Risk Engine V2."""

from datetime import UTC, datetime, timedelta

import pytest

from project_alpha.risk_engine import (
    PortfolioState,
    RiskMode,
    RiskPolicyV2,
    evaluate_proposal,
)

NOW = datetime(2026, 11, 6, 21, 0, tzinfo=UTC)


def _state(
    *,
    current: dict[str, float] | None = None,
    equity: float = 10_000.0,
    peak: float = 10_000.0,
    daily_return: float = 0.0,
    data_age_days: float = 0.5,
) -> PortfolioState:
    return PortfolioState(
        equity=equity,
        peak_equity=peak,
        daily_return=daily_return,
        current_weights=current or {},
        data_timestamp=NOW - timedelta(days=data_age_days),
        as_of=NOW,
    )


def test_allows_compliant_proposal() -> None:
    decision = evaluate_proposal({"SPY": 0.20, "QQQ": 0.15}, _state())

    assert decision.allowed
    assert decision.mode is RiskMode.NORMAL
    assert decision.violations == ()
    assert decision.gross_exposure == pytest.approx(0.35)
    assert decision.turnover == pytest.approx(0.35)


def test_rejects_oversized_position_and_explains_why() -> None:
    decision = evaluate_proposal({"NVDA": 0.35}, _state())

    assert not decision.allowed
    assert decision.violations == ("MAX_POSITION_WEIGHT",)
    assert "NVDA" in decision.details[0]


def test_rejects_short_by_default() -> None:
    decision = evaluate_proposal({"SPY": 0.10, "QQQ": -0.10}, _state())

    assert "SHORT_NOT_ALLOWED" in decision.violations


def test_leverage_requires_explicit_policy() -> None:
    with pytest.raises(ValueError, match="Leva"):
        RiskPolicyV2(max_gross_exposure=1.5)


def test_rejects_gross_exposure_increase_above_limit() -> None:
    policy = RiskPolicyV2(max_gross_exposure=0.50, max_turnover=None)
    proposal = {"A": 0.20, "B": 0.20, "C": 0.20}

    decision = evaluate_proposal(proposal, _state(), policy)

    assert decision.violations == ("MAX_GROSS_EXPOSURE",)


def test_reduction_never_violates_upper_limits() -> None:
    """Dopo un rialzo NVDA pesa il 30%: ridurla al 25% deve essere ammesso."""
    decision = evaluate_proposal(
        {"NVDA": 0.25},
        _state(current={"NVDA": 0.30}),
    )

    assert decision.allowed


def test_reduction_is_allowed_even_with_high_turnover() -> None:
    policy = RiskPolicyV2(max_turnover=0.10)
    current = {"A": 0.20, "B": 0.20, "C": 0.20}

    liquidation = evaluate_proposal({}, _state(current=current), policy)
    rebalance = evaluate_proposal({"A": 0.20, "D": 0.20}, _state(current=current), policy)

    assert liquidation.allowed
    assert liquidation.turnover == pytest.approx(0.60)
    assert "MAX_TURNOVER" in rebalance.violations


def test_drawdown_breach_switches_to_reduce_only() -> None:
    state = _state(current={"SPY": 0.20}, equity=8_400.0, peak=10_000.0)

    add_risk = evaluate_proposal({"SPY": 0.20, "QQQ": 0.10}, state)
    reduce_risk = evaluate_proposal({"SPY": 0.10}, state)

    assert add_risk.mode is RiskMode.REDUCE_ONLY
    assert add_risk.violations == ("REDUCE_ONLY_MODE",)
    assert reduce_risk.allowed
    assert reduce_risk.mode is RiskMode.REDUCE_ONLY


def test_daily_loss_breach_switches_to_reduce_only() -> None:
    state = _state(current={"SPY": 0.20}, daily_return=-0.04)

    decision = evaluate_proposal({"SPY": 0.20, "QQQ": 0.05}, state)

    assert decision.mode is RiskMode.REDUCE_ONLY
    assert not decision.allowed


def test_switching_side_is_not_a_reduction() -> None:
    policy = RiskPolicyV2(allow_short=True, max_gross_exposure=1.0)
    state = _state(current={"SPY": 0.20}, equity=8_000.0)

    decision = evaluate_proposal({"SPY": -0.10}, state, policy)

    assert "REDUCE_ONLY_MODE" in decision.violations


def test_kill_switch_allows_only_reductions() -> None:
    policy = RiskPolicyV2(kill_switch=True)
    state = _state(current={"SPY": 0.20})

    assert evaluate_proposal({}, state, policy).allowed
    blocked = evaluate_proposal({"SPY": 0.20, "QQQ": 0.01}, state, policy)
    assert blocked.mode is RiskMode.HALTED
    assert blocked.violations == ("KILL_SWITCH",)


def test_stale_data_blocks_new_risk_but_not_reductions() -> None:
    state = _state(current={"SPY": 0.20}, data_age_days=10)

    assert "STALE_DATA" in evaluate_proposal({"SPY": 0.20, "QQQ": 0.10}, state).violations
    assert evaluate_proposal({"SPY": 0.05}, state).allowed


def test_group_limit_blocks_correlated_concentration() -> None:
    policy = RiskPolicyV2(
        max_group_weight=0.40,
        max_turnover=None,
        groups={"AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "XOM": "energy"},
    )

    decision = evaluate_proposal(
        {"AAPL": 0.20, "MSFT": 0.20, "NVDA": 0.15, "XOM": 0.10},
        _state(),
        policy,
    )

    assert decision.violations == ("MAX_GROUP_WEIGHT",)
    assert "tech" in decision.details[0]


def test_allowed_symbols_whitelist() -> None:
    policy = RiskPolicyV2(allowed_symbols=frozenset({"spy", "qqq"}))

    assert evaluate_proposal({"SPY": 0.20}, _state(), policy).allowed
    decision = evaluate_proposal({"GME": 0.05}, _state(), policy)
    assert decision.violations == ("UNKNOWN_SYMBOL",)

    # Vendere un simbolo fuori lista già in portafoglio è sempre ammesso.
    assert evaluate_proposal({}, _state(current={"GME": 0.05}), policy).allowed


def test_order_notional_limit() -> None:
    policy = RiskPolicyV2(max_order_notional=1_000.0)

    assert evaluate_proposal({"SPY": 0.10}, _state(), policy).allowed
    decision = evaluate_proposal({"SPY": 0.15}, _state(), policy)
    assert decision.violations == ("MAX_ORDER_NOTIONAL",)


def test_min_cash_weight() -> None:
    policy = RiskPolicyV2(min_cash_weight=0.10, max_position_weight=0.5, max_group_weight=None)

    decision = evaluate_proposal({"A": 0.5, "B": 0.45}, _state(), policy)

    assert "MIN_CASH_WEIGHT" in decision.violations


def test_multiple_violations_are_all_reported() -> None:
    decision = evaluate_proposal({"NVDA": 0.40, "AMD": -0.05}, _state(daily_return=-0.05))

    assert set(decision.violations) == {
        "REDUCE_ONLY_MODE",
        "SHORT_NOT_ALLOWED",
        "MAX_POSITION_WEIGHT",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_position_weight": 0.0},
        {"max_daily_loss": float("nan")},
        {"max_turnover": -0.1},
        {"min_cash_weight": 1.0},
        {"max_data_age_days": 0},
    ],
)
def test_policy_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        RiskPolicyV2(**kwargs)


def test_state_validation() -> None:
    with pytest.raises(ValueError, match="peak_equity"):
        _state(equity=11_000.0, peak=10_000.0)

    with pytest.raises(ValueError, match="duplicato"):
        _state(current={"spy": 0.1, "SPY": 0.1})

    naive = PortfolioState(
        equity=1.0,
        peak_equity=1.0,
        daily_return=0.0,
        current_weights={},
        data_timestamp=datetime(2026, 1, 1),
        as_of=NOW,
    )
    with pytest.raises(ValueError, match="fuso"):
        evaluate_proposal({}, naive)


def test_policy_is_immutable() -> None:
    policy = RiskPolicyV2()

    with pytest.raises(AttributeError):
        policy.max_position_weight = 0.9  # type: ignore[misc]
