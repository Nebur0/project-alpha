"""Test del motore di esecuzione (broker simulato, nessuna rete)."""

from datetime import UTC, datetime, timedelta

import pytest

from project_alpha.execution.broker import OrderStatus, SimulatedBroker
from project_alpha.execution.engine import rebalance
from project_alpha.execution.orders import (
    Order,
    OrderPlanSettings,
    Side,
    client_order_id,
    plan_orders,
)
from project_alpha.risk_engine import RiskPolicyV2

NOW = datetime(2026, 11, 9, 21, 30, tzinfo=UTC)
PRICES = {"SPY": 500.0, "QQQ": 400.0, "GLD": 200.0}
POLICY = RiskPolicyV2(max_position_weight=0.5, max_group_weight=None, max_turnover=None)


# --------------------------------------------------------------------------
# Pianificazione
# --------------------------------------------------------------------------


def test_plan_orders_whole_shares_rounds_down() -> None:
    orders = plan_orders(
        {"SPY": 0.5, "QQQ": 0.3},
        {},
        PRICES,
        10_000.0,
        strategy="s",
        as_of="2026-11-09",
        settings=OrderPlanSettings(cash_buffer=0.0),
    )

    quantities = {order.symbol: order.quantity for order in orders}
    assert quantities == {"SPY": 10.0, "QQQ": 7.0}  # 7,5 -> 7: mai oltre il target
    assert all(order.side is Side.BUY for order in orders)


def test_plan_orders_sells_first_and_closes_positions() -> None:
    orders = plan_orders(
        {"QQQ": 0.4},
        {"SPY": 10.0, "GLD": 0.3},
        PRICES,
        10_000.0,
        strategy="s",
        as_of="2026-11-09",
        settings=OrderPlanSettings(cash_buffer=0.0),
    )

    assert [order.side for order in orders] == [Side.SELL, Side.SELL, Side.BUY]
    sells = {order.symbol: order.quantity for order in orders if order.side is Side.SELL}
    # Anche la frazione residua di GLD (60 €) viene chiusa nonostante il minimo.
    assert sells == {"GLD": 0.3, "SPY": 10.0}


def test_plan_orders_fractional_and_minimum_notional() -> None:
    settings = OrderPlanSettings(allow_fractional=True, cash_buffer=0.0, min_order_notional=50)

    orders = plan_orders(
        {"SPY": 0.1, "QQQ": 0.002},
        {},
        PRICES,
        1_000.0,
        strategy="s",
        as_of="2026-11-09",
        settings=settings,
    )

    assert [(order.symbol, order.quantity) for order in orders] == [("SPY", 0.2)]


def test_plan_orders_validation() -> None:
    with pytest.raises(ValueError, match="long-only"):
        plan_orders({"SPY": -0.1}, {}, PRICES, 1000.0, strategy="s", as_of="d")

    with pytest.raises(ValueError, match="Prezzo"):
        plan_orders({"XYZ": 0.1}, {}, PRICES, 1000.0, strategy="s", as_of="d")

    with pytest.raises(ValueError, match="equity"):
        plan_orders({}, {}, PRICES, 0.0, strategy="s", as_of="d")


def test_client_order_id_is_deterministic() -> None:
    first = client_order_id("v1", "2026-11-09", "SPY", Side.BUY, 10)
    again = client_order_id("v1", "2026-11-09", "SPY", Side.BUY, 10)
    other_day = client_order_id("v1", "2026-11-10", "SPY", Side.BUY, 10)

    assert first == again != other_day
    assert first.startswith("pa-")


# --------------------------------------------------------------------------
# Broker simulato
# --------------------------------------------------------------------------


def _order(symbol: str, side: Side, quantity: float, oid: str = "x") -> Order:
    return Order(oid, symbol, side, quantity, PRICES[symbol])


def test_simulated_broker_fills_and_charges_commission() -> None:
    broker = SimulatedBroker(cash=10_000.0, commission_rate=0.001)

    report = broker.submit(_order("SPY", Side.BUY, 10), fill_price=500.0)

    assert report.status is OrderStatus.FILLED
    assert report.commission == pytest.approx(5.0)
    assert broker.cash == pytest.approx(10_000 - 5_000 - 5)
    assert broker.account(PRICES).equity == pytest.approx(9_995.0)


def test_simulated_broker_is_idempotent() -> None:
    broker = SimulatedBroker(cash=10_000.0)

    broker.submit(_order("SPY", Side.BUY, 10, "same"), fill_price=500.0)
    broker.submit(_order("SPY", Side.BUY, 10, "same"), fill_price=500.0)

    assert broker.positions == {"SPY": 10.0}


def test_simulated_broker_rejects_unsafe_orders() -> None:
    broker = SimulatedBroker(cash=1_000.0, positions={"QQQ": 1})

    no_cash = broker.submit(_order("SPY", Side.BUY, 10, "a"), fill_price=500.0)
    short = broker.submit(_order("QQQ", Side.SELL, 2, "b"), fill_price=400.0)

    assert no_cash.status is OrderStatus.REJECTED and "Cash" in no_cash.reason
    assert short.status is OrderStatus.REJECTED and "scoperto" in short.reason
    assert broker.positions == {"QQQ": 1.0}
    assert broker.order_report("a") == no_cash


# --------------------------------------------------------------------------
# Ribilanciamento
# --------------------------------------------------------------------------


def _rebalance(broker, targets, **kwargs):
    defaults = {
        "strategy": "v1",
        "as_of": NOW,
        "data_timestamp": NOW - timedelta(hours=2),
        "peak_equity": 10_000.0,
        "daily_return": 0.0,
        "policy": POLICY,
        "settings": OrderPlanSettings(cash_buffer=0.01),
    }
    defaults.update(kwargs)
    return rebalance(broker, targets, PRICES, **defaults)


def test_dry_run_plans_but_does_not_trade() -> None:
    broker = SimulatedBroker(cash=10_000.0)

    result = _rebalance(broker, {"SPY": 0.4, "QQQ": 0.3})

    assert result.status == "DRY_RUN"
    assert len(result.orders) == 2
    assert result.reports == ()
    assert broker.positions == {}


def test_execute_reaches_targets_within_rounding() -> None:
    broker = SimulatedBroker(cash=10_000.0, commission_rate=0.0)

    result = _rebalance(broker, {"SPY": 0.4, "QQQ": 0.3}, dry_run=False)

    assert result.status == "EXECUTED"
    assert all(report.status is OrderStatus.FILLED for report in result.reports)
    assert broker.positions == {"SPY": 7.0, "QQQ": 7.0}
    # Con azioni intere l'errore massimo è il peso di una singola azione SPY.
    assert result.max_weight_error <= 500.0 / 10_000.0 + 1e-9


def test_repeating_the_same_rebalance_does_not_duplicate_orders() -> None:
    broker = SimulatedBroker(cash=10_000.0)

    first = _rebalance(broker, {"SPY": 0.4}, dry_run=False)
    second = _rebalance(broker, {"SPY": 0.4}, dry_run=False)

    assert first.status == "EXECUTED"
    assert second.status == "NOTHING_TO_DO"
    assert broker.positions == {"SPY": 7.0}


def test_risk_rejection_sends_no_orders() -> None:
    broker = SimulatedBroker(cash=10_000.0)

    result = _rebalance(broker, {"SPY": 0.9}, dry_run=False)

    assert result.status == "REJECTED_BY_RISK"
    assert "MAX_POSITION_WEIGHT" in result.risk.violations
    assert result.orders == ()
    assert broker.positions == {}


def test_drawdown_allows_only_reductions() -> None:
    broker = SimulatedBroker(cash=5_000.0, positions={"SPY": 8})  # equity 9.000
    increase = _rebalance(broker, {"SPY": 0.4, "QQQ": 0.3}, peak_equity=12_000.0, dry_run=False)
    decrease = _rebalance(broker, {"SPY": 0.2}, peak_equity=12_000.0, dry_run=False)

    assert increase.status == "REJECTED_BY_RISK"
    assert "REDUCE_ONLY_MODE" in increase.risk.violations
    assert decrease.status == "EXECUTED"
    assert broker.positions["SPY"] < 8


def test_stale_data_blocks_new_positions() -> None:
    broker = SimulatedBroker(cash=10_000.0)

    result = _rebalance(broker, {"SPY": 0.3}, data_timestamp=NOW - timedelta(days=10))

    assert result.status == "REJECTED_BY_RISK"
    assert "STALE_DATA" in result.risk.violations


def test_partial_execution_stops_at_first_rejection() -> None:
    class FailingBroker(SimulatedBroker):
        def submit(self, order, *, fill_price):
            if order.symbol == "QQQ":
                return super().submit(
                    Order(order.client_order_id, order.symbol, order.side, 10_000, 400.0),
                    fill_price=fill_price,
                )
            return super().submit(order, fill_price=fill_price)

    broker = FailingBroker(cash=10_000.0)
    result = _rebalance(broker, {"QQQ": 0.3, "SPY": 0.3}, dry_run=False)

    assert result.status == "PARTIAL"
    assert result.reports[-1].status is OrderStatus.REJECTED
    assert len(result.reports) < len(result.orders)


def test_queued_orders_are_reported_as_submitted() -> None:
    from project_alpha.execution.broker import OrderReport

    class QueuedBroker(SimulatedBroker):
        def submit(self, order, *, fill_price):
            return OrderReport(
                order.client_order_id,
                order.symbol,
                order.side,
                order.quantity,
                OrderStatus.SUBMITTED,
                None,
                0.0,
            )

    result = _rebalance(QueuedBroker(cash=10_000.0), {"SPY": 0.3, "QQQ": 0.3}, dry_run=False)

    assert result.status == "SUBMITTED"
    assert len(result.reports) == len(result.orders) == 2
