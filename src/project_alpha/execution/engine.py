"""Ribilanciamento: rischio → piano ordini → invio → riconciliazione.

Sequenza, sempre nello stesso ordine:

1. legge conto e posizioni dal broker (il broker è la fonte di verità);
2. calcola i pesi attuali e chiede al Risk Engine V2 se i pesi target sono
   ammessi; se no, **nessun ordine** e il motivo viene restituito;
3. pianifica gli ordini (vendite prima, quantità per difetto);
4. se non è un dry-run, li invia con id deterministici (ripetere lo stesso
   ribilanciamento non duplica nulla) e si ferma al primo rifiuto;
5. rilegge il conto e confronta i pesi ottenuti con quelli target.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from project_alpha.execution.broker import Broker, OrderReport, OrderStatus
from project_alpha.execution.orders import Order, OrderPlanSettings, plan_orders
from project_alpha.risk_engine import (
    PortfolioState,
    RiskDecisionV2,
    RiskPolicyV2,
    evaluate_proposal,
)


@dataclass(frozen=True)
class RebalanceResult:
    status: str  # "REJECTED_BY_RISK" | "DRY_RUN" | "EXECUTED" | "PARTIAL" | "NOTHING_TO_DO"
    risk: RiskDecisionV2
    orders: tuple[Order, ...]
    reports: tuple[OrderReport, ...]
    weights_before: Mapping[str, float]
    weights_after: Mapping[str, float]
    max_weight_error: float | None


def _weights(positions: Mapping[str, float], prices: Mapping[str, float], equity: float):
    return {symbol: quantity * prices[symbol] / equity for symbol, quantity in positions.items()}


def rebalance(
    broker: Broker,
    target_weights: Mapping[str, float],
    prices: Mapping[str, float],
    *,
    strategy: str,
    as_of: datetime | pd.Timestamp,
    data_timestamp: datetime | pd.Timestamp,
    peak_equity: float,
    daily_return: float,
    policy: RiskPolicyV2 | None = None,
    settings: OrderPlanSettings | None = None,
    dry_run: bool = True,
) -> RebalanceResult:
    """Porta il conto verso ``target_weights`` se il Risk Engine lo consente.

    ``dry_run=True`` (default): calcola tutto ma non invia ordini.
    """
    price_map = {symbol.upper(): float(price) for symbol, price in prices.items()}
    targets = {symbol.upper(): float(weight) for symbol, weight in target_weights.items()}

    before = broker.account(price_map)
    weights_before = _weights(before.positions, price_map, before.equity)

    state = PortfolioState(
        equity=before.equity,
        peak_equity=max(peak_equity, before.equity),
        daily_return=daily_return,
        current_weights=weights_before,
        data_timestamp=data_timestamp,
        as_of=as_of,
    )
    decision = evaluate_proposal(targets, state, policy)

    if not decision.allowed:
        return RebalanceResult(
            status="REJECTED_BY_RISK",
            risk=decision,
            orders=(),
            reports=(),
            weights_before=weights_before,
            weights_after=weights_before,
            max_weight_error=None,
        )

    orders = plan_orders(
        targets,
        before.positions,
        price_map,
        before.equity,
        strategy=strategy,
        as_of=pd.Timestamp(as_of).date().isoformat(),
        settings=settings,
    )

    if not orders:
        status = "NOTHING_TO_DO"
    elif dry_run:
        status = "DRY_RUN"
    else:
        status = "EXECUTED"

    reports: list[OrderReport] = []

    if orders and not dry_run:
        for order in orders:
            report = broker.submit(order, fill_price=price_map[order.symbol])
            reports.append(report)

            if report.status is not OrderStatus.FILLED:
                status = "PARTIAL"
                break

    after = broker.account(price_map)
    weights_after = _weights(after.positions, price_map, after.equity)
    symbols = set(targets) | set(weights_after)
    error = max(
        (abs(weights_after.get(symbol, 0.0) - targets.get(symbol, 0.0)) for symbol in symbols),
        default=0.0,
    )

    return RebalanceResult(
        status=status,
        risk=decision,
        orders=tuple(orders),
        reports=tuple(reports),
        weights_before=weights_before,
        weights_after=weights_after,
        max_weight_error=error,
    )
