"""Test delle modalità di esecuzione (blocco 2: realismo dell'esecuzione)."""

import numpy as np
import pandas as pd
import pytest

from project_alpha import cli
from project_alpha.backtest import (
    DEFAULT_EXECUTION,
    ExecutionConfig,
    run_long_only_backtest,
)
from project_alpha.batch_optimization import (
    run_batch_walk_forward_optimization,
)
from project_alpha.execution_stress_analysis import (
    analyze_frozen_selection_execution_stress,
)
from project_alpha.portfolio_walk_forward import (
    run_walk_forward_portfolio_backtest,
)


def _frame() -> pd.DataFrame:
    """Cinque sedute con aperture e chiusure scelte a mano."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=5,
                freq="D",
                tz="UTC",
            ),
            "open": [100.0, 102.0, 108.0, 110.0, 100.0],
            "close": [100.0, 105.0, 110.0, 104.0, 104.0],
            "target_position": [1, 1, 0, 0, 0],
        }
    )


def _market_frame(symbol: str, periods: int = 100) -> pd.DataFrame:
    index = np.arange(periods)
    close = 100.0 + index * 0.30 + np.sin(index / 3.0) * 2.0

    return pd.DataFrame(
        {
            "symbol": [symbol] * periods,
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=periods,
                freq="D",
                tz="UTC",
            ),
            "open": close - 0.50,
            "high": close + 1.00,
            "low": close - 1.00,
            "close": close,
            "volume": 1_000.0 + (index % 10) * 100.0,
        }
    )


def _small_grid() -> dict[str, list[int | float]]:
    return {
        "fast_ema_period": [3, 4],
        "slow_ema_period": [6],
        "trend_ema_period": [12],
        "rsi_period": [5],
        "rsi_lower": [40.0],
        "rsi_upper": [80.0],
        "atr_period": [5],
        "volume_period": [5],
        "breakout_period": [5],
    }


# --------------------------------------------------------------------------
# ExecutionConfig
# --------------------------------------------------------------------------


def test_execution_config_defaults_match_historical_behavior() -> None:
    assert DEFAULT_EXECUTION == ExecutionConfig()
    assert ExecutionConfig().is_default
    assert not ExecutionConfig(mode="next_open").is_default


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mode": "vwap"}, "modalità di esecuzione"),
        ({"delay_days": -1}, "delay_days"),
        ({"delay_days": 1.5}, "delay_days"),
        ({"delay_days": True}, "delay_days"),
        ({"cash_rate_annual": float("nan")}, "cash_rate_annual"),
        ({"cash_rate_annual": -1.0}, "cash_rate_annual"),
    ],
)
def test_execution_config_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ExecutionConfig(**kwargs)


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------


def test_default_execution_is_identical_to_explicit_close() -> None:
    implicit = run_long_only_backtest(_frame(), cost_per_side=0.001)
    explicit = run_long_only_backtest(
        _frame(),
        cost_per_side=0.001,
        execution=ExecutionConfig(mode="close"),
    )

    pd.testing.assert_series_equal(
        implicit.equity_curve["equity"],
        explicit.equity_curve["equity"],
    )
    assert implicit.metrics["execution_next_open"] == 0
    assert implicit.metrics["execution_delay_days"] == 0


def test_close_execution_uses_close_to_close_returns() -> None:
    result = run_long_only_backtest(_frame(), cost_per_side=0.0)
    curve = result.equity_curve

    # Posizione 1 nelle sedute 1 e 2 (segnali di 0 e 1).
    assert curve["executed_position"].tolist() == [0, 1, 1, 0, 0]
    assert curve.loc[1, "strategy_return"] == pytest.approx(105 / 100 - 1)
    assert curve.loc[2, "strategy_return"] == pytest.approx(110 / 105 - 1)
    assert result.metrics["total_return"] == pytest.approx(0.10)


def test_next_open_execution_pays_the_gaps() -> None:
    result = run_long_only_backtest(
        _frame(),
        cost_per_side=0.0,
        execution=ExecutionConfig(mode="next_open"),
    )
    curve = result.equity_curve

    assert curve["executed_position"].tolist() == [0, 1, 1, 0, 0]
    assert curve["overnight_position"].tolist() == [0, 0, 1, 1, 0]

    # Seduta 1: compra all'apertura (102), tiene fino alla chiusura (105).
    assert curve.loc[1, "strategy_return"] == pytest.approx(105 / 102 - 1)
    # Seduta 2: tiene tutta la seduta, gap notturno incluso.
    assert curve.loc[2, "strategy_return"] == pytest.approx(110 / 105 - 1)
    # Seduta 3: porta la posizione nella notte e vende all'apertura (110).
    assert curve.loc[3, "strategy_return"] == pytest.approx(110 / 110 - 1)
    # Totale: compra a 102, vende a 110.
    assert result.metrics["total_return"] == pytest.approx(110 / 102 - 1)
    assert result.metrics["execution_next_open"] == 1


def test_next_open_costs_are_charged_once_per_side() -> None:
    result = run_long_only_backtest(
        _frame(),
        cost_per_side=0.001,
        execution=ExecutionConfig(mode="next_open"),
    )

    assert result.equity_curve["trading_cost"].sum() == pytest.approx(0.002)
    assert result.equity_curve["turnover"].tolist() == [0, 1, 0, 1, 0]


def test_delay_shifts_execution_by_extra_days() -> None:
    result = run_long_only_backtest(
        _frame(),
        cost_per_side=0.0,
        execution=ExecutionConfig(mode="close", delay_days=1),
    )

    assert result.equity_curve["executed_position"].tolist() == [0, 0, 1, 1, 0]
    assert result.metrics["total_return"] == pytest.approx(104 / 105 - 1)
    assert result.metrics["execution_delay_days"] == 1


def test_cash_rate_accrues_only_on_uninvested_capital() -> None:
    frame = _frame()
    frame["target_position"] = 0

    result = run_long_only_backtest(
        frame,
        cost_per_side=0.0,
        execution=ExecutionConfig(cash_rate_annual=0.05),
    )

    daily = 1.05 ** (1 / 252) - 1
    assert result.metrics["total_return"] == pytest.approx((1 + daily) ** 5 - 1)
    assert result.metrics["cash_rate_annual"] == pytest.approx(0.05)

    invested = run_long_only_backtest(
        _frame(),
        cost_per_side=0.0,
        execution=ExecutionConfig(cash_rate_annual=0.05),
    )
    cash_return = invested.equity_curve["cash_return"].tolist()
    assert cash_return[1] == pytest.approx(0.0)
    assert cash_return[3] == pytest.approx(daily)


def test_next_open_requires_valid_open_prices() -> None:
    frame = _frame().drop(columns=["open"])

    with pytest.raises(ValueError, match="open"):
        run_long_only_backtest(
            frame,
            execution=ExecutionConfig(mode="next_open"),
        )

    bad = _frame()
    bad.loc[2, "open"] = 0.0

    with pytest.raises(ValueError, match="prezzi"):
        run_long_only_backtest(
            bad,
            execution=ExecutionConfig(mode="next_open"),
        )


def test_turnover_and_holding_metrics() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
            "close": np.linspace(100.0, 109.0, 10),
            "target_position": [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        }
    )

    result = run_long_only_backtest(frame, cost_per_side=0.0)

    # Due posizioni chiuse: 3 e 2 sedute.
    assert result.metrics["average_holding_days"] == pytest.approx(2.5)
    # Quattro operazioni (2 entrate + 2 uscite) su 9/252 anni.
    assert result.metrics["annual_turnover"] == pytest.approx(4 / (9 / 252))


def test_holding_days_is_none_without_completed_trades() -> None:
    frame = _frame()
    frame["target_position"] = 0

    result = run_long_only_backtest(frame)

    assert result.metrics["average_holding_days"] is None


# --------------------------------------------------------------------------
# Walk-forward, portafoglio e stress a selezione congelata
# --------------------------------------------------------------------------


def _random_market_frame(
    symbol: str,
    seed: int,
    periods: int = 160,
) -> pd.DataFrame:
    """Prezzi casuali con gap notturni, così la strategia apre posizioni."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.015, periods))
    open_ = close / (1.0 + rng.normal(0.0, 0.006, periods))

    return pd.DataFrame(
        {
            "symbol": [symbol] * periods,
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=periods,
                freq="D",
                tz="UTC",
            ),
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 3_000, periods).astype(float),
        }
    )


def _batch():
    frame = pd.concat(
        [
            _random_market_frame("AAA", seed=1),
            _random_market_frame("BBB", seed=2),
        ],
        ignore_index=True,
    )

    batch = run_batch_walk_forward_optimization(
        frame,
        ["AAA", "BBB"],
        _small_grid(),
        train_size=60,
        test_size=50,
        score_metric="total_return",
        minimum_completed_trades=0,
    )

    return frame, batch


def test_portfolio_execution_override_keeps_frozen_selections() -> None:
    frame, batch = _batch()

    close = run_walk_forward_portfolio_backtest(frame, batch)
    next_open = run_walk_forward_portfolio_backtest(
        frame,
        batch,
        execution=ExecutionConfig(mode="next_open"),
    )

    assert close.portfolio.metrics["mean_gross_exposure"] > 0.0

    # Stessi segnali e stessa esposizione: cambia solo il prezzo di esecuzione.
    assert next_open.portfolio.metrics["mean_gross_exposure"] == pytest.approx(
        close.portfolio.metrics["mean_gross_exposure"]
    )
    assert next_open.portfolio.metrics["total_return"] != pytest.approx(
        close.portfolio.metrics["total_return"]
    )


def test_walk_forward_records_execution_config() -> None:
    frame = pd.concat([_market_frame("AAA")], ignore_index=True)
    execution = ExecutionConfig(mode="next_open")

    batch = run_batch_walk_forward_optimization(
        frame,
        ["AAA"],
        _small_grid(),
        train_size=40,
        test_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
        execution=execution,
    )

    assert batch.results["AAA"].execution == execution

    # Senza override, il portafoglio usa l'esecuzione del walk-forward.
    implicit = run_walk_forward_portfolio_backtest(frame, batch)
    explicit = run_walk_forward_portfolio_backtest(
        frame,
        batch,
        execution=execution,
    )
    assert implicit.portfolio.metrics["total_return"] == pytest.approx(
        explicit.portfolio.metrics["total_return"]
    )


def test_execution_stress_analysis_builds_scenario_table() -> None:
    frame, batch = _batch()

    result = analyze_frozen_selection_execution_stress(frame, batch)
    table = result.scenarios

    assert table["scenario"].tolist() == [
        "close",
        "next_open",
        "next_open_delay_1",
        "close_delay_1",
    ]
    assert table.loc[0, "total_return"] == pytest.approx(
        run_walk_forward_portfolio_backtest(frame, batch)
        .portfolio.metrics["total_return"]
    )
    assert table.loc[0, "mean_gross_exposure"] == pytest.approx(
        table.loc[1, "mean_gross_exposure"]
    )


def test_execution_stress_rejects_bad_scenarios() -> None:
    frame, batch = _batch()

    with pytest.raises(ValueError, match="almeno uno"):
        analyze_frozen_selection_execution_stress(frame, batch, scenarios=[])

    with pytest.raises(ValueError, match="duplicato"):
        analyze_frozen_selection_execution_stress(
            frame,
            batch,
            scenarios=[
                ("a", ExecutionConfig()),
                ("a", ExecutionConfig()),
            ],
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_execution_arguments_default_to_none() -> None:
    args = cli.build_parser().parse_args(
        ["backtest", "--symbol", "SPY"]
    )

    assert args.execution_mode == "close"
    assert cli._execution_from_args(args) is None


def test_cli_execution_arguments_build_config() -> None:
    args = cli.build_parser().parse_args(
        [
            "batch-optimize-walk-forward",
            "--symbols",
            "SPY",
            "--train-size",
            "40",
            "--test-size",
            "20",
            "--execution-mode",
            "next_open",
            "--execution-delay-days",
            "1",
            "--cash-rate",
            "0.03",
        ]
    )

    assert cli._execution_from_args(args) == ExecutionConfig(
        mode="next_open",
        delay_days=1,
        cash_rate_annual=0.03,
    )
