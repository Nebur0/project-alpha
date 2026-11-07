import pandas as pd
import pytest

from project_alpha.portfolio import (
    PortfolioResult,
)
from project_alpha.portfolio_blend_analysis import (
    analyze_walk_forward_portfolio_blends,
)
from project_alpha.portfolio_walk_forward import (
    WalkForwardPortfolioResult,
)


def _walk_forward_result() -> WalkForwardPortfolioResult:
    timestamps = pd.date_range(
        "2025-01-01",
        periods=8,
        freq="D",
        tz="UTC",
    )

    portfolio_curve = pd.DataFrame(
        {
            "timestamp": timestamps,
            "portfolio_return": [
                0.00,
                -0.01,
                0.00,
                0.01,
                0.00,
                0.01,
                0.01,
                0.00,
            ],
            "benchmark_return": [
                0.00,
                -0.04,
                -0.02,
                0.01,
                0.00,
                0.03,
                0.02,
                0.01,
            ],
        }
    )

    portfolio = PortfolioResult(
        equity_curve=portfolio_curve,
        metrics={
            "initial_capital": 1000.0,
        },
        weights={
            "AAA": 0.5,
            "BBB": 0.5,
        },
    )

    window_numbers = [
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ]

    symbol_curves = {
        "AAA": pd.DataFrame(
            {
                "timestamp": timestamps,
                "window_number": (window_numbers),
            }
        ),
        "BBB": pd.DataFrame(
            {
                "timestamp": timestamps,
                "window_number": (window_numbers),
            }
        ),
    }

    return WalkForwardPortfolioResult(
        portfolio=portfolio,
        symbol_curves=symbol_curves,
    )


def test_analysis_reports_default_weight_grid() -> None:
    result = analyze_walk_forward_portfolio_blends(_walk_forward_result())

    assert result.overall["defensive_weight"].tolist() == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]

    assert result.overall["passive_weight"].tolist() == [
        1.0,
        0.75,
        0.5,
        0.25,
        0.0,
    ]


def test_analysis_reports_each_window_and_leave_one_out() -> None:
    result = analyze_walk_forward_portfolio_blends(_walk_forward_result())

    assert len(result.by_window) == 10

    assert set(result.by_window["window_number"]) == {
        1,
        2,
    }

    assert len(result.leave_one_window_out) == 10

    assert set(result.leave_one_window_out["omitted_window_number"]) == {
        1,
        2,
    }


def test_blend_returns_match_endpoints() -> None:
    source = _walk_forward_result()

    result = analyze_walk_forward_portfolio_blends(
        source,
        defensive_weights=[
            0.0,
            1.0,
        ],
    )

    passive = result.overall.loc[result.overall["defensive_weight"].eq(0.0)].iloc[0]

    defensive = result.overall.loc[result.overall["defensive_weight"].eq(1.0)].iloc[0]

    expected_passive_return = (1.0 + source.portfolio.equity_curve["benchmark_return"]).prod() - 1.0

    expected_defensive_return = (
        1.0 + source.portfolio.equity_curve["portfolio_return"]
    ).prod() - 1.0

    assert passive["total_return"] == pytest.approx(expected_passive_return)

    assert defensive["total_return"] == pytest.approx(expected_defensive_return)


@pytest.mark.parametrize(
    "weights",
    [
        [],
        [-0.1],
        [1.1],
        [float("nan")],
    ],
)
def test_analysis_rejects_invalid_weights(
    weights: list[float],
) -> None:
    with pytest.raises(ValueError):
        analyze_walk_forward_portfolio_blends(
            _walk_forward_result(),
            defensive_weights=weights,
        )


def test_analysis_rejects_inconsistent_window_mapping() -> None:
    source = _walk_forward_result()

    inconsistent = source.symbol_curves["BBB"].copy()

    inconsistent.loc[
        0,
        "window_number",
    ] = 99

    broken = WalkForwardPortfolioResult(
        portfolio=source.portfolio,
        symbol_curves={
            "AAA": source.symbol_curves["AAA"],
            "BBB": inconsistent,
        },
    )

    with pytest.raises(
        ValueError,
        match="non coincidono",
    ):
        analyze_walk_forward_portfolio_blends(broken)
