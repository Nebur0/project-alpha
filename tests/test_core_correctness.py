"""Test di regressione per le correzioni di correttezza del core.

Coprono tre bug emersi in revisione:

1. metriche non finite (Sharpe senza trade) che facevano fallire la
   serializzazione JSON e lasciavano file di report troncati;
2. griglie con combinazioni invalide (es. fast >= slow) che facevano
   fallire l'intera ricerca invece di scartare le sole combinazioni;
3. finestre walk-forward senza candidati validi che interrompevano
   l'intera ottimizzazione (ora configurabile con ``no_candidate_policy``).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_alpha import cli
from project_alpha.batch_optimization import (
    batch_walk_forward_comparison_to_frame,
    run_batch_walk_forward_optimization,
)
from project_alpha.io_utils import (
    dumps_json,
    to_json_safe,
    write_json_atomic,
    write_text_atomic,
)
from project_alpha.optimization import (
    generate_strategy_parameters,
    inspect_parameter_grid,
    run_walk_forward_optimization,
    walk_forward_optimization_to_dict,
    walk_forward_optimization_to_frame,
)
from project_alpha.pipeline import backtest_and_save
from project_alpha.portfolio_walk_forward import (
    reconstruct_walk_forward_oos_curve,
    run_walk_forward_portfolio_backtest,
)


def _market_frame(
    symbol: str = "TEST",
    periods: int = 100,
) -> pd.DataFrame:
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


def _no_trade_features_frame(
    symbol: str = "FLAT",
    periods: int = 30,
) -> pd.DataFrame:
    """Dataset già 'analizzato' in cui la strategia non entra mai."""
    return pd.DataFrame(
        {
            "symbol": [symbol] * periods,
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=periods,
                freq="D",
                tz="UTC",
            ),
            "close": np.linspace(100.0, 110.0, periods),
            "target_position": [0] * periods,
        }
    )


# --------------------------------------------------------------------------
# 1. JSON sicuro e scritture atomiche
# --------------------------------------------------------------------------


def test_to_json_safe_converts_non_finite_and_numpy_values() -> None:
    payload = {
        "sharpe": float("nan"),
        "upside": float("inf"),
        "downside": -np.inf,
        "trades": np.int64(3),
        "ratio": np.float64(0.25),
        "flag": np.bool_(True),
        "when": pd.Timestamp("2024-01-02", tz="UTC"),
        "nested": [float("nan"), (1, 2.5)],
    }

    safe = to_json_safe(payload)

    assert safe["sharpe"] is None
    assert safe["upside"] is None
    assert safe["downside"] is None
    assert safe["trades"] == 3 and isinstance(safe["trades"], int)
    assert safe["ratio"] == pytest.approx(0.25)
    assert safe["flag"] is True
    assert safe["when"] == "2024-01-02T00:00:00+00:00"
    assert safe["nested"] == [None, [1, 2.5]]

    # Il risultato deve essere JSON standard (niente NaN letterali).
    json.loads(dumps_json(payload))


def test_to_json_safe_rejects_unknown_types() -> None:
    with pytest.raises(TypeError, match="non serializzabile"):
        to_json_safe({"value": object()})


def test_write_json_atomic_keeps_previous_file_on_failure(
    tmp_path: Path,
) -> None:
    target = tmp_path / "report.json"
    write_json_atomic(target, {"version": 1})

    with pytest.raises(TypeError):
        write_json_atomic(target, {"version": object()})

    assert json.loads(target.read_text()) == {"version": 1}
    leftovers = [path.name for path in tmp_path.iterdir()]
    assert leftovers == ["report.json"]


def test_write_text_atomic_creates_parent_and_overwrites(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "out.txt"

    write_text_atomic(target, "first")
    write_text_atomic(target, "second")

    assert target.read_text() == "second"


def test_backtest_and_save_without_trades_writes_valid_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    features_path = tmp_path / "features.csv"
    _no_trade_features_frame().to_csv(features_path, index=False)

    result = backtest_and_save(
        features_path,
        "FLAT",
        initial_capital=10_000.0,
        cost_per_side=0.001,
    )

    # Nessuna posizione: Sharpe non definito.
    assert math.isnan(result.metrics["annualized_sharpe"])

    metrics = json.loads((tmp_path / "reports" / "FLAT_metrics.json").read_text())
    assert metrics["annualized_sharpe"] is None
    assert metrics["completed_trades"] == 0
    assert metrics["total_return"] == pytest.approx(0.0)
    assert (tmp_path / "reports" / "FLAT_equity.csv").exists()


def test_cli_batch_backtest_survives_symbol_without_trades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    traded = _no_trade_features_frame("LONG")
    traded["target_position"] = [0] * 5 + [1] * 20 + [0] * 5
    frame = pd.concat(
        [traded, _no_trade_features_frame("FLAT")],
        ignore_index=True,
    )
    features_path = tmp_path / "features.csv"
    frame.to_csv(features_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project-alpha",
            "batch-backtest",
            "--input",
            str(features_path),
            "--symbols",
            "LONG",
            "FLAT",
        ],
    )

    cli.main()

    report = json.loads((tmp_path / "reports" / "batch_metrics.json").read_text())
    assert set(report) == {"LONG", "FLAT"}
    assert report["FLAT"]["annualized_sharpe"] is None
    assert report["LONG"]["annualized_sharpe"] is not None
    assert "Report JSON salvato" in capsys.readouterr().out


# --------------------------------------------------------------------------
# 2. Griglia con combinazioni invalide
# --------------------------------------------------------------------------


def test_generate_strategy_parameters_skips_invalid_combinations() -> None:
    grid = {
        "fast_ema_period": [10, 30],
        "slow_ema_period": [30, 50],
    }

    combinations = generate_strategy_parameters(grid)
    pairs = {
        (parameters.fast_ema_period, parameters.slow_ema_period) for parameters in combinations
    }

    # (30, 30) viola fast < slow e viene scartata.
    assert pairs == {(10, 30), (10, 50), (30, 50)}

    inspection = inspect_parameter_grid(grid)
    assert inspection.invalid_count == 1
    assert inspection.total_count == 4
    assert "fast < slow < trend" in inspection.invalid_reasons[0]


def test_generate_strategy_parameters_raise_policy_keeps_old_behavior() -> None:
    with pytest.raises(ValueError, match="fast < slow < trend"):
        generate_strategy_parameters(
            {
                "fast_ema_period": [10, 30],
                "slow_ema_period": [30],
            },
            on_invalid="raise",
        )


def test_generate_strategy_parameters_fails_when_nothing_is_valid() -> None:
    with pytest.raises(ValueError, match="Nessuna combinazione valida"):
        generate_strategy_parameters(
            {
                "fast_ema_period": [60],
                "slow_ema_period": [50],
            }
        )


def test_generate_strategy_parameters_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="on_invalid"):
        generate_strategy_parameters(
            {"fast_ema_period": [10]},
            on_invalid="ignore",
        )


# --------------------------------------------------------------------------
# 3. Finestre walk-forward senza candidati
# --------------------------------------------------------------------------


def test_walk_forward_flat_policy_keeps_windows_in_cash() -> None:
    frame = _market_frame()

    result = run_walk_forward_optimization(
        frame,
        _small_grid(),
        train_size=40,
        test_size=20,
        score_metric="total_return",
        minimum_completed_trades=10_000,
        no_candidate_policy="flat",
    )

    assert result.windows
    assert result.flat_window_count == len(result.windows)

    for window in result.windows:
        assert window.is_flat
        assert window.selected_parameters is None
        assert window.selected_score is None
        assert window.in_sample_metrics == {}
        assert "Nessuna combinazione" in window.flat_reason
        assert window.out_of_sample_metrics["total_return"] == pytest.approx(0.0)
        assert window.out_of_sample_metrics["completed_trades"] == 0
        assert window.out_of_sample_metrics["annualized_sharpe"] is None

    assert result.summary["compounded_total_return"] == pytest.approx(0.0)
    assert result.summary["valid_sharpe_window_count"] == 0


def test_walk_forward_flat_windows_serialize() -> None:
    result = run_walk_forward_optimization(
        _market_frame(),
        _small_grid(),
        train_size=40,
        test_size=20,
        score_metric="total_return",
        minimum_completed_trades=10_000,
        no_candidate_policy="flat",
    )

    report = walk_forward_optimization_to_dict(result)
    assert report["no_candidate_policy"] == "flat"
    assert report["flat_window_count"] == len(result.windows)
    assert report["windows"][0]["selected_parameters"] is None
    assert report["windows"][0]["is_flat"] is True
    json.loads(dumps_json(report))

    frame = walk_forward_optimization_to_frame(result)
    assert frame["is_flat"].all()
    assert frame["fast_ema_period"].isna().all()


def test_walk_forward_mixed_windows_only_flatten_failing_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import project_alpha.optimization as optimization_module

    original = optimization_module._select_best_training_result
    calls = {"count": 0}

    def fail_first_window(results, *, minimum_completed_trades):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("Nessuna combinazione soddisfa i requisiti.")
        return original(
            results,
            minimum_completed_trades=minimum_completed_trades,
        )

    monkeypatch.setattr(
        optimization_module,
        "_select_best_training_result",
        fail_first_window,
    )

    result = run_walk_forward_optimization(
        _market_frame(),
        _small_grid(),
        train_size=40,
        test_size=20,
        score_metric="total_return",
        minimum_completed_trades=0,
        no_candidate_policy="flat",
    )

    assert result.windows[0].is_flat
    assert not any(window.is_flat for window in result.windows[1:])
    assert result.flat_window_count == 1


def test_walk_forward_default_policy_still_raises() -> None:
    with pytest.raises(ValueError, match="Nessuna combinazione soddisfa"):
        run_walk_forward_optimization(
            _market_frame(),
            _small_grid(),
            train_size=40,
            test_size=20,
            score_metric="total_return",
            minimum_completed_trades=10_000,
        )


def test_walk_forward_rejects_unknown_no_candidate_policy() -> None:
    with pytest.raises(ValueError, match="no_candidate_policy"):
        run_walk_forward_optimization(
            _market_frame(),
            _small_grid(),
            train_size=40,
            test_size=20,
            no_candidate_policy="skip",
        )


def test_portfolio_reconstruction_handles_flat_windows() -> None:
    frame = pd.concat(
        [_market_frame("AAA"), _market_frame("BBB")],
        ignore_index=True,
    )

    batch = run_batch_walk_forward_optimization(
        frame,
        ["AAA", "BBB"],
        _small_grid(),
        train_size=40,
        test_size=20,
        score_metric="total_return",
        minimum_completed_trades=10_000,
        no_candidate_policy="flat",
    )

    curve = reconstruct_walk_forward_oos_curve(
        frame,
        batch.results["AAA"],
    )
    assert (curve["executed_position"] == 0).all()
    assert curve["strategy_return"].abs().sum() == pytest.approx(0.0)

    portfolio = run_walk_forward_portfolio_backtest(frame, batch)
    assert portfolio.portfolio.metrics["mean_gross_exposure"] == pytest.approx(0.0)
    assert portfolio.portfolio.metrics["total_return"] == pytest.approx(0.0)

    comparison = batch_walk_forward_comparison_to_frame(batch)
    assert comparison["flat_window_count"].tolist() == [3, 3]


def test_cli_exposes_no_candidate_policy() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "optimize-walk-forward",
            "--symbol",
            "SPY",
            "--train-size",
            "40",
            "--test-size",
            "20",
            "--no-candidate-policy",
            "flat",
        ]
    )
    assert args.no_candidate_policy == "flat"

    batch_args = parser.parse_args(
        [
            "batch-optimize-walk-forward",
            "--symbols",
            "SPY",
            "--train-size",
            "40",
            "--test-size",
            "20",
        ]
    )
    assert batch_args.no_candidate_policy == "raise"
