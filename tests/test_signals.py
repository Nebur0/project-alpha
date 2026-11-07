"""Test del registro forward dei segnali."""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_alpha.research import load_research_config
from project_alpha.risk_engine import RiskPolicyV2
from project_alpha.signals import (
    SIGNAL_LOG_COLUMNS,
    read_signal_log,
    record_forward_signals,
)

CONFIG = """
name = "signals_test"
data = "data/bars.csv"
output_dir = "reports/research/signals_test"
registry = ""
symbols = ["AAA", "BBB"]

[walk_forward]
train_size = 60
test_size = 40
score_metric = "total_return"
minimum_completed_trades = 0

[grid]
fast_ema_period = [3, 4]
slow_ema_period = [6]
trend_ema_period = [12]
rsi_period = [5]
rsi_lower = [40.0]
rsi_upper = [80.0]
atr_period = [5]
volume_period = [5]
breakout_period = [5]
"""


def _write_data(path: Path, periods: int = 150, cut_last_of: str | None = None) -> pd.DataFrame:
    frames = []

    for seed, symbol in enumerate(["AAA", "BBB"], start=1):
        rng = np.random.default_rng(seed)
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.003, 0.015, periods))
        open_ = close / (1.0 + rng.normal(0.0, 0.005, periods))
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "timestamp": pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC"),
                "open": open_,
                "high": np.maximum(open_, close) * 1.01,
                "low": np.minimum(open_, close) * 0.99,
                "close": close,
                "volume": rng.integers(1_000, 3_000, periods).astype(float),
            }
        )
        if symbol == cut_last_of:
            frame = frame.iloc[:-1]
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(path, index=False)
    return data


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "config.toml").write_text(CONFIG, encoding="utf-8")
    return tmp_path


def _config(workspace: Path):
    return load_research_config(workspace / "config.toml", base_dir=workspace)


def test_records_one_row_per_symbol_and_is_idempotent(workspace: Path) -> None:
    data = _write_data(workspace / "data" / "bars.csv")
    log = workspace / "signals.csv"
    now = data["timestamp"].max().to_pydatetime() + timedelta(hours=20)

    first = record_forward_signals(_config(workspace), log_path=log, now=now)
    second = record_forward_signals(_config(workspace), log_path=log, now=now)

    table = read_signal_log(log)
    assert list(table.columns) == list(SIGNAL_LOG_COLUMNS)
    assert len(table) == 2
    assert set(table["symbol"]) == {"AAA", "BBB"}
    assert first.appended and not second.appended
    assert first.data_timestamp == data["timestamp"].max()
    assert set(table["target_weight"]) <= {0.0, 0.5}
    assert table["strategy"].unique().tolist() == ["signals_test"]


def test_fresh_data_passes_risk_engine(workspace: Path) -> None:
    data = _write_data(workspace / "data" / "bars.csv")
    now = data["timestamp"].max().to_pydatetime() + timedelta(hours=20)

    snapshot = record_forward_signals(
        _config(workspace),
        log_path=workspace / "signals.csv",
        policy=RiskPolicyV2(max_position_weight=0.5, max_group_weight=None, max_turnover=None),
        now=now,
    )

    assert snapshot.risk_allowed
    assert snapshot.risk_violations == ()


def test_stale_data_is_flagged_not_hidden(workspace: Path) -> None:
    data = _write_data(workspace / "data" / "bars.csv")
    now = data["timestamp"].max().to_pydatetime() + timedelta(days=30)

    snapshot = record_forward_signals(
        _config(workspace),
        log_path=workspace / "signals.csv",
        policy=RiskPolicyV2(max_position_weight=0.5, max_turnover=None),
        now=now,
    )

    # Se la strategia vuole posizioni, i dati vecchi devono bloccarle.
    if any(snapshot.rows["target_position"] > 0):
        assert "STALE_DATA" in snapshot.risk_violations
        assert not snapshot.risk_allowed
    assert read_signal_log(workspace / "signals.csv")["risk_mode"].iloc[0] == "NORMAL"


def test_misaligned_data_is_rejected(workspace: Path) -> None:
    _write_data(workspace / "data" / "bars.csv", cut_last_of="BBB")

    with pytest.raises(ValueError, match="non allineati"):
        record_forward_signals(_config(workspace), log_path=workspace / "signals.csv")


def test_missing_symbol_is_rejected(workspace: Path) -> None:
    _write_data(workspace / "data" / "bars.csv")
    (workspace / "config.toml").write_text(
        CONFIG.replace('["AAA", "BBB"]', '["AAA", "ZZZ"]'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ZZZ"):
        record_forward_signals(_config(workspace), log_path=workspace / "signals.csv")


def test_new_day_appends_and_uses_previous_weights(workspace: Path) -> None:
    data = _write_data(workspace / "data" / "bars.csv", periods=150)
    log = workspace / "signals.csv"
    first_now = data["timestamp"].max().to_pydatetime() + timedelta(hours=20)
    record_forward_signals(_config(workspace), log_path=log, now=first_now)

    _write_data(workspace / "data" / "bars.csv", periods=151)
    record_forward_signals(_config(workspace), log_path=log, now=first_now + timedelta(days=1))

    table = read_signal_log(log)
    assert len(table) == 4
    assert table["data_timestamp"].nunique() == 2
