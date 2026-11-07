"""Test della pipeline di ricerca riproducibile (blocco 4)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_alpha import cli
from project_alpha.research import (
    REGISTRY_COLUMNS,
    config_fingerprint,
    file_sha256,
    load_research_config,
    read_registry,
    registry_trials,
    run_research,
)


def _write_market_data(path: Path) -> None:
    frames = []

    for seed, symbol in enumerate(["AAA", "BBB"], start=1):
        rng = np.random.default_rng(seed)
        periods = 200
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.015, periods))
        open_ = close / (1.0 + rng.normal(0.0, 0.006, periods))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "timestamp": pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC"),
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.01,
                    "low": np.minimum(open_, close) * 0.99,
                    "close": close,
                    "volume": rng.integers(1_000, 3_000, periods).astype(float),
                }
            )
        )

    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


CONFIG = """
name = "test_run"
data = "data/bars.csv"
output_dir = "reports/research/test_run"
registry = "reports/research/experiment_registry.csv"
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

[analysis]
costs_bps_per_side = [0.0, 10.0]
blend_defensive_weights = [0.0, 1.0]
permutations = 50
"""


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    _write_market_data(tmp_path / "data" / "bars.csv")
    (tmp_path / "research.toml").write_text(CONFIG, encoding="utf-8")
    return tmp_path


def test_load_research_config_resolves_paths(workspace: Path) -> None:
    config = load_research_config(workspace / "research.toml", base_dir=workspace)

    assert config.name == "test_run"
    assert config.data == workspace / "data" / "bars.csv"
    assert config.symbols == ("AAA", "BBB")
    assert config.walk_forward.train_size == 60
    assert config.execution.is_default
    assert config.analysis.costs_bps_per_side == (0.0, 10.0)
    assert config.registry == workspace / "reports" / "research" / "experiment_registry.csv"


def test_load_research_config_rejects_unknown_keys(workspace: Path) -> None:
    bad = workspace / "bad.toml"
    bad.write_text('unexpected = "x"\n' + CONFIG, encoding="utf-8")

    with pytest.raises(ValueError, match="chiavi sconosciute"):
        load_research_config(bad, base_dir=workspace)

    # Chiave sbagliata dentro una sezione: messaggio con la sezione.
    typo = workspace / "typo.toml"
    typo.write_text(CONFIG + "\npermutation = 10\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[analysis\]"):
        load_research_config(typo, base_dir=workspace)


def test_load_research_config_requires_walk_forward(workspace: Path) -> None:
    bad = workspace / "bad.toml"
    bad.write_text(
        'name = "x"\ndata = "a"\noutput_dir = "b"\nsymbols = ["A"]\n[grid]\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="walk_forward"):
        load_research_config(bad, base_dir=workspace)


def test_run_research_writes_outputs_manifest_and_registry(workspace: Path) -> None:
    config = load_research_config(workspace / "research.toml", base_dir=workspace)

    result = run_research(config, repo_dir=workspace)
    output = workspace / "reports" / "research" / "test_run"

    for name in [
        "walk_forward.csv",
        "walk_forward_comparison.csv",
        "portfolio_metrics.json",
        "portfolio_equity.csv",
        "evaluation.json",
        "evaluation_by_window.csv",
        "evaluation_on_off.csv",
        "cost_stress.csv",
        "execution_stress.csv",
        "blend_overall.csv",
        "policy_baselines.csv",
        "manifest.json",
    ]:
        assert (output / name).exists(), name

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["run_id"] == result.run_id
    assert manifest["data"]["sha256"] == file_sha256(workspace / "data" / "bars.csv")
    assert manifest["config"]["grid_size"] == 2
    assert manifest["trials_on_same_data"] == 1
    # La cartella temporanea non è un repository git.
    assert manifest["git"]["commit"] is None

    registry = read_registry(workspace / "reports" / "research" / "experiment_registry.csv")
    assert list(registry.columns) == list(REGISTRY_COLUMNS)
    assert registry["run_id"].tolist() == [result.run_id]
    assert result.summary["trials"] == 1

    baselines = pd.read_csv(output / "policy_baselines.csv")
    assert baselines["scenario"].tolist() == [
        "strategy",
        "passive_100",
        "passive_static_same_exposure",
        "passive_vol_target_10",
        "passive_vol_target_6",
        "half_passive_half_strategy",
    ]


def test_registry_counts_distinct_configurations(workspace: Path) -> None:
    config = load_research_config(workspace / "research.toml", base_dir=workspace)

    run_research(config, repo_dir=workspace)
    repeat = run_research(config, repo_dir=workspace)

    # Rieseguire la stessa configurazione non è una prova nuova.
    assert repeat.summary["trials"] == 1

    variant_path = workspace / "variant.toml"
    variant_path.write_text(
        CONFIG.replace("fast_ema_period = [3, 4]", "fast_ema_period = [2, 3]"),
        encoding="utf-8",
    )
    variant = run_research(
        load_research_config(variant_path, base_dir=workspace),
        repo_dir=workspace,
    )

    registry = read_registry(workspace / "reports" / "research" / "experiment_registry.csv")
    assert len(registry) == 3
    assert registry["config_sha256"].nunique() == 2
    assert variant.summary["trials"] == 2

    trials, _ = registry_trials(registry, file_sha256(workspace / "data" / "bars.csv"))
    assert trials == 2
    assert registry_trials(registry, "altro-hash") == (0, None)


def test_config_fingerprint_ignores_name_and_outputs(workspace: Path) -> None:
    base = load_research_config(workspace / "research.toml", base_dir=workspace)
    renamed_path = workspace / "renamed.toml"
    renamed_path.write_text(
        CONFIG.replace('name = "test_run"', 'name = "altro"').replace(
            "reports/research/test_run", "reports/research/altro"
        ),
        encoding="utf-8",
    )
    renamed = load_research_config(renamed_path, base_dir=workspace)

    assert config_fingerprint(base) == config_fingerprint(renamed)


def test_run_research_rejects_missing_symbols(workspace: Path) -> None:
    config_path = workspace / "missing.toml"
    config_path.write_text(CONFIG.replace('["AAA", "BBB"]', '["AAA", "ZZZ"]'), encoding="utf-8")
    config = load_research_config(config_path, base_dir=workspace)

    with pytest.raises(ValueError, match="ZZZ"):
        run_research(config, repo_dir=workspace)


def test_cli_research_command(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        sys,
        "argv",
        ["project-alpha", "research", "--config", "research.toml"],
    )

    cli.main()

    printed = json.loads(capsys.readouterr().out)
    assert printed["summary"]["trials"] == 1
    assert (workspace / "reports" / "research" / "test_run" / "manifest.json").exists()
