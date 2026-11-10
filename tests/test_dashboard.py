"""Test della dashboard locale (dati e server, nessuna rete esterna)."""

import json
import shutil
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_alpha.dashboard.data import overview
from project_alpha.dashboard.server import create_server, page_html
from project_alpha.execution.broker import SimulatedBroker
from project_alpha.paper import load_paper_config, run_paper

REPO = Path(__file__).resolve().parents[1]


def _bars(symbols, days=120, seed=0, end="2026-11-09"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=days, tz="UTC") + pd.Timedelta(hours=5)
    rows = []
    for index, symbol in enumerate(symbols):
        price = 100.0 + 10 * index
        for timestamp in dates:
            price *= 1 + rng.normal(0, 0.008)
            rows.append({"symbol": symbol, "timestamp": timestamp, "close": price})
    return pd.DataFrame(rows)


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO / "configs" / "paper_core_v1.toml", tmp_path / "configs")
    shutil.copytree(REPO / "reports" / "research", tmp_path / "reports" / "research")
    return tmp_path


def test_overview_without_paper_runs_explains_what_is_missing(repo) -> None:
    data = overview(repo, "configs/paper_core_v1.toml")

    assert data["paper"]["available"] is False
    assert data["signals"]["available"] is False
    assert data["ips"]["limits"]["stop_drawdown"] == -0.12
    assert data["ips"]["limits"]["ips_max_drawdown"] == -0.15
    assert data["ips"]["group_weights"] == pytest.approx(
        {"azioni": 0.6, "obbligazioni": 0.3, "reali": 0.1}
    )
    names = {run["name"] for run in data["research"]["runs"]}
    assert {"dev15_v1", "etf_multiasset_v1"} <= names
    assert "etf_multiasset_v1" in data["research"]["baselines"]
    json.dumps(data, allow_nan=False)


def test_overview_reads_paper_log(repo) -> None:
    config = load_paper_config(repo / "configs" / "paper_core_v1.toml", root=repo)
    broker = SimulatedBroker(cash=10_000.0, commission_rate=0.0)
    now = datetime(2026, 11, 9, 22, 30, tzinfo=UTC)
    run_paper(config, broker, frame=_bars(config.symbols), submit=True, now=now)

    paper = overview(repo, "configs/paper_core_v1.toml")["paper"]

    assert paper["available"] and paper["runs"] == 1 and paper["submitted_runs"] == 1
    assert paper["last"]["status"] == "EXECUTED"
    assert set(paper["last"]["target_weights"]) == set(config.weights)
    assert paper["last"]["orders"][0]["status"] == "FILLED"
    assert paper["state"]["peak_equity"] == pytest.approx(paper["last"]["equity"], abs=0.01)


def test_old_paper_log_gains_new_columns(repo) -> None:
    config = load_paper_config(repo / "configs" / "paper_core_v1.toml", root=repo)
    config.log.parent.mkdir(parents=True)
    config.log.write_text("run_timestamp_utc,mode,status\n2026-11-01,DRY_RUN,DRY_RUN\n")
    now = datetime(2026, 11, 9, 22, 30, tzinfo=UTC)

    run_paper(replace(config), SimulatedBroker(cash=10_000.0), frame=_bars(config.symbols), now=now)

    header = config.log.read_text().splitlines()[0]
    assert "target_weights" in header and "current_weights" in header


def test_server_is_local_only_and_read_only(repo) -> None:
    with pytest.raises(ValueError, match="locale"):
        create_server(repo, "configs/paper_core_v1.toml", host="0.0.0.0", port=0)

    server = create_server(repo, "configs/paper_core_v1.toml", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        with urllib.request.urlopen(f"{base}/") as response:
            assert "Project Alpha" in response.read().decode("utf-8")
            assert "default-src 'none'" in response.headers["Content-Security-Policy"]

        with urllib.request.urlopen(f"{base}/api/overview") as response:
            assert json.loads(response.read())["ips"]["name"] == "core_bilanciato_v1"

        post = urllib.request.Request(f"{base}/api/overview", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(post)
        assert error.value.code == 404  # nessuna azione sui dati via POST

        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{base}/../.env")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_page_has_no_external_resources() -> None:
    html = page_html()

    assert "http://" not in html.replace("http://127.0.0.1", "")
    assert "https://" not in html
