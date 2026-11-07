"""Test dei dati: client Alpaca robusto, controllo qualità, chiavi nascoste."""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from project_alpha import cli
from project_alpha.config import Settings
from project_alpha.data.alpaca import fetch_daily_bars
from project_alpha.data_quality import check_bars

SETTINGS = Settings(
    alpaca_api_key="KEY-SEGRETA",
    alpaca_api_secret="SECRET-SEGRETO",
    alpaca_data_base_url="https://example.invalid",
)


# --------------------------------------------------------------------------
# Server finto
# --------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, headers, params, timeout):
        self.calls.append(dict(params))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _bar(day: int, close: float) -> dict:
    return {
        "t": f"2026-01-{day:02d}T05:00:00Z",
        "o": close,
        "h": close + 1,
        "l": close - 1,
        "c": close,
        "v": 1000,
        "n": 10,
        "vw": close,
    }


def _fetch(session, **kwargs):
    sleeps: list[float] = []
    frame = fetch_daily_bars(
        SETTINGS,
        ["spy"],
        date(2026, 1, 1),
        session=session,
        sleep=sleeps.append,
        **kwargs,
    )
    return frame, sleeps


def test_fetch_follows_pagination() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"bars": {"SPY": [_bar(2, 100)]}, "next_page_token": "abc"}),
            FakeResponse(200, {"bars": {"SPY": [_bar(5, 101)]}, "next_page_token": None}),
        ]
    )

    frame, sleeps = _fetch(session)

    assert frame["close"].tolist() == [100, 101]
    assert session.calls[1]["page_token"] == "abc"
    assert sleeps == []


def test_fetch_retries_rate_limit_with_retry_after() -> None:
    session = FakeSession(
        [
            FakeResponse(429, headers={"Retry-After": "7"}),
            FakeResponse(503),
            FakeResponse(200, {"bars": {"SPY": [_bar(2, 100)]}}),
        ]
    )

    frame, sleeps = _fetch(session, backoff_seconds=0.5)

    assert len(frame) == 1
    assert sleeps == [7.0, 1.0]  # Retry-After, poi backoff 0.5 * 2^1


def test_fetch_retries_network_errors() -> None:
    session = FakeSession(
        [
            requests.ConnectionError("giù"),
            FakeResponse(200, {"bars": {"SPY": [_bar(2, 100)]}}),
        ]
    )

    frame, sleeps = _fetch(session)

    assert len(frame) == 1
    assert len(sleeps) == 1


def test_fetch_gives_up_after_max_retries() -> None:
    session = FakeSession([FakeResponse(500)] * 3)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        _fetch(session, max_retries=2)

    assert len(session.calls) == 3


def test_fetch_does_not_retry_client_errors() -> None:
    session = FakeSession([FakeResponse(403, {"message": "forbidden"})])

    with pytest.raises(RuntimeError, match="HTTP 403"):
        _fetch(session)

    assert len(session.calls) == 1


def test_fetch_feed_parameter() -> None:
    session = FakeSession([FakeResponse(200, {"bars": {"SPY": [_bar(2, 100)]}})])
    _fetch(session, feed="iex")
    assert session.calls[0]["feed"] == "iex"

    with pytest.raises(ValueError, match="feed"):
        _fetch(FakeSession([]), feed="otc")


def test_fetch_without_bars_raises() -> None:
    with pytest.raises(RuntimeError, match="Nessuna candela"):
        _fetch(FakeSession([FakeResponse(200, {"bars": None})]))


def test_settings_repr_hides_keys() -> None:
    text = repr(SETTINGS)

    assert "KEY-SEGRETA" not in text
    assert "SECRET-SEGRETO" not in text
    assert "example.invalid" in text


# --------------------------------------------------------------------------
# Controllo qualità
# --------------------------------------------------------------------------


def _bars(symbol: str, days: int = 10, start: str = "2026-01-05") -> pd.DataFrame:
    timestamps = pd.bdate_range(start, periods=days, tz="America/New_York").tz_convert("UTC")
    close = np.linspace(100, 110, days)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": timestamps,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000.0,
        }
    )


def test_clean_data_passes() -> None:
    report = check_bars(pd.concat([_bars("AAA"), _bars("BBB")]))

    assert report.ok
    assert report.issues == ()
    assert report.symbols["rows"].tolist() == [10, 10]


def test_detects_symbol_that_ends_early() -> None:
    report = check_bars(pd.concat([_bars("AAA"), _bars("BBB", days=8)]))

    assert not report.ok
    assert [(issue.code, issue.symbol) for issue in report.errors] == [("ENDS_EARLY", "BBB")]


def test_detects_structural_errors() -> None:
    broken = _bars("AAA")
    broken.loc[1, "close"] = np.nan
    broken.loc[2, "low"] = 0.0
    broken.loc[3, "high"] = broken.loc[3, "low"] - 5
    broken = pd.concat([broken, broken.iloc[[4]]])

    codes = {issue.code for issue in check_bars(broken).errors}

    assert codes == {"MISSING_PRICES", "NON_POSITIVE_PRICES", "HIGH_BELOW_LOW", "DUPLICATES"}


def test_detects_warnings() -> None:
    aaa = _bars("AAA")
    aaa.loc[5, "close"] = aaa.loc[4, "close"] * 2  # +100%: split non aggiustato?
    aaa.loc[5, "high"] = aaa.loc[5, "close"] + 1
    aaa.loc[6, "open"] = aaa.loc[6, "high"] + 5  # open fuori range
    aaa.loc[7, "volume"] = 0
    bbb = _bars("BBB").drop(index=3)  # data mancante
    ccc = _bars("CCC", days=8, start="2026-01-07")  # inizia dopo

    report = check_bars(pd.concat([aaa, bbb, ccc]))
    codes = {(issue.code, issue.symbol) for issue in report.warnings}

    assert ("EXTREME_RETURNS", "AAA") in codes
    assert ("OUTSIDE_HIGH_LOW", "AAA") in codes
    assert ("ZERO_VOLUME", "AAA") in codes
    assert ("MISSING_DATES", "BBB") in codes
    assert ("STARTS_LATE", "CCC") in codes
    assert report.ok  # solo avvisi


def test_missing_requested_symbol_is_an_error() -> None:
    report = check_bars(_bars("AAA"), ["aaa", "zzz"])

    assert [(issue.code, issue.symbol) for issue in report.errors] == [("MISSING_SYMBOL", "ZZZ")]


def test_requires_columns() -> None:
    with pytest.raises(ValueError, match="Colonne mancanti"):
        check_bars(pd.DataFrame({"symbol": ["A"]}))


def test_cli_check_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = tmp_path / "good.csv"
    pd.concat([_bars("AAA"), _bars("BBB")]).to_csv(good, index=False)
    monkeypatch.setattr(sys, "argv", ["project-alpha", "check-data", "--input", str(good)])
    cli.main()
    assert json.loads(capsys.readouterr().out)["ok"] is True

    bad = tmp_path / "bad.csv"
    pd.concat([_bars("AAA"), _bars("BBB", days=7)]).to_csv(bad, index=False)
    monkeypatch.setattr(sys, "argv", ["project-alpha", "check-data", "--input", str(bad)])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["error_count"] == 1
