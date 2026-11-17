"""Motore dell'app: lavori serali e orari e pianificatore interno.

Un solo processo (``project-alpha app``) tiene accesa la dashboard ed esegue
da sé, all'orario giusto e con i recuperi:

- **lavoro serale** (dopo la chiusura USA, ~23:00 in Italia): scarica i dati,
  li controlla, esegue tutte le strategie in paper trading
  (``configs/paper_*.toml``), scrive i riepiloghi, manda la notifica e
  aggiorna le notizie. È **idempotente per seduta**: se la seduta è già stata
  elaborata non fa nulla, quindi può girare anche dall'attività di Windows
  senza duplicare ordini; un lucchetto evita due esecuzioni contemporanee;
- **lavoro orario**: notizie e sentinella (solo con la borsa aperta);
- **recupero**: se il PC (o il server) era spento all'ora prevista, il lavoro
  serale parte appena possibile, ma mai a borsa aperta.

Funziona uguale sul PC e su un server nel cloud.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from project_alpha.data_quality import check_bars
from project_alpha.io_utils import dumps_json, write_text_atomic
from project_alpha.monitor import check_paper, load_inputs, render_report
from project_alpha.paper import (
    MARKET_TZ,
    PaperConfig,
    load_paper_config,
    run_paper,
    us_market_open,
)

DATA_START = date(2016, 1, 1)
CORE_CONFIG = "configs/paper_core_v1.toml"
# Ora di New York dopo la quale la barra giornaliera è definitiva (23:00 in Italia).
DAILY_HOUR_NY = 17
LOCK_STALE_SECONDS = 3600
RETRY_MINUTES = 30
MAX_ATTEMPTS_PER_SESSION = 3


def last_complete_session(now: datetime) -> date:
    """Ultima seduta USA (lun–ven) già chiusa e con dati definitivi a ``now``."""
    local = pd.Timestamp(now).tz_convert(MARKET_TZ)
    day = local.date()
    if local.weekday() >= 5 or local.hour < DAILY_HOUR_NY:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _default_fetch(symbols: list[str], output: Path) -> pd.DataFrame:
    from project_alpha.pipeline import fetch_and_save

    return fetch_and_save(symbols, DATA_START, None, output)


def _default_alpaca_broker():
    from project_alpha.config import Settings
    from project_alpha.execution.alpaca_paper import AlpacaPaperBroker

    return AlpacaPaperBroker(Settings.from_env())


def _default_news(root: Path, symbols: list[str]) -> dict[str, int]:
    from project_alpha.config import Settings
    from project_alpha.news import update_news

    return update_news(Settings.from_env(), root, symbols, days=2)


@dataclass
class JobContext:
    root: Path
    fetch: Callable[[list[str], Path], pd.DataFrame] = _default_fetch
    alpaca_broker: Callable[[], Any] = _default_alpaca_broker
    news: Callable[[Path, list[str]], dict[str, int]] | None = _default_news
    post: Callable[..., Any] | None = None  # invio push (None = requests.post)
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))

    @property
    def paper_dir(self) -> Path:
        return self.root / "reports" / "paper"

    def log(self, message: str) -> None:
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.clock().strftime("%Y-%m-%dT%H:%M:%SZ")
        with (self.paper_dir / "engine.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")


def paper_configs(root: Path) -> list[PaperConfig]:
    """Configurazioni paper: il core per primo, poi le prove in parallelo."""
    paths = sorted((root / "configs").glob("paper_*.toml"))
    paths.sort(key=lambda p: p.name != Path(CORE_CONFIG).name)
    return [load_paper_config(path, root=root) for path in paths]


# ----------------------------------------------------------------------
# Lucchetto e marcatore della seduta
# ----------------------------------------------------------------------


class RunLock:
    """Lucchetto su file: impedisce due lavori serali contemporanei."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and time.time() - self.path.stat().st_mtime > LOCK_STALE_SECONDS:
            self.path.unlink(missing_ok=True)  # lasciato da un'esecuzione interrotta
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return self
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, *exc) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


def _marker_path(ctx: JobContext) -> Path:
    return ctx.paper_dir / "last_daily.json"


def read_marker(ctx: JobContext) -> dict[str, Any]:
    path = _marker_path(ctx)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ----------------------------------------------------------------------
# Lavoro serale
# ----------------------------------------------------------------------


def _run_config(
    ctx: JobContext, config: PaperConfig, frame: pd.DataFrame, now: datetime, submit: bool
):
    if config.broker == "ledger":
        from project_alpha.execution.ledger import LedgerBroker

        ledger = LedgerBroker.load(config.ledger, initial_cash=config.budget, now=now)
        ledger.settle(frame)
        outcome = run_paper(config, ledger, frame=frame, submit=submit, now=now)
        if submit:
            ledger.save()
        return outcome
    return run_paper(config, ctx.alpaca_broker(), frame=frame, submit=submit, now=now)


def daily_run(ctx: JobContext, *, submit: bool = True, force: bool = False) -> dict[str, Any]:
    """Lavoro serale completo. Restituisce un riepilogo (anche in caso di salto)."""
    now = ctx.clock()
    session = last_complete_session(now).isoformat()

    if submit and us_market_open(now):
        return {"status": "WAIT_FOR_CLOSE", "session": session}

    marker = read_marker(ctx)
    if submit and not force and marker.get("session") == session and marker.get("ok"):
        return {"status": "ALREADY_DONE", "session": session}

    with RunLock(ctx.paper_dir / "daily.lock") as lock:
        if not lock.acquired:
            return {"status": "LOCKED", "session": session}
        summary: dict[str, Any] = {"status": "EXCEPTION", "session": session}
        try:
            summary = _daily_run_locked(ctx, now, session, submit)
            return summary
        except Exception as exc:
            ctx.log(f"lavoro serale fallito: {exc}\n{traceback.format_exc()}")
            summary["error"] = f"{type(exc).__name__}: {exc}"
            _notify_problem(
                ctx,
                f"Lavoro serale fallito ({type(exc).__name__}): riprovo tra {RETRY_MINUTES} minuti.",
            )
            return summary
        finally:
            if submit:
                _write_marker(ctx, session, ok=summary.get("status") == "OK", now=now)


def _write_marker(ctx: JobContext, session: str, *, ok: bool, now: datetime) -> None:
    previous = read_marker(ctx)
    attempts = previous.get("attempts", 0) if previous.get("session") == session else 0
    write_text_atomic(
        _marker_path(ctx),
        json.dumps({"session": session, "ok": ok, "attempts": attempts + 1, "at": now.isoformat()}),
    )


def _daily_run_locked(ctx: JobContext, now: datetime, session: str, submit: bool) -> dict[str, Any]:
    configs = paper_configs(ctx.root)
    summary: dict[str, Any] = {
        "session": session,
        "mode": "SUBMIT" if submit else "DRY_RUN",
        "strategies": {},
    }
    ctx.log(f"lavoro serale: seduta {session} ({summary['mode']})")

    by_data: dict[Path, set[str]] = {}
    for config in configs:
        by_data.setdefault(config.data, set()).update(config.symbols)

    frames: dict[Path, pd.DataFrame] = {}
    for path, symbols in by_data.items():
        frame = ctx.fetch(sorted(symbols), path)
        quality = check_bars(frame, sorted(symbols))
        write_text_atomic(
            ctx.paper_dir / "last_data_check.json", dumps_json(quality.summary(), indent=2)
        )
        if not quality.ok:
            summary["status"] = "DATA_ERROR"
            ctx.log(f"dati con errori: {[i.code for i in quality.errors]}")
            _notify_problem(ctx, "Dati di mercato con errori: nessuna operazione stasera.")
            return summary
        frames[path] = frame

    failures = 0
    for config in configs:
        try:
            outcome = _run_config(ctx, config, frames[config.data], now, submit)
            summary["strategies"][config.name] = outcome.summary()
            ctx.log(f"{config.name}: {outcome.status}")
        except Exception as exc:  # una strategia in errore non ferma le altre
            failures += 1
            summary["strategies"][config.name] = {"error": f"{type(exc).__name__}: {exc}"}
            ctx.log(f"{config.name}: ERRORE {exc}\n{traceback.format_exc()}")

        try:
            log, state, bars = load_inputs(config)
            result = check_paper(config, log, state, bars=bars, now=now)
            write_text_atomic(config.log.parent / "daily_report.md", render_report(config, result))
        except Exception as exc:
            ctx.log(f"{config.name}: riepilogo non scritto ({exc})")

    try:
        from project_alpha.notifications import notify_latest_run

        notify_latest_run(configs[0], ctx.root, post=ctx.post)
    except Exception as exc:
        ctx.log(f"notifica non inviata ({exc})")

    if failures:
        _notify_problem(ctx, f"{failures} strategie in errore nel lavoro serale: vedi engine.log.")

    if ctx.news is not None:
        try:
            ctx.news(ctx.root, sorted(set().union(*by_data.values())))
        except Exception as exc:
            ctx.log(f"notizie non aggiornate ({exc})")

    summary["status"] = "OK" if not failures else "PARTIAL"
    return summary


def _notify_problem(ctx: JobContext, text: str) -> None:
    try:
        from project_alpha.notifications import push_directory
        from project_alpha.push import SubscriptionStore, VapidKeys, send_notification

        directory = push_directory(ctx.root)
        send_notification(
            SubscriptionStore(directory),
            VapidKeys.load_or_create(directory),
            {
                "title": "⛔ Project Alpha: intervenire",
                "body": text,
                "tag": "project-alpha-error",
                "url": "/",
            },
            urgency="high",
            post=ctx.post,
        )
    except Exception as exc:
        ctx.log(f"notifica di errore non inviata ({exc})")


# ----------------------------------------------------------------------
# Lavoro orario
# ----------------------------------------------------------------------


def hourly_run(ctx: JobContext) -> dict[str, Any]:
    result: dict[str, Any] = {}
    configs = paper_configs(ctx.root)
    if ctx.news is not None:
        try:
            symbols = sorted(set().union(*(c.symbols for c in configs)))
            result["news"] = ctx.news(ctx.root, symbols)
        except Exception as exc:
            result["news"] = f"errore: {exc}"
    try:
        from project_alpha.watch import watch_paper

        core = configs[0]
        if core.broker == "alpaca" and us_market_open(ctx.clock()):
            result["watch"] = watch_paper(
                core, ctx.alpaca_broker(), ctx.root, now=ctx.clock(), post=ctx.post
            ).to_dict()
    except Exception as exc:
        result["watch"] = f"errore: {exc}"
    return result


# ----------------------------------------------------------------------
# Pianificatore
# ----------------------------------------------------------------------


def daily_due(ctx: JobContext) -> bool:
    """Il lavoro serale va fatto ora? (seduta nuova, borsa chiusa, tentativi residui)."""
    now = ctx.clock()
    if us_market_open(now):
        return False
    session = last_complete_session(now).isoformat()
    marker = read_marker(ctx)
    if marker.get("session") != session:
        return True
    if marker.get("ok"):
        return False
    last = datetime.fromisoformat(marker["at"])
    return marker.get("attempts", 0) < MAX_ATTEMPTS_PER_SESSION and now - last >= timedelta(
        minutes=RETRY_MINUTES
    )


class Scheduler:
    """Ciclo del motore: controlla ogni ``tick`` secondi cosa c'è da fare."""

    def __init__(self, ctx: JobContext, *, tick: float = 60.0) -> None:
        self.ctx = ctx
        self.tick = tick
        self.stop = threading.Event()
        self._last_hour: str | None = None

    def step(self) -> list[str]:
        done = []
        now = self.ctx.clock()
        if daily_due(self.ctx):
            try:
                result = daily_run(self.ctx)
                done.append(f"daily:{result.get('status')}")
            except Exception as exc:
                self.ctx.log(f"lavoro serale fallito: {exc}\n{traceback.format_exc()}")
                done.append("daily:EXCEPTION")
        hour = now.strftime("%Y-%m-%dT%H")
        if hour != self._last_hour:
            self._last_hour = hour
            try:
                result = hourly_run(self.ctx)
                for key, value in result.items():
                    if isinstance(value, str) and value.startswith("errore"):
                        self.ctx.log(f"lavoro orario, {key}: {value}")
                done.append("hourly")
            except Exception as exc:
                self.ctx.log(f"lavoro orario fallito: {exc}")
        return done

    def run_forever(self) -> None:
        self.ctx.log("motore avviato")
        while not self.stop.is_set():
            self.step()
            self.stop.wait(self.tick)


def run_app(root: Path | str, config_path: Path | str = CORE_CONFIG, *, port: int = 8050) -> None:
    """Dashboard + motore nello stesso processo (PC o server)."""
    from project_alpha.dashboard.server import create_server

    repo = Path(root)
    server = create_server(repo, config_path, port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(
        f"App su http://127.0.0.1:{server.server_address[1]} — motore attivo (Ctrl+C per fermare)"
    )
    scheduler = Scheduler(JobContext(repo))
    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.stop.set()
        server.shutdown()
        server.server_close()
