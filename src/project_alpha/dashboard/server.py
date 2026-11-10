"""Server locale della dashboard e della web app (solo libreria standard).

- ascolta solo su ``127.0.0.1``: dall'iPhone ci si arriva tramite
  ``tailscale serve`` (rete privata tra i propri dispositivi, https);
- in sola lettura sul sistema: nessuna azione di trading, nessun ordine
  parte da qui; non legge ``.env`` né chiama il broker;
- le uniche scritture sono le iscrizioni dei dispositivi alle notifiche
  (``POST /api/push/...``), accettate solo dalla pagina stessa (controllo
  dell'origine e JSON obbligatorio) e solo verso i servizi push dei browser.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from project_alpha.dashboard.data import overview
from project_alpha.io_utils import dumps_json
from project_alpha.notifications import push_directory
from project_alpha.push import SubscriptionStore, VapidKeys, send_notification

LOCAL_HOSTS = {"127.0.0.1", "localhost"}
MAX_BODY_BYTES = 10_000
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; manifest-src 'self'; worker-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
STATIC_FILES = {
    "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
    "/sw.js": ("sw.js", "text/javascript; charset=utf-8"),
    "/icons/icon-192.png": ("icon-192.png", "image/png"),
    "/icons/icon-512.png": ("icon-512.png", "image/png"),
    "/icons/apple-touch-icon.png": ("apple-touch-icon.png", "image/png"),
}


def page_html() -> str:
    return files("project_alpha.dashboard").joinpath("index.html").read_text(encoding="utf-8")


def static_bytes(name: str) -> bytes:
    return files("project_alpha.dashboard").joinpath("static", name).read_bytes()


def origin_allowed(origin: str | None, host: str | None) -> bool:
    """La richiesta arriva dalla pagina stessa (stesso host) o da Tailscale/localhost."""
    if not origin:
        return False
    origin_host = (urlparse(origin).hostname or "").lower()
    host_name = (host or "").split(":")[0].lower()
    return origin_host == host_name or origin_host in LOCAL_HOSTS or origin_host.endswith(".ts.net")


def make_handler(root: Path, config_path: Path) -> type[BaseHTTPRequestHandler]:
    push_dir = push_directory(root)

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "ProjectAlphaDashboard"

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            if self.path.split("?", 1)[0] == "/sw.js":
                self.send_header("Service-Worker-Allowed", "/")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: Any) -> None:
            self._send(
                status, dumps_json(payload).encode("utf-8"), "application/json; charset=utf-8"
            )

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]

            if path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, page_html().encode("utf-8"), "text/html; charset=utf-8")
                return

            if path in STATIC_FILES:
                name, content_type = STATIC_FILES[path]
                self._send(HTTPStatus.OK, static_bytes(name), content_type)
                return

            if path == "/api/overview":
                try:
                    payload = overview(root, config_path)
                    status = HTTPStatus.OK
                except Exception as exc:  # la pagina mostra l'errore invece di restare vuota
                    payload = {"error": f"{type(exc).__name__}: {exc}"}
                    status = HTTPStatus.INTERNAL_SERVER_ERROR
                self._json(status, payload)
                return

            if path == "/api/push/key":
                keys = VapidKeys.load_or_create(push_dir)
                devices = len(SubscriptionStore(push_dir).all())
                self._json(HTTPStatus.OK, {"publicKey": keys.public_key_b64, "devices": devices})
                return

            self._send(HTTPStatus.NOT_FOUND, b"Non trovato", "text/plain; charset=utf-8")

        def do_HEAD(self) -> None:
            self.do_GET()

        def _read_json(self) -> Any | None:
            if not origin_allowed(self.headers.get("Origin"), self.headers.get("Host")):
                self._json(HTTPStatus.FORBIDDEN, {"error": "Origine non ammessa."})
                return None
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Serve JSON."})
                return None
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY_BYTES:
                self._json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Richiesta troppo grande."}
                )
                return None
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "JSON non valido."})
                return None

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path not in {"/api/push/subscribe", "/api/push/unsubscribe", "/api/push/test"}:
                self._send(HTTPStatus.NOT_FOUND, b"Non trovato", "text/plain; charset=utf-8")
                return

            payload = self._read_json()
            if payload is None:
                return

            store = SubscriptionStore(push_dir)

            if path == "/api/push/subscribe":
                try:
                    store.add(payload.get("subscription"), label=str(payload.get("label", "")))
                except (ValueError, AttributeError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._json(HTTPStatus.OK, {"ok": True, "devices": len(store.all())})
                return

            if path == "/api/push/unsubscribe":
                endpoint = payload.get("endpoint") if isinstance(payload, dict) else None
                removed = store.remove(str(endpoint)) if endpoint else False
                self._json(HTTPStatus.OK, {"ok": removed, "devices": len(store.all())})
                return

            delivery = send_notification(
                store,
                VapidKeys.load_or_create(push_dir),
                {
                    "title": "✅ Project Alpha",
                    "body": "Notifica di prova: da qui arriveranno gli esiti di ogni sera.",
                    "tag": "project-alpha-test",
                    "url": "/",
                },
            )
            self._json(
                HTTPStatus.OK,
                {
                    "sent": delivery.sent,
                    "removed": delivery.removed,
                    "failed": list(delivery.failed),
                },
            )

        def log_message(self, format: str, *args) -> None:
            return  # niente rumore nel terminale a ogni aggiornamento automatico

    return DashboardHandler


def create_server(
    root: Path | str,
    config_path: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
) -> ThreadingHTTPServer:
    if host not in LOCAL_HOSTS:
        raise ValueError("La dashboard ascolta solo in locale (127.0.0.1).")

    return ThreadingHTTPServer((host, port), make_handler(Path(root), Path(config_path)))


def serve(root: Path | str, config_path: Path | str, *, port: int = 8050) -> None:
    server = create_server(root, config_path, port=port)
    print(f"Dashboard su http://127.0.0.1:{server.server_address[1]}  (Ctrl+C per fermare)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
