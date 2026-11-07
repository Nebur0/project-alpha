"""Server locale della dashboard (solo libreria standard, sola lettura).

- ascolta solo su ``127.0.0.1``: la dashboard non è raggiungibile da altri
  dispositivi della rete;
- accetta solo ``GET``: nessuna azione, nessun ordine parte da qui;
- non legge ``.env`` né chiama il broker: mostra i file che il sistema scrive
  (registro paper, stato, segnali, registro delle prove).
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

from project_alpha.dashboard.data import overview
from project_alpha.io_utils import dumps_json

LOCAL_HOSTS = {"127.0.0.1", "localhost"}
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def page_html() -> str:
    return files("project_alpha.dashboard").joinpath("index.html").read_text(encoding="utf-8")


def make_handler(root: Path, config_path: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "ProjectAlphaDashboard"

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]

            if path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, page_html().encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/overview":
                try:
                    payload = overview(root, config_path)
                    status = HTTPStatus.OK
                except Exception as exc:  # la pagina mostra l'errore invece di restare vuota
                    payload = {"error": f"{type(exc).__name__}: {exc}"}
                    status = HTTPStatus.INTERNAL_SERVER_ERROR
                body = dumps_json(payload).encode("utf-8")
                self._send(status, body, "application/json; charset=utf-8")
                return

            self._send(HTTPStatus.NOT_FOUND, b"Non trovato", "text/plain; charset=utf-8")

        def do_HEAD(self) -> None:
            self.do_GET()

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
