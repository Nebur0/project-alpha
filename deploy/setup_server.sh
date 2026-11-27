#!/usr/bin/env bash
# Project Alpha — installazione su un server Ubuntu nuovo (docs/CLOUD.md).
# Uso, dal server, con alpha_stato.tgz nella stessa cartella:
#   curl -fsSLO https://raw.githubusercontent.com/Nebur0/project-alpha/main/deploy/setup_server.sh
#   sudo bash setup_server.sh
# Chiede le chiavi PAPER di Alpaca (non le stampa) e l'accesso a Tailscale.
set -euo pipefail

REPO_URL="https://github.com/Nebur0/project-alpha.git"
TARGET="/opt/project-alpha"
STATE="$(pwd)/alpha_stato.tgz"
COMPOSE=(docker compose -f "$TARGET/deploy/compose.yaml")

if [[ $EUID -ne 0 ]]; then
  echo "Avvia con sudo: sudo bash setup_server.sh" >&2
  exit 1
fi

echo "== 1/6 Docker e Tailscale"
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
tailscale status >/dev/null 2>&1 || tailscale up

echo "== 2/6 Codice"
if [[ -d "$TARGET/.git" ]]; then git -C "$TARGET" pull --ff-only; else git clone "$REPO_URL" "$TARGET"; fi

echo "== 3/6 Chiavi paper di Alpaca (non vengono mostrate)"
if [[ ! -f "$TARGET/.env" ]]; then
  read -rp "ALPACA_API_KEY (inizia con PK): " key
  [[ "$key" == PK* ]] || { echo "Serve una chiave paper (PK...)." >&2; exit 1; }
  read -rsp "ALPACA_API_SECRET: " secret; echo
  umask 077
  printf 'ALPACA_API_KEY=%s\nALPACA_API_SECRET=%s\n' "$key" "$secret" > "$TARGET/.env"
fi

echo "== 4/6 Container (creato, non ancora avviato)"
"${COMPOSE[@]}" build
"${COMPOSE[@]}" create

echo "== 5/6 Stato dei portafogli dal PC"
if [[ -f "$STATE" ]]; then
  tmp="$(mktemp -d)"
  tar -xzf "$STATE" -C "$tmp"
  for dir in reports/paper reports/paper_permanente reports/paper_crescita reports/paper_bitcoin \
             data/push data/paper_permanente data/paper_crescita data/paper_bitcoin; do
    [[ -d "$tmp/$dir" ]] && "${COMPOSE[@]}" cp "$tmp/$dir/." "alpha:/app/$dir/"
  done
  rm -rf "$tmp"
  "${COMPOSE[@]}" run --rm --no-deps -u root --entrypoint chown alpha -R alpha:alpha /app/data /app/reports
else
  echo "ATTENZIONE: alpha_stato.tgz non trovato: i portafogli ripartono da zero." >&2
  read -rp "Continuare comunque? [s/N] " answer
  [[ "$answer" == s ]] || exit 1
fi

echo "== 6/6 Avvio e accesso dall'iPhone"
"${COMPOSE[@]}" up -d
tailscale serve --bg 8050
echo
echo "Fatto. Indirizzo per l'iPhone:"
tailscale serve status | grep -o 'https://[^ ]*' | head -1
echo "Registro: ${COMPOSE[*]} logs -f"
