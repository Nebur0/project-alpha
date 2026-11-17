#!/usr/bin/env bash
# Project Alpha — ogni ora: notizie aggiornate e sentinella sul conto paper.
# La sentinella NON fa operazioni: con la borsa USA aperta controlla capitale,
# calo della giornata e distanza dal massimo, e manda una notifica solo se
# una soglia è superata (una volta per livello al giorno).
#
# Uso (Git Bash): scripts/hourly_watch.sh

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

for candidate in ".venv" "../.venv"; do
  if [[ -f "$candidate/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source "$candidate/Scripts/activate"
    break
  elif [[ -f "$candidate/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$candidate/bin/activate"
    break
  fi
done

LOG="reports/paper/hourly_watch.log"
mkdir -p reports/paper
{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python -m project_alpha news-fetch --days 1
  python -m project_alpha paper-watch --config configs/paper_core_v1.toml
} >> "$LOG" 2>&1
