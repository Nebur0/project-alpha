#!/usr/bin/env bash
# Project Alpha — ogni ora: notizie aggiornate e sentinella sul conto paper.
# La sentinella NON fa operazioni: con la borsa USA aperta controlla capitale,
# calo della giornata e distanza dal massimo, e manda una notifica solo se
# una soglia è superata (una volta per livello al giorno).
#
# Uso (Git Bash): scripts/hourly_watch.sh

set -uo pipefail

# Output Python scritto subito nel registro (anche se il processo si blocca).
export PYTHONUNBUFFERED=1

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

mkdir -p reports/paper
python -m project_alpha hourly-run >> reports/paper/hourly_watch.log 2>&1
