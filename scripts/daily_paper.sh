#!/usr/bin/env bash
# Project Alpha — paper trading giornaliero del core bilanciato (soldi finti).
#
# Esegue il lavoro serale (project-alpha daily-run): dati, controllo,
# strategie in paper trading, riepiloghi, notifica e notizie.
# Di norma lo fa già l'app (project-alpha app) all'orario giusto; questo
# script resta come riserva e per le prove manuali.
#
# Uso (Git Bash):  scripts/daily_paper.sh           -> invia ordini al conto paper
#                  scripts/daily_paper.sh --dry-run -> solo prova, nessun ordine
# Va eseguito dopo la chiusura del mercato USA (dopo le 22:30 ora italiana).
# Le chiavi Alpaca (solo paper, iniziano con PK) vengono lette da .env.

set -euo pipefail

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

MODE=""
if [[ "${1:-}" == "--dry-run" ]]; then
  MODE="--dry-run"
fi

mkdir -p reports/paper
# Tutto il lavoro serale è nel motore Python (dati, controllo, strategie,
# riepiloghi, notifica, notizie). È idempotente per seduta: se l'app l'ha già
# fatto, qui non succede nulla.
python -m project_alpha daily-run $MODE 2>&1 | tee -a reports/paper/daily_runs.log
