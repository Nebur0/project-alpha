#!/usr/bin/env bash
# Project Alpha — paper trading giornaliero del core bilanciato (soldi finti).
#
# 1. scarica le barre aggiornate degli ETF del core in data/raw/bars_core.csv;
# 2. controlla la qualità dei dati (si ferma se ci sono errori);
# 3. ribilancia il conto Alpaca PAPER se i pesi sono fuori banda
#    (gli ordini restano in coda e vengono eseguiti all'apertura successiva).
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

SUBMIT="--submit"
if [[ "${1:-}" == "--dry-run" ]]; then
  SUBMIT=""
fi

SYMBOLS=(SPY QQQ IWM EFA EEM IEF LQD TLT GLD VNQ)
DATA="data/raw/bars_core.csv"
LOG_DIR="reports/paper"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/daily_runs.log"

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) avvio paper trading ${SUBMIT:-(prova)}" | tee -a "$RUN_LOG"

python -m project_alpha fetch \
  --symbols "${SYMBOLS[@]}" \
  --start 2016-01-01 \
  --output "$DATA" 2>&1 | tee -a "$RUN_LOG"

python -m project_alpha check-data \
  --input "$DATA" \
  --symbols "${SYMBOLS[@]}" > "$LOG_DIR/last_data_check.json"

python -m project_alpha paper-rebalance \
  --config configs/paper_core_v1.toml $SUBMIT 2>&1 | tee -a "$RUN_LOG"

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) completato" | tee -a "$RUN_LOG"
