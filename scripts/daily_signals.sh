#!/usr/bin/env bash
# Project Alpha — registro forward giornaliero (nessun ordine).
#
# 1. scarica le barre giornaliere aggiornate in data/raw/bars_live.csv
#    (il dataset di ricerca bars_dev15.csv non viene toccato);
# 2. controlla la qualità dei dati (si ferma se ci sono errori);
# 3. aggiunge i segnali del giorno a reports/signals/signal_log.csv.
#
# Uso (Git Bash, dalla radice del repository o da qualunque cartella):
#   scripts/daily_signals.sh
# Va eseguito dopo la chiusura del mercato USA (dopo le 22:30 ora italiana).
# Le chiavi Alpaca (solo paper) vengono lette da .env dal programma.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Ambiente virtuale: nella cartella del repo o in quella superiore.
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

SYMBOLS=(SPY QQQ AAPL MSFT NVDA AMZN GOOGL JPM XOM JNJ COST CAT NEE HD AMD)
LIVE_DATA="data/raw/bars_live.csv"
LOG_DIR="reports/signals"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/daily_runs.log"

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) avvio registro giornaliero" | tee -a "$RUN_LOG"

python -m project_alpha fetch \
  --symbols "${SYMBOLS[@]}" \
  --start 2018-01-01 \
  --output "$LIVE_DATA" 2>&1 | tee -a "$RUN_LOG"

python -m project_alpha check-data \
  --input "$LIVE_DATA" \
  --symbols "${SYMBOLS[@]}" > "$LOG_DIR/last_data_check.json"

python -m project_alpha signals \
  --config configs/signals_dev15_v1.toml \
  --log "$LOG_DIR/signal_log.csv" 2>&1 | tee -a "$RUN_LOG"

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) completato" | tee -a "$RUN_LOG"
