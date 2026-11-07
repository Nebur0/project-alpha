[![Tests](https://github.com/Nebur0/project-alpha/actions/workflows/tests.yml/badge.svg)](https://github.com/Nebur0/project-alpha/actions/workflows/tests.yml)

# Project Alpha

Piattaforma personale per gestire un portafoglio di investimento in modo
**sistematico e sotto limiti di rischio deterministici**. Prima dimostra con
test onesti quali regole funzionano davvero, poi le prova senza soldi
(paper trading) e solo alla fine, se superano criteri decisi in anticipo,
le applica con capitale reale piccolo.

- Visione, fasi e criteri: [`ROADMAP.md`](ROADMAP.md)
- Regole per lo sviluppo (anche per Claude): [`CLAUDE.md`](CLAUDE.md)

> Progetto di ricerca. Non esegue ordini reali e non è una raccomandazione
> finanziaria. Un buon risultato storico non garantisce risultati futuri.

## Stato attuale

La baseline **V1** (trend/breakout long-only su azioni USA) è stata
analizzata su DEV15 (15 titoli, 2018 → 2026) con walk-forward, portafoglio
equal-weight, stress di costi ed esecuzione e una valutazione a pari
esposizione:

| Esecuzione | CAGR | Sharpe | Max DD | vs benchmark statico a pari esposizione |
|---|---|---|---|---|
| Chiusura del giorno del segnale | 5,29% | 0,98 | −4,8% | +1,30 pt/anno |
| Apertura del giorno dopo | 4,91% | 0,91 | −5,7% | +0,92 pt/anno |
| Apertura + 1 giorno di ritardo | 4,22% | 0,80 | −7,8% | +0,25 pt/anno |

Il vantaggio non è distinguibile dal caso (test di permutazione p ≈ 0,18):
la protezione nei ribassi viene soprattutto dall'esposizione media bassa
(~21%). Dettagli in `reports/research/dev15_v1/` e `reports/evaluation/`.

## Architettura

```
Dati di mercato (Alpaca, daily OHLCV)
  → Indicatori e strategia            indicators.py, strategy.py
  → Backtest ed esecuzione            backtest.py (ExecutionConfig)
  → Walk-forward e ottimizzazione     validation.py, optimization.py, batch_optimization.py
  → Portafoglio                       portfolio.py, portfolio_walk_forward.py
  → Valutazione standard              evaluation.py
  → Stress e analisi                  cost_stress_analysis.py, execution_stress_analysis.py,
                                      portfolio_blend_analysis.py
  → Rischio deterministico            risk.py
  → Pipeline riproducibile            research.py (+ manifest e registro delle prove)
```

Analisi di robustezza storiche (usate per i report in
`reports/advanced_robustness/`): `robustness.py`, `regime_analysis.py`,
`temporal_robustness.py`, `parameter_stability.py`, `selection_skill.py`,
`grid_flatness.py`, `statistical_robustness.py`.

## Installazione

Python 3.11 o superiore.

```bash
py -m venv .venv
source .venv/Scripts/activate        # Git Bash su Windows
python -m pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env                 # inserire le chiavi Alpaca (solo paper)
```

Non pubblicare mai il file `.env`.

## Uso

### Pipeline di ricerca (consigliato)

Un solo comando rigenera tutta l'analisi da una configurazione TOML:

```bash
project-alpha research --config configs/dev15_v1.toml
```

Scrive in `reports/research/<nome>/`: walk-forward, curva del portafoglio,
valutazione (`evaluation.json`), stress di costi ed esecuzione, blend e un
`manifest.json` con configurazione, commit git e hash SHA-256 dei dati.
Ogni esecuzione aggiunge una riga a `reports/research/experiment_registry.csv`:
il numero di prove sugli stessi dati entra nel Deflated Sharpe Ratio.

### Qualità dei dati

```bash
project-alpha check-data --input data/raw/bars_dev15.csv
```

Errori (duplicati, prezzi mancanti o non positivi, high < low, simboli che
finiscono prima degli altri) e avvisi (date mancanti, rendimenti estremi,
volumi nulli). `fetch` esegue lo stesso controllo dopo ogni download e
riprova da solo in caso di errori temporanei di Alpaca (429/5xx);
`--feed iex|sip` fissa la fonte dei dati. Il risultato del controllo
finisce anche nel `manifest.json` di ogni ricerca.

### Registro forward dei segnali (nessun ordine)

```bash
scripts/daily_signals.sh
```

Ogni sera, dopo la chiusura USA: scarica i dati aggiornati in
`data/raw/bars_live.csv` (il dataset di ricerca `bars_dev15.csv` resta
congelato), ne controlla la qualità e aggiunge a
`reports/signals/signal_log.csv` il target di domani per ogni simbolo, con
i parametri che la prossima finestra walk-forward sceglierebbe e la
decisione del Risk Engine V2. Idempotente: rilanciarlo lo stesso giorno non
duplica. Nessun ordine viene inviato.

Per eseguirlo in automatico su Windows (Utilità di pianificazione →
Crea attività di base → ogni giorno, lun–ven, ore 23:00):

- Programma: `C:\Program Files\Git\bin\bash.exe`
- Argomenti: `-lc "<percorso del repo>/scripts/daily_signals.sh"`

Il registro va versionato (`git add reports/signals/signal_log.csv`): è la
prova in avanti della strategia.

### Fase 1: V1 congelata su ETF multi-asset

```bash
project-alpha fetch --symbols SPY QQQ IWM EFA EEM VNQ TLT IEF LQD GLD DBC \
  --start 2016-01-01 --output data/raw/bars_etf.csv
project-alpha research --config configs/etf_multiasset_v1.toml
```

`policy_baselines.csv` confronta la strategia con il passivo al 100%, il
passivo statico alla stessa esposizione e il passivo con volatility
targeting (10% e 6%).

### Comandi singoli

```bash
project-alpha fetch --symbols SPY QQQ AAPL --start 2018-01-01 --output data/raw/bars.csv
project-alpha analyze --input data/raw/bars.csv
project-alpha backtest --input data/processed/features.csv --symbol SPY
project-alpha optimize-walk-forward --symbol SPY --train-size 1000 --test-size 252
project-alpha --help
```

Opzioni di esecuzione (su `backtest`, `batch-backtest`, `run`, `grid-search`
e i comandi `optimize-walk-forward`):

- `--execution-mode close|next_open` — chiusura del giorno del segnale
  (default storico) o apertura del giorno dopo;
- `--execution-delay-days N` — giorni di ritardo aggiuntivi;
- `--cash-rate 0.03` — rendimento annuo della quota in cash.

Con `--no-candidate-policy flat` una finestra walk-forward senza candidati
validi resta in cash invece di interrompere l'analisi.

## Test e qualità

```bash
python -m pytest -q
ruff check src tests
ruff format --check src tests
```

La CI esegue lint e test su Python 3.11, 3.12 e 3.13.

## Limitazioni note

- Universo di 15 titoli large cap scelti oggi: survivorship bias.
- Pochi regimi di mercato indipendenti (2018 → 2026).
- Esecuzione modellata su barre giornaliere: niente spread dinamico,
  impatto di mercato o fill parziali.
- Il registro delle prove parte da novembre 2026: le varianti precedenti
  (V2–V5) non sono conteggiate nel Deflated Sharpe Ratio.
