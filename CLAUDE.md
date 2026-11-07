# Project Alpha — regole per Claude

Piattaforma personale di ricerca quantitativa e controllo del rischio
(Python, `src/project_alpha`). Obiettivo: rendimento corretto per il rischio
sotto vincoli deterministici. Non è un generatore di "alpha garantito".

## Regole non negoziabili

- **Mai leggere `.env`** né stampare chiavi o segreti.
- **Validation set bloccato.** Non leggere, staggare o usare
  `data/raw/bars_validation.csv` e `data/raw/bars_independent_validation.csv`
  (META, TSLA, AVGO, UNH, PG, WMT, DIS, BA, GS, CVX). Solo il proprietario
  decide se e quando usarli, una volta, su una versione congelata.
- **V1 è congelata.** Non modificare parametri, griglia o score di V1
  guardando i risultati. Nuove idee = nuove varianti separate.
- **Nessun ordine, nessun broker, nessun capitale reale.** Nessun codice che
  invii ordini (nemmeno paper) senza richiesta esplicita del proprietario.
- **Default invariati.** Ogni novità che cambia i numeri va dietro un
  parametro opzionale; il comportamento di default deve riprodurre i report
  esistenti (DEV15 V1: 10.000 → 12.287,34, Sharpe 0,979, esposizione 20,82%).

## Flusso di lavoro

- Un branch per blocco (`feature/<nome>`), PR verso il branch di
  integrazione corrente; mai commit diretti su `main`. Il merge lo fa il
  proprietario.
- Prima di ogni commit: `python -m pytest -q` verde.
- Ogni nuova funzionalità ha test in `tests/`.
- Messaggi, docstring ed errori in italiano, come il resto del codice.
- Esperimenti e prove: annotarli (cosa, perché, quante combinazioni) in un
  registro, così il numero di prove resta tracciabile per i test di
  significatività.

## Dati

- Sviluppo: `data/raw/bars_dev15.csv` (15 simboli, 2018 → 2026).
- I dati non sono versionati (`.gitignore`); i report in `reports/` sì.

## Comandi utili

```bash
python -m pytest -q
python -m project_alpha --help
```
