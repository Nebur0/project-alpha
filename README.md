[![Tests](https://github.com/Nebur0/project-alpha/actions/workflows/tests.yml/badge.svg)](https://github.com/Nebur0/project-alpha/actions/workflows/tests.yml)

# Project Alpha — Starter MVP

Primo nucleo verificabile di un assistente quantitativo per il trading.

## Obiettivo del primo sprint

1. Scaricare candele giornaliere OHLCV da Alpaca Market Data.
2. Calcolare EMA 20/50/200, RSI 14, ATR 14 e volume medio 20.
3. Generare una baseline long-only trasparente.
4. Eseguire un backtest applicando i segnali dal giorno successivo, con costi configurabili.
5. Salvare dati e risultati in CSV/JSON.

> Questo progetto è destinato inizialmente a ricerca e paper trading. Non esegue ordini reali.

## Requisiti

- Python 3.11 o superiore
- Un account Alpaca con API key per i dati azionari
- Git Bash, PowerShell o terminale equivalente

## Installazione su Windows con Git Bash

```bash
cd project-alpha-starter
py -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env
```

Apri `.env` e inserisci le credenziali. Non pubblicare mai il file `.env`.

## Installazione con PowerShell

```powershell
cd project-alpha-starter
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

## Utilizzo

Scarica i dati giornalieri:

```bash
project-alpha fetch --symbols SPY QQQ AAPL MSFT NVDA --start 2018-01-01
```

Calcola indicatori e segnali:

```bash
project-alpha analyze --input data/raw/bars.csv
```

Esegue il backtest su un simbolo:

```bash
project-alpha backtest \
  --input data/processed/features.csv \
  --symbol SPY \
  --initial-capital 10000 \
  --cost-per-side 0.001
```

Oppure esegui l'intera pipeline:

```bash
project-alpha run \
  --symbols SPY QQQ AAPL MSFT NVDA \
  --start 2018-01-01 \
  --initial-capital 10000 \
  --cost-per-side 0.001
```

## Strategia baseline

Ingresso long quando, alla chiusura:

- prezzo sopra EMA 200;
- EMA 20 sopra EMA 50;
- RSI tra 50 e 70;
- chiusura sopra il massimo dei 20 giorni precedenti;
- volume sopra la media a 20 giorni.

Uscita quando:

- chiusura sotto EMA 20; oppure
- EMA 20 sotto EMA 50.

Il segnale calcolato alla chiusura viene applicato dalla seduta successiva. Questo evita l'uso involontario di informazioni future.

## Risultati

- `data/raw/bars.csv`
- `data/processed/features.csv`
- `reports/<SIMBOLO>_equity.csv`
- `reports/<SIMBOLO>_metrics.json`

## Test

```bash
pytest
```

## Limitazioni note

- La baseline non è una raccomandazione finanziaria.
- Il backtest non modella ancora spread variabile, liquidità, gap di esecuzione, tasse o sizing ATR.
- Il dataset deve essere controllato per split, dividendi, simboli delistati e survivorship bias.
- Un buon risultato storico non garantisce risultati futuri.
