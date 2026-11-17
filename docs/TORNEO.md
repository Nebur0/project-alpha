# Torneo delle strategie sugli ETF

## Pre-registrazione (2026-11-17, scritta prima di vedere i risultati)

**Domanda.** Esiste una strategia nota e documentata che, sugli stessi ETF
del paper trading, batta in modo solido la strategia attuale (core
bilanciato con vol target 8%)?

**Perché le regole prima.** Provando molte strategie, una che sembra ottima
sul passato si trova sempre per caso. Regole, parametri e criteri sono
fissati qui e non si cambiano dopo aver visto i numeri. I parametri sono
quelli "da manuale" della letteratura, non ottimizzati.

### Dati ed esecuzione (uguali per tutti)

- Universo: i 10 ETF del core (SPY QQQ IWM EFA EEM IEF LQD TLT GLD VNQ),
  chiusure aggiustate Alpaca, `data/raw/bars_etf.csv` (2016 → 2026).
- Decisioni **una volta al mese**, sull'ultima chiusura del mese, con i soli
  dati disponibili fino a quel giorno.
- I nuovi pesi entrano in vigore alla chiusura del giorno di borsa
  successivo (ritardo di un giorno); tra un ribilanciamento e l'altro i pesi
  derivano con i prezzi.
- Costi: 10 punti base per lato sul controvalore scambiato.
- Liquidità: rendimento 0% (prudente). Sensibilità: liquidità al 2% annuo.
- Periodo di valutazione comune: dal primo giorno in cui tutte le strategie
  hanno 12 mesi di storia (inizio 2017) alla fine dei dati.
- Long-only, nessuna leva, somma dei pesi ≤ 100%.

### Riferimento

**R. Attuale** — pesi del core (azioni 60%, obbligazioni 30%, reali 10%)
scalati dal vol target 8% (volatilità del core a 63 giorni, esposizione
massima 100%). Nel torneo è ribilanciato mensilmente (il paper trading
usa bande giornaliere).

### Candidati (8 prove)

| | Strategia | Regola (parametri fissi) | Fonte |
|---|---|---|---|
| C1 | Core al 100% | pesi del core, senza vol target | portafoglio di policy |
| C2 | Pesi uguali (1/N) | 10% per ETF | DeMiguel, Garlappi, Uppal (2009) |
| C3 | Momentum assoluto 12 mesi | ogni ETF al peso del core se il rendimento a 12 mesi > 0, altrimenti liquidità | Moskowitz, Ooi, Pedersen (2012) |
| C4 | Media mobile 10 mesi | ogni ETF al peso del core se la chiusura mensile > media delle ultime 10 chiusure mensili, altrimenti liquidità | Faber (2007) |
| C5 | Momentum relativo top 3 | i 3 ETF con il miglior rendimento a 12 mesi, 1/3 ciascuno | Jegadeesh, Titman (1993); Faber (2010) |
| C6 | Dual momentum | se SPY a 12 mesi > 0: il migliore tra SPY ed EFA a 12 mesi, al 100%; altrimenti IEF al 100% | Antonacci (2014) |
| C7 | Inverso della volatilità | pesi ∝ 1 / volatilità a 63 giorni | risk parity "naïve" |
| C8 | Portafoglio permanente | SPY 25%, TLT 25%, GLD 25%, liquidità 25% | Browne (1987) |

### Criteri (decisi ora)

Un candidato **batte il riferimento** solo se soddisfa **tutti** i punti:

1. **Sharpe più alto** del riferimento sull'intero periodo, con p ≤ 0,05/8
   (correzione di Bonferroni per 8 prove) in un bootstrap a blocchi di 21
   giorni (5.000 ricampionamenti) sui rendimenti giornalieri appaiati.
2. Sharpe più alto del riferimento in **almeno 2 dei 3 sottoperiodi**
   (2017–2019, 2020–2022, 2023–2026).
3. **Drawdown massimo non peggiore** del riferimento.
4. Compatibile con i limiti dell'IPS (peso massimo 20% per posizione) oppure
   adattabile senza cambiare la regola.

Un candidato è **promettente** se soddisfa 2 e 3 e ha Sharpe più alto
sull'intero periodo ma non supera la soglia statistica 1: può entrare in
paper trading **in parallelo** (con capitale virtuale separato) per la
verifica in avanti, che è il test vero. Tutti gli altri sono scartati.

Si riporta anche il Deflated Sharpe Ratio con 9 prove (riferimento + 8).

---

## Risultati (2026-11-17, dopo la pre-registrazione)

Periodo: 2017-02-01 → 2026-11-06 (9 anni e 9 mesi). Comando:
`project-alpha strategy-lab` (report in `reports/lab/etf_torneo_v1/`,
sensibilità con liquidità al 2% in `reports/lab/etf_torneo_v1_cash2/`).

| Strategia | Rend. annuo | Volatilità | Sharpe | Max DD | Sottoperiodi vinti | p | Esito |
|---|---|---|---|---|---|---|---|
| R. Attuale (vol target 8%) | 5,7% | 9,1% | 0,66 | −20,2% | — | — | riferimento |
| C1 Core al 100% | 9,0% | 12,0% | 0,77 | −25,2% | 3/3 | 0,15 | scartata (drawdown) |
| C2 Pesi uguali | 8,5% | 11,7% | 0,76 | −25,8% | 3/3 | 0,23 | scartata (drawdown) |
| C3 Momentum assoluto 12m | 6,3% | 9,1% | 0,71 | −18,5% | 1/3 | 0,36 | scartata |
| C4 Media mobile 10 mesi | 6,1% | 7,7% | 0,81 | −14,0% | 1/3 | 0,27 | scartata |
| C5 Momentum relativo top 3 | 8,8% | 15,2% | 0,63 | −31,2% | 2/3 | 0,59 | scartata |
| C6 Dual momentum | 8,3% | 16,6% | 0,56 | −33,8% | 1/3 | 0,73 | scartata |
| C7 Inverso della volatilità | 6,4% | 9,2% | 0,72 | −24,3% | 3/3 | 0,36 | scartata (drawdown) |
| **C8 Portafoglio permanente** | 6,9% | 7,6% | **0,92** | −17,7% | **3/3** | 0,17 | **promettente** |

Soglia statistica: p ≤ 0,00625 (0,05/8). Con la liquidità al 2% le
conclusioni non cambiano (C8 Sharpe 0,98, p 0,15; tutti gli altri esiti
uguali).

### Lettura

1. **Nessuna strategia batte il riferimento in modo statisticamente
   solido.** Con meno di 10 anni di dati, differenze di Sharpe di questa
   ampiezza non si distinguono dal caso.
2. **Le strategie di timing e rotazione più famose non aiutano su questi
   ETF:** momentum relativo e dual momentum hanno rendimenti simili ma
   rischio e cadute molto maggiori (fino a −34%); momentum assoluto e media
   mobile riducono le cadute ma vincono in un solo sottoperiodo su tre.
3. **C8, portafoglio permanente** (azioni USA, Treasury lunghi, oro e
   liquidità, 25% ciascuno), è l'unico **promettente**: Sharpe più alto in
   tutti e tre i sottoperiodi e cadute minori del riferimento.
   Non compatibile così com'è con l'IPS: peso del 25% per posizione contro
   il limite del 20% (criterio 4), e la caduta massima storica (−17,7%)
   supera il limite del 15%, gestito però dallo stop a −12%.
4. **Il vol target attuale costa rendimento:** il core al 100% ha Sharpe più
   alto del riferimento in tutti i sottoperiodi; il vol target serve solo a
   ridurre le cadute (−20% contro −25%), coerente con la Fase 1.

### Decisioni proposte (al proprietario)

- **Nessun cambio al portafoglio in paper trading** sulla base del passato.
- **C8 in paper trading in parallelo**, con capitale virtuale separato, per
  la verifica in avanti; per rispettare l'IPS serve una decisione esplicita
  (eccezione al limite del 20% per C8, oppure non avviarlo).
- Nuove idee = nuove prove registrate qui, con la stessa procedura.
