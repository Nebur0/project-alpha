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
