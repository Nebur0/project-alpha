# Laboratorio intraday: si può prevedere la direzione delle prossime candele?

## Pre-registrazione (2026-11-18, scritta prima di scaricare i dati)

**Idea del proprietario.** Analizzare le candele e prevedere la direzione
delle prossime 2–3, con piccoli investimenti giornalieri ("10 € al giorno")
che nel complesso diano un risultato positivo.

**Regole e criteri sono fissati qui e non si cambiano dopo aver visto i
numeri.** Una variante è una nuova prova, da registrare.

### Dati

- Candele da **15 minuti** di **SPY** e **QQQ** (gli ETF più scambiati al
  mondo: spread minimi, il caso più favorevole), da Alpaca, aggiustate per
  dividendi e split, dal 2020-01-01 all'ultimo dato disponibile.
- Solo seduta regolare (9:30–16:00 di New York), 26 candele al giorno.

### Esecuzione simulata (uguale per tutte le regole)

- Il segnale si calcola alla **chiusura** di una candela; si entra al prezzo
  di chiusura di quella candela e si esce alla chiusura di **2 candele dopo**
  (30 minuti), oppure prima, alla chiusura dell'ultima candela del giorno:
  **nessuna posizione overnight**.
- Long o short (si scommette sulla direzione), una posizione alla volta per
  simbolo, importo fisso per operazione.
- **Costi: 1 punto base per lato** (spread + slittamento; Alpaca non ha
  commissioni). Sensibilità: 0,5 e 2 punti base.

### Regole (5 prove × 2 simboli = 10 prove)

| | Regola | Segnale |
|---|---|---|
| L1 | Momentum di 1 candela | candela verde → long; rossa → short |
| L2 | Inversione di 1 candela | candela verde → short; rossa → long |
| L3 | Engulfing | corpo che "ingloba" il corpo precedente di colore opposto: rialzista → long, ribassista → short |
| L4 | Rottura del range d'apertura | dopo le prime 2 candele (30 minuti): prima chiusura sopra il massimo del range → long fino a fine giornata; sotto il minimo → short (una sola operazione al giorno) |
| L5 | Modello statistico (regressione logistica) | caratteristiche delle ultime 8 candele (rendimenti, ampiezze, volume relativo, orario); prevede il segno del rendimento delle 2 candele successive; addestrato su **12 mesi** di dati precedenti, riaddestrato ogni mese; long se probabilità > 0,55, short se < 0,45 |

### Periodo di giudizio

- **Fuori campione: dal 2022-01-01** alla fine dei dati (L5 usa il 2021 come
  primo anno di addestramento; per tutte le regole conta solo il periodo
  dal 2022).

### Criteri (tutti necessari per "funziona")

1. **Guadagno medio giornaliero netto > 0**, con test t (errori robusti,
   Newey-West 5 giorni) e **p ≤ 0,005** (0,05/10 prove).
2. Positivo in **almeno 3 anni su 4** (2022, 2023, 2024, 2025–2026).
3. Positivo anche con **costi di 2 punti base per lato**.

Una regola che passa entra nel **salvadanaio** in paper trading (10 € virtuali
al giorno di rischio, conto interno separato) per la verifica in avanti.
Nessun uso di denaro reale senza mesi di verifica e una decisione esplicita
del proprietario.
