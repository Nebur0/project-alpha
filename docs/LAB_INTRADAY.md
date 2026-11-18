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

---

## Risultati (2026-11-18)

Dati: 218.444 candele da 15 minuti di SPY e QQQ (Alpaca, dal 2020-01-01);
giudizio su 1.225 giorni di borsa dal 2022-01-01. Comandi:
`project-alpha fetch-intraday` e `project-alpha intraday-lab`
(report in `reports/lab/intraday_v1/`). Nessuna regola modificata.

| Simbolo | Regola | Operazioni/giorno | Vinte | Media/giorno | € al giorno ogni 1.000 € per operazione | t | p | Anni positivi | Con costi doppi | Esito |
|---|---|---|---|---|---|---|---|---|---|---|
| SPY | L1 momentum 1 candela | 12,9 | 44,7% | −24,4 bp | −2,44 | −9,8 | 1,00 | 0/4 | −50,2 bp | scartata |
| SPY | L2 inversione 1 candela | 12,9 | 43,6% | −27,3 bp | −2,73 | −11,0 | 1,00 | 0/4 | −53,1 bp | scartata |
| SPY | L3 engulfing | 3,5 | 44,0% | −7,0 bp | −0,70 | −6,7 | 1,00 | 0/4 | −13,9 bp | scartata |
| SPY | L4 rottura apertura | 1,0 | 51,6% | +0,7 bp | +0,07 | 0,3 | 0,38 | 2/4 | −1,3 bp | scartata |
| SPY | L5 logistica | 3,9 | 48,3% | −5,1 bp | −0,51 | −3,6 | 1,00 | 0/4 | −12,9 bp | scartata |
| QQQ | L1 momentum 1 candela | 12,9 | 45,8% | −23,4 bp | −2,34 | −7,4 | 1,00 | 0/4 | −49,2 bp | scartata |
| QQQ | L2 inversione 1 candela | 12,9 | 45,1% | −28,2 bp | −2,82 | −8,9 | 1,00 | 0/4 | −54,0 bp | scartata |
| QQQ | L3 engulfing | 3,3 | 45,2% | −6,6 bp | −0,66 | −4,7 | 1,00 | 0/4 | −13,1 bp | scartata |
| QQQ | L4 rottura apertura | 1,0 | 53,7% | +2,5 bp | +0,25 | 1,1 | 0,14 | 3/4 | +0,6 bp | scartata |
| QQQ | L5 logistica | 3,6 | 48,7% | −5,6 bp | −0,56 | −2,7 | 1,00 | 0/4 | −12,9 bp | scartata |

Soglia: p ≤ 0,005. La colonna "totale_periodo_per_1000" del CSV è la somma
semplice dei risultati giornalieri (senza capitalizzazione), utile solo come
ordine di grandezza.

### Lettura

1. **Nessuna regola funziona.** Tutte e 10 le prove sono scartate.
2. **Momentum e inversione perdono entrambe ~25 bp al giorno**: quasi
   esattamente il costo delle loro ~13 operazioni (13 × 2 bp). Prima dei
   costi la direzione della candela successiva è indovinata come lanciando
   una moneta; dopo i costi la perdita è sistematica.
3. **Il modello statistico (L5)**, addestrato e riaddestrato ogni mese,
   indovina il 48% delle operazioni: meno del caso, e perde in tutti gli
   anni.
4. **La rottura del range d'apertura (L4)** è l'unica vicina allo zero
   (+0,25 € al giorno ogni 1.000 € su QQQ), ma non si distingue dal caso
   (p 0,14) e con costi doppi sparisce.
5. Conseguenza: **niente salvadanaio intraday**. Prove registrate: 10.
