# Project Alpha — Roadmap

## Visione

Un'app che in futuro gestisce in autonomia un capitale (es. 10.000 €),
comprando e vendendo azioni ed ETF, analizzando il mercato in modo continuo
e facendo crescere il capitale nel lungo periodo **dentro limiti di rischio
fissati dal proprietario**. A regime, un livello di AI addestrata/agentica
potrà proporre e poi decidere allocazioni, sempre filtrate da un Risk Engine
deterministico che l'AI non può modificare né aggirare.

Principi:

- **Crescita nel lungo periodo con cadute contenute**, non guadagni continui
  o garantiti. Metro di giudizio: battere un portafoglio passivo semplice
  (ETF) a parità di rischio, dopo costi e tasse.
- **La fiducia si guadagna per gradini.** Ogni livello di autonomia si
  sblocca solo quando il livello precedente ha superato criteri decisi prima
  di vedere i risultati.
- **Prima la verità, poi l'infrastruttura.** Backend, dashboard e AI si
  costruiscono sopra regole che hanno dimostrato di funzionare.

## Dove siamo (novembre 2026)

- Ricerca su V1 (trend/breakout long-only) completa su DEV15; risultati
  riproducibili.
- Valutazione a pari esposizione: V1 fa circa +1,3% annuo rispetto a tenere
  il 21% del paniere e il resto in cash; il vantaggio **non è distinguibile
  dal caso** (permutazioni p ≈ 0,18; alfa CAPM t ≈ 1,3). La protezione nei
  ribassi viene soprattutto dall'esposizione bassa.

## Fase 0 — Fondamenta affidabili *(in corso)*

| # | Blocco | Stato |
|---|---|---|
| 1 | Correttezza del core: JSON sicuro, scritture atomiche, griglie e finestre WF tolleranti | ✅ PR #16 |
| 2 | Realismo dell'esecuzione: next open, ritardo, cash, turnover, durata posizioni | ✅ PR #17 |
| 3 | Modulo di valutazione: pari esposizione/beta, decomposizione timing/selezione, permutazioni, PSR/DSR | ✅ PR #18 |
| 4 | Pipeline riproducibile: un comando + config, manifest (hash dati e codice), registro prove | ✅ PR #19 |
| 5 | Pulizia: ruff in CI, formattazione, README aggiornato (merge su `main` dal proprietario) | ✅ PR #20 |

**Uscita:** ogni risultato si rigenera con un comando ed è accompagnato dalla
valutazione standard.

**Fasi successive, stato (novembre 2026):**

- Fase 1: eseguita il 2026-11-07 (run `26cee9214d95`, report in
  `reports/research/etf_multiasset_v1/`). Esito: V1 non supera il test
  (vedi sotto); decisione del proprietario in attesa.
- Fase 2: IPS compilato dal proprietario il 2026-11-07 (10.000 €, max −15%,
  profilo intermedio): core bilanciato 60/30/10 con vol target 8% e stop a
  −12% (`configs/paper_core_v1.toml`).
- Fase 3: registro forward dei segnali (PR #22) e paper trading del core su
  Alpaca paper, autorizzato dal proprietario il 2026-11-07
  (`scripts/daily_paper.sh`, registro `reports/paper/paper_log.csv`).

## Fase 1 — La verità su V1

V1 congelata, rivalutata con esecuzione realistica, su più storia possibile
e su ETF multi-asset.

**Decisione:** V1 resta come overlay difensivo oppure viene archiviata.
In entrambi i casi la piattaforma resta e diventa la base per altri motori.

**Risultati (11 ETF, 2016 → 2026, esecuzione next_open, 6 finestre OOS):**

| Portafoglio | CAGR | Sharpe |
|---|---|---|
| V1 (esposizione media 20,7%) | 0,52% | 0,18 |
| Statico alla stessa esposizione | 1,88% | 0,71 |
| Passivo equal-weight 100% | 8,68% | 0,71 (MaxDD −22,7%) |
| Vol target 10% | 5,6% | 0,60 |

- Alfa CAPM −0,33% (t −0,28); test di permutazione del timing p = 0,94.
- V1 peggio dello statico in 4 finestre su 6; nel 2022 solo +0,4 punti.
- Con esecuzione a close −0,13%, con un giorno di ritardo 0,09%.
- Il vantaggio del vol targeting visto su DEV15 (Sharpe 1,04) non si
  ripete sugli ETF (0,60, sotto il passivo): non è robusto.

## Fase 2 — Portafoglio di policy

- Investment Policy Statement scritto: obiettivo, perdita massima
  tollerata, orizzonte, vincoli, tasse.
- Core passivo multi-asset + overlay di trend, pesi decisi a priori,
  volatility targeting.
- Risk Gate V2: modalità solo-riduzione in drawdown, divieto di short,
  limiti per cluster/settore, turnover, dati stantii, notional per ordine.

## Fase 3 — Operatività senza soldi

- Registro forward giornaliero dei segnali (con timestamp, senza ordini).
- Paper trading su Alpaca: gestione ordini idempotente, riconciliazione con
  il broker, kill switch, alert. Durata minima 3–6 mesi.
- **Criterio:** scostamento contenuto tra paper e backtest, nessun errore
  operativo grave.

## Fase 4 — Micro-live a gradini

Capitale minimo, chiavi con permessi minimi, niente leva/short/opzioni,
stop automatici. Aumento a gradini (es. 1% → 5% → 25% del budget) solo se
ogni gradino rispetta i criteri.

## Fase 5 — AI e autonomia

Autonomia crescente, sempre dietro il Risk Engine deterministico:

1. **Copilot** (sola lettura): analisi, report, spiegazione di anomalie.
2. **Proponente:** proposte strutturate (JSON con schema) che passano da
   test, backtest e approvazione.
3. **Autonomo in paper trading:** decide allocazioni entro limiti, con
   budget di scostamento dal portafoglio di policy.
4. **Autonomo con capitale piccolo e supervisionato.**
5. **Autonomia limitata e controllata** sul budget dedicato.

Regole per l'AI: può ridurre il rischio liberamente, aumentarlo solo entro
vincoli; non tocca limiti, chiavi o codice di produzione; nessun accesso
diretto al broker. Le decisioni di un LLM si validano **solo in avanti**
(i backtest su dati passati sono contaminati da ciò che il modello ha già
visto in addestramento).

Backend (FastAPI + PostgreSQL), monitoraggio e dashboard (Next.js) si
costruiscono in questa fase, quando servono davvero.

## Criterio di stop

Se dopo la Fase 1 e 12 mesi di registro forward nessun overlay migliora il
portafoglio a parità di rischio, il budget va su un portafoglio passivo
semplice e Project Alpha resta un laboratorio di ricerca.
