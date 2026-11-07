# Investment Policy Statement (IPS) — Project Alpha

Documento scritto **prima** di guardare i risultati. Fissa obiettivi e
limiti; il codice li fa rispettare (`RiskPolicyV2`, `run_policy_portfolio`).
Si modifica solo con una decisione esplicita e datata, mai durante un
drawdown per "recuperare".

Valori decisi dal proprietario il 2026-11-07 (capitale circa 10.000 €,
perdita massima 15%, profilo intermedio tra crescita e tranquillità).
Configurazione eseguibile: `configs/paper_core_v1.toml`.

## 1. Obiettivo

- Far crescere il capitale dedicato nel lungo periodo con cadute contenute.
- Metro di giudizio: battere un portafoglio passivo semplice a parità di
  rischio (volatilità o drawdown), al netto di costi e tasse.
- Non è un obiettivo: guadagni continui o garantiti.

## 2. Capitale e orizzonte

| Voce | Valore |
|---|---|
| Capitale dedicato (budget massimo gestito dal sistema) | 10.000 € (`budget`) |
| Versamenti/prelievi | manuali, mai automatici |
| Orizzonte minimo di valutazione | 3 anni |
| Perdita massima accettabile del budget | 15% dal massimo |
| Profilo | intermedio: né il più rischioso né il più prudente |

## 3. Universo e strumenti

- Ammessi: ETF e azioni quotate, solo long, senza leva.
- Esclusi: derivati, short, margine, criptovalute, titoli illiquidi.
- Lista dei simboli ammessi: `allowed_symbols` della policy di rischio.
- Verificare l'operatività del broker e la fiscalità nel paese di residenza
  (es. limiti UE sugli ETF USA: possono richiedere ETF UCITS).

## 4. Allocazione di policy (decisa a priori)

| Sleeve | Peso | Note |
|---|---|---|
| Core bilanciato — azioni | 60% | SPY 20, QQQ 10, IWM 5, EFA 15, EEM 10 |
| Core bilanciato — obbligazioni | 30% | IEF 15, LQD 10, TLT 5 |
| Core bilanciato — attivi reali | 10% | GLD 5, VNQ 5 |
| Overlay di trend/difensivo | 0% | V1 non ha superato la Fase 1 |
| Cash | resto | quota lasciata dal volatility targeting |

Volatility targeting del portafoglio: 8% annuo, finestra 63 giorni,
esposizione massima 100% (mai leva). Ribilanciamento solo fuori banda:
un peso scostato di almeno 3 punti o l'esposizione totale di almeno 5.

Per il paper trading si usano ETF USA (Alpaca). Per il conto reale in
Italia servono gli equivalenti UCITS (regole PRIIPs), da scegliere prima
del micro-live.

## 5. Limiti di rischio (Risk Engine V2)

| Limite | Default prudente | Valore scelto |
|---|---|---|
| Peso massimo per posizione | 20% | 20% |
| Esposizione lorda massima | 100% | 100% |
| Peso massimo per gruppo (settore/cluster) | 40% | 65% (gruppi = asset class) |
| Perdita giornaliera → modalità solo-riduzione | −3% | −4% |
| Drawdown → modalità solo-riduzione | −15% | −10% |
| Drawdown → tutto in cash e stop fino allo sblocco manuale | — | −12% |
| Turnover massimo per ribilanciamento | 50% | 100% (serve per costruire il portafoglio da cash) |
| Notional massimo per ordine | — | — |
| Età massima dei dati | 4 giorni | 4 giorni |

Il limite del 15% non si può garantire (un'apertura in forte gap può
superarlo): per questo il sistema smette di aumentare il rischio a −10% ed
esce a −12%, lasciando 3 punti di margine.

## 6. Gradini di autonomia

1. Ricerca e registro forward dei segnali (nessun ordine).
2. Paper trading: almeno 3–6 mesi, scostamento contenuto dal backtest.
3. Micro-live: 1% del budget → 5% → 25% → 100%, un gradino alla volta,
   solo se il gradino precedente rispetta i criteri.
4. AI: prima sola lettura, poi proposte vincolate (vedi `ROADMAP.md`).

## 7. Criteri di stop

- Drawdown oltre il limite: solo-riduzione automatica, revisione manuale.
- Dopo 12 mesi di registro forward e la Fase 1, se nessun overlay migliora
  il portafoglio a parità di rischio: budget su passivo semplice.

## 8. Registro delle modifiche

| Data | Modifica | Motivo |
|---|---|---|
| | Prima versione | |
| 2026-11-07 | Valori del proprietario: 10.000 €, max −15%, profilo intermedio; core bilanciato 60/30/10 con vol target 8%, stop a −12% | V1 non ha superato la Fase 1 (overlay a 0%); serve una base passiva per il paper trading |

## 9. Verifica storica (una sola prova, non un'ottimizzazione)

Configurazione definita sopra, verificata una volta sugli 11 ETF
2016 → 2026 (`reports/policy/core_bilanciato_v1/historical_check.json`):

| Variante | CAGR | Volatilità | Sharpe | Drawdown massimo |
|---|---|---|---|---|
| Core al 100% (senza vol target) | 9,3% | 11,9% | 0,81 | −25,0% |
| Core con vol target 8% | 5,9% | 8,1% | 0,75 | −16,3% |
| Core con vol target e stop a −12% | — | — | — | −12,4% (stop il 2020-03-12) |

Lettura: senza protezioni il core avrebbe superato il limite del 15%
(2020 e 2022). Con vol target resta poco sopra; lo stop lo tiene sotto ma
in quel caso il sistema si sarebbe fermato a marzo 2020 in attesa di una
decisione manuale. Sono numeri passati, non una promessa.
