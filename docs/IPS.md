# Investment Policy Statement (IPS) — Project Alpha

Documento scritto **prima** di guardare i risultati. Fissa obiettivi e
limiti; il codice li fa rispettare (`RiskPolicyV2`, `run_policy_portfolio`).
Si modifica solo con una decisione esplicita e datata, mai durante un
drawdown per "recuperare".

Campi marcati **[DA DECIDERE]**: li compila il proprietario.

## 1. Obiettivo

- Far crescere il capitale dedicato nel lungo periodo con cadute contenute.
- Metro di giudizio: battere un portafoglio passivo semplice a parità di
  rischio (volatilità o drawdown), al netto di costi e tasse.
- Non è un obiettivo: guadagni continui o garantiti.

## 2. Capitale e orizzonte

| Voce | Valore |
|---|---|
| Capitale dedicato (budget massimo gestito dal sistema) | **[DA DECIDERE]** es. 10.000 € |
| Versamenti/prelievi | manuali, mai automatici |
| Orizzonte minimo di valutazione | 3 anni |
| Perdita massima accettabile del budget | **[DA DECIDERE]** es. 15% dal massimo |

## 3. Universo e strumenti

- Ammessi: ETF e azioni quotate, solo long, senza leva.
- Esclusi: derivati, short, margine, criptovalute, titoli illiquidi.
- Lista dei simboli ammessi: `allowed_symbols` della policy di rischio.
- Verificare l'operatività del broker e la fiscalità nel paese di residenza
  (es. limiti UE sugli ETF USA: possono richiedere ETF UCITS).

## 4. Allocazione di policy (decisa a priori)

| Sleeve | Peso | Note |
|---|---|---|
| Core passivo diversificato | **[DA DECIDERE]** | es. ETF azionario globale + obbligazionario |
| Overlay di trend/difensivo | **[DA DECIDERE]** | solo se supera la Fase 1 |
| Cash | resto | |

Volatility targeting del portafoglio: **[DA DECIDERE]** (es. 10% annuo,
finestra 63 giorni, esposizione massima 100%).

## 5. Limiti di rischio (Risk Engine V2)

| Limite | Default prudente | Valore scelto |
|---|---|---|
| Peso massimo per posizione | 20% | |
| Esposizione lorda massima | 100% | |
| Peso massimo per gruppo (settore/cluster) | 40% | |
| Perdita giornaliera → modalità solo-riduzione | −3% | |
| Drawdown → modalità solo-riduzione | −15% | |
| Turnover massimo per ribilanciamento | 50% | |
| Notional massimo per ordine | — | |
| Età massima dei dati | 4 giorni | |

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
