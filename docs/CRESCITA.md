# Portafoglio di crescita — prova in avanti (`crescita_v1`)

Registrato il 2026-11-23, **prima** di qualsiasi risultato in avanti.
Configurazione: `configs/paper_crescita_v1.toml`. Non si modifica guardando
i risultati; un'idea nuova diventa una variante separata.

## Perché

Il proprietario ha chiesto di dare priorità alla crescita del capitale e ha
dichiarato di accettare più rischio ("non mi importa molto del rischio, a me
importerebbe vedere i soldi salire bene"), lasciando a Claude la scelta del
livello. Il core resta invariato; questa è una prova in parallelo con
capitale virtuale separato (conto interno, 10.000).

## Regole

- 80% azioni, 15% obbligazioni, 5% attivi reali, con le stesse proporzioni
  interne del core (SPY 26,67%, QQQ 13,33%, IWM 6,67%, EFA 20%, EEM 13,33%,
  IEF 7,5%, LQD 5%, TLT 2,5%, GLD 2,5%, VNQ 2,5%).
- Nessun freno di volatilità, nessuna leva, solo posizioni lunghe.
- Ribilanciamento con le stesse bande del core (3 punti per posizione,
  5 punti di esposizione).
- Protezioni: −25% dal massimo → solo riduzioni; −30% → tutto in cash e stop
  fino allo sblocco manuale; perdita massima dichiarata per questa prova 35%.
- Esecuzione: conto interno, eseguito all'apertura della seduta successiva,
  costo 5 punti base.

## Verifica storica descrittiva (2017 → 2026-11-20)

Sei varianti guardate **insieme**, una volta, il 2026-11-23, sui prezzi di
`bars_core.csv` (prezzi senza dividendi: i rendimenti sono sottostimati,
il confronto tra varianti regge). Ribilanciamento mensile, senza costi.

| Variante | CAGR | Volatilità | Drawdown max | Anno peggiore |
|---|---|---|---|---|
| Core attuale 60/30/10, vol target 8% | 5,9% | 9,3% | −20,3% | −12,0% (2022) |
| 60/30/10 senza vol target | 8,8% | 12,2% | −25,0% | −18,6% (2022) |
| 80/15/5, vol target 12% | 7,7% | 12,9% | −27,0% | −14,0% (2022) |
| **80/15/5 senza vol target (scelta)** | **10,4%** | **15,0%** | **−27,6%** | **−19,3% (2022)** |
| 100% azioni | 11,9% | 18,0% | −32,7% | −20,1% (2022) |
| 100% SPY+QQQ | 16,4% | 20,1% | −30,9% | −25,6% (2022) |

Avvertenza: la scelta è stata fatta **dopo** aver visto questa tabella, quindi
i numeri storici della variante scelta sono ottimistici per costruzione.
Il giudizio vale solo sulla prova in avanti. Il 100% SPY+QQQ è escluso
apposta: è il migliore solo perché oggi sappiamo che la tecnologia americana
ha vinto il decennio (concentrazione e senno di poi). Prove guardate: 6.

## Criteri di giudizio (dopo almeno 6 mesi)

- La prova "funziona" se resta sotto lo stop e il rendimento in avanti è
  coerente con un portafoglio 80/20 (scostamento dal modello contenuto).
- Non si confronta con il core su poche settimane: il rischio più alto vuol
  dire anche oscillazioni più ampie, in entrambe le direzioni.
- Nessun capitale reale senza una decisione esplicita e separata del
  proprietario.
