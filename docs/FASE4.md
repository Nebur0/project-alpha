# Fase 4 — Preparazione: broker, ETF UCITS, fiscalità

Documento di preparazione (novembre 2026). **Nessun collegamento al conto
reale è implementato**: l'adattatore verso un broker reale si scrive solo su
richiesta esplicita del proprietario (vedi `CLAUDE.md`). Le informazioni su
broker e fiscalità vanno riverificate al momento della scelta: offerte e
regole cambiano. Non è consulenza finanziaria né fiscale.

## 1. Decisioni da prendere (proprietario)

| # | Decisione | Opzioni | Dipende da |
|---|---|---|---|
| D1 | Broker | BG Saxo · Directa · Interactive Brokers | §2 |
| D2 | Regime fiscale | amministrato (il broker paga le tasse) · dichiarativo (le dichiari tu) | D1 |
| D3 | Obbligazioni in dollari o in euro | come nel paper (USD) · titoli di Stato in euro / coperti dal cambio | §5 |
| D4 | Numero di posizioni | 10 come nel paper · 5–6 per ridurre le commissioni | §4 |
| D5 | Primo gradino di capitale | 1% (100 €) non sostenibile con commissioni fisse · proposto 1.000 € | §6 |

## 2. Broker con API utilizzabili dall'Italia

Requisito minimo: API per inviare ordini da programma. Fineco, Trade
Republic e Scalable Capital sono sostituti d'imposta ma senza API pubbliche
di trading: esclusi per il sistema automatico.

| | **BG Saxo** | **Directa SIM** | **Interactive Brokers** |
|---|---|---|---|
| Regime fiscale | amministrato di default (dichiarativo su richiesta) | solo amministrato | dichiarativo |
| API | OpenAPI (REST) | socket TCP locali tramite la piattaforma Darwin | TWS API e Web API |
| Ambiente di prova | sì: simulazione con conto da 100.000 | **no**: si sviluppa sul conto reale | sì: conto paper con utente dedicato |
| Requisiti API | chiave applicazione; l'accesso live richiede autorizzazione e test (da verificare per clienti italiani) | abilitazione dal sito; Darwin acceso sul PC, si disconnette ogni notte verso mezzanotte | conto IBKR; gateway o TWS in esecuzione |
| Costo API | — | trading gratis; dati 20 €/mese (gratis i primi 2 mesi o con oltre 200 € di commissioni al mese) | dati di mercato a pagamento secondo le borse |
| Commissioni ETF (indicative) | 2 € per ordine su Borsa Italiana/Euronext (profilo Bronze) | profilo Semplice 5 €/ordine; Variabile 1,9‰ (min 1,50 €, max 18 €) | basse, variabili per borsa |

**Lettura:**

- **BG Saxo** è l'unico che, sulla carta, unisce sostituto d'imposta, API
  moderna e ambiente di prova: è il primo da verificare (chiedere al supporto
  OpenAPI se i clienti BG Saxo Italia possono usare l'accesso live).
- **Directa** è sostituto d'imposta e ha un'API gratuita, ma senza ambiente
  di prova e con la piattaforma Darwin da tenere accesa sul PC.
- **Interactive Brokers** ha l'API e il paper trading migliori, ma il regime
  è dichiarativo: quadri RT/RM/RW in dichiarazione ogni anno (di solito con
  un commercialista).

## 3. Fiscalità italiana (ETF e ETC)

- **Plusvalenze sugli ETF:** redditi di capitale, 26% (12,5% sulla quota di
  titoli di Stato italiani ed europei contenuti nell'ETF).
- **Minusvalenze sugli ETF:** redditi diversi. **Non compensano le
  plusvalenze di ETF**; compensano plusvalenze di azioni, obbligazioni, ETC
  e certificati entro 4 anni.
- **ETC (es. oro fisico):** non sono fondi; guadagni e perdite si
  compensano tra loro e con altri redditi diversi.
- **ETF ad accumulazione:** nessuna tassazione dei proventi finché non si
  vende; con le regole attuali conviene rispetto a quelli a distribuzione.
- **Amministrato:** il broker trattiene le imposte e tiene lo "zainetto"
  fiscale delle minusvalenze. **Dichiarativo:** tutto a carico
  dell'investitore.

**Conseguenze per il sistema:**

1. Ogni ribilanciamento che vende un ETF in guadagno paga il 26% e le perdite
   su altri ETF non lo compensano. Nel conto reale servono **bande più larghe**
   e ribilanciamento preferibilmente con nuovi versamenti, non con vendite.
2. Preferire classi **ad accumulazione**.
3. L'oro come **ETC** è fiscalmente più flessibile di un ETF.

## 4. Costi delle commissioni fisse

Con 10.000 € e 10 posizioni, un ribilanciamento completo costa per esempio
10 × 5 € = 50 €, cioè lo 0,5% del capitale. Con 5–6 posizioni e un
importo minimo per ordine (es. 300–500 €) il costo scende di circa la metà.
Il parametro `min_order_notional` esiste già nel motore di esecuzione.

## 5. ETF UCITS equivalenti ai 10 del paper trading

Candidati con ISIN verificati sul nome del fondo. **Da verificare prima
dell'uso:** quotazione su Borsa Italiana o sulla borsa del broker scelto,
TER, dimensione del fondo, ticker presso il broker.

| Paper (USA) | Esposizione | Candidato UCITS | ISIN | Classe |
|---|---|---|---|---|
| SPY | S&P 500 | iShares Core S&P 500 UCITS ETF | IE00B5BMR087 | Acc |
| QQQ | Nasdaq-100 | iShares Nasdaq 100 UCITS ETF | IE00B53SZB19 | Acc |
| IWM | Russell 2000 | SPDR Russell 2000 US Small Cap UCITS ETF | IE00BJ38QD84 | Acc |
| EFA | Paesi sviluppati esclusi USA | Xtrackers MSCI World ex USA UCITS ETF 1C | IE0006WW1TQ4 | Acc |
| EEM | Mercati emergenti | iShares Core MSCI EM IMI UCITS ETF | IE00BKM4GZ66 | Acc |
| IEF | Treasury USA 7–10 anni | iShares $ Treasury Bond 7-10yr UCITS ETF | IE00B3VWN518 | Acc |
| LQD | Obbligazioni societarie USD | iShares $ Corp Bond UCITS ETF | IE0032895942 | Dist (cercare la classe Acc) |
| TLT | Treasury USA 20+ anni | iShares $ Treasury Bond 20+yr UCITS ETF | IE00BFM6TC58 | Acc |
| GLD | Oro | iShares Physical Gold **ETC** | IE00B4ND3602 | ETC |
| VNQ | Immobiliare USA | iShares US Property Yield UCITS ETF | IE00B1FZSF77 | Dist |

**Nota sul cambio (D3).** Nel paper le obbligazioni sono in dollari: per chi
spende in euro, il 30% "prudente" del portafoglio è esposto al cambio
EUR/USD, che può muoversi più delle obbligazioni stesse. Le alternative
(titoli di Stato dell'area euro, ETF coperti dal cambio) cambiano il
portafoglio: vanno trattate come **nuova variante**, decisa a priori e
verificata una volta, non come ritocco del core attuale.

## 6. Gradini di capitale

L'IPS prevede 1% → 5% → 25% → 100%. Con commissioni fisse, l'1% (100 €)
non è operabile (5 € di commissione = 5%). Proposta da decidere (D5):
primo gradino a 1.000 € su 3–4 ETF, poi 2.500 €, 5.000 €, 10.000 €, ognuno
solo se il precedente rispetta i criteri (nessun errore operativo,
scostamento dal modello entro le soglie del monitoraggio).

## 7. Cosa cambierà nel codice (quando richiesto)

1. Adattatore del broker scelto che implementa l'interfaccia `Broker`
   esistente, con le stesse barriere dell'adattatore paper (nessun reinvio
   alla cieca, id ordine deterministici, conto bloccato = stop).
2. **Fonte dati europea:** Alpaca non copre le borse europee; servono i
   prezzi dal broker (Directa 20 €/mese, IBKR a pagamento) o un'altra fonte
   di chiusure giornaliere.
3. Configurazione `live_core_ucits_v1.toml` con ISIN/ticker, valuta EUR,
   bande e importo minimo per ordine adatti a commissioni e tasse.
4. Ribilanciamento attento alle tasse: soglie più larghe, priorità ai
   versamenti, nessuna vendita solo per piccoli scostamenti.

## Fonti

- [Broker sostituti d'imposta 2026 — letraderdudimanche.com](https://letraderdudimanche.com/it/broker-sostituto-imposta) (aggiornato 29/10/2026)
- [Migliori broker con regime amministrato — promarketer.it](https://www.promarketer.it/investimenti/broker/migliori-broker-regime-amministrato/) (aggiornato 02/11/2026)
- [Directa — API e servizi Darwin](https://www.directa.it/pub2/en/darwin/api.html)
- [Directa — documentazione tecnica API](https://directa-it.github.io/documentation/en.html)
- [Saxo OpenAPI — ambienti SIM e live](https://www.developer.saxo/openapi/learn/environments)
- [IBKR Web API — paper trading](https://www.interactivebrokers.com/docs/web-api/authentication/paper.md)
- [justETF — la fiscalità degli ETF in Italia](https://www.justetf.com/it/news/etf/la-fiscalita-degli-etf-in-italia.html)
- [justETF — la tassazione degli ETC in Italia](https://www.justetf.com/it/news/etf/la-tassazione-degli-etc-in-italia.html)
- Schede degli ETF per ISIN: [IE00B5BMR087](https://www.justetf.com/uk/etf-profile.html?isin=IE00B5BMR087),
  [IE00B53SZB19](https://www.trackingdifferences.com/ETF/ISIN/IE00B53SZB19),
  [IE00BJ38QD84](https://extraetf.com/at/etf-profile/IE00BJ38QD84),
  [IE0006WW1TQ4](https://fiscoinvestimenti.it/xtrackers-msci-world-ex-usa-ucits-etf-analisi-isin-ie0006ww1tq4/),
  [IE00BKM4GZ66](https://www.justetf.com/de-en/etf-profile.html?isin=IE00BKM4GZ66),
  [IE00B3VWN518](https://extraetf.com/ch/etf-profile/IE00B3VWN518),
  [IE0032895942](https://www.finanzen.ch/etf/daten-gebuehr/ishares-corp-bond-etf-ie0032895942),
  [IE00BFM6TC58](https://extraetf.com/at/etf-profile/IE00BFM6TC58),
  [IE00B4ND3602](https://www.justetf.com/de-en/etf-profile.html?isin=IE00B4ND3602),
  [IE00B1FZSF77](https://www.finanzen.at/etf/ishares-us-property-yield-etf-ie00b1fzsf77)
