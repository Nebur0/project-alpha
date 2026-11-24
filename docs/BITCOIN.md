# Satellite bitcoin con filtro di tendenza — prova in avanti (`bitcoin_trend_v1`)

Registrato il 2026-11-24, **prima** di qualsiasi risultato in avanti.
Configurazione: `configs/paper_bitcoin_trend_v1.toml`. Non si modifica
guardando i risultati; un'idea nuova diventa una variante separata.

## Perché

Il proprietario ha chiesto un portafoglio sulle cripto ("vai, creiamo un nuovo
portafoglio"). Il trading veloce è escluso (laboratorio intraday: nessuna
regola supera i costi). Si prova invece l'unica regola semplice con evidenza
documentata anche sulle cripto: seguire la tendenza di lungo periodo.

## Regole

- Strumento: IBIT (ETF spot su bitcoin quotato a New York), dati giornalieri
  Alpaca in `data/raw/bars_bitcoin.csv`.
- Ogni sera, a seduta chiusa: se la chiusura di IBIT è **sopra** la media
  delle ultime 200 sedute → 100% IBIT; altrimenti → 100% liquidità.
  Con meno di 200 sedute di storia → liquidità.
- Ordini sul conto interno, eseguiti all'apertura della seduta successiva,
  costo 5 punti base. Capitale virtuale 1.000.
- Protezioni: −40% dal massimo → solo riduzioni; −50% → tutto in cash e stop
  fino allo sblocco manuale; perdita massima dichiarata 60%.

## Cosa aspettarsi

Il bitcoin ha avuto più volte cali del 70–80%. Il filtro di tendenza di solito
evita la parte centrale dei crolli lunghi, ma entra ed esce in ritardo e nei
mercati laterali paga piccoli falsi segnali. Settimane da −20% sono normali.

Nessuna verifica storica è stata fatta: la storia di IBIT (dal 2024) è troppo
breve per giudicare. Prove guardate: 0.

## Criteri di giudizio (dopo almeno 6 mesi)

- Il filtro deve aver ridotto il calo massimo rispetto a tenere IBIT sempre,
  senza perdere la maggior parte della crescita.
- Nessun capitale reale senza una decisione esplicita e separata del
  proprietario (tasse italiane sulle cripto e sugli ETF USA da verificare).
