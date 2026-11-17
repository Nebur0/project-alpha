# Spostare l'app su un server nel cloud

Obiettivo: il PC può restare spento. Sul server gira lo **stesso** comando
del PC (`project-alpha app`: dashboard + motore con lavoro serale,
recuperi, sentinella, notizie), dentro un container; l'iPhone lo raggiunge
con Tailscale come oggi.

## Costi e scelta del server

Basta il server Linux più piccolo (1–2 vCPU, 2–4 GB di RAM, Ubuntu): l'app
usa pochissime risorse. Da provider europei (es. Hetzner) costa **pochi euro
al mese**; i prezzi cambiano, va verificato sul sito al momento
dell'acquisto. Serve un account a tuo nome con un metodo di pagamento.

## ⚠️ Una sola istanza alla volta

PC e server **non devono mai girare insieme**: ciascuno pensa di essere
l'unico gestore del conto paper e gli ordini verrebbero duplicati. Prima di
avviare il server, sul PC:

1. elimina il collegamento `ProjectAlpha App` in
   `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`;
2. elimina l'attività pianificata "ProjectAlpha Paper"
   (`schtasks /Delete /TN "ProjectAlpha Paper" /F`);
3. chiudi il processo dell'app (Gestione attività → pythonw).

## Passi

1. **Server**: crea un server Ubuntu, accedi via SSH, installa Docker e
   Tailscale (`curl -fsSL https://tailscale.com/install.sh | sh`, poi
   `sudo tailscale up` con il tuo account).
2. **Codice**: `git clone https://github.com/Nebur0/project-alpha.git && cd project-alpha`.
3. **Chiavi**: crea `.env` con `ALPACA_API_KEY=PK...` e
   `ALPACA_API_SECRET=...` (le stesse chiavi paper del PC; mai chiavi live).
4. **Stato dal PC** (così il server continua da dove era il PC, stesso
   capitale, massimo, conto interno e notifiche): copia sul server, nei
   volumi, il contenuto di
   - `reports/paper/` (registro, `state.json`, `last_daily.json`),
   - `reports/paper_permanente/`,
   - `data/push/` (chiave delle notifiche e iPhone iscritti),
   - `data/paper_permanente/` (conto interno del portafoglio permanente).
   Senza `data/push/` l'iPhone va iscritto di nuovo; senza `state.json` il
   sistema riparte con un nuovo massimo.
5. **Avvio**: `docker compose -f deploy/compose.yaml up -d --build`.
6. **Accesso**: `sudo tailscale serve --bg 8050` → nuovo indirizzo
   `https://nome-server.….ts.net`; aggiungilo alla schermata Home
   dell'iPhone e riattiva le notifiche se non hai copiato `data/push/`.
7. **Controllo**: `docker compose -f deploy/compose.yaml logs -f` e
   `reports/paper/engine.log` nel volume.

## Aggiornamenti

`git pull && docker compose -f deploy/compose.yaml up -d --build`. I volumi
conservano dati e stato tra un aggiornamento e l'altro.

## Backup

Il registro `paper_log.csv` è la prova in avanti: copialo ogni tanto nel
repository (o fai uno snapshot del server) per non perderlo.
