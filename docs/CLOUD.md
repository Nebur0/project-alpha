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
l'unico gestore dei conti e gli ordini verrebbero duplicati. Il passaggio
dal PC si fa con un solo script, che ferma l'app e prepara lo stato.

## Passi

1. **Server**: crea un server Ubuntu (24.04) dal provider, con il tuo
   account e metodo di pagamento, e annota indirizzo e password o chiave SSH.
2. **PC, il giorno del passaggio (non tra le 22 e le 24)**: doppio clic su
   `scripts\windows\passa_al_cloud.bat`. Ferma l'app, ne disattiva l'avvio
   automatico (reversibile: `schtasks /Change /TN "ProjectAlpha App" /ENABLE`)
   e crea `alpha_stato.tgz` nella cartella del progetto con lo stato dei
   quattro portafogli (`reports/paper*`, `data/paper_*`) e delle notifiche
   (`data/push`). Il file `.env` non viene copiato.
3. **Copia lo stato sul server** (dal PC, in un terminale):
   `scp alpha_stato.tgz root@<indirizzo-server>:~/`
4. **Server** (via SSH, nella cartella dove hai copiato il file):

   ```bash
   curl -fsSLO https://raw.githubusercontent.com/Nebur0/project-alpha/main/deploy/setup_server.sh
   sudo bash setup_server.sh
   ```

   Lo script installa Docker e Tailscale (ti chiede di accedere a Tailscale
   con il tuo account), scarica il codice in `/opt/project-alpha`, chiede le
   chiavi **paper** di Alpaca (mai chiavi live; non vengono mostrate), crea
   il container, ci copia lo stato **prima** di avviarlo, lo avvia e
   pubblica la dashboard con `tailscale serve`.
5. **iPhone**: apri il nuovo indirizzo `https://<nome-server>.….ts.net`
   stampato alla fine, aggiungilo alla schermata Home e cancella la vecchia
   icona del PC. Le notifiche restano attive (la chiave è in `data/push`).
6. **Controllo**: `docker compose -f /opt/project-alpha/deploy/compose.yaml logs -f`
   e, la mattina dopo, `engine.log` nel volume del registro paper.

Ogni portafoglio ha il suo volume (`deploy/compose.yaml`); un test
(`tests/test_deploy.py`) fallisce se un portafoglio nuovo non ha il volume.

## Aggiornamenti

`cd /opt/project-alpha && git pull && docker compose -f deploy/compose.yaml up -d --build`. I volumi
conservano dati e stato tra un aggiornamento e l'altro.

## Backup

Il registro `paper_log.csv` è la prova in avanti: copialo ogni tanto nel
repository (o fai uno snapshot del server) per non perderlo.
