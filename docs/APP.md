# Web app sull'iPhone (con notifiche)

La dashboard è una web app installabile (PWA): sull'iPhone si apre a schermo
intero dall'icona **Alpha** e riceve le notifiche push. Il server gira sul PC
Windows e ascolta solo in locale; l'iPhone ci arriva tramite **Tailscale**,
una rete privata gratuita tra i tuoi dispositivi (https, nessuna porta aperta
su internet).

Limite attuale: con il PC spento la dashboard dall'iPhone non si apre e le
notifiche non partono (non parte nemmeno il paper trading). Il passo
successivo previsto è spostare server e paper trading su un piccolo server
nel cloud: la web app resta la stessa.

## 1. Installazione sul PC (una volta)

```bash
pip install -e .                     # aggiunge la libreria crittografica per le notifiche
```

Server della web app sempre acceso (parte a ogni accesso a Windows):

```bat
schtasks /Create /TN "ProjectAlpha App" /SC ONLOGON /RL LIMITED ^
  /TR "\"<percorso del repo>\scripts\windows\app_server.bat\""
schtasks /Run /TN "ProjectAlpha App"
```

Controllo: http://127.0.0.1:8050 nel browser del PC.

## 2. Tailscale (una volta)

1. Installa Tailscale su Windows (tailscale.com/download) e accedi.
2. Installa **Tailscale** sull'iPhone dall'App Store e accedi con **lo
   stesso account**; lascia la VPN attiva.
3. Nella console di Tailscale (login.tailscale.com → DNS) attiva
   **MagicDNS** e **HTTPS Certificates**.
4. Sul PC, in un terminale:

   ```bash
   tailscale serve --bg 8050
   ```

   Stampa un indirizzo tipo `https://nome-pc.tailXXXX.ts.net`: è
   l'indirizzo della web app, raggiungibile solo dai tuoi dispositivi.

## 3. iPhone (una volta)

1. Apri l'indirizzo `https://nome-pc.tailXXXX.ts.net` in **Safari**.
2. Tocca **Condividi** → **Aggiungi alla schermata Home** → **Aggiungi**.
3. Apri **Alpha** dall'icona (non da Safari: su iPhone le notifiche
   funzionano solo dall'app installata, iOS 16.4 o successivo).
4. Tocca **Attiva notifiche** e consenti.
5. Tocca **Invia una prova**: deve arrivare una notifica.

## Cosa arriva

Dopo ogni esecuzione serale (`scripts/daily_paper.sh`) una notifica con
l'esito e il capitale; titolo ✅ / ⚠️ / ⛔ secondo gli avvisi del
monitoraggio, con urgenza alta per quelli gravi (stop, ordini rifiutati,
esecuzioni mancanti). A mano: `project-alpha paper-notify --config
configs/paper_core_v1.toml`.

## Sicurezza

- Il server ascolta solo su `127.0.0.1`; Tailscale lo espone solo ai
  dispositivi del tuo account.
- La web app non può fare operazioni di trading. Le uniche scritture sono
  le iscrizioni alle notifiche, accettate solo dalla pagina stessa e solo
  verso i servizi push dei browser (Apple, Google, Mozilla, Microsoft).
- I messaggi sono cifrati end-to-end (RFC 8291): Apple li consegna senza
  poterli leggere. Chiave privata e iscrizioni stanno in `data/push/`, fuori
  da git.
