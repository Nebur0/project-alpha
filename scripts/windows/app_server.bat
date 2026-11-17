@echo off
rem Project Alpha - avvia l'app in background: dashboard + motore (lavoro serale,
rem recuperi, sentinella oraria, notizie). Nessuna finestra.
rem Pensato per l'Utilita' di pianificazione: "All'accesso" dell'utente.
setlocal
cd /d "%~dp0\..\.."

set "PYW="
if exist ".venv\Scripts\pythonw.exe" set "PYW=.venv\Scripts\pythonw.exe"
if not defined PYW if exist "..\.venv\Scripts\pythonw.exe" set "PYW=..\.venv\Scripts\pythonw.exe"
if not defined PYW (
  echo Ambiente Python .venv non trovato.
  exit /b 1
)

start "" "%PYW%" -m project_alpha app --port 8050
