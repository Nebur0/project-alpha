@echo off
rem Project Alpha - avvia il server della web app in background (nessuna finestra).
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

start "" "%PYW%" -m project_alpha dashboard --port 8050
