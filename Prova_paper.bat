@echo off
rem Project Alpha - aggiorna l'installazione e fa una PROVA del paper trading.
rem Nessun ordine viene inviato. Doppio clic per avviarlo.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else if exist "..\.venv\Scripts\activate.bat" (
  call "..\.venv\Scripts\activate.bat"
) else (
  echo Ambiente Python .venv non trovato accanto al progetto.
  goto :fine
)

set "BASH=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%BASH%" for /f "delims=" %%B in ('where bash 2^>nul') do if not defined BASHFOUND set "BASH=%%B" & set BASHFOUND=1
if not exist "%BASH%" (
  echo Git Bash non trovato: installare Git per Windows.
  goto :fine
)

echo === Aggiornamento dell'installazione
python -m pip install -e . --quiet
if errorlevel 1 goto :errore

echo === Prova del paper trading (nessun ordine)
"%BASH%" scripts/daily_paper.sh --dry-run
if errorlevel 1 goto :errore

echo.
echo === Fatto. Risultato in reports\paper\paper_log.csv (e nella dashboard).
goto :fine

:errore
echo.
echo === Qualcosa non e' andato: leggere i messaggi qui sopra.

:fine
echo.
pause
