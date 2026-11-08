@echo off
rem Project Alpha - apre la dashboard (sola lettura) su http://127.0.0.1:8050
rem Doppio clic per avviarla; chiudere questa finestra per spegnerla.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else if exist "..\.venv\Scripts\activate.bat" (
  call "..\.venv\Scripts\activate.bat"
) else (
  echo Ambiente Python .venv non trovato accanto al progetto.
  pause
  exit /b 1
)

rem Apre il browser dopo due secondi, il tempo che il server parta.
start "" /min cmd /c "timeout /t 2 /nobreak >nul & start "" http://127.0.0.1:8050"
python -m project_alpha dashboard
pause
