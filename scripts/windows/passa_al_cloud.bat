@echo off
rem Project Alpha - passaggio dal PC al server nel cloud (docs/CLOUD.md).
rem 1) ferma l'app sul PC e ne disattiva l'avvio automatico (reversibile);
rem 2) impacchetta lo stato dei portafogli in alpha_stato.tgz da copiare sul server.
rem Non legge e non copia il file .env: le chiavi si inseriscono sul server.
setlocal
cd /d "%~dp0\..\.."

echo [1/3] Fermo l'app sul PC...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' OR Name='python.exe'\" | Where-Object { $_.CommandLine -match 'project_alpha' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"

echo [2/3] Disattivo l'avvio automatico (si riattiva con: schtasks /Change /TN "ProjectAlpha App" /ENABLE)...
schtasks /Change /TN "ProjectAlpha App" /DISABLE >nul 2>&1
schtasks /Change /TN "ProjectAlpha Paper" /DISABLE >nul 2>&1

echo [3/3] Impacchetto lo stato in alpha_stato.tgz...
set "LIST="
for %%D in (reports\paper reports\paper_permanente reports\paper_crescita reports\paper_bitcoin data\push data\paper_permanente data\paper_crescita data\paper_bitcoin) do (
  if exist "%%D" call set "LIST=%%LIST%% %%D"
)
tar -czf alpha_stato.tgz %LIST%
if errorlevel 1 (
  echo Errore durante la creazione del pacchetto.
  exit /b 1
)
echo.
echo Fatto: alpha_stato.tgz creato. Da adesso l'app sul PC e' ferma:
echo non riavviarla mentre gira il server (gli ordini verrebbero doppi).
endlocal
