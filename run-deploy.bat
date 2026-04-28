@echo off
setlocal
title CPose - lephongphu.cloud

echo Starting CPose for lephongphu.cloud...
if exist package.json (
  call npm.cmd run build:css >nul 2>nul
)

set "CPOSE_DASHBOARD_URL=https://lephongphu.cloud"
set "CPOSE_CORS_ORIGINS=https://lephongphu.cloud,https://www.lephongphu.cloud,http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000"

start "CPose Backend" cmd /k ""%~dp0.venv\Scripts\python.exe" "%~dp0main.py""
start "Cloudflare Tunnel" cmd /k ""cloudflared" tunnel --config "%~dp0deploy\cloudflared.lephongphu.yml" run cpose-lephongphu"

endlocal
exit /b
