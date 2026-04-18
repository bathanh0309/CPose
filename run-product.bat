@echo off
title CPose - Product Server
echo Starting CPose Main Dashboard...
if exist package.json (
  call npm.cmd run build:css >nul 2>nul
)
start "CPose Backend" cmd /k ""%~dp0.venv\Scripts\python.exe" "%~dp0main.py""
exit /b
