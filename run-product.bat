@echo off
title CPose - Product Server
echo Starting CPose Main Dashboard...
if exist package.json (
  call npm.cmd run build:css >nul 2>nul
)
.venv\Scripts\python.exe main.py
pause
