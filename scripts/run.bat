@echo off
chcp 65001 >nul 2>&1
title HAVEN Shared Dashboard

echo ========================================
echo    HAVEN Shared Dashboard
echo    http://127.0.0.1:8000
echo ========================================
echo.

set "ROOT_DIR=%~dp0\.."
for %%I in ("%ROOT_DIR%") do set "ROOT_DIR=%%~fI"

call "%ROOT_DIR%\run-phase1.bat" %*
exit /b %ERRORLEVEL%
