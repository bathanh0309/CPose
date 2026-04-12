echo off
setlocal enabledelayedexpansion

cd /d %~dp0

echo === Git push script for bathanh0309/CPose ===

git --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Git is not installed or not in PATH.
  pause
  exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo [ERROR] This folder is not a Git repository.
  echo Run this file inside your CPose project folder.
  pause
  exit /b 1
)

git config user.name "bathanh0309"
git config user.email "bathanh1234asd@gmail.com"

git remote get-url origin >nul 2>&1
if errorlevel 1 (
  git remote add origin https://github.com/bathanh0309/CPose.git
) else (
  git remote set-url origin https://github.com/bathanh0309/CPose.git
)

for /f %%i in ('git branch --show-current') do set CUR_BRANCH=%%i
if "%CUR_BRANCH%"=="" set CUR_BRANCH=main

echo Current branch: %CUR_BRANCH%

git add -A
git commit -m "update project" 2>nul
if errorlevel 1 (
  echo No new commit created. Continuing to push current state...
)

echo Pushing all local content to origin/%CUR_BRANCH% without pull...
git push -u origin %CUR_BRANCH% --force

if errorlevel 1 (
  echo.
  echo [ERROR] Push failed.
  echo If GitHub asks for authentication, use a Personal Access Token instead of password.
  pause
  exit /b 1
)

echo.
echo [DONE] Push completed successfully.
pause