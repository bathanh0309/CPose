@echo off
setlocal
cd /d %~dp0
git --version >nul 2>&1 || (echo [ERROR] Git not found. & exit /b 1)
git rev-parse --is-inside-work-tree >nul 2>&1 || (echo [ERROR] Not a git repo. & exit /b 1)
git config user.name "bathanh0309"
git config user.email "bathanh1234asd@gmail.com"
git config core.autocrlf false
git remote get-url origin >nul 2>&1 || git remote add origin https://github.com/bathanh0309/CPose.git
git add .
git commit -m "[auto] quick push"
git push origin HEAD
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

echo Push only to origin/%CUR_BRANCH%...
echo This script does not run git pull.
git push -u origin %CUR_BRANCH% --force

if errorlevel 1 (
  echo.
  echo [ERROR] Push failed.
  echo This script only pushes. It does not pull remote code.
  echo If GitHub rejects the push, resolve it manually or keep using force push intentionally.
  pause
  exit /b 1
)

echo.
echo [DONE] Push completed successfully.
pause
