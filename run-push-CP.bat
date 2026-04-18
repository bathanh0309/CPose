@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo === Git push script for MrPhu/MultiCam_Surveillance_App ===

set "TARGET_REPO_URL=https://github.com/MrPhu/MultiCam_Surveillance_App.git"
set "TARGET_BRANCH=feat/research-ADL"

git --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Git is not installed or not in PATH.
  pause
  exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo [ERROR] This folder is not a Git repository.
  echo Run this file inside your MultiCam_Surveillance_App project folder.
  pause
  exit /b 1
)

git remote get-url origin >nul 2>&1
if errorlevel 1 (
  git remote add origin %TARGET_REPO_URL%
) else (
  git remote set-url origin %TARGET_REPO_URL%
)

set "CUR_BRANCH="
for /f %%i in ('git branch --show-current') do set "CUR_BRANCH=%%i"
if "%CUR_BRANCH%"=="" set "CUR_BRANCH=main"

echo Current branch: %CUR_BRANCH%
echo Target branch: %TARGET_BRANCH%

git add -A

git diff --cached --quiet
if not errorlevel 1 (
  echo No new changes to commit.
) else (
  git commit -m "feat[update app Pose vs ADL ]"
  if errorlevel 1 (
    echo [ERROR] Commit failed.
    echo Check your Git user.name and user.email settings, or inspect the staged changes.
    pause
    exit /b 1
  )
)

echo Pushing local content to origin/%TARGET_BRANCH% without pull...
git push -u origin %CUR_BRANCH%:%TARGET_BRANCH% --force

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
