@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo === Git push script for MrPhu/MultiCam_Surveillance_App ===

set "TARGET_REPO_URL=https://github.com/MrPhu/MultiCam_Surveillance_App.git"
set "TARGET_BRANCH=feat/research-ADL"
rem Only source and config files are included here; generated media stays out of the push.
set "CODE_ONLY_INDEX=%TEMP%\cp_code_only_%RANDOM%_%RANDOM%.idx"
set "CODE_ONLY_PATHS=.gitignore README.md main.py package.json package-lock.json requirements.txt run-product.bat run-push-CP.bat run-push-git.bat run-research.bat app cpose configs research shared static tools"
set "SNAPSHOT_MESSAGE=feat: code-only sync snapshot"

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
echo Code-only paths: %CODE_ONLY_PATHS%

set "HAS_CODE_CHANGES="
for /f "delims=" %%i in ('git status --porcelain --untracked-files=normal -- %CODE_ONLY_PATHS%') do (
  set "HAS_CODE_CHANGES=1"
  goto :code_changes_detected
)
:code_changes_detected

if not defined HAS_CODE_CHANGES (
  echo No code changes to push.
  goto :cleanup_ok
)

echo Building code-only snapshot...
set "GIT_INDEX_FILE=%CODE_ONLY_INDEX%"
git read-tree --empty
if errorlevel 1 (
  echo [ERROR] Failed to initialize the temporary Git index.
  goto :cleanup_fail
)

git add -A -- %CODE_ONLY_PATHS%
if errorlevel 1 (
  echo [ERROR] Failed to stage the code-only paths.
  goto :cleanup_fail
)

for /f %%i in ('git write-tree') do set "TREE_SHA=%%i"
if "%TREE_SHA%"=="" (
  echo [ERROR] Failed to create the snapshot tree.
  goto :cleanup_fail
)

for /f %%i in ('git commit-tree %TREE_SHA% -m "%SNAPSHOT_MESSAGE%"') do set "SNAPSHOT_COMMIT=%%i"
if "%SNAPSHOT_COMMIT%"=="" (
  echo [ERROR] Failed to create the snapshot commit.
  goto :cleanup_fail
)

set "GIT_INDEX_FILE="

echo Pushing code-only snapshot to origin/%TARGET_BRANCH% without pull...
git push origin %SNAPSHOT_COMMIT%:refs/heads/%TARGET_BRANCH% --force

if errorlevel 1 (
  echo.
  echo [ERROR] Push failed.
  echo If GitHub asks for authentication, use a Personal Access Token instead of password.
  goto :cleanup_fail
)

echo.
echo [DONE] Code-only push completed successfully.
goto :cleanup_ok

:cleanup_fail
set "GIT_INDEX_FILE="
if exist "%CODE_ONLY_INDEX%" del /f /q "%CODE_ONLY_INDEX%" >nul 2>&1
pause
exit /b 1

:cleanup_ok
set "GIT_INDEX_FILE="
if exist "%CODE_ONLY_INDEX%" del /f /q "%CODE_ONLY_INDEX%" >nul 2>&1
pause
exit /b 0
