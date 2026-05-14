@echo off
setlocal
cd /d "%~dp0"
set "TARGET_REPO_URL=https://github.com/MrPhu/MultiCam_Surveillance_App.git"
set "TARGET_BRANCH=feat/research-ADL"
git --version >nul 2>&1 || (echo [ERROR] Git not found. & exit /b 1)
git rev-parse --is-inside-work-tree >nul 2>&1 || (echo [ERROR] Not a git repo. & exit /b 1)
git remote get-url origin >nul 2>&1 || git remote add origin %TARGET_REPO_URL%
git add .
git commit -m "[auto] quick push"
git push origin HEAD:%TARGET_BRANCH%
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
