@echo off
setlocal


cd /d %~dp0


git --version >nul 2>&1 || (
    echo [ERROR] Git not found.
    exit /b 1
)


:: ============================================================
:: INIT: Nếu chưa có .git thì khởi tạo repo mới
:: ============================================================
if not exist ".git" (
    echo [INIT] No git repo found. Initializing...
    git init
    git branch -M main
    echo [INIT] Git repo initialized.
)


git config user.name "bathanh0309"
git config user.email "bathanh1234asd@gmail.com"
git config core.autocrlf false


git remote get-url origin >nul 2>&1
if errorlevel 1 (
    git remote add origin https://github.com/bathanh0309/CPose.git
) else (
    git remote set-url origin https://github.com/bathanh0309/CPose.git
)


if exist .git\index.lock (
    echo Removing stale Git lock...
    del /f /q .git\index.lock
)


echo Current remote:
git remote -v


echo.
echo Adding all files...
git add -A


git commit -m "update project" 2>nul
if errorlevel 1 (
    echo No new commit created. Continuing to force push current state...
)


echo.
echo Force pushing current HEAD to origin/main ...
git push -u origin HEAD:main --force


if errorlevel 1 (
    echo.
    echo [ERROR] Push failed.
    pause
    exit /b 1
)


echo.
echo [DONE] Force push to origin/main completed successfully.
pause