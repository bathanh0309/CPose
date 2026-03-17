@echo off
echo Configuring git user...

if not exist .git (
    echo Initializing local Git repository...
    git init
)

git config --local user.name "bathanh0309"
git config --local user.email "bathanh1234asd@gmail.com"

git remote get-url origin >nul 2>&1
if %errorlevel% neq 0 (
    echo Adding GitHub remote origin...
    git remote add origin https://github.com/bathanh0309/HavenNet
)

echo.
set /p commit_msg="Enter commit message (Default: HavenNet): "
if "%commit_msg%"=="" set commit_msg=HavenNet

echo.
echo Adding files...
git add .

echo.
echo Committing changes...
git commit -m "%commit_msg%"

echo.
echo Pushing to GitHub...
git push -u origin main 2>nul || git push -u origin master 2>nul || git push

echo.
echo Done!
pause
