@echo off
echo ========================================================
echo   Pushing VisionAttend AI to GitHub
echo   Repository: am0008474-beep/automated-vision-attendance
echo ========================================================
cd /d "%~dp0"
git branch -M main
git push -u origin main
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo   SUCCESS! Project has been pushed to GitHub!
    echo   View repository at:
    echo   https://github.com/am0008474-beep/automated-vision-attendance
    echo ========================================================
) else (
    echo.
    echo [!] Push failed. Please make sure the repo 'automated-vision-attendance'
    echo     is created on GitHub at https://github.com/new
)
pause
