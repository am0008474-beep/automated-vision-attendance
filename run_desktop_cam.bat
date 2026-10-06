@echo off
title VisionAttend AI - Standalone Desktop Camera Launcher
echo ========================================================
echo   Starting VisionAttend AI Desktop Kiosk
echo   Author: Ashish Kumar Maurya (B.Tech CSE - AI)
echo   Controls: Press 'Q' or ESC to Exit, 'R' to reload model
echo ========================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" desktop_cam.py
) else (
    python desktop_cam.py
)
pause
