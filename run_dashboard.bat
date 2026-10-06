@echo off
title VisionAttend AI - Web Dashboard Launcher
echo ========================================================
echo   Starting VisionAttend AI Web Dashboard
echo   Author: Ashish Kumar Maurya (B.Tech CSE - AI)
echo ========================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8503
) else (
    python -m streamlit run app.py --server.port 8503
)
pause
