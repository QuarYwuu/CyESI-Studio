@echo off
cd /d "%~dp0"
title CyESI Studio - English Edition
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py --studio %*
) else (
    python main.py --studio %*
)
if errorlevel 1 pause
