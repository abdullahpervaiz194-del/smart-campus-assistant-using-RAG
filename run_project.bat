@echo off
title Smart Campus Assistant (RAG + Ollama) - CSC-320(L) Group 1
cd /d "%~dp0"

echo ======================================================================
echo   Smart Campus Assistant using RAG + LLM (100%% Offline Local Mode)
echo   Course: CSC-320(L) Artificial Intelligence - Section B - Group 1
echo ======================================================================
echo.

REM Check if Ollama is already listening on port 11434; if not, start it in background
netstat -ano | findstr ":11434" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Starting local Ollama server in the background...
    start "" /B ollama serve >nul 2>&1
    timeout /t 3 /nobreak >nul
) else (
    echo [*] Local Ollama server is already running on port 11434.
)

echo [*] Opening http://127.0.0.1:5000 in your browser in 2 seconds...
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:5000"

echo [*] Starting Flask server (Keep this window open while using the app)...
echo.
".\.venv\Scripts\python.exe" app.py
pause
