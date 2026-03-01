@echo off
title Tank Vision AI - HUD
echo ==========================================
echo   TANK VISION AI - Askeri HUD Sistemi
echo   Cikmak icin 'q' tusuna bas
echo ==========================================
echo.

set PYTHON="C:\Users\orhun\AppData\Local\Programs\Python\Python312\python.exe"

%PYTHON% "%~dp0scripts\tank_vision_hud.py" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo HATA olustu!
    pause
)
