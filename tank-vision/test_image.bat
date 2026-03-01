@echo off
title Tank Vision AI - Image Test
echo ==========================================
echo   Tank Vision AI - Image Test
echo ==========================================
echo.

if "%~1"=="" (
    echo Kullanim: Bu dosyanin uzerine bir resim surukle-birak
    echo   veya: test_image.bat resim_yolu.jpg
    pause
    exit /b 1
)

"C:\Users\orhun\AppData\Local\Programs\Python\Python312\python.exe" "%~dp0scripts\quick_test.py" --source "%~1"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo HATA olustu!
    pause
)
