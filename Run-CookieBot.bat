@echo off
setlocal
cd /d "%~dp0"

py bot_image_match.py
if errorlevel 1 (
    echo.
    echo CookieBot stopped with an error.
    pause
)

endlocal