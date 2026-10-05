@echo off
title TensorBoard - DriverSafeIoT Visualization
echo ========================================================
echo       DRIVER SAFE IOT - TENSORBOARD MONITOR
echo ========================================================
echo.
set "APP_DIR=D:\Workspace\DuanIoT"
cd /d "%APP_DIR%"

echo Dang khoi dong TensorBoard tren thu muc: %APP_DIR%\runs ...
echo Vui long mo trinh duyet truy cap: http://localhost:6006
echo Bam Ctrl+C de dung TensorBoard.
echo.

start "" http://localhost:6006
"%APP_DIR%\.venv\Scripts\tensorboard.exe" --logdir="%APP_DIR%\runs" --port=6006
pause
