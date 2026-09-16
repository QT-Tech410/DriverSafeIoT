@echo off
REM Khoi dong Mosquitto broker voi config cua du an (spec L1: 0.0.0.0:1883, anonymous).
REM Service "mosquitto" cua ban Windows chiem port 1883 nen phai stop truoc.

setlocal
set REPO=%~dp0..
set MOSQ=C:\Program Files\mosquitto\mosquitto.exe

if not exist "%MOSQ%" (
    echo [LOI] Khong tim thay mosquitto tai "%MOSQ%"
    echo        Sua bien MOSQ trong file nay cho dung duong dan cai dat.
    exit /b 1
)

sc query mosquitto | findstr /C:"RUNNING" >nul
if %errorlevel%==0 (
    echo [i] Dang stop service mosquitto de giai phong port 1883...
    sc stop mosquitto >nul
    REM dung ping thay timeout: ten "timeout" co the bi Git Bash/PATH che khuat
    ping -n 3 127.0.0.1 >nul
)

echo [i] Khoi dong broker voi %REPO%\mqtt\mosquitto.conf
"%MOSQ%" -c "%REPO%\mqtt\mosquitto.conf" -v
