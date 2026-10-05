@echo off
    title Driver Safe IoT - Khoi dong he thong
    echo ========================================================
    echo        DRIVER SAFE IOT - STARTUP ALL SERVICES
    echo ========================================================
    echo.

    set "APP_DIR=D:\Workspace\DuanIoT"
    set "PY_BIN=%APP_DIR%\.venv\Scripts\python.exe"
    set "MOSQUITTO_DIR=C:\Program Files\mosquitto"

    cd /d "%APP_DIR%"
    set "PYTHONPATH=%APP_DIR%"

    rem 1. Kiem tra va khoi dong Mosquitto MQTT Broker
    echo [0/4] Dang kiem tra Mosquitto MQTT Broker...
    netstat -ano | findstr :1883 > nul
    if %errorlevel% equ 0 (
        echo [OK] Mosquitto dang chay tren Port 1883.
    ) else (
        if exist "%MOSQUITTO_DIR%\mosquitto.exe" (
            echo [OK] Khoi dong Mosquitto tu %MOSQUITTO_DIR%
            start "Mosquitto Broker" "%MOSQUITTO_DIR%\mosquitto.exe" -v
            timeout /t 2 /nobreak > nul
        ) else (
            echo [CANH BAO] Khong tim thay mosquitto.exe!
        )
    )

    rem 2. Kiem tra Python .venv
    if exist "%PY_BIN%" (
        echo [OK] Python .venv: %PY_BIN%
    ) else (
        set "PY_BIN=python"
        echo [CANH BAO] Dung python he thong.
    )

    rem 3. Khoi dong Web Dashboard Server
    echo.
    echo [1/3] Dang khoi dong Web Dashboard tai http://localhost:8000 ...
    start "Web Dashboard" cmd /k "%PY_BIN% host\dashboard\server.py"

    timeout /t 2 /nobreak > nul

    rem 4. Khoi dong Fusion Engine
    echo [2/3] Dang khoi dong Fusion Engine ...
    start "Fusion Engine" cmd /k "%PY_BIN% host\fusion\fusion.py"

    timeout /t 2 /nobreak > nul

    rem 5. Khoi dong Vision Worker (Chay dung file publisher.py co MediaPipe + Drawing)
    echo [3/3] Dang khoi dong Vision Worker (MediaPipe Face Mesh)...
    start "Vision Worker" cmd /k "%PY_BIN% host\vision\publisher.py"

    echo.
    echo ========================================================
    echo   TAT CA CAC DICH VU DA DUOC KHOI DONG THANH CONG!
    echo   - Web UI : http://localhost:8000
    echo ========================================================
    echo.
    pause