@echo off
    title Driver Safe IoT - Khoi dong he thong
    echo ========================================================
    echo        DRIVER SAFE IOT - STARTUP ALL SERVICES
    echo ========================================================
    echo.

    set "ROOT_DIR=D:\Workspace\DuanIoT"
    set "APP_DIR=%ROOT_DIR%\.kilo\worktrees\uncovered-sheet"
    set "PY_BIN=%ROOT_DIR%\.venv\Scripts\python.exe"
    set "MOSQUITTO_DIR=C:\Program Files\mosquitto"

    rem 1. Khoi dong Mosquitto MQTT Broker (Port 1883)
    echo [0/4] Dang kiem tra va khoi dong Mosquitto MQTT Broker...
    netstat -ano | findstr :1883 > nul
    if %errorlevel% equ 0 (
        echo [OK] Mosquitto (hoac Service 1883) da dang chay tren Port 1883.
    ) else (
        if exist "%MOSQUITTO_DIR%\mosquitto.exe" (
            echo [OK] Tim thay Mosquitto tai %MOSQUITTO_DIR%
            start "Mosquitto Broker" cmd /k "title Mosquitto Broker && cd /d "%MOSQUITTO_DIR%" && mosquitto.exe -v"
            timeout /t 2 /nobreak > nul
        ) else (
            echo [CANH BAO] Khong tim thay mosquitto.exe tai %MOSQUITTO_DIR%!
            echo Vui long kiem tra lai duong dan hoac khoi dong Mosquitto Service thu cong.
        )
    )

    echo.
    rem 2. Kiem tra Python .venv
    if exist "%PY_BIN%" (
        echo [OK] Su dung Python .venv: %PY_BIN%
    ) else (
        set "PY_BIN=python"
        echo [CANH BAO] Khong tim thay .venv, su dung python he thong.
    )

    rem 3. Khoi dong Web Dashboard Server (Port 8000)
    echo.
    echo [1/3] Dang khoi dong Web Dashboard tai http://localhost:8000 ...
    start "Web Dashboard" cmd /k "title Web Dashboard && cd /d %APP_DIR% && set PYTHONPATH=%APP_DIR% && %PY_BIN% host\dashboard\server.py"

    timeout /t 2 /nobreak > nul

    rem 4. Khoi dong Fusion Engine
    echo [2/3] Dang khoi dong Fusion Engine ...
    start "Fusion Engine" cmd /k "title Fusion Engine && cd /d %APP_DIR% && set PYTHONPATH=%APP_DIR% && %PY_BIN% host\fusion\fusion.py"

    timeout /t 2 /nobreak > nul

    rem 5. Khoi dong Vision Camera MediaPipe
    echo [3/3] Dang khoi dong Vision Worker (MediaPipe) ...
    start "Vision Worker" cmd /k "title Vision Worker && cd /d %APP_DIR% && set PYTHONPATH=%APP_DIR% && %PY_BIN% host\vision\capture.py"

    echo.
    echo ========================================================
    echo   TAT CA DICH VU (MOSQUITTO, DASHBOARD, FUSION, VISION)
    echo   DA DUOC KHOI DONG THANH CONG!
    echo   - Web UI : http://localhost:8000
    echo   - Broker : 127.0.0.1:1883 (Local) / Lan IP:1883
    echo ========================================================
    echo.
    pause