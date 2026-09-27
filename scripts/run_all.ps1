# DriverSafe-IoT - One-Click Launch PowerShell Script (SYS-01)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "      DRIVER SAFE IoT - HE THONG AN TOAN NGUOI LAI XE" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Khoi dong toan bo cac tien trinh theo thu tu chuan:"
Write-Host "   1. Mosquitto MQTT Broker (127.0.0.1:1883)"
Write-Host "   2. Fusion Mamdani Engine (FUS-03/04)"
Write-Host "   3. FastAPI Dashboard Server (DASH-01/02/03/04) port 8000"
Write-Host "   4. Vision Face Mesh Publisher (VIS-05)"
Write-Host "=================================================================="

# Chay tu thu muc chua script nay (launch tu bat ky noi nao cung dung)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $ScriptDir

$pythonBin = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonBin)) {
    Write-Host "[LOI] Khong tim thay Python tai $pythonBin!" -ForegroundColor Red
    Pop-Location
    exit 1
}

# 1. Khoi dong Mosquitto Broker
Write-Host "[1/4] Khoi dong / Kiem tra Mosquitto Broker..." -ForegroundColor Yellow
Start-Process -FilePath "cmd.exe" -ArgumentList "/c if exist mqtt\mosquitto.conf ( mosquitto -c mqtt\mosquitto.conf -v ) else ( mosquitto -v )"
Start-Sleep -Seconds 2

# 2. Khoi dong Fusion Engine
Write-Host "[2/4] Khoi dong Fusion Engine (Mamdani Policy)..." -ForegroundColor Yellow
Start-Process -FilePath $pythonBin -ArgumentList "-m host.fusion.fusion --policy mamdani"
Start-Sleep -Seconds 2

# 3. Khoi dong Dashboard Web Server
Write-Host "[3/4] Khoi dong Dashboard Server (http://localhost:8000)..." -ForegroundColor Yellow
Start-Process -FilePath $pythonBin -ArgumentList "-m host.dashboard.server --port 8000"
Start-Sleep -Seconds 2

# 4. Khoi dong Vision Camera Publisher
Write-Host "[4/4] Khoi dong Vision Camera Publisher..." -ForegroundColor Yellow
Start-Process -FilePath $pythonBin -ArgumentList "-m host.vision.publisher --src webcam --index 0"
Start-Sleep -Seconds 2

Pop-Location

# 5. Mo trinh duyet
Write-Host "`n[V] TOAN BO HE THONG DA DUOC KHOI DONG THANH CONG!" -ForegroundColor Green
Write-Host "Dang mo giao dien Dashboard tai http://localhost:8000 ..." -ForegroundColor Cyan
Start-Process "http://localhost:8000"

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host " KICH BAN DIEN TAP DEMO 8 BUOC (Theo Spec 13):" -ForegroundColor Magenta
Write-Host "   1. Gioi thieu kien truc 3 nguon: Vision - ESP32 - Fusion (1.5p)"
Write-Host "   2. Bam 'Enroll Driver' tren Dashboard de hieu chuan 60s (1p)"
Write-Host "   3. Lai xe tinh tao: Dashboard Xanh (SAFE), Risk thap (1p)"
Write-Host "   4. Mo phong buon ngu: Nham mat 2-3s -> Microsleep -> Coi reo (2p)"
Write-Host "   5. Mo phong con: MQ-3 Level 2 -> Dong co bi KHOA -> Bam Unlock (2p)"
Write-Host "   6. Mo phong cabin: Che LDR hoac nong -> Risk Surface tang nhe (1p)"
Write-Host "   7. Kiem tra Edge Autonomy: Rut mang laptop -> ESP32 van canh bao (1p)"
Write-Host "   8. Xuat bao cao CSV tren Dashboard (Export CSV) hoan tat (1p)"
Write-Host "==================================================================" -ForegroundColor Cyan
