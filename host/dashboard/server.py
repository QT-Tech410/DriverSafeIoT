"""FastAPI Server & Realtime Web Bridge (DASH-01).

Cung cấp Backend Web cho DriverSafe-IoT:
- REST APIs:
    + GET  /api/status            : Tình trạng node online, risk hiện tại, engine lock
    + GET  /api/telemetry/recent  : Lịch sử telemetry 1Hz (mặc định 60s cho chart)
    + GET  /api/events            : Nhật ký các sự kiện rủi ro
    + POST /api/cmd               : Phát lệnh điều khiển xuống ESP32 qua MQTT
- WebSocket Endpoint:
    + /ws                         : Stream dữ liệu thời gian thực độ trễ <= 50ms
- Tích hợp SQLite Database và MQTT Bridge tự động lưu trữ và đồng bộ
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Cho phép import các module cùng cấp
DASHBOARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DASHBOARD_DIR))

from camera_stream import WebCameraStreamer
from db import Database
from models import SystemStatus
from mqtt_bridge import MqttBridge
from ws_manager import WebSocketManager

STATIC_DIR = DASHBOARD_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Khởi tạo các singleton service
db = Database()
ws_manager = WebSocketManager()
camera_streamer = WebCameraStreamer()
mqtt_bridge: MqttBridge | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời khởi động và kết thúc của FastAPI server."""
    global mqtt_bridge
    loop = asyncio.get_running_loop()

    # Khởi tạo MQTT Bridge với async loop hiện tại
    mqtt_bridge = MqttBridge(
        broker_host="127.0.0.1",
        broker_port=1883,
        client_id="driversafe_dashboard_backend",
        db=db,
        ws_manager=ws_manager,
        loop=loop,
    )
    print("[dashboard.server] Dang ket noi MQTT Bridge...")
    mqtt_bridge.connect_broker(timeout=3.0)
    camera_streamer.start()

    yield

    # Cleanup khi shutdown
    camera_streamer.stop()
    if mqtt_bridge:
        print("[dashboard.server] Dang ngat ket noi MQTT Bridge...")
        mqtt_bridge.disconnect_broker()


app = FastAPI(
    title="DriverSafe-IoT Dashboard API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware hỗ trợ test local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CommandRequest(BaseModel):
    """Schema request gửi lệnh điều khiển xuống thiết bị IoT."""

    cmd: str
    n: int | None = None
    sig: str | None = None


@app.get("/api/status")
async def get_system_status() -> dict:
    """Trả về trạng thái tổng quan của hệ thống và các node."""
    if mqtt_bridge:
        status = mqtt_bridge.get_system_status()
    else:
        status = SystemStatus()
    return status.as_dict()


@app.get("/api/telemetry/recent")
async def get_recent_telemetry(limit: int = 60) -> list[dict]:
    """Lấy N bản ghi telemetry gần nhất (mặc định 60 điểm thời gian thực)."""
    records = db.get_recent_telemetry(limit=min(limit, 300))
    return [r.as_dict() for r in records]


@app.get("/api/events")
async def get_recent_events(limit: int = 50) -> list[dict]:
    """Lấy danh sách các sự kiện rủi ro đã ghi nhận."""
    records = db.get_recent_events(limit=min(limit, 200))
    return [r.as_dict() for r in records]


@app.post("/api/cmd")
async def send_command(req: CommandRequest) -> dict:
    """Gửi lệnh điều khiển (unlock, enroll, beep...) xuống ESP32 qua MQTT."""
    if not mqtt_bridge or not mqtt_bridge.is_connected():
        raise HTTPException(status_code=503, detail="MQTT Broker chua ket noi, khong the gui lenh")

    cmd_dict = {"cmd": req.cmd}
    if req.n is not None:
        cmd_dict["n"] = req.n
    if req.sig is not None:
        cmd_dict["sig"] = req.sig

    ok = mqtt_bridge.send_command(cmd_dict)
    if not ok:
        raise HTTPException(status_code=500, detail="Loi khi publish lenh MQTT")

    return {"status": "ok", "sent": cmd_dict}


@app.get("/api/video_feed")
async def video_feed():
    """Stream luồng video MJPEG có vẽ landmarks và metrics cho Dashboard."""
    return StreamingResponse(
        camera_streamer.stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Kênh WebSocket stream dữ liệu thời gian thực cho trình duyệt."""
    await ws_manager.connect(websocket)

    # Gửi ngay gói tin initial state ngay khi client vừa kết nối
    try:
        init_telemetry = [r.as_dict() for r in db.get_recent_telemetry(limit=60)]
        init_events = [e.as_dict() for e in db.get_recent_events(limit=20)]
        status = mqtt_bridge.get_system_status().as_dict() if mqtt_bridge else SystemStatus().as_dict()
        latest_telemetry = mqtt_bridge.build_current_telemetry().as_dict() if mqtt_bridge else None

        init_msg = {
            "type": "init",
            "status": status,
            "recent_telemetry": init_telemetry,
            "recent_events": init_events,
            "latest_telemetry": latest_telemetry,
        }
        await websocket.send_json(init_msg)

        # Lắng nghe các thông điệp từ client (nếu client gửi lệnh qua WS)
        while True:
            data = await websocket.receive_text()
            # Echo hoặc xử lý ping/pong từ frontend
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)


# Phục vụ file tĩnh giao diện Web
if (STATIC_DIR / "index.html").is_file():
    @app.get("/")
    async def serve_index():
        return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main() -> int:
    parser = argparse.ArgumentParser(description="DriverSafe-IoT Dashboard Server (DASH-01)")
    parser.add_argument("--host", default="127.0.0.1", help="Host lắng nghe (mặc định: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port lắng nghe (mặc định: 8000)")
    args = parser.parse_args()

    print(f"[dashboard.server] Khoi chay FastAPI server tai http://{args.host}:{args.port}")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
