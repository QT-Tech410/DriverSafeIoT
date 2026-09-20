"""MQTT Bridge Service for Dashboard (DASH-01).

Chịu trách nhiệm:
- Kết nối tới Mosquitto Broker, subscribe các topic của hệ thống:
    + ds/fusion/level     (kết quả suy luận Risk, Band, Drivers, Action)
    + ds/esp32/sensors    (telemetry cảm biến ESP32: MQ3, NTC, LDR, RSSI)
    + ds/vision/metrics   (metrics AI: EAR, PERCLOS, CLES, MAR, Head Pitch)
    + ds/events           (nhật ký sự kiện rủi ro)
    + ds/vision/events    (sự kiện tức thời AI)
    + ds/esp32/events     (sự kiện tức thời ESP32)
- Tự động lưu trữ vào SQLite Database (bảng telemetry và bảng events)
- Chuyển tiếp tức thời (bridge) xuống giao diện Web thông qua WebSocketManager (độ trễ <= 50ms)
- Hỗ trợ phát lệnh điều khiển ngược lại cho ESP32 qua topic ds/esp32/cmd
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any

import paho.mqtt.client as mqtt

from db import Database
from models import EventRecord, SystemStatus, TelemetryRecord
from ws_manager import WebSocketManager

# Topic MQTT chuẩn của dự án (spec §5, §6.1)
TOPIC_FUSION_LEVEL = "ds/fusion/level"
TOPIC_ESP32_SENSORS = "ds/esp32/sensors"
TOPIC_VISION_METRICS = "ds/vision/metrics"
TOPIC_EVENTS = "ds/events"
TOPIC_VISION_EVENTS = "ds/vision/events"
TOPIC_ESP32_EVENTS = "ds/esp32/events"
TOPIC_ESP32_CMD = "ds/esp32/cmd"

try:
    _CALLBACK_API = mqtt.CallbackAPIVersion.VERSION2
except AttributeError:  # pragma: no cover
    _CALLBACK_API = None


class MqttBridge(mqtt.Client):
    """Cầu nối MQTT giữa phần cứng IoT / AI Engine với Web Dashboard."""

    def __init__(
        self,
        broker_host: str = "127.0.0.1",
        broker_port: int = 1883,
        client_id: str = "dashboard_mqtt_bridge",
        db: Database | None = None,
        ws_manager: WebSocketManager | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        if _CALLBACK_API is not None:
            super().__init__(_CALLBACK_API, client_id=client_id)
        else:  # pragma: no cover
            super().__init__(client_id=client_id)

        self.broker_host = broker_host
        self.broker_port = broker_port
        self.db = db or Database()
        self.ws_manager = ws_manager or WebSocketManager()
        self.loop = loop

        # Cache dữ liệu mới nhất từ từng nguồn để tổng hợp thành bản ghi 1Hz
        self.latest_fusion: dict[str, Any] = {}
        self.latest_esp32: dict[str, Any] = {}
        self.latest_vision: dict[str, Any] = {}

        self.last_fusion_ts: float = 0.0
        self.last_esp32_ts: float = 0.0
        self.last_vision_ts: float = 0.0
        self.engine_locked: bool = False

        self.on_connect = self._handle_connect
        self.on_disconnect = self._handle_disconnect
        self.on_message = self._handle_message

    def _handle_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        is_ok = (reason_code == 0) if isinstance(reason_code, int) else (not reason_code.is_failure)
        if is_ok:
            print(f"[dashboard.mqtt] Da ket noi Mosquitto Broker tai {self.broker_host}:{self.broker_port}")
            self.subscribe(
                [
                    (TOPIC_FUSION_LEVEL, 1),
                    (TOPIC_ESP32_SENSORS, 0),
                    (TOPIC_VISION_METRICS, 0),
                    (TOPIC_EVENTS, 1),
                    (TOPIC_VISION_EVENTS, 1),
                    (TOPIC_ESP32_EVENTS, 1),
                    (TOPIC_ESP32_CMD, 1),
                ]
            )
        else:
            print(f"[dashboard.mqtt] Ket noi broker that bai: code {reason_code}", file=sys.stderr)

    def _handle_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None) -> None:
        print(f"[dashboard.mqtt] Ngat ket noi khoi broker (code {reason_code}). Dang thu ket noi lai...")

    def _handle_message(self, client, userdata, msg) -> None:
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return

        now = time.time()

        if topic == TOPIC_FUSION_LEVEL:
            self.latest_fusion = payload
            self.last_fusion_ts = now
            action = payload.get("action", "none")
            if action == "lock":
                self.engine_locked = True
            elif action == "unlock":
                self.engine_locked = False
            self._on_fusion_cycle(payload)

        elif topic == TOPIC_ESP32_SENSORS:
            self.latest_esp32 = payload
            self.last_esp32_ts = now
            # Nếu Fusion Engine chưa chạy, tự động phát sóng telemetry cảm biến xuống Web
            # để người dùng thấy ngay nhiệt độ, nồng độ cồn, ánh sáng nhảy realtime
            if not self.is_fusion_online():
                self._on_standalone_sensor_update()

        elif topic == TOPIC_VISION_METRICS:
            self.latest_vision = payload
            self.last_vision_ts = now
            if not self.is_fusion_online():
                self._on_standalone_sensor_update()

        elif topic in (TOPIC_EVENTS, TOPIC_VISION_EVENTS, TOPIC_ESP32_EVENTS):
            self._on_event_message(topic, payload)

        elif topic == TOPIC_ESP32_CMD:
            cmd = payload.get("cmd", "")
            if cmd == "lock":
                self.engine_locked = True
            elif cmd == "unlock":
                self.engine_locked = False
            # Forward cmd tới websocket
            self._dispatch_ws({"type": "command", "data": payload})

    def is_fusion_online(self) -> bool:
        """Kiểm tra node Fusion có đang hoạt động trong vòng 3 giây gần nhất không."""
        return (time.time() - self.last_fusion_ts) <= 3.0 if self.last_fusion_ts > 0 else False

    def build_current_telemetry(self) -> TelemetryRecord:
        """Tổng hợp bản ghi Telemetry mới nhất từ các nguồn Fusion, ESP32 và Vision."""
        now_ms = int(time.time() * 1000)
        return TelemetryRecord(
            ts=self.latest_fusion.get("ts", now_ms),
            risk=float(self.latest_fusion.get("risk", 0.0)),
            band=str(self.latest_fusion.get("band", "SAFE")),  # type: ignore
            drivers=list(self.latest_fusion.get("drivers", [])),
            action=str(self.latest_fusion.get("action", "none")),
            perclos_60s=float(self.latest_vision.get("perclos_60s", 0.0)),
            ear=float(self.latest_vision.get("ear", 0.0)),
            cles_dur_ms=float(self.latest_vision.get("cles_dur_ms", 0.0)),
            mar=float(self.latest_vision.get("mar", 0.0)),
            yawn_per_min=float(self.latest_vision.get("yawn_per_min", 0.0)),
            head_pitch_deg=float(self.latest_vision.get("head_pitch_deg", 0.0)),
            head_drop=bool(self.latest_vision.get("head_drop", False)),
            alcohol_g_l=float(self.latest_esp32.get("alcohol_g_l", 0.0)),
            alco_level=int(self.latest_esp32.get("alco_level", 0)),
            temp_c=float(self.latest_esp32.get("temp_c", 25.0)),
            lux_mode=str(self.latest_esp32.get("lux_mode", "day")),
            ldr_pct=int(self.latest_esp32.get("ldr_pct", 50)),
            rssi=int(self.latest_esp32.get("rssi", -60)),
        )

    def _on_standalone_sensor_update(self) -> None:
        """Cập nhật dữ liệu tức thời xuống Web khi Fusion Engine chưa khởi động."""
        record = self.build_current_telemetry()
        try:
            self.db.insert_telemetry(record)
        except Exception:
            pass

        ws_msg = {
            "type": "telemetry",
            "data": record.as_dict(),
            "status": self.get_system_status().as_dict(),
        }
        self._dispatch_ws(ws_msg)

    def _on_fusion_cycle(self, fusion_payload: dict) -> None:
        """Được gọi mỗi khi nhận được kết quả Fusion 1Hz -> Lưu DB và Broadcast WS."""
        record = self.build_current_telemetry()

        # 1. Lưu SQLite Database
        try:
            self.db.insert_telemetry(record)
        except Exception as e:
            print(f"[dashboard.db] Loi luu telemetry: {e}", file=sys.stderr)

        # 2. Phát sóng WebSocket tới Frontend
        ws_msg = {
            "type": "telemetry",
            "data": record.as_dict(),
            "status": self.get_system_status().as_dict(),
        }
        self._dispatch_ws(ws_msg)

    def _on_event_message(self, topic: str, payload: dict) -> None:
        """Lưu trữ và phát sóng sự kiện rủi ro tức thời."""
        now_ms = int(time.time() * 1000)
        source = "vision" if "vision" in topic else ("esp32" if "esp32" in topic else "fusion")
        evt_name = payload.get("name") or payload.get("event") or payload.get("kind") or "risk_alert"

        record = EventRecord(
            ts=payload.get("ts", now_ms),
            source=source,
            event_name=str(evt_name),
            risk=float(payload.get("risk", self.latest_fusion.get("risk", 0.0))),
            band=str(payload.get("band", self.latest_fusion.get("band", "SAFE"))),  # type: ignore
            action=str(payload.get("action", self.latest_fusion.get("action", "none"))),
            details=payload,
        )

        try:
            self.db.insert_event(record)
        except Exception as e:
            print(f"[dashboard.db] Loi luu event: {e}", file=sys.stderr)

        self._dispatch_ws({"type": "event", "data": record.as_dict()})

    def _dispatch_ws(self, msg: dict) -> None:
        """Đẩy dữ liệu vào event loop của asyncio để phát qua WebSocket."""
        if self.loop is not None and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.ws_manager.broadcast(msg), self.loop)

    def get_system_status(self) -> SystemStatus:
        """Tổng hợp tình trạng node online / offline theo thời gian thực."""
        now = time.time()
        return SystemStatus(
            ts=int(now * 1000),
            vision_online=(now - self.last_vision_ts) <= 2.0 if self.last_vision_ts > 0 else False,
            esp32_online=(now - self.last_esp32_ts) <= 5.0 if self.last_esp32_ts > 0 else False,
            fusion_online=(now - self.last_fusion_ts) <= 3.0 if self.last_fusion_ts > 0 else False,
            mqtt_online=self.is_connected(),
            current_risk=float(self.latest_fusion.get("risk", 0.0)),
            current_band=str(self.latest_fusion.get("band", "SAFE")),  # type: ignore
            current_action=str(self.latest_fusion.get("action", "none")),
            engine_locked=self.engine_locked,
            active_drivers=list(self.latest_fusion.get("drivers", [])),
        )

    def send_command(self, cmd_dict: dict) -> bool:
        """Publish lệnh điều khiển xuống MQTT topic ds/esp32/cmd."""
        payload_str = json.dumps(cmd_dict)
        info = self.publish(TOPIC_ESP32_CMD, payload_str, qos=1)
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def connect_broker(self, timeout: float = 3.0) -> bool:
        """Kết nối tới broker và chạy thread loop."""
        try:
            self.connect(self.broker_host, self.broker_port, keepalive=60)
            self.loop_start()
        except Exception as e:
            print(f"[dashboard.mqtt] Khong the ket noi broker: {e}", file=sys.stderr)
            return False

        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            if self.is_connected():
                return True
            time.sleep(0.05)
        return self.is_connected()

    def disconnect_broker(self) -> None:
        """Dừng network thread và ngắt kết nối."""
        try:
            self.loop_stop()
            self.disconnect()
        except Exception:
            pass
