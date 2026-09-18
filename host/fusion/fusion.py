"""Telemetry Ingestion & Data Caching for Fusion Subsystem (FUS-01).

Lắng nghe đồng thời 3 luồng MQTT:
    - ds/esp32/sensors (1Hz): Telemetry từ cảm biến ESP32 (MQ-3, NTC, LDR, FSM)
    - ds/esp32/events (tức thời): Sự kiện cồn tức thời từ ESP32
    - ds/vision/metrics (1Hz): Metrics từ Vision AI (PERCLOS, CLES, MAR, Pitch)
    - ds/vision/events (tức thời): Sự kiện thị giác (microsleep, yawn, head_drop, face_lost)

Quản lý bộ nhớ đệm (TelemetryCache) kèm timestamp cho từng nguồn, kiểm tra độ tươi
của dữ liệu (vision_online <= 2s, esp32_online <= 5s) và xử lý an toàn chống crash
khi gói tin JSON bị lỗi hoặc thiếu trường (try/except KeyError/TypeError/JSONDecodeError).
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Callable

import paho.mqtt.client as mqtt

# Cho phép import các module cùng cấp trong thư mục host/fusion
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FusionConfig, load_config  # noqa: E402

# Topic MQTT theo chuẩn spec §5 / spec §6.1
TOPIC_ESP32_SENSORS = "ds/esp32/sensors"
TOPIC_ESP32_EVENTS = "ds/esp32/events"
TOPIC_VISION_METRICS = "ds/vision/metrics"
TOPIC_VISION_EVENTS = "ds/vision/events"

# Tương thích Paho MQTT v1 và v2
try:
    _CALLBACK_API = mqtt.CallbackAPIVersion.VERSION2
except AttributeError:  # pragma: no cover
    _CALLBACK_API = None


@dataclass
class VisionTelemetry:
    """Cache bản tin metrics mới nhất từ Vision subsystem (spec §5)."""

    ts: float = 0.0
    face: bool = False
    ear: float = 0.0
    perclos_60s: float = 0.0
    cles_dur_ms: float = 0.0
    mar: float = 0.0
    yawn_per_min: float = 0.0
    head_pitch_deg: float = 0.0
    head_drop: bool = False
    lux_mode: str = "day"
    received_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["received_at"] = round(self.received_at, 2)
        return d


@dataclass
class Esp32Telemetry:
    """Cache bản tin cảm biến mới nhất từ ESP32-S3 (spec §5)."""

    ts: float = 0.0
    mq3_ao_v: float = 0.0
    rs_r0: float = 1.0
    alcohol_g_l: float = 0.0
    temp_c: float = 25.0
    ldr_pct: int = 50
    lux_mode: str = "day"
    rssi: int = -60
    uptime_s: int = 0
    degraded: bool = False
    received_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["received_at"] = round(self.received_at, 2)
        return d


@dataclass
class TelemetryEvent:
    """Sự kiện tức thời ghi nhận từ Vision hoặc ESP32."""

    source: str           # "vision" | "esp32"
    name: str             # Tên sự kiện: microsleep, yawn, alcohol_level2,...
    details: dict | str   # Dữ liệu bổ sung
    ts: float = 0.0
    received_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "name": self.name,
            "details": self.details,
            "ts": round(self.ts, 1),
            "received_at": round(self.received_at, 2),
        }


class TelemetryCache:
    """Bộ nhớ đệm lưu trữ dữ liệu telemetry và lịch sử sự kiện của Fusion."""

    def __init__(
        self,
        max_events: int = 100,
        history_len: int = 60,
        vision_timeout_s: float | None = 2.0,
        esp32_timeout_s: float | None = 5.0,
    ) -> None:
        self.vision_timeout_s = vision_timeout_s if vision_timeout_s is not None else 2.0
        self.esp32_timeout_s = esp32_timeout_s if esp32_timeout_s is not None else 5.0

        self.vision: VisionTelemetry | None = None
        self.esp32: Esp32Telemetry | None = None
        self.last_vision_ts: float = 0.0
        self.last_esp32_ts: float = 0.0

        self.events: deque[TelemetryEvent] = deque(maxlen=max_events)
        self.history_vision: deque[VisionTelemetry] = deque(maxlen=history_len)
        self.history_esp32: deque[Esp32Telemetry] = deque(maxlen=history_len)

    def update_vision(self, payload: dict) -> VisionTelemetry:
        """Cập nhật bản tin Vision metrics với xử lý an toàn (fallback default)."""
        now = time.time()
        try:
            entry = VisionTelemetry(
                ts=float(payload.get("ts", now * 1000)),
                face=bool(payload.get("face", False)),
                ear=float(payload.get("ear", 0.0)),
                perclos_60s=float(payload.get("perclos_60s", 0.0)),
                cles_dur_ms=float(payload.get("cles_dur_ms", 0.0)),
                mar=float(payload.get("mar", 0.0)),
                yawn_per_min=float(payload.get("yawn_per_min", 0.0)),
                head_pitch_deg=float(payload.get("head_pitch_deg", 0.0)),
                head_drop=bool(payload.get("head_drop", False)),
                lux_mode=str(payload.get("lux_mode", "day")),
                received_at=now,
            )
        except (ValueError, TypeError) as e:
            print(f"[fusion.cache] Loi ep kieu VisionTelemetry: {e}. Dung default.", file=sys.stderr)
            entry = VisionTelemetry(received_at=now)

        self.vision = entry
        self.last_vision_ts = now
        self.history_vision.append(entry)
        return entry

    def update_esp32(self, payload: dict) -> Esp32Telemetry:
        """Cập nhật bản tin ESP32 sensors với xử lý an toàn (fallback default)."""
        now = time.time()
        try:
            entry = Esp32Telemetry(
                ts=float(payload.get("ts", now * 1000)),
                mq3_ao_v=float(payload.get("mq3_ao_v", 0.0)),
                rs_r0=float(payload.get("rs_r0", 1.0)),
                alcohol_g_l=float(payload.get("alcohol_g_l", 0.0)),
                temp_c=float(payload.get("temp_c", 25.0)),
                ldr_pct=int(payload.get("ldr_pct", 50)),
                lux_mode=str(payload.get("lux_mode", "day")),
                rssi=int(payload.get("rssi", -60)),
                uptime_s=int(payload.get("uptime_s", 0)),
                degraded=bool(payload.get("degraded", False)),
                received_at=now,
            )
        except (ValueError, TypeError) as e:
            print(f"[fusion.cache] Loi ep kieu Esp32Telemetry: {e}. Dung default.", file=sys.stderr)
            entry = Esp32Telemetry(received_at=now)

        self.esp32 = entry
        self.last_esp32_ts = now
        self.history_esp32.append(entry)
        return entry

    def add_event(self, source: str, name: str, details: dict | str, ts: float | None = None) -> TelemetryEvent:
        """Ghi nhận một sự kiện vào hàng đợi sự kiện."""
        now = time.time()
        event_ts = ts if ts is not None else (now * 1000)
        ev = TelemetryEvent(source=source, name=name, details=details, ts=event_ts, received_at=now)
        self.events.append(ev)
        return ev

    def is_vision_online(self, timeout: float | None = None) -> bool:
        """Kiểm tra nguồn Vision có đang hoạt động trong timeout không."""
        t = timeout if timeout is not None else self.vision_timeout_s
        return (time.time() - self.last_vision_ts) <= t if self.last_vision_ts > 0 else False

    def is_esp32_online(self, timeout: float | None = None) -> bool:
        """Kiểm tra node ESP32 có đang gửi dữ liệu trong timeout không."""
        t = timeout if timeout is not None else self.esp32_timeout_s
        return (time.time() - self.last_esp32_ts) <= t if self.last_esp32_ts > 0 else False

    def get_snapshot(self) -> dict:
        """Tổng hợp toàn bộ trạng thái hiện tại thành snapshot dictionary."""
        return {
            "ts": int(time.time() * 1000),
            "vision_online": self.is_vision_online(),
            "esp32_online": self.is_esp32_online(),
            "vision": self.vision.as_dict() if self.vision else None,
            "esp32": self.esp32.as_dict() if self.esp32 else None,
            "events_count": len(self.events),
            "latest_event": self.events[-1].as_dict() if self.events else None,
        }


class FusionSubscriber(mqtt.Client):
    """Subscriber MQTT chuyên thu nạp telemetry từ Vision và ESP32."""

    def __init__(
        self,
        broker_host: str = "127.0.0.1",
        broker_port: int = 1883,
        client_id: str = "fusion_subscriber",
        cache: TelemetryCache | None = None,
        vision_timeout_s: float | None = None,
        esp32_timeout_s: float | None = None,
    ) -> None:
        if _CALLBACK_API is not None:
            super().__init__(_CALLBACK_API, client_id=client_id)
        else:  # pragma: no cover
            super().__init__(client_id=client_id)

        self.broker_host = broker_host
        self.broker_port = broker_port
        self.cache = cache or TelemetryCache(
            vision_timeout_s=vision_timeout_s,
            esp32_timeout_s=esp32_timeout_s,
        )

        # Hooks cho module Fusion Engine (FUS-02 / FUS-03)
        self.on_vision_update: Callable[[VisionTelemetry], None] | None = None
        self.on_esp32_update: Callable[[Esp32Telemetry], None] | None = None
        self.on_event_received: Callable[[TelemetryEvent], None] | None = None

        # Callbacks MQTT
        self.on_connect = self._handle_connect
        self.on_disconnect = self._handle_disconnect
        self.on_message = self._handle_message

    @property
    def connected(self) -> bool:
        return self.is_connected()

    def _handle_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        is_ok = (reason_code == 0) if isinstance(reason_code, int) else (not reason_code.is_failure)
        if is_ok:
            print(f"[fusion.sub] Da ket noi Broker tai {self.broker_host}:{self.broker_port}")
            # Lắng nghe đồng thời 4 topic (spec §5)
            topics = [
                (TOPIC_ESP32_SENSORS, 0),
                (TOPIC_ESP32_EVENTS, 1),
                (TOPIC_VISION_METRICS, 0),
                (TOPIC_VISION_EVENTS, 1),
            ]
            self.subscribe(topics)
            print("[fusion.sub] Da subscribe cac topic: ESP32 sensors/events & Vision metrics/events")
        else:
            print(f"[fusion.sub] Ket noi broker that bai: code {reason_code}", file=sys.stderr)

    def _handle_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None) -> None:
        print(f"[fusion.sub] Ngat ket noi broker ({reason_code}). Dang tu dong ket noi lai...")

    def _handle_message(self, client, userdata, msg) -> None:
        """Xử lý gói tin nhận được an toàn tuyệt đối chống crash."""
        payload_str = ""
        try:
            payload_str = msg.payload.decode("utf-8")
            data = json.loads(payload_str)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"[fusion.sub] CANH BAO: Bo qua goi tin JSON loi tai topic '{msg.topic}': {e}", file=sys.stderr)
            return

        topic = msg.topic
        try:
            if topic == TOPIC_VISION_METRICS:
                entry_v = self.cache.update_vision(data)
                if self.on_vision_update:
                    self.on_vision_update(entry_v)

            elif topic == TOPIC_ESP32_SENSORS:
                entry_e = self.cache.update_esp32(data)
                if self.on_esp32_update:
                    self.on_esp32_update(entry_e)

            elif topic == TOPIC_VISION_EVENTS:
                kind = data.get("kind", "unknown_vision_event")
                ev = self.cache.add_event(source="vision", name=kind, details=data, ts=data.get("t_ms"))
                print(f"[fusion.sub EVENT] VISION -> {kind.upper()} ({data.get('duration_ms', 0):.0f}ms)")
                if self.on_event_received:
                    self.on_event_received(ev)

            elif topic == TOPIC_ESP32_EVENTS:
                evt_name = data.get("event", "unknown_esp32_event")
                ev = self.cache.add_event(source="esp32", name=evt_name, details=data, ts=data.get("ts"))
                print(f"[fusion.sub EVENT] ESP32 -> {evt_name.upper()} ({data.get('raw', '')})")
                if self.on_event_received:
                    self.on_event_received(ev)

        except Exception as e:
            # Phòng ngừa lỗi lập trình bất ngờ, đảm bảo subscriber không chết
            print(f"[fusion.sub] LOI xu ly noi dung topic '{topic}': {e}", file=sys.stderr)

    def connect_broker(self, timeout: float = 3.0) -> bool:
        """Kết nối broker và khởi chạy network loop nền."""
        try:
            self.connect(self.broker_host, self.broker_port, keepalive=60)
            self.loop_start()
        except Exception as e:
            print(f"[fusion.sub] Loi ket noi: {e}", file=sys.stderr)
            return False

        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            if self.is_connected():
                return True
            time.sleep(0.05)
        return self.is_connected()

    def disconnect_broker(self) -> None:
        """Dừng network loop và ngắt kết nối."""
        try:
            self.loop_stop()
            self.disconnect()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fusion Telemetry Ingestion Node (FUS-01)")
    parser.add_argument("--host", default="127.0.0.1", help="Mosquitto broker host (override config)")
    parser.add_argument("--port", type=int, default=1883, help="Mosquitto broker port (override config)")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: autodetect)")
    parser.add_argument("--duration", type=float, default=0.0, help="Thoi gian chay (0 = lien tuc)")
    args = parser.parse_args(argv)

    # 1. Nạp cấu hình (config.yaml > defaults). CLI args override host/port.
    cfg: FusionConfig = load_config(args.config)
    broker_host = args.host if args.host != "127.0.0.1" else cfg.broker_host
    broker_port = args.port if args.port != 1883 else cfg.broker_port

    cache = TelemetryCache(
        vision_timeout_s=cfg.vision_timeout_s,
        esp32_timeout_s=cfg.esp32_timeout_s,
    )
    sub = FusionSubscriber(
        broker_host=broker_host,
        broker_port=broker_port,
        client_id=cfg.client_id,
        cache=cache,
        vision_timeout_s=cfg.vision_timeout_s,
        esp32_timeout_s=cfg.esp32_timeout_s,
    )

    print(f"[fusion.ingestion] Dang ket noi Mosquitto broker tai {broker_host}:{broker_port}...")
    if not sub.connect_broker(timeout=3.0):
        print(f"[fusion.ingestion] CANH BAO: Chua ket noi duoc broker {broker_host}:{broker_port}", file=sys.stderr)

    print("[fusion.ingestion] Da san sang lang nghe 3 luong du lieu. Nhan Ctrl+C de dung.\n")

    t_start = time.perf_counter()
    try:
        while True:
            time.sleep(1.0)
            now = time.perf_counter()

            # Trạng thái Vision
            v_ok = cache.is_vision_online()
            if v_ok and cache.vision:
                v_str = f"ONLINE (P78:{cache.vision.perclos_60s*100:4.1f}%, EAR:{cache.vision.ear:.3f}, CLES:{cache.vision.cles_dur_ms:.0f}ms)"
            else:
                v_str = "OFFLINE (>2s)"

            # Trạng thái ESP32
            e_ok = cache.is_esp32_online()
            if e_ok and cache.esp32:
                e_str = f"ONLINE (BrAC:{cache.esp32.alcohol_g_l:.2f}g/L, T:{cache.esp32.temp_c}°C, LUX:{cache.esp32.lux_mode.upper()})"
            else:
                e_str = "OFFLINE (>5s)"

            n_ev = len(cache.events)
            print(f"[FUSION CACHE] Vision: {v_str:<45} | ESP32: {e_str:<45} | Events: {n_ev:02d}")

            if args.duration > 0 and (now - t_start) >= args.duration:
                print(f"\n[fusion.ingestion] Da chay het thoi gian dinh san ({args.duration}s).")
                break

    except KeyboardInterrupt:
        print("\n[fusion.ingestion] Dung boi nguoi dung (Ctrl+C).")
    finally:
        sub.disconnect_broker()

    print("[fusion.ingestion] Da dung tien trinh an toan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
