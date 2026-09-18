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
from policy import (  # noqa: E402
    FusionResult,
    MamdaniPolicy,
    RuleWeightedPolicy,
    TOPIC_ESP32_CMD,
    TOPIC_EVENTS,
    TOPIC_FUSION_LEVEL,
)

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


# Tách riêng module cache để tuân thủ kiến trúc chia nhỏ Single Responsibility
from cache import (  # noqa: E402
    Esp32Telemetry,
    TelemetryCache,
    TelemetryEvent,
    VisionTelemetry,
)

# Re-export để tương thích ngược 100%
__all__ = [
    "Esp32Telemetry",
    "TelemetryCache",
    "TelemetryEvent",
    "VisionTelemetry",
    "FusionSubscriber",
    "main",
]


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
        policy: MamdaniPolicy | RuleWeightedPolicy | None = None,
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
        self.policy = policy or MamdaniPolicy()
        self.level_publish_count = 0
        self.cmd_publish_count = 0
        self.last_fusion_result: FusionResult | None = None

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

    def step_fusion(self) -> FusionResult:
        """Thực hiện 1 chu kỳ suy luận Fusion, publish ds/fusion/level (QoS1 retain) và phát lệnh ds/esp32/cmd."""
        v_dict = self.cache.vision.as_dict() if self.cache.vision else None
        e_dict = self.cache.esp32.as_dict() if self.cache.esp32 else None
        v_online = self.cache.is_vision_online()
        e_online = self.cache.is_esp32_online()

        res = self.policy.evaluate(
            vision_data=v_dict,
            esp32_data=e_dict,
            vision_online=v_online,
            esp32_online=e_online,
        )
        self.last_fusion_result = res

        # Publish ds/fusion/level (QoS 1, retain=True theo spec §5, §8)
        lvl_payload = json.dumps(res.as_level_payload())
        self.publish(TOPIC_FUSION_LEVEL, lvl_payload, qos=1, retain=True)
        self.level_publish_count += 1

        # Phát lệnh ds/esp32/cmd và log ds/events nếu có lệnh
        if res.cmd_payload is not None:
            cmd_str = json.dumps(res.cmd_payload)
            self.publish(TOPIC_ESP32_CMD, cmd_str, qos=1)
            self.cmd_publish_count += 1
            print(f"\n>>> [FUSION PHAT LENH] {cmd_str} (Risk={res.risk}, Band={res.band}, Drivers={res.drivers})")

            # Log sự kiện cảnh báo lên ds/events (spec §5, §9)
            evt_dict = {
                "ts": res.ts,
                "risk": res.risk,
                "band": res.band,
                "drivers": res.drivers,
                "cmd": res.cmd_payload,
                "metrics": {
                    "vision": v_dict,
                    "esp32": e_dict,
                },
            }
            self.publish(TOPIC_EVENTS, json.dumps(evt_dict), qos=1)

        return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fusion Telemetry Ingestion & Engine Node (FUS-01/02/03)")
    parser.add_argument("--host", default="127.0.0.1", help="Mosquitto broker host (override config)")
    parser.add_argument("--port", type=int, default=1883, help="Mosquitto broker port (override config)")
    parser.add_argument("--policy", choices=["mamdani", "rule_weighted"], default="mamdani", help="Chon thuat toan Fusion (mac dinh: mamdani)")
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
    pol = MamdaniPolicy() if args.policy == "mamdani" else RuleWeightedPolicy()
    sub = FusionSubscriber(
        broker_host=broker_host,
        broker_port=broker_port,
        client_id=cfg.client_id,
        cache=cache,
        vision_timeout_s=cfg.vision_timeout_s,
        esp32_timeout_s=cfg.esp32_timeout_s,
        policy=pol,
    )

    print(f"[fusion.engine] Dang ket noi Mosquitto broker tai {broker_host}:{broker_port}...")
    if not sub.connect_broker(timeout=3.0):
        print(f"[fusion.engine] CANH BAO: Chua ket noi duoc broker {broker_host}:{broker_port}", file=sys.stderr)

    print(f"[fusion.engine] Da san sang thuc hien suy luan Fusion ({args.policy.upper()}) 1Hz. Nhan Ctrl+C de dung.\n")

    t_start = time.perf_counter()
    try:
        while True:
            time.sleep(1.0)
            now = time.perf_counter()

            # Thực hiện chu kỳ suy luận Fusion và publish ds/fusion/level + ds/esp32/cmd
            res = sub.step_fusion()

            v_ok = cache.is_vision_online()
            e_ok = cache.is_esp32_online()

            print(
                f"[FUSION 1Hz #{sub.level_publish_count:03d}] "
                f"RISK: {res.risk:4.1f} ({res.band:<8}) | "
                f"Drivers: {str(res.drivers):<28} | "
                f"Action: {res.action:<12} | "
                f"Vision:{'ON' if v_ok else 'OFF'} ESP:{'ON' if e_ok else 'OFF'}"
            )

            if args.duration > 0 and (now - t_start) >= args.duration:
                print(f"\n[fusion.engine] Da chay het thoi gian dinh san ({args.duration}s).")
                break

    except KeyboardInterrupt:
        print("\n[fusion.engine] Dung boi nguoi dung (Ctrl+C).")
    finally:
        sub.disconnect_broker()

    print("[fusion.engine] Da dung tien trinh an toan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
