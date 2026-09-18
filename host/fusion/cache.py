"""Telemetry Caching & Freshness Management (FUS-01 / FUS-04).

Quản lý bộ nhớ đệm luồng dữ liệu thời gian thực từ Vision và ESP32:
- Cache bản tin metrics mới nhất từ Vision subsystem
- Cache bản tin cảm biến mới nhất từ ESP32-S3
- Quản lý danh sách sự kiện tức thời
- Đánh giá độ tươi (freshness) và timeout kết nối (spec §8, §3)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any


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
    alco_level: int = 0
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
    details: dict[str, Any] = field(default_factory=dict)
    ts: float | int = 0
    received_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["received_at"] = round(self.received_at, 2)
        return d


class TelemetryCache:
    """Bộ nhớ đệm lưu trữ trạng thái telemetry mới nhất từ cả hai nguồn."""

    def __init__(
        self,
        vision_timeout_s: float = 2.0,
        esp32_timeout_s: float = 5.0,
        max_events_history: int = 100,
    ) -> None:
        self.vision_timeout_s = vision_timeout_s
        self.esp32_timeout_s = esp32_timeout_s
        self.max_events_history = max_events_history

        self.vision: VisionTelemetry | None = None
        self.esp32: Esp32Telemetry | None = None
        self.events: list[TelemetryEvent] = []

        self.last_vision_ts: float = 0.0
        self.last_esp32_ts: float = 0.0

    def update_vision(self, data: dict) -> VisionTelemetry:
        """Cập nhật dữ liệu từ ds/vision/metrics (chống crash nếu thiếu trường)."""
        now = time.time()
        self.last_vision_ts = now

        entry = VisionTelemetry(
            ts=float(data.get("ts", now * 1000)),
            face=bool(data.get("face", False)),
            ear=float(data.get("ear", 0.0)),
            perclos_60s=float(data.get("perclos_60s", 0.0)),
            cles_dur_ms=float(data.get("cles_dur_ms", 0.0)),
            mar=float(data.get("mar", 0.0)),
            yawn_per_min=float(data.get("yawn_per_min", 0.0)),
            head_pitch_deg=float(data.get("head_pitch_deg", 0.0)),
            head_drop=bool(data.get("head_drop", False)),
            lux_mode=str(data.get("lux_mode", "day")),
            received_at=now,
        )
        self.vision = entry
        return entry

    def update_esp32(self, data: dict) -> Esp32Telemetry:
        """Cập nhật dữ liệu từ ds/esp32/sensors (chống crash nếu thiếu trường)."""
        now = time.time()
        self.last_esp32_ts = now

        alco_val = float(data.get("alcohol_g_l", 0.0))
        default_lvl = 2 if alco_val >= 0.3 else (1 if alco_val >= 0.1 else 0)

        entry = Esp32Telemetry(
            ts=float(data.get("ts", now * 1000)),
            mq3_ao_v=float(data.get("mq3_ao_v", 0.0)),
            rs_r0=float(data.get("rs_r0", 1.0)),
            alcohol_g_l=alco_val,
            alco_level=int(data.get("alco_level", default_lvl)),
            temp_c=float(data.get("temp_c", 25.0)),
            ldr_pct=int(data.get("ldr_pct", 50)),
            lux_mode=str(data.get("lux_mode", "day")),
            rssi=int(data.get("rssi", -60)),
            uptime_s=int(data.get("uptime_s", 0)),
            degraded=bool(data.get("degraded", False)),
            received_at=now,
        )
        self.esp32 = entry
        return entry

    def add_event(self, source: str, name: str, details: dict | None = None, ts: float | None = None) -> TelemetryEvent:
        """Thêm sự kiện mới vào lịch sử."""
        now = time.time()
        evt = TelemetryEvent(
            source=source,
            name=name,
            details=details or {},
            ts=ts if ts is not None else int(now * 1000),
            received_at=now,
        )
        self.events.append(evt)
        if len(self.events) > self.max_events_history:
            self.events.pop(0)
        return evt

    def is_vision_online(self, timeout: float | None = None) -> bool:
        """Kiểm tra nguồn Vision có đang gửi dữ liệu trong timeout không (spec §8)."""
        t = timeout if timeout is not None else self.vision_timeout_s
        return (time.time() - self.last_vision_ts) <= t if self.last_vision_ts > 0 else False

    def is_esp32_online(self, timeout: float | None = None) -> bool:
        """Kiểm tra node ESP32 có đang gửi dữ liệu trong timeout không (spec §3)."""
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
