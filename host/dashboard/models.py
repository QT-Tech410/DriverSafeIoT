"""Data Models & Schemas for Dashboard Subsystem (DASH-01).

Định nghĩa cấu trúc dữ liệu cho:
- Bản ghi Telemetry lưu trữ trong SQLite (bảng telemetry)
- Bản ghi Sự kiện rủi ro lưu trữ trong SQLite (bảng events)
- Gói tin phát sóng WebSocket (WS Broadcast)
- Request gửi lệnh điều khiển (REST API)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any, Literal

BandType = Literal["SAFE", "WARN", "ALARM", "CRITICAL"]


@dataclass
class TelemetryRecord:
    """Bản ghi tổng hợp telemetry 1Hz từ Vision, ESP32 và Fusion."""

    ts: int | float = field(default_factory=lambda: int(time.time() * 1000))
    risk: float = 0.0
    band: BandType = "SAFE"
    drivers: list[str] = field(default_factory=list)
    action: str = "none"

    # Vision metrics
    perclos_60s: float = 0.0
    ear: float = 0.0
    cles_dur_ms: float = 0.0
    mar: float = 0.0
    yawn_per_min: float = 0.0
    head_pitch_deg: float = 0.0
    head_drop: bool = False

    # ESP32 sensor telemetry
    alcohol_g_l: float = 0.0
    alco_level: int = 0
    temp_c: float = 25.0
    lux_mode: str = "day"
    ldr_pct: int = 50
    rssi: int = -60

    id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class EventRecord:
    """Bản ghi sự kiện rủi ro tức thời (microsleep, cồn cao, khóa động cơ...)."""

    ts: int | float = field(default_factory=lambda: int(time.time() * 1000))
    source: str = "fusion"          # "vision" | "esp32" | "fusion"
    event_name: str = "risk_alert"  # microsleep, yawn, alcohol_level2, lock,...
    risk: float = 0.0
    band: BandType = "SAFE"
    action: str = "none"
    details: dict[str, Any] = field(default_factory=dict)
    id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class SystemStatus:
    """Trạng thái tổng quan của toàn bộ hệ thống DriverSafe-IoT."""

    ts: int = field(default_factory=lambda: int(time.time() * 1000))
    vision_online: bool = False
    esp32_online: bool = False
    fusion_online: bool = False
    mqtt_online: bool = False
    current_risk: float = 0.0
    current_band: BandType = "SAFE"
    current_action: str = "none"
    engine_locked: bool = False
    active_drivers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
