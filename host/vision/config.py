"""Viết cấu hình cho Vision subsystem (VIS-02..VIS-05).

File này thực hiện hai vai trò:
1. Tải config từ YAML (config.yaml) với fallback default values
2. Cung cấp các topic MQTT từ spec §5 / spec §6.1 / spec §6.4
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class MqttConfig:
    """Cấu hình MQTT cho Vision subsystem."""

    broker_host: str = "127.0.0.1"
    broker_port: int = 1883
    client_id: str = "vision_publisher"
    keepalive: int = 60


@dataclass
class VisionTelemetryConfig:
    """Schema JSON cho `ds/vision/metrics` theo spec §5 (10 chỉ số)."""

    topic: str = "ds/vision/metrics"
    qos: int = 0  # QoS0 là định kỳ, không cần đảm bảo

    # Quốc tắc JSON (tất cả numeric: int/float, bool, str)
    required_fields = [
        "ts",
        "face",
        "ear",
        "perclos_60s",
        "cles_dur_ms",
        "mar",
        "yawn_per_min",
        "head_pitch_deg",
        "head_drop",
        "lux_mode",
    ]


@dataclass
class VisionEventsConfig:
    """Topic + QoS cho vision events.

    LƯU Ý KIẾN TRÚC (extension over spec §5):
      - Spec §5 table KHÔNG có `ds/vision/events`, chỉ `ds/esp32/events`.
      - Spec §6.1 nói "publish ngay khi có event (microsleep, yawn, head_drop)"
      - Vision subsystem quản lý event lifecycle độc lập, nên extract thành topic riêng.
    """

    topic: str = "ds/vision/events"
    qos: int = 1  # Event phải đảm bảo đến đích (no-loss)
    data_class: str = "VisionEvent"  # Tên class dataclass trong temporal.py


@dataclass
class Esp32Config:
    """Topic để subscribe lux_mode từ ESP32 sensor (spec §6.4)."""

    topic_sensors: str = "ds/esp32/sensors"
    topic_events: str = "ds/esp32/events"
    qos_sensors: int = 0
    qos_events: int = 1


@dataclass
class TemporalConfig:
    """Cấu hình class TemporalMetrics (VIS-04)."""

    # Thời gian buffer (giây)
    buffer_seconds: int = 60
    # Giảm mẫu cho PERCLOS (thang 10Hz từ 300 FPS)
    sample_freq_hz: int = 10

    # Ngưỡng event (giây = ms / 1000)
    T_CLOSED_SEC: float = 0.22  # Eye closure (chuẩn hóa người châu Á / mắt mí lót)
    MICROSLEEP_MS: int = 1200  # ≥ 1200ms eye closure equals microsleep (1.2s giấc ngủ trắng)
    EYE_CLOSURE_MS: int = 250  # ≥ 250ms eye closure is an event
    HEAD_DROP_MS: int = 800  # Miễn tối thiếu 800ms

    # Yawn detection
    YAWN_MS: int = 400  # MAR ≥ T_Yawn duy trì ≥ 400ms
    HEAD_DROP_DEG: float = 15.0  # >15° from neutral

    # Face lost detection
    FACE_LOST_MS: int = 2000  # Không thấy mặt ≥ 2s


@dataclass
class VisionConfig:
    """Tổng hợp cấu hình cho Vision subsystem."""

    mqtt: MqttConfig
    telemetry: VisionTelemetryConfig
    events: VisionEventsConfig
    esp32: Esp32Config
    temporal: TemporalConfig

    models_dir: str = ""  # Path đến thư mục chứa model, override bằng --model


# Đường dẫn mặc định: config.yaml ở root của repo (cùng cấp với host/)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config(config_path: str | None = None) -> VisionConfig:
    """Tải config từ YAML hoặc dùng default values.

    Thứ tự ưu tiên:
        1. Path chỉ định qua `config_path`
        2. `config.yaml` ở gốc repository
        3. Hard-coded defaults
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            return _merge_defaults(cfg)

    # No file — use all defaults
    return VisionConfig(
        mqtt=MqttConfig(),
        telemetry=VisionTelemetryConfig(),
        events=VisionEventsConfig(),
        esp32=Esp32Config(),
        temporal=TemporalConfig(),
        models_dir=str(Path(__file__).parent / "models"),
    )


def _merge_defaults(cfg: dict) -> VisionConfig:
    """Merge user-provided config with hard-coded defaults.

    Cấu trúc YAML (xem config.yaml ở root repo):
        mqtt: {...}
        vision:
          telemetry: {topic, qos}
          events:    {topic, qos}
        esp32: {...}
        temporal: {...}
        models_dir: ...
    """
    mqtt_cfg = cfg.get("mqtt", {})
    mqtt = MqttConfig(
        broker_host=mqtt_cfg.get("broker_host", "127.0.0.1"),
        broker_port=mqtt_cfg.get("broker_port", 1883),
        client_id=mqtt_cfg.get("client_id", "vision_publisher"),
        keepalive=mqtt_cfg.get("keepalive", 60),
    )

    vision_cfg = cfg.get("vision", {})

    telemetry_cfg = vision_cfg.get("telemetry", {})
    telemetry = VisionTelemetryConfig(
        topic=telemetry_cfg.get("topic", "ds/vision/metrics"),
        qos=telemetry_cfg.get("qos", 0),
    )

    events_cfg = vision_cfg.get("events", {})
    events = VisionEventsConfig(
        topic=events_cfg.get("topic", "ds/vision/events"),
        qos=events_cfg.get("qos", 1),
    )

    esp32_cfg = cfg.get("esp32", {})
    esp32 = Esp32Config(
        topic_sensors=esp32_cfg.get("topic_sensors", "ds/esp32/sensors"),
        topic_events=esp32_cfg.get("topic_events", "ds/esp32/events"),
        qos_sensors=esp32_cfg.get("qos_sensors", 0),
        qos_events=esp32_cfg.get("qos_events", 1),
    )

    temporal_cfg = cfg.get("temporal", {})
    temporal = TemporalConfig(
        buffer_seconds=temporal_cfg.get("buffer_seconds", 60),
        sample_freq_hz=temporal_cfg.get("sample_freq_hz", 10),
        T_CLOSED_SEC=temporal_cfg.get("T_CLOSED_SEC", 0.20),
        MICROSLEEP_MS=temporal_cfg.get("MICROSLEEP_MS", 500),
        EYE_CLOSURE_MS=temporal_cfg.get("EYE_CLOSURE_MS", 250),
        HEAD_DROP_MS=temporal_cfg.get("HEAD_DROP_MS", 800),
        YAWN_MS=temporal_cfg.get("YAWN_MS", 400),
        HEAD_DROP_DEG=temporal_cfg.get("HEAD_DROP_DEG", 15.0),
        FACE_LOST_MS=temporal_cfg.get("FACE_LOST_MS", 2000),
    )

    default_models = str(Path(__file__).parent / "models")
    models_dir = cfg.get("models_dir", default_models)

    return VisionConfig(
        mqtt=mqtt,
        telemetry=telemetry,
        events=events,
        esp32=esp32,
        temporal=temporal,
        models_dir=models_dir,
    )


# Export cùng cấp cho code cũ (backward compatible)
TOPIC_VISION_METRICS = VisionTelemetryConfig.topic
TOPIC_VISION_EVENTS = VisionEventsConfig.topic
TOPIC_ESP32_SENSORS = Esp32Config.topic_sensors
TOPIC_ESP32_EVENTS = Esp32Config.topic_events

VISION_TIMEOUT_S = TemporalConfig.FACE_LOST_MS / 1000  # Derived
ESP32_TIMEOUT_S = 5.0  # Spec §8, §3
