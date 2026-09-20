"""Configuration for Fusion subsystem (FUS-01).

Nạp cấu hình từ file YAML (mặc định tìm `config.yaml` ở gốc repo, hoặc
path do người dùng chỉ định qua `--config`). Fallback về default values
nếu không tìm thấy file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Đường dẫn mặc định: config.yaml ở root của repo (cùng cấp với host/)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


@dataclass
class FusionConfig:
    """Tổng hợp cấu hình cho Fusion subsystem."""

    # MQTT connection
    broker_host: str = "127.0.0.1"
    broker_port: int = 1883
    client_id: str = "fusion_subscriber"
    keepalive: int = 60

    # Timeout độ tươi dữ liệu (spec §8, §3)
    vision_timeout_s: float = 2.0
    esp32_timeout_s: float = 5.0


def load_config(config_path: str | None = None) -> FusionConfig:
    """Nạp config từ YAML hoặc dùng default values.

    Thứ tự ưu tiên:
        1. Path chỉ định qua `--config`
        2. `config.yaml` ở root repo
        3. Defaults hard-coded
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        return FusionConfig()

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _merge_defaults(cfg)


def _merge_defaults(cfg: dict) -> FusionConfig:
    """Merge user-provided config với hard-coded defaults."""
    mqtt_cfg = cfg.get("mqtt", {})
    fusion_cfg = cfg.get("fusion", {})

    return FusionConfig(
        broker_host=mqtt_cfg.get("broker_host", "127.0.0.1"),
        broker_port=mqtt_cfg.get("broker_port", 1883),
        client_id=fusion_cfg.get("client_id", "fusion_engine"),
        keepalive=mqtt_cfg.get("keepalive", 60),
        vision_timeout_s=fusion_cfg.get("vision_timeout_s", 2.0),
        esp32_timeout_s=fusion_cfg.get("esp32_timeout_s", 5.0),
    )
