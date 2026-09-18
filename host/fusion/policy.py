"""Fusion Policy Engine & Algorithm Dispatcher (FUS-02 / FUS-03 / FUS-04).

Module phụ trách suy luận Risk Score 0–100 và điều phối chính sách an toàn:
- Hỗ trợ 2 phương pháp suy luận:
  1. MamdaniPolicy: Động cơ suy luận mờ Mamdani 6 inputs, 12 rules (chính thức, FUS-03).
  2. RuleWeightedPolicy: Động cơ xấp xỉ tuyến tính theo trọng số (dự phòng, FUS-02).
- Tích hợp SafetyEnforcer (enforcer.py) để thực thi chính sách khóa xe 2/3 lần đo,
  chống khóa xe khi mệt mỏi và quản lý cooldown phát lệnh MQTT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal

# Tách riêng module enforcer để tuân thủ kiến trúc chia nhỏ Single Responsibility
from enforcer import BandType, SafetyEnforcer

# Topic MQTT theo chuẩn spec §5
TOPIC_FUSION_LEVEL = "ds/fusion/level"
TOPIC_ESP32_CMD = "ds/esp32/cmd"
TOPIC_EVENTS = "ds/events"

# Re-export để đảm bảo tương thích ngược
__all__ = [
    "BandType",
    "FusionResult",
    "SafetyEnforcer",
    "RuleWeightedPolicy",
    "MamdaniPolicy",
    "TOPIC_FUSION_LEVEL",
    "TOPIC_ESP32_CMD",
    "TOPIC_EVENTS",
]


@dataclass
class FusionResult:
    """Kết quả suy luận của Fusion Engine tại 1 chu kỳ."""

    ts: int | float
    risk: float                # 0.0 - 100.0
    band: BandType             # SAFE, WARN, ALARM, CRITICAL
    drivers: list[str]         # ["perclos", "alcohol", "fatigue", "cabin", ...]
    action: str                # "none", "beep_1", "beep_3", "lock", "stop_driving", "confirming_alco"
    cmd_payload: dict | None = None  # Gói tin gửi cho ds/esp32/cmd (nếu cần phát lệnh)

    def as_level_payload(self) -> dict:
        """JSON payload publish lên ds/fusion/level (spec §5)."""
        return {
            "ts": int(self.ts),
            "risk": round(self.risk, 1),
            "band": self.band,
            "drivers": self.drivers,
            "action": self.action,
        }


class RuleWeightedPolicy:
    """Bộ suy luận Fusion dạng Rule-Weighted kèm Action Policy phân tầng (FUS-02/04)."""

    def __init__(
        self,
        w_perclos: float = 0.35,
        w_blink: float = 0.25,
        w_alco: float = 0.30,
        w_cabin: float = 0.10,
        cooldown_s: float = 10.0,
        enforcer: SafetyEnforcer | None = None,
    ) -> None:
        self.w_perclos = w_perclos
        self.w_blink = w_blink
        self.w_alco = w_alco
        self.w_cabin = w_cabin
        self.enforcer = enforcer or SafetyEnforcer(
            cooldown_warn_s=cooldown_s,
            cooldown_alarm_s=cooldown_s * 0.8,
            cooldown_critical_s=cooldown_s * 0.5,
        )

    def scale_perclos(self, perclos_60s: float) -> float:
        """Ánh xạ PERCLOS 0 -> 40% lên thang điểm 0 -> 100 (spec §8)."""
        return max(0.0, min(100.0, (perclos_60s / 0.40) * 100.0))

    def scale_blink_fatigue(
        self,
        cles_dur_ms: float = 0.0,
        head_drop: bool = False,
        yawn_per_min: float = 0.0,
    ) -> float:
        """Tính điểm rủi ro từ chớp mắt dài, microsleep, ngáp và cúi đầu."""
        score = 0.0
        if cles_dur_ms >= 500.0:
            score = max(score, 100.0)
        elif cles_dur_ms >= 250.0:
            score = max(score, 60.0)

        if head_drop:
            score = max(score, 80.0)

        if yawn_per_min >= 2.0:
            score = max(score, 50.0)
        elif yawn_per_min >= 1.0:
            score = max(score, 30.0)

        return score

    def scale_alcohol(self, alcohol_g_l: float) -> float:
        """Ánh xạ nồng độ cồn lên thang điểm 0 -> 100 (spec §7.1.5)."""
        if alcohol_g_l >= 0.30:
            return 100.0
        if alcohol_g_l >= 0.10:
            return 50.0 + ((alcohol_g_l - 0.10) / 0.20) * 20.0
        return max(0.0, (alcohol_g_l / 0.10) * 20.0)

    def scale_cabin(self, temp_c: float, lux_mode: str, ldr_pct: int = 50) -> float:
        """Tính chỉ số CabinStress từ nhiệt độ NTC và ánh sáng LDR (spec §7.2, §7.3)."""
        stress = 0.0
        if temp_c >= 35.0:
            stress += 60.0
        elif temp_c <= 16.0:
            stress += 40.0

        if lux_mode in ("dark", "night") or ldr_pct <= 20:
            stress += 40.0
        elif lux_mode == "dim" or ldr_pct <= 45:
            stress += 20.0

        return min(100.0, stress)

    def evaluate(
        self,
        vision_data: dict | None,
        esp32_data: dict | None,
        vision_online: bool = True,
        esp32_online: bool = True,
        current_time_ms: int | float | None = None,
    ) -> FusionResult:
        """Thực hiện suy luận Risk Score và xác định hành động chính sách."""
        now_ms = current_time_ms if current_time_ms is not None else int(time.time() * 1000)
        now_s = now_ms / 1000.0 if current_time_ms is not None else time.time()

        v_dict = vision_data or {}
        e_dict = esp32_data or {}

        # 1. Trích xuất các biến thành phần
        perclos_val = float(v_dict.get("perclos_60s", 0.0))
        cles_val = float(v_dict.get("cles_dur_ms", 0.0))
        head_drop_val = bool(v_dict.get("head_drop", False))
        yawn_val = float(v_dict.get("yawn_per_min", 0.0))

        alco_val = float(e_dict.get("alcohol_g_l", 0.0))
        alco_lvl = int(e_dict.get("alco_level", 2 if alco_val >= 0.3 else (1 if alco_val >= 0.1 else 0)))
        temp_val = float(e_dict.get("temp_c", 25.0))
        lux_val = str(e_dict.get("lux_mode", "day"))
        ldr_val = int(e_dict.get("ldr_pct", 50))

        # 2. Chuẩn hóa thang điểm 0 - 100
        s_perclos = self.scale_perclos(perclos_val)
        s_blink = self.scale_blink_fatigue(cles_val, head_drop_val, yawn_val)
        s_alco = self.scale_alcohol(alco_val)
        s_cabin = self.scale_cabin(temp_val, lux_val, ldr_val)

        drivers: list[str] = []

        # 3. Tính toán Risk Score cơ sở (Hỗ trợ Fallback khi vision offline)
        if vision_online:
            risk = (
                self.w_perclos * s_perclos
                + self.w_blink * s_blink
                + self.w_alco * s_alco
                + self.w_cabin * s_cabin
            )

            # Quy tắc rủi ro mệt mỏi (Rule-Weighted Safety Overrides):
            if (cles_val >= 500.0 or head_drop_val) and (s_perclos >= 65.0 or yawn_val >= 2.0):
                risk = max(risk, 85.0)
            elif s_perclos >= 60.0 and s_blink >= 50.0:
                risk = max(risk, 58.0)
            elif cles_val >= 500.0 or head_drop_val or s_perclos >= 75.0:
                risk = max(risk, 55.0)
        else:
            # Fallback: Mất vision > 2s -> suy luận thuần cảm biến (spec §8)
            risk = 0.75 * s_alco + 0.25 * s_cabin
            drivers.append("vision_offline")

        # Cồn mức 2 luôn kích hoạt rủi ro khẩn cấp CRITICAL
        if alco_val >= 0.30 or alco_lvl >= 2:
            risk = max(risk, 85.0)
        elif alco_val >= 0.10 and (s_perclos >= 50.0 or s_cabin >= 50.0):
            risk = max(risk, 65.0)

        risk = max(0.0, min(100.0, round(risk, 1)))

        # 4. Xác định các nguyên nhân rủi ro chính (Drivers)
        if s_alco >= 50.0 or alco_val >= 0.10 or alco_lvl >= 1:
            drivers.append("alcohol")
        if vision_online:
            if s_perclos >= 40.0:
                drivers.append("perclos")
            if s_blink >= 50.0 or cles_val >= 500.0 or head_drop_val:
                drivers.append("fatigue")
        if s_cabin >= 50.0:
            drivers.append("cabin")
        if not drivers:
            drivers.append("normal")

        # 5. Phân dải Risk thành 4 band
        if risk < 25.0:
            band: BandType = "SAFE"
        elif risk < 50.0:
            band: BandType = "WARN"
        elif risk <= 75.0:
            band: BandType = "ALARM"
        else:
            band: BandType = "CRITICAL"

        # 6. Áp dụng SafetyEnforcer (xác nhận cồn 2/3 lần đo & cooldown)
        action, cmd = self.enforcer.decide_action(
            band=band,
            drivers=drivers,
            alco_val=alco_val,
            alco_lvl=alco_lvl,
            now_s=now_s,
        )

        return FusionResult(
            ts=now_ms,
            risk=risk,
            band=band,
            drivers=drivers,
            action=action,
            cmd_payload=cmd,
        )


class MamdaniPolicy:
    """Bộ suy luận Fusion ứng dụng Fuzzy Mamdani Engine & SafetyEnforcer (FUS-03/04)."""

    def __init__(
        self,
        rules_path: str | Path | None = None,
        cooldown_s: float = 10.0,
        enforcer: SafetyEnforcer | None = None,
    ) -> None:
        from fuzzy import MamdaniEngine, normalize_inputs

        self.engine = MamdaniEngine(rules_path)
        self.normalize_inputs = normalize_inputs
        self.enforcer = enforcer or SafetyEnforcer(
            cooldown_warn_s=cooldown_s,
            cooldown_alarm_s=cooldown_s * 0.8,
            cooldown_critical_s=cooldown_s * 0.5,
        )

    def evaluate(
        self,
        vision_data: dict | None,
        esp32_data: dict | None,
        vision_online: bool = True,
        esp32_online: bool = True,
        current_time_ms: int | float | None = None,
    ) -> FusionResult:
        """Thực hiện suy luận Risk Score bằng Mamdani Fuzzy Engine và áp dụng Action Policy an toàn."""
        now_ms = current_time_ms if current_time_ms is not None else int(time.time() * 1000)
        now_s = now_ms / 1000.0 if current_time_ms is not None else time.time()

        v_dict = vision_data or {}
        e_dict = esp32_data or {}

        # 1. Trích xuất chỉ số
        perclos_val = float(v_dict.get("perclos_60s", 0.0)) if vision_online else 0.0
        cles_val = float(v_dict.get("cles_dur_ms", 0.0)) if vision_online else 0.0
        head_drop_val = bool(v_dict.get("head_drop", False)) if vision_online else False
        yawn_val = float(v_dict.get("yawn_per_min", 0.0)) if vision_online else 0.0

        alco_val = float(e_dict.get("alcohol_g_l", 0.0))
        alco_lvl = int(e_dict.get("alco_level", 2 if alco_val >= 0.3 else (1 if alco_val >= 0.1 else 0)))
        temp_val = float(e_dict.get("temp_c", 25.0))
        lux_val = str(e_dict.get("lux_mode", "day"))
        ldr_pct = int(e_dict.get("ldr_pct", 50))
        ldr_lux = 20.0 if lux_val in ("dark", "night") or ldr_pct <= 20 else 300.0

        # 2. Chuẩn hóa thang 0 - 100 cho 6 inputs
        fuzzy_inputs = self.normalize_inputs(
            perclos_pct=perclos_val,
            cles_dur_ms=cles_val,
            yawn_per_min=yawn_val,
            head_drop=head_drop_val,
            alco_level=alco_lvl,
            ntc_temp_c=temp_val,
            ldr_lux=ldr_lux,
        )

        drivers: list[str] = []

        # 3. Suy luận Mamdani
        if vision_online:
            risk, fired_rules = self.engine.infer(fuzzy_inputs)

            # Phân loại drivers từ các luật fired
            fired_ids = {r.id for r in fired_rules if r.weight >= 0.1}
            if alco_val >= 0.10 or alco_lvl >= 1 or any(r in fired_ids for r in ("R1", "R7", "R8")):
                drivers.append("alcohol")
            if any(r in fired_ids for r in ("R1", "R2", "R4", "R5", "R6", "R7", "R10")):
                drivers.append("perclos")
            if any(r in fired_ids for r in ("R3", "R4", "R5")) or head_drop_val:
                drivers.append("fatigue")
            if any(r in fired_ids for r in ("R6", "R9", "R11")):
                drivers.append("cabin")
        else:
            # Fallback khi mất Vision > 2s: suy luận thuần cảm biến
            s_alco = fuzzy_inputs["alco"]
            s_cabin = fuzzy_inputs["cabin"]
            risk = round(0.75 * s_alco + 0.25 * s_cabin, 1)
            drivers.append("vision_offline")
            if s_alco >= 40.0:
                drivers.append("alcohol")
            if s_cabin >= 40.0:
                drivers.append("cabin")

        # Cồn mức 2 luôn kích hoạt rủi ro khẩn cấp CRITICAL
        if alco_val >= 0.30 or alco_lvl >= 2:
            risk = max(risk, 85.0)
            if "alcohol" not in drivers:
                drivers.append("alcohol")

        if not drivers:
            drivers.append("normal")

        # 4. Phân dải Risk thành 4 band chính xác
        if risk < 25.0:
            band: BandType = "SAFE"
        elif risk < 50.0:
            band: BandType = "WARN"
        elif risk <= 75.0:
            band: BandType = "ALARM"
        else:
            band: BandType = "CRITICAL"

        # 5. Áp dụng SafetyEnforcer (xác nhận cồn 2/3 lần đo & cooldown)
        action, cmd = self.enforcer.decide_action(
            band=band,
            drivers=drivers,
            alco_val=alco_val,
            alco_lvl=alco_lvl,
            now_s=now_s,
        )

        return FusionResult(
            ts=now_ms,
            risk=risk,
            band=band,
            drivers=drivers,
            action=action,
            cmd_payload=cmd,
        )
