"""Rule-Weighted Temporary Fusion Engine & Action Policy (FUS-02).

Tính toán Risk Score 0–100 theo công thức Rule-Weighted tạm (spec §11, FUS-02):
    Risk = w1·PERCLOS + w2·blink_events + w3·ALCO + w4·CABIN

Phân dải Risk thành 4 band chính xác:
    - SAFE     (< 25):   Bình thường, log trạng thái
    - WARN     (25–50):  Cảnh báo mệt mỏi/nhiệt độ nhẹ, beep 1 nhát
    - ALARM    (50–75):  Nguy cơ cao, buzzer 3 nhịp + LED đỏ
    - CRITICAL (> 75):   Khẩn cấp:
                         + Nếu rủi ro do Cồn (ALCO=2) -> Khóa động cơ ({"cmd":"lock"})
                         + Nếu rủi ro do Mệt mỏi -> KHÔNG khóa, chỉ cảnh báo ("NGUNG LAI NGAY")

Quản lý cooldown giữa các lệnh gửi cho ESP32 và hỗ trợ fallback khi vision_offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Literal

BandType = Literal["SAFE", "WARN", "ALARM", "CRITICAL"]

# Topic MQTT (spec §5)
TOPIC_FUSION_LEVEL = "ds/fusion/level"
TOPIC_ESP32_CMD = "ds/esp32/cmd"
TOPIC_EVENTS = "ds/events"


@dataclass
class FusionResult:
    """Kết quả suy luận của Fusion Engine tại 1 chu kỳ."""

    ts: int | float
    risk: float                # 0.0 - 100.0
    band: BandType             # SAFE, WARN, ALARM, CRITICAL
    drivers: list[str]         # ["perclos", "alcohol", "fatigue", "cabin", ...]
    action: str                # "none", "beep_1", "beep_3", "lock", "stop_driving"
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
    """Bộ suy luận Fusion dạng Rule-Weighted kèm Action Policy phân tầng."""

    def __init__(
        self,
        w_perclos: float = 0.35,
        w_blink: float = 0.25,
        w_alco: float = 0.30,
        w_cabin: float = 0.10,
        cooldown_s: float = 10.0,
    ) -> None:
        self.w_perclos = w_perclos
        self.w_blink = w_blink
        self.w_alco = w_alco
        self.w_cabin = w_cabin
        self.cooldown_s = cooldown_s

        self.last_action_ts: float = 0.0
        self.last_band: BandType = "SAFE"
        self.last_action: str = "none"

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
        # Microsleep >= 500ms
        if cles_dur_ms >= 500.0:
            score = max(score, 100.0)
        # Eye-closure >= 250ms
        elif cles_dur_ms >= 250.0:
            score = max(score, 60.0)

        # Cúi đầu head_drop
        if head_drop:
            score = max(score, 80.0)

        # Ngáp nhiều (>= 2 lần/phút)
        if yawn_per_min >= 2.0:
            score = max(score, 50.0)
        elif yawn_per_min >= 1.0:
            score = max(score, 30.0)

        return score

    def scale_alcohol(self, alcohol_g_l: float) -> float:
        """Ánh xạ nồng độ cồn lên thang điểm 0 -> 100 (spec §7.1.5).

        ALCO=0: < 0.1 g/L -> điểm thấp (0-20)
        ALCO=1: 0.1 - 0.3 g/L -> điểm vừa (50-70)
        ALCO=2: > 0.3 g/L -> điểm tối đa (100)
        """
        if alcohol_g_l >= 0.30:
            return 100.0
        if alcohol_g_l >= 0.10:
            return 50.0 + ((alcohol_g_l - 0.10) / 0.20) * 20.0
        return max(0.0, (alcohol_g_l / 0.10) * 20.0)

    def scale_cabin(self, temp_c: float, lux_mode: str, ldr_pct: int = 50) -> float:
        """Tính chỉ số CabinStress từ nhiệt độ NTC và ánh sáng LDR (spec §7.2, §7.3)."""
        stress = 0.0
        # Nhiệt độ cabin: >35°C là NÓNG (spec §7.4 WARN); <16°C là LẠNH
        if temp_c >= 35.0:
            stress += 60.0
        elif temp_c <= 16.0:
            stress += 40.0

        # Ánh sáng: dark / night làm tăng nguy cơ buồn ngủ khi lái xe đêm
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
        now_s = time.time()

        v_dict = vision_data or {}
        e_dict = esp32_data or {}

        # 1. Trích xuất các biến thành phần
        perclos_val = float(v_dict.get("perclos_60s", 0.0))
        cles_val = float(v_dict.get("cles_dur_ms", 0.0))
        head_drop_val = bool(v_dict.get("head_drop", False))
        yawn_val = float(v_dict.get("yawn_per_min", 0.0))

        alco_val = float(e_dict.get("alcohol_g_l", 0.0))
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
            # Ngủ gật cực độ: Microsleep (>=500ms) kết hợp PERCLOS cao hoặc cúi đầu -> CRITICAL
            if (cles_val >= 500.0 or head_drop_val) and (s_perclos >= 65.0 or yawn_val >= 2.0):
                risk = max(risk, 85.0)
            # Mệt mỏi cao: PERCLOS cao kết hợp nhắm mắt dài (>=250ms) -> ALARM
            elif s_perclos >= 60.0 and s_blink >= 50.0:
                risk = max(risk, 58.0)
            # Sự kiện lẻ: Microsleep hoặc Head-drop đơn lẻ -> ALARM
            elif cles_val >= 500.0 or head_drop_val or s_perclos >= 75.0:
                risk = max(risk, 55.0)
        else:
            # Fallback: Mất vision > 2s -> suy luận thuần cảm biến (spec §8)
            risk = 0.75 * s_alco + 0.25 * s_cabin
            drivers.append("vision_offline")

        # Cồn mức 2 (ALCO=2, >=0.3 g/L) luôn kích hoạt rủi ro khẩn cấp CRITICAL
        if alco_val >= 0.30:
            risk = max(risk, 85.0)
        # Cồn mức 1 kết hợp mệt mỏi hoặc cabin nóng -> ALARM+
        elif alco_val >= 0.10 and (s_perclos >= 50.0 or s_cabin >= 50.0):
            risk = max(risk, 65.0)

        risk = max(0.0, min(100.0, round(risk, 1)))

        # 4. Xác định các nguyên nhân rủi ro chính (Drivers)
        if s_alco >= 50.0:
            drivers.append("alcohol")
        if vision_online:
            if s_perclos >= 40.0:
                drivers.append("perclos")
            if s_blink >= 50.0:
                drivers.append("fatigue")
        if s_cabin >= 50.0:
            drivers.append("cabin")
        if not drivers:
            drivers.append("normal")

        # 5. Phân dải Risk thành 4 band chính xác
        if risk < 25.0:
            band: BandType = "SAFE"
        elif risk < 50.0:
            band: BandType = "WARN"
        elif risk <= 75.0:
            band: BandType = "ALARM"
        else:
            band: BandType = "CRITICAL"

        # 6. Action Policy & Safety Enforcement (spec §8, §7.1.5, §7.4)
        action = "none"
        cmd: dict | None = None

        if band == "SAFE":
            action = "none"
            cmd = None
        elif band == "WARN":
            action = "beep_1"
            cmd = {"cmd": "beep", "n": 1}
        elif band == "ALARM":
            action = "beep_3"
            cmd = {"cmd": "beep", "n": 3}
        elif band == "CRITICAL":
            # Phân biệt rõ Alcohol vs Fatigue (spec §8):
            # Khóa động cơ CHỈ KHI có cồn mức cao (ALCO >= 2)
            if "alcohol" in drivers and alco_val >= 0.30:
                action = "lock"
                cmd = {"cmd": "lock", "sig": "1"}
            else:
                # Do buồn ngủ / mệt mỏi cực độ: Tuyệt đối KHÔNG khóa động cơ xe đang chạy!
                action = "stop_driving"
                cmd = {"cmd": "beep", "n": 5}

        # 7. Quản lý Cooldown phát lệnh
        # Chỉ gửi cmd nếu:
        # - Đổi band (chuyển tầng nguy cơ), HOẶC
        # - Đã qua thời gian cooldown_s
        should_send_cmd = False
        if cmd is not None:
            if band != self.last_band:
                should_send_cmd = True
            elif (now_s - self.last_action_ts) >= self.cooldown_s:
                should_send_cmd = True

        if should_send_cmd:
            self.last_action_ts = now_s
        else:
            cmd = None

        self.last_band = band
        self.last_action = action

        return FusionResult(
            ts=now_ms,
            risk=risk,
            band=band,
            drivers=drivers,
            action=action,
            cmd_payload=cmd,
        )
