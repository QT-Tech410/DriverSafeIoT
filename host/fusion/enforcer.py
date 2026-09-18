"""Safety Enforcement Logic & Vehicle Lock Policy (FUS-04).

Chịu trách nhiệm thực thi các chính sách an toàn nghiêm ngặt:
1. Quy tắc khóa động cơ an toàn: Chỉ gửi lệnh {"cmd":"lock"} khi ALCO=2 (>=0.3g/L)
   được xác nhận >= 2/3 lần đo liên tiếp (chống false-positive giật cảm biến).
2. Quy tắc an toàn mệt mỏi: Khi rủi ro do Fatigue (dù ở mức CRITICAL):
   Tuyệt đối KHÔNG khóa động cơ xe đang chạy, chỉ phát còi beep cảnh báo dồn dập
   và hiển thị OLED "NGUNG LAI NGAY" (action = "stop_driving").
3. Quản lý Cooldown phân cấp:
   - Khi chuyển tầng rủi ro (đổi band) hoặc đổi hành động (action thay đổi): Phát lệnh tức thời.
   - Khi cùng tầng rủi ro: Áp dụng thời gian cooldown riêng biệt (CRITICAL: 5s, ALARM: 8s, WARN: 10s).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Literal

BandType = Literal["SAFE", "WARN", "ALARM", "CRITICAL"]


@dataclass
class SafetyEnforcer:
    """Quản lý chính sách an toàn, xác nhận cồn 2/3 lần đo và cooldown (FUS-04)."""

    cooldown_warn_s: float = 10.0
    cooldown_alarm_s: float = 8.0
    cooldown_critical_s: float = 5.0
    window_size: int = 3
    confirm_threshold: int = 2

    alco_history: deque[bool] = field(default_factory=lambda: deque(maxlen=3))
    last_action_ts: float = 0.0
    last_band: BandType = "SAFE"
    last_action: str = "none"

    def record_alcohol(self, is_alco2: bool) -> bool:
        """Ghi nhận 1 lần đo cồn và trả về True nếu đã xác nhận >= 2/3 lần."""
        self.alco_history.append(bool(is_alco2))
        return self.is_alcohol_confirmed()

    def is_alcohol_confirmed(self) -> bool:
        """Kiểm tra điều kiện: ALCO=2 xuất hiện ít nhất 2 trong 3 lần đo gần nhất."""
        if len(self.alco_history) < self.confirm_threshold:
            return False
        return sum(1 for x in self.alco_history if x) >= self.confirm_threshold

    def reset_history(self) -> None:
        """Reset lịch sử đo cồn và trạng thái cooldown."""
        self.alco_history.clear()
        self.last_band = "SAFE"
        self.last_action = "none"
        self.last_action_ts = 0.0

    def decide_action(
        self,
        band: BandType,
        drivers: list[str],
        alco_val: float,
        alco_lvl: int,
        now_s: float | None = None,
    ) -> tuple[str, dict | None]:
        """Quyết định hành động theo chính sách an toàn phân tầng FUS-04.

        Returns:
            (action_name, cmd_payload_or_None)
        """
        now = time.time() if now_s is None else now_s

        is_alco2 = (alco_val >= 0.30 or alco_lvl >= 2)
        alco_confirmed = self.record_alcohol(is_alco2)

        action = "none"
        raw_cmd: dict | None = None

        if band == "SAFE":
            action = "none"
            raw_cmd = None
        elif band == "WARN":
            action = "beep_1"
            raw_cmd = {"cmd": "beep", "n": 1}
        elif band == "ALARM":
            action = "beep_3"
            raw_cmd = {"cmd": "beep", "n": 3}
        elif band == "CRITICAL":
            # TIÊU CHÍ AN TOÀN QUAN TRỌNG NHẤT (spec §7.1.5, §8, FUS-04):
            # 1. Khóa động cơ CHỈ ĐƯỢC PHÁT KHI ALCO=2 được xác nhận >= 2/3 lần đo.
            # 2. Khi rủi ro do Fatigue (dù CRITICAL): Tuyệt đối KHÔNG khóa động cơ, chỉ cảnh báo beep 5.
            if "alcohol" in drivers and alco_confirmed:
                action = "lock"
                raw_cmd = {"cmd": "lock", "sig": "1"}
            elif "alcohol" in drivers and not alco_confirmed:
                # Đang chờ xác nhận 2/3 lần đo: Cảnh báo mạnh nhưng CHƯA KHÓA XE
                action = "confirming_alco"
                raw_cmd = {"cmd": "beep", "n": 3}
            else:
                # Do ngủ gật / mệt mỏi cực độ (Fatigue) -> KHÔNG KHÓA ĐỘNG CƠ KHI ĐANG LÁI
                action = "stop_driving"
                raw_cmd = {"cmd": "beep", "n": 5}

        # Quản lý Cooldown phát lệnh
        should_send_cmd = False
        if raw_cmd is not None:
            if band != self.last_band or action != self.last_action:
                should_send_cmd = True
            else:
                # Cùng band và cùng action: kiểm tra cooldown tương ứng từng mức
                cooldown = self.cooldown_critical_s if band == "CRITICAL" else (
                    self.cooldown_alarm_s if band == "ALARM" else self.cooldown_warn_s
                )
                if (now - self.last_action_ts) >= cooldown:
                    should_send_cmd = True

        if should_send_cmd:
            self.last_action_ts = now
            final_cmd = raw_cmd
        else:
            final_cmd = None

        self.last_band = band
        self.last_action = action
        return action, final_cmd
