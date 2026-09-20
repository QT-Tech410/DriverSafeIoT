"""Simulated Sensors & MQTT Test Utility for ESP32-S3 (TEST-01).

Giả lập node cảm biến phần cứng ESP32-S3 (Phần A) để Phần B test độc lập logic
Fusion & Dashboard trước khi có phần cứng thật.

Publish 1Hz lên topic `ds/esp32/sensors` (spec §5, §7).
Publish tức thời lên `ds/esp32/events` khi phát hiện cồn mức cao.
Lắng nghe lệnh điều khiển từ Fusion trên `ds/esp32/cmd` (lock, unlock, beep, oled, reset).

Phím điều khiển nhanh (Interactive Terminal):
    1 : Mức cồn 0 (SAFE, 0.00 g/L)
    2 : Mức cồn 1 (WARN, 0.18 g/L)
    3 : Mức cồn 2 (CRITICAL, 0.45 g/L -> gửi event)
    t : Đổi nhiệt độ (OK 26.5°C -> NÓNG 36.5°C -> LẠNH 14.5°C)
    l : Đổi ánh sáng (DAY 75% -> DIM 40% -> DARK 12%)
    d : Bật/tắt chế độ mất mạng (degraded_mode)
    h : Hiển thị trợ giúp phím tắt
    q : Thoát
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import threading
import time

import paho.mqtt.client as mqtt

# Topic MQTT (spec §5)
TOPIC_ESP32_SENSORS = "ds/esp32/sensors"
TOPIC_ESP32_EVENTS = "ds/esp32/events"
TOPIC_ESP32_CMD = "ds/esp32/cmd"

try:
    _CALLBACK_API = mqtt.CallbackAPIVersion.VERSION2
except AttributeError:  # pragma: no cover
    _CALLBACK_API = None


class SensorSimulator:
    """Mô phỏng các cảm biến MQ-3, NTC 10K, LDR và FSM cục bộ ESP32."""

    def __init__(
        self,
        broker_host: str = "127.0.0.1",
        broker_port: int = 1883,
        client_id: str = "esp32_simulator",
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id

        # Thông số mô phỏng
        self.alco_level: int = 0          # 0: SAFE, 1: WARN, 2: CRITICAL
        self.alcohol_g_l: float = 0.0     # BrAC g/L
        self.mq3_ao_v: float = 0.85       # Điện áp AO MQ-3 (V)
        self.rs_r0: float = 1.15          # Tỷ số trở kháng Rs/R0

        self.temp_mode_idx: int = 0
        self.temp_modes = [
            (26.5, "OK (26.5°C)"),
            (36.5, "NONG (>35°C)"),
            (14.5, "LANH (<16°C)"),
        ]

        self.lux_mode_idx: int = 0
        self.lux_modes = [
            ("day", 75, "SANG / DAY (75%)"),
            ("dim", 40, "DU / DIM (40%)"),
            ("dark", 12, "TOI / DARK (12%)"),
        ]

        self.degraded: bool = False
        self.rssi: int = -56
        self.t_start = time.time()
        self.publish_count: int = 0

        # MQTT client
        if _CALLBACK_API is not None:
            self.client = mqtt.Client(_CALLBACK_API, client_id=self.client_id)
        else:  # pragma: no cover
            self.client = mqtt.Client(client_id=self.client_id)

        self.client.on_connect = self._handle_connect
        self.client.on_disconnect = self._handle_disconnect
        self.client.on_message = self._handle_message
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def _handle_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        is_ok = (reason_code == 0) if isinstance(reason_code, int) else (not reason_code.is_failure)
        if is_ok:
            self._connected = True
            print(f"\n[sim_sensors] Da ket noi Broker tai {self.broker_host}:{self.broker_port}")
            # Lắng nghe lệnh điều khiển từ host/fusion
            self.client.subscribe(TOPIC_ESP32_CMD, qos=1)
        else:
            self._connected = False
            print(f"\n[sim_sensors] Ket noi broker that bai: code {reason_code}", file=sys.stderr)

    def _handle_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None) -> None:
        self._connected = False
        print(f"\n[sim_sensors] Ngat ket noi broker ({reason_code})")

    def _handle_message(self, client, userdata, msg) -> None:
        """Xử lý lệnh nhận được từ ds/esp32/cmd."""
        if msg.topic == TOPIC_ESP32_CMD:
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                cmd = payload.get("cmd")
                print(f"\n>>> [ESP32 CHAP HANH CMD] Nhan lenh: {payload}")
                if cmd == "lock":
                    print("    [ACT] -> SERVO QUAY 90° (KHOA DONG CO) + COI LIEN TUC + OLED DO")
                elif cmd == "unlock":
                    print("    [ACT] -> SERVO VE 0° (MO KHOA DONG CO) + OLED XANH")
                elif cmd == "beep":
                    n = payload.get("n", 1)
                    print(f"    [ACT] -> BUZZER BEEP {n} NHIP")
                elif cmd == "oled":
                    m = payload.get("msg", "")
                    print(f"    [ACT] -> OLED DISPLAY: '{m}'")
                elif cmd == "reset":
                    print("    [ACT] -> FSM RESET VE SAFE")
            except Exception as e:
                print(f"[sim_sensors] Loi parse cmd: {e}", file=sys.stderr)

    def set_alcohol_level(self, level: int) -> None:
        """Đặt mức cồn 0, 1 hoặc 2 (spec §7.1.5)."""
        self.alco_level = max(0, min(2, level))
        if self.alco_level == 0:
            self.alcohol_g_l = 0.0
            self.mq3_ao_v = 0.85
            self.rs_r0 = 1.15
            print(f"\n[SIMULATOR] Dat muc con: LEVEL 0 (SAFE - {self.alcohol_g_l:.2f} g/L)")
        elif self.alco_level == 1:
            self.alcohol_g_l = 0.18
            self.mq3_ao_v = 2.05
            self.rs_r0 = 2.40
            print(f"\n[SIMULATOR] Dat muc con: LEVEL 1 (WARN - {self.alcohol_g_l:.2f} g/L)")
        else:
            self.alcohol_g_l = 0.45
            self.mq3_ao_v = 3.15
            self.rs_r0 = 4.30
            print(f"\n[SIMULATOR] Dat muc con: LEVEL 2 (CRITICAL - {self.alcohol_g_l:.2f} g/L)")
            # Phát event tức thời khi đạt mức 2
            self.publish_event("alcohol_level2", f"mq3_ao_v={self.mq3_ao_v:.2f}")

    def cycle_temperature(self) -> None:
        """Chuyển đổi trạng thái nhiệt độ xoay vòng."""
        self.temp_mode_idx = (self.temp_mode_idx + 1) % len(self.temp_modes)
        temp, label = self.temp_modes[self.temp_mode_idx]
        print(f"\n[SIMULATOR] Chuyen nhiet do: {label}")

    def cycle_lux(self) -> None:
        """Chuyển đổi trạng thái ánh sáng LDR xoay vòng."""
        self.lux_mode_idx = (self.lux_mode_idx + 1) % len(self.lux_modes)
        mode, _, label = self.lux_modes[self.lux_mode_idx]
        print(f"\n[SIMULATOR] Chuyen che do sang: {label}")

    def toggle_degraded(self) -> None:
        """Bật/tắt cờ degraded_mode."""
        self.degraded = not self.degraded
        st = "BAT (Mat song/Tu chu)" if self.degraded else "TAT (Binh thuong)"
        print(f"\n[SIMULATOR] Degraded Mode: {st}")

    def connect(self, timeout: float = 3.0) -> bool:
        """Kết nối tới Mosquitto broker."""
        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"[sim_sensors] Loi ket noi: {e}", file=sys.stderr)
            return False

        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            if self._connected:
                return True
            time.sleep(0.05)
        return self._connected

    def disconnect(self) -> None:
        """Ngắt kết nối broker."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass

    def build_telemetry(self) -> dict:
        """Tạo payload telemetry 1Hz chuẩn schema spec §5."""
        now_ms = int(time.time() * 1000)
        uptime = int(time.time() - self.t_start)

        # Thêm chút jitter thực tế
        v_jitter = self.mq3_ao_v + (random.uniform(-0.02, 0.02) if self.alco_level > 0 else random.uniform(0.0, 0.01))
        r_jitter = self.rs_r0 + random.uniform(-0.05, 0.05)
        alco_jitter = max(0.0, self.alcohol_g_l + (random.uniform(-0.01, 0.01) if self.alco_level > 0 else 0.0))

        base_temp = self.temp_modes[self.temp_mode_idx][0]
        temp_val = round(base_temp + random.uniform(-0.2, 0.2), 1)

        lux_mode, base_ldr, _ = self.lux_modes[self.lux_mode_idx]
        ldr_val = int(max(0, min(100, base_ldr + random.randint(-2, 2))))
        rssi_val = int(self.rssi + random.randint(-2, 2))

        return {
            "ts": now_ms,
            "mq3_ao_v": round(v_jitter, 2),
            "rs_r0": round(r_jitter, 2),
            "alcohol_g_l": round(alco_jitter, 2),
            "temp_c": temp_val,
            "ldr_pct": ldr_val,
            "lux_mode": lux_mode,
            "rssi": rssi_val,
            "uptime_s": uptime,
            "degraded": self.degraded,
        }

    def publish_telemetry(self) -> bool:
        """Publish gói tin 1Hz lên ds/esp32/sensors (QoS 0)."""
        data = self.build_telemetry()
        payload = json.dumps(data)
        info = self.client.publish(TOPIC_ESP32_SENSORS, payload, qos=0)
        self.publish_count += 1
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def publish_event(self, event_name: str, raw_info: str) -> bool:
        """Publish sự kiện tức thời lên ds/esp32/events (QoS 1)."""
        now_ms = int(time.time() * 1000)
        event_dict = {
            "ts": now_ms,
            "event": event_name,
            "raw": raw_info,
        }
        info = self.client.publish(TOPIC_ESP32_EVENTS, json.dumps(event_dict), qos=1)
        return info.rc == mqtt.MQTT_ERR_SUCCESS


def print_banner() -> None:
    print("=" * 66)
    print("    ESP32-S3 SENSOR SIMULATOR — DriverSafe-IoT (TEST-01)")
    print("=" * 66)
    print(" Phim tat dieu khien nhanh:")
    print("   [1] : Muc con LEVEL 0 (SAFE - 0.00 g/L)")
    print("   [2] : Muc con LEVEL 1 (WARN - 0.18 g/L)")
    print("   [3] : Muc con LEVEL 2 (CRITICAL - 0.45 g/L -> Event)")
    print("   [t] : Doi nhiet do cabin (OK / NONG / LANH)")
    print("   [l] : Doi do sang LDR (DAY / DIM / DARK)")
    print("   [d] : Bat/tat degraded_mode (Mat song)")
    print("   [h] : In lai menu tro giup")
    print("   [q] : Thoat chuong trinh")
    print("=" * 66)


def read_key_windows() -> str | None:
    """Đọc ký tự phím không blocking trên Windows."""
    try:
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b"\x00", b"\xe0"):  # Phím đặc biệt (mũi tên, function key)
                msvcrt.getch()
                return None
            try:
                return ch.decode("utf-8").lower()
            except Exception:
                return None
    except ImportError:  # pragma: no cover
        pass
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ESP32-S3 Sensor Simulator (TEST-01)")
    parser.add_argument("--host", default="127.0.0.1", help="Broker host (127.0.0.1)")
    parser.add_argument("--port", type=int, default=1883, help="Broker port (1883)")
    parser.add_argument("--duration", type=float, default=0.0, help="Thoi gian chay tu dong (0 = chay lien tuc)")
    parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2], help="Khoi tao muc con ban dau (0, 1, 2)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Che do tuong tac ban phim (mac dinh da bat)")
    args = parser.parse_args(argv)

    sim = SensorSimulator(broker_host=args.host, broker_port=args.port)
    sim.set_alcohol_level(args.level)

    print(f"[sim_sensors] Dang ket noi Mosquitto broker tai {args.host}:{args.port}...")
    if not sim.connect(timeout=3.0):
        print(f"[sim_sensors] CANH BAO: Chua the ket noi broker {args.host}:{args.port}. "
              "Hay khoi dong Mosquitto broker!", file=sys.stderr)

    print_banner()

    t_start = time.perf_counter()
    last_pub = 0.0

    try:
        while True:
            now = time.perf_counter()

            # Đọc phím điều khiển từ bàn phím terminal
            key = read_key_windows()
            if key:
                if key == "1":
                    sim.set_alcohol_level(0)
                elif key == "2":
                    sim.set_alcohol_level(1)
                elif key == "3":
                    sim.set_alcohol_level(2)
                elif key == "t":
                    sim.cycle_temperature()
                elif key == "l":
                    sim.cycle_lux()
                elif key == "d":
                    sim.toggle_degraded()
                elif key in ("h", "?"):
                    print_banner()
                elif key in ("q", "\x1b"):  # q hoặc ESC
                    print("\n[sim_sensors] Nhanh phim thoat.")
                    break

            # Publish định kỳ 1Hz
            if now - last_pub >= 1.0:
                data = sim.build_telemetry()
                sim.publish_telemetry()
                last_pub = now

                # In log 1Hz súc tích
                alco_str = f"ALCO={sim.alco_level} ({data['alcohol_g_l']:.2f}g/L)"
                temp_str = f"T={data['temp_c']}°C"
                lux_str = f"LUX={data['lux_mode'].upper()}({data['ldr_pct']}%)"
                deg_str = f"DEG={'ON' if data['degraded'] else 'off'}"
                mqtt_str = "OK" if sim.connected else "DISC"

                sys.stdout.write(
                    f"\r[1Hz #{sim.publish_count:03d}] {alco_str} | {temp_str} | {lux_str} | {deg_str} | MQTT:{mqtt_str}  "
                )
                sys.stdout.flush()

            time.sleep(0.05)

            if args.duration > 0 and (now - t_start) >= args.duration:
                print(f"\n[sim_sensors] Da chay het thoi gian dinh san ({args.duration}s).")
                break

    except KeyboardInterrupt:
        print("\n[sim_sensors] Dung boi nguoi dung (Ctrl+C).")
    finally:
        sim.disconnect()

    print(f"[sim_sensors] Da dung tien trinh an toan. Tong so ban tin da gui: {sim.publish_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
