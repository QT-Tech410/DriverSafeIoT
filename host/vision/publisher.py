"""Vision Metrics MQTT Publisher (VIS-05).

Tổng hợp các chỉ số Vision ở tầng thời gian (TemporalMetrics) và publish dữ liệu
định kỳ 1Hz lên topic MQTT `ds/vision/metrics` với QoS 0, theo đúng schema JSON
thống nhất với Phần A (spec §5).

Đồng thời publish tức thời các sự kiện (microsleep, yawn, head_drop, face_lost)
lên `ds/vision/events` với QoS 1 (spec §6.1).

Chạy:
    python host/vision/publisher.py                     # webcam + mở cửa sổ camera trực quan
    python host/vision/publisher.py --headless          # chạy ngầm không mở GUI
    python host/vision/publisher.py --host 127.0.0.1 --port 1883
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt

# Cho phép import các module cùng cấp trong thư mục host/vision
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import FrameSource, FpsMeter, draw_hud  # noqa: E402
from face_metrics import (  # noqa: E402
    FaceMesh,
    MODEL_PATH,
    compute_geo_metrics,
    draw_landmarks,
    draw_metrics,
    draw_status,
)
from temporal import (  # noqa: E402
    Event,
    MetricsSnapshot,
    TemporalMetrics,
    draw_temporal_hud,
)

# MQTT Topics theo đặc tả spec §5
TOPIC_VISION_METRICS = "ds/vision/metrics"
TOPIC_VISION_EVENTS = "ds/vision/events"
TOPIC_ESP32_SENSORS = "ds/esp32/sensors"


# Tương thích Paho MQTT v1 và v2
try:
    _CALLBACK_API = mqtt.CallbackAPIVersion.VERSION2
except AttributeError:  # pragma: no cover
    _CALLBACK_API = None


class VisionPublisher(mqtt.Client):
    """Client MQTT chuyên trách publish chỉ số và sự kiện từ Vision subsystem."""

    def __init__(
        self,
        broker_host: str = "127.0.0.1",
        broker_port: int = 1883,
        client_id: str = "vision_publisher",
        keepalive: int = 60,
    ) -> None:
        if _CALLBACK_API is not None:
            super().__init__(_CALLBACK_API, client_id=client_id)
        else:  # pragma: no cover
            super().__init__(client_id=client_id)

        self.broker_host = broker_host
        self.broker_port = broker_port
        self.keepalive = keepalive
        self.publish_count = 0
        self.event_publish_count = 0
        self.last_published_ts: float = 0.0
        self.current_lux_mode: str = "day"

        self.on_connect = self._handle_connect
        self.on_disconnect = self._handle_disconnect
        self.on_message = self._handle_message

    @property
    def connected(self) -> bool:
        return self.is_connected()

    def _handle_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        is_ok = (reason_code == 0) if isinstance(reason_code, int) else (not reason_code.is_failure)
        if is_ok:
            print(f"[publisher] Da ket noi Mosquitto broker tai {self.broker_host}:{self.broker_port}")
            # Đăng ký nhận lux_mode từ ESP32 sensor nếu có (spec §6.4)
            self.subscribe(TOPIC_ESP32_SENSORS, qos=0)
        else:
            print(f"[publisher] Ket noi broker that bai: code {reason_code}", file=sys.stderr)

    def _handle_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None) -> None:
        print(f"[publisher] Da ngat ket noi khoi Mosquitto broker (code {reason_code})")

    def _handle_message(self, client, userdata, msg) -> None:
        if msg.topic == TOPIC_ESP32_SENSORS:
            try:
                data = json.loads(msg.payload.decode("utf-8"))
                lux = data.get("lux_mode")
                if lux in ("dark", "night"):
                    self.current_lux_mode = "night"
                elif lux in ("day", "light"):
                    self.current_lux_mode = "day"
            except Exception:
                pass

    def connect_broker(self, timeout: float = 5.0) -> bool:
        """Kết nối tới Mosquitto broker và khởi chạy network loop background."""
        try:
            self.connect(self.broker_host, self.broker_port, self.keepalive)
            self.loop_start()
        except Exception as e:
            print(f"[publisher] Khong the ket noi den {self.broker_host}:{self.broker_port} -> {e}", file=sys.stderr)
            return False

        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            if self.connected:
                return True
            time.sleep(0.05)
        return self.connected

    def disconnect_broker(self) -> None:
        """Dừng network loop và ngắt kết nối an toàn."""
        try:
            self.loop_stop()
            self.disconnect()
        except Exception:
            pass

    def publish_metrics(self, snap: MetricsSnapshot) -> bool:
        """Publish gói tin 1Hz lên ds/vision/metrics (QoS 0).

        JSON schema chuẩn (spec §5):
          ts, face, ear, perclos_60s, cles_dur_ms, mar, yawn_per_min,
          head_pitch_deg, head_drop, lux_mode
        """
        payload_dict = snap.as_dict()
        payload_dict["lux_mode"] = self.current_lux_mode
        payload_str = json.dumps(payload_dict)

        info = self.publish(TOPIC_VISION_METRICS, payload_str, qos=0)
        self.publish_count += 1
        self.last_published_ts = time.time()
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def publish_event(self, ev: Event) -> bool:
        """Publish sự kiện tức thời lên ds/vision/events (QoS 1)."""
        payload_dict = ev.as_dict()
        payload_str = json.dumps(payload_dict)
        info = self.publish(TOPIC_VISION_EVENTS, payload_str, qos=1)
        self.event_publish_count += 1
        return info.rc == mqtt.MQTT_ERR_SUCCESS


def draw_publisher_hud(frame: np.ndarray, pub: VisionPublisher) -> np.ndarray:
    """Vẽ trạng thái kết nối MQTT và số lượng gói tin đã gửi lên màn hình camera."""
    h = frame.shape[0]

    # Trạng thái MQTT:
    if pub.connected:
        mqtt_status = f"MQTT: ONLINE ({pub.broker_host}:{pub.broker_port})"
        c_mqtt = (0, 255, 0)
    else:
        mqtt_status = f"MQTT: OFFLINE ({pub.broker_host}:{pub.broker_port})"
        c_mqtt = (0, 0, 255)

    info_line = f"PUB: {pub.publish_count} pkts @ 1Hz | EV: {pub.event_publish_count} | LUX: {pub.current_lux_mode.upper()}"

    cv2.putText(frame, mqtt_status, (12, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, mqtt_status, (12, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_mqtt, 2, cv2.LINE_AA)

    cv2.putText(frame, info_line, (12, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, info_line, (12, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)

    return frame


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DriverSafe-IoT Vision Metrics MQTT Publisher (VIS-05)")
    p.add_argument("--host", default="127.0.0.1", help="Dia chi Mosquitto Broker (mac dinh: 127.0.0.1)")
    p.add_argument("--port", type=int, default=1883, help="Port MQTT Broker (mac dinh: 1883)")
    p.add_argument("--src", choices=["webcam", "mjpeg"], default="webcam")
    p.add_argument("--index", type=int, default=0, help="Camera index")
    p.add_argument("--url", default=None, help="MJPEG stream URL")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model", default=None, help="Duong dan den face_landmarker.task")
    p.add_argument("--headless", action="store_true", help="Chay khong mo cua so GUI")
    p.add_argument("--duration", type=float, default=0.0, help="Thoi gian chay (giay), 0 = chay mai")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # 1. Khởi tạo MQTT Publisher
    pub = VisionPublisher(broker_host=args.host, broker_port=args.port)
    print(f"[publisher] Dang ket noi toi broker {args.host}:{args.port}...")
    connected = pub.connect_broker(timeout=3.0)
    if not connected:
        print(f"[publisher] CANH BAO: Chua ket noi duoc broker {args.host}:{args.port}. "
              "Hay chac chan mosquitto dang chay! (tiep tuc thu ket noi lai)", file=sys.stderr)

    # 2. Khởi tạo Face Mesh
    model_path = args.model or str(MODEL_PATH)
    try:
        mesh = FaceMesh(model_path, running_mode="video")
    except (FileNotFoundError, ValueError) as e:
        print(f"[publisher] LOI khoi tao FaceMesh: {e}", file=sys.stderr)
        pub.disconnect_broker()
        return 1

    # 3. Khởi tạo nguồn Camera
    try:
        source = FrameSource(args.src, args.index, args.url, args.width, args.height)
        source.open()
    except (RuntimeError, ValueError) as e:
        print(f"[publisher] LOI khoi tao Camera: {e}", file=sys.stderr)
        mesh.close()
        pub.disconnect_broker()
        return 1

    tm = TemporalMetrics()
    meter = FpsMeter()
    t_start = time.perf_counter()
    last_1hz_pub = t_start
    src_label = args.url if args.src == "mjpeg" else f"webcam[{args.index}]"
    recent_events: list[tuple[str, float]] = []

    print("[publisher] Pipeline Vision Publisher san sang. Nhan q/ESC tren cua so camera de dung.")

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("[publisher] Mat frame tu camera — thoat", file=sys.stderr)
                break

            meter.tick()
            now = time.perf_counter()
            t_ms = (now - t_start) * 1000.0

            # Xử lý FaceMesh và tính toán hình học
            result = mesh.process(frame, int(t_ms))
            geo = compute_geo_metrics(result, (frame.shape[1], frame.shape[0]))

            # Cập nhật lớp thời gian
            tm.update(t_ms, result.present, geo.ear, geo.mar, geo.pitch_deg)

            # Kiểm tra các sự kiện tức thời và publish ngay (QoS 1)
            events = tm.drain_events()
            for ev in events:
                recent_events.append((ev.kind, now))
                pub.publish_event(ev)
                print(f"[publisher EVENT] {ev.kind.upper():<12} | bat dau: {ev.t_ms:.0f}ms | do dai: {ev.duration_ms:.0f}ms")

            # Publish metrics 1Hz định kỳ (QoS 0)
            if now - last_1hz_pub >= 1.0:
                snap = tm.snapshot()
                pub.publish_metrics(snap)
                last_1hz_pub = now
                if args.headless:
                    print(
                        f"[publisher 1Hz #{pub.publish_count}] FPS {meter.fps:4.1f} | "
                        f"EAR {snap.ear:.3f} | PERCLOS {snap.perclos_60s*100:4.1f}% | "
                        f"CLES {snap.cles_dur_ms:4.0f}ms | PITCH {snap.head_pitch_deg:+5.1f}° | "
                        f"HD {snap.head_drop} | MQTT {'OK' if pub.connected else 'DISC'}"
                    )

            # Hiển thị cửa sổ camera trực quan
            if not args.headless:
                draw_landmarks(frame, result)
                draw_status(frame, result)
                draw_metrics(frame, result, geo)
                draw_temporal_hud(frame, tm.snapshot(), recent_events)
                draw_publisher_hud(frame, pub)
                frame = draw_hud(frame, meter.fps, src_label)
                cv2.imshow("DriverSafe-IoT | Vision Publisher", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if args.duration and (now - t_start) >= args.duration:
                break

    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        mesh.close()
        cv2.destroyAllWindows()
        pub.disconnect_broker()

    print(f"[publisher] Da dung chuong trinh an toan. Tong so goi tin da gui: {pub.publish_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
