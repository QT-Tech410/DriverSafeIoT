"""Lớp thời gian — PERCLOS/CLES/microsleep/yawn/head_drop/face_lost (VIS-04).

Circular buffer 60 giây lấy mẫu 10 Hz (spec §6.2): mỗi mẫu lưu trạng thái mắt,
MAR, pitch, và timestamp. Tần số camera ~30 FPS được downsample xuống 10 Hz
trước khi vào buffer, đúng như spec yêu cầu ("mẫu chuẩn hóa xuống 10Hz để
buffer gọn").

Chống nhiễu trạng thái mắt khép: cần >= 3 frame liên tiếp (~100ms) mới công
nhận "mắt khép" — tránh nháy mắt vã (blink thường 100-250ms) bị tính nhầm
thành buồn ngủ.

Các sự kiện (publish ngay, QoS1):
  - eye_closure: mắt khép >= 250ms (CLES >= 250ms)
  - microsleep:  mắt khép >= 500ms
  - yawn:        MAR >= T_yawn duy trì >= 400ms (1 chu kỳ mở->khép)
  - head_drop:   pitch lệch khỏi neutral >= 15° về phía cúi xuống, >= 0.8s
  - face_lost:   không thấy mặt >= 2s

LỜI CAM KẾT CHỮ KÝ
------------------
Spec §6.2 viết "head_drop = pitch > neutral + 15°". Đã kiểm chứng thực
nghiệm (VIS-03, trên camera thật, model 3D theo convention OpenCV y-xuống)
rằng: nhìn thẳng ≈ 0, CÚI ĐẦU ra phía trước là pitch ÂM, ngẩng lên là pitch
DƯƠNG. Tài xế ngủ gật = cúi xuống, nên điều kiện so sánh phải là
"pitch <= neutral − 15°" (nhỏ hơn). Đây là sự dịch dấu có chủ đích, giải
thích tại đây và tại compute_pitch — nếu dùng y-xuống thì phải đảo dấu.

Tương tự, EAR/MAR dùng làm đầu vào so ngưỡng là bản KHÔNG THỨ NGUYÊN
(ear, mar), không phải ear_norm/mar_norm của spec — lý do dimensional-analysis
ghi tại face_metrics.GeoMetrics.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# --- Tần số lấy mẫu & kích thước buffer ---
SAMPLE_HZ = 10                       # chuẩn hóa xuống 10 Hz (spec §6.2)
WINDOW_S = 60                        # cửa sổ trượt 60 giây
BUFFER_LEN = SAMPLE_HZ * WINDOW_S    # 600 mẫu

# --- Ngưỡng thời gian (spec §6.2) ---
T_FRAME_MS = 1000.0 / SAMPLE_HZ      # 100ms mỗi mẫu
EYE_DEBOUNCE = 3                     # 3 mẫu liên tiếp (~100ms) mới công nhận khép
EYE_CLOSURE_MS = 250.0               # CLES >= 250ms
MICROSLEEP_MS = 500.0                # >= 500ms
YAWN_MS = 400.0                      # MAR >= T_yawn >= 400ms
HEAD_DROP_MS = 800.0                 # |pitch - neutral| >= 15° >= 0.8s
FACE_LOST_MS = 2000.0               # không thấy mặt >= 2s

# --- Ngưỡng hình học mặc định (sẽ bị ghi đè bởi enroll, spec §6.3) ---
# EAR/MAR ở đây là bản KHÔNG THỨ NGUYÊN (xem face_metrics.GeoMetrics):
#   - EAR < 0.20 ≈ mắt khép (giá trị tham khảo, màn hình máy tính).
#   - MAR >= 0.50 ≈ miệng mở quá mức nghỉ — kiểm chứng trực quan trên camera
#     (ngậm miệng ~0.00-0.05, ngáp/há miệng tới 0.9+). Chỉ nên coi là dự phòng:
#     spec §6.3 yêu cầu enroll 60s rồi T_yawn = mu_mar_open + 1.5 sigma.
DEFAULT_T_CLOSED = 0.20
DEFAULT_T_YAWN = 0.50
HEAD_DROP_DEG = 15.0


@dataclass
class Sample:
    """1 mẫu 10 Hz: trạng thái mắt/miếng/đầu tại một thời điểm."""

    t_ms: float            # timestamp gốc (ms), từ đầu phiên
    face: bool
    closed: bool           # mắt khép (đã qua khử nhiễu)
    mar: float
    pitch: float           # độ
    head_drop: bool        # pitch lệch đủ lớn so với neutral tại mẫu này


@dataclass
class Event:
    """Một sự kiện phát hiện được, để publish lên MQTT."""

    kind: str              # eye_closure | microsleep | yawn | head_drop | face_lost
    t_ms: float            # thời điểm bắt đầu (ms)
    duration_ms: float     # thời lượng

    def as_dict(self) -> dict:
        return {"kind": self.kind, "t_ms": round(self.t_ms, 0),
                "duration_ms": round(self.duration_ms, 0)}


@dataclass
class MetricsSnapshot:
    """Đặc tả các chỉ số để publish ds/vision/metrics (spec §5)."""

    ts: float
    face: bool
    ear: float
    perclos_60s: float
    cles_dur_ms: float
    mar: float
    yawn_per_min: float
    head_pitch_deg: float
    head_drop: bool
    lux_mode: str = "day"

    def as_dict(self) -> dict:
        return {
            "ts": round(self.ts, 1),
            "face": self.face,
            "ear": round(self.ear, 4),
            "perclos_60s": round(self.perclos_60s, 4),
            "cles_dur_ms": round(self.cles_dur_ms, 0),
            "mar": round(self.mar, 4),
            "yawn_per_min": round(self.yawn_per_min, 2),
            "head_pitch_deg": round(self.head_pitch_deg, 1),
            "head_drop": self.head_drop,
            "lux_mode": self.lux_mode,
        }


class TemporalMetrics:
    """Xử lý thời gian: khử nhiễu mắt, tính PERCLOS, phát hiện sự kiện.

    Sử dụng:
        tm = TemporalMetrics()
        # mỗi frame camera (~30 FPS):
        tm.update(t_ms, face, ear, mar, pitch)
        # mỗi 1 giây:
        snap = tm.snapshot(); events = tm.drain_events()
    """

    def __init__(
        self,
        t_closed: float = DEFAULT_T_CLOSED,
        t_yawn: float = DEFAULT_T_YAWN,
        pitch_neutral: float = 0.0,
        window_s: int = WINDOW_S,
        sample_hz: int = SAMPLE_HZ,
    ) -> None:
        self.t_closed = t_closed
        self.t_yawn = t_yawn
        self.pitch_neutral = pitch_neutral
        self._sample_period_ms = 1000.0 / sample_hz

        # Circular buffer 60s: deque giới hạn độ dài tự loại mẫu cũ.
        self._buffer: deque[Sample] = deque(maxlen=int(sample_hz * window_s))

        # --- Khử nhiễu mắt: đếm mẫu liên tiếp có EAR < ngưỡng ---
        self._closed_run = 0
        self._closed_state = False
        self._eye_run_start: float | None = None   # onset thực tế (chưa qua debounce)

        # --- Theo dõi CLES (độ dài lần khép hiện tại) ---
        self._cles_start: float | None = None
        self._cles_fired: str | None = None   # "eye_closure" | "microsleep" | None
        self.cles_dur_ms: float = 0.0

        # --- Yawn: chu kỳ MAR >= T_yawn ---
        self._yawn_open = False
        self._yawn_start: float | None = None
        self._yawn_counted = False
        self._yawn_events: deque[float] = deque(maxlen=1200)  # toàn bộ phiên

        # --- Head drop ---
        self._hd_start: float | None = None
        self._hd_fired = False

        # --- Face lost ---
        # Khởi tạo = 0 (đầu phiên): ngay cả khi chưa từng thấy mặt, phiên được
        # tính là "đang theo dõi" từ giây 0, nên mất mặt >= 2s vẫn phải báo.
        self._face_last_seen: float = 0.0
        self._face_lost_fired = False

        # Sự kiện chờ publish.
        self._events: deque[Event] = deque()

        # Mẫu cuối cùng & tần số update.
        self._last_sample_t: float | None = None
        self._last_ear = 0.0
        self._last_mar = 0.0
        self._last_pitch = 0.0
        self._last_face = False

        # Cho yawn_per_min: thống kê số ngáp mỗi phút (đếm trên 60s).
        self._yawn_count_window: deque[float] = deque(maxlen=600)

    # ------------------------------------------------------------------
    # Cập nhật
    # ------------------------------------------------------------------
    def _retract_event(self, kind: str, t_ms: float) -> None:
        """Rút lại sự kiện chưa publish (dùng khi microsleep thay eye_closure).

        Khi CLES từ 250ms tiến tới 500ms, sự kiện đã chốt là eye_closure phải
        được nâng cấp thành microsleep — tài xế ngủ gật thật sự.
        """
        for i, e in enumerate(self._events):
            if e.kind == kind and e.t_ms == t_ms:
                del self._events[i]
                return
    def update(
        self,
        t_ms: float,
        face: bool,
        ear: float = 0.0,
        mar: float = 0.0,
        pitch: float = 0.0,
    ) -> None:
        """Nạp 1 mẫu. Chấp nhận tần số bất kỳ, tự downsample xuống 10 Hz.

        t_ms: timestamp từ đầu phiên (ms). ear/mar/pitch: từ face_metrics.
        """
        # Downsample: bỏ qua mẫu quá gần mẫu trước — nhưng vẫn cập nhật biến
        # "trạng thái hiện tại" (mắt/miếng/đầu) để snapshot() không bị giật cục.
        is_sample = True
        if self._last_sample_t is not None:
            if t_ms - self._last_sample_t < self._sample_period_ms - 1.0:
                is_sample = False
        if is_sample:
            self._last_sample_t = t_ms

        # Cập nhật trạng thái hiện tại cho dù có bị downsample.
        if face:
            self._last_ear, self._last_mar, self._last_pitch = ear, mar, pitch
            self._last_face = True
            self._face_last_seen = t_ms
            self._face_lost_fired = False
        else:
            self._last_face = False
            # Mở mắt sau một CLES dài phải được xử lý để cles_dur_ms về 0,
            # kể cả khi mẫu này bị loại do downsample.
            if self._cles_start is not None:
                self._cles_start = None
                self._cles_fired = None
                self.cles_dur_ms = 0.0
            self._closed_run = 0
            self._eye_run_start = None

        if not is_sample:
            return

        # --- Khử nhiễu mắt khép: >= 3 mẫu liên tiếp ---
        # Ghi timestamp bắt đầu chu kỳ khép THỰC SỰ (chưa qua debounce):
        # khi debounce xong, CLES phải tính từ mẫu đầu tiên khép, chứ không
        # phải mẫu thứ 3 — nếu không sẽ bị thiếu mất 200ms độ dài.
        eye_closed_raw = self._last_face and ear < self.t_closed
        if eye_closed_raw:
            if self._closed_run == 0:
                self._eye_run_start = t_ms
            self._closed_run += 1
        else:
            self._closed_run = 0
            self._eye_run_start = None
        closed = self._closed_run >= EYE_DEBOUNCE
        self._closed_state = closed

        # --- CLES duration: đo độ dài lần khép hiện tại ---
        # Chốt sự kiện NGAY KHI đủ ngưỡng (không chờ mở mắt): nếu tài xế
        # nhắm mắt và không bao giờ mở lại (ngủ gật cuối phiên), sự kiện
        # vẫn phải được phát đi. Cờ _cles_fired tránh phát 2 lần.
        if eye_closed_raw:
            if self._cles_start is None:
                self._cles_start = self._eye_run_start
                self._cles_fired = None
            self.cles_dur_ms = t_ms - self._cles_start
            if self.cles_dur_ms >= MICROSLEEP_MS:
                if self._cles_fired != "microsleep":
                    # Ưu tiên microsleep: nếu đã chốt eye_closure trước đó,
                    # rút lại sự kiện đó và thay bằng microsleep.
                    if self._cles_fired == "eye_closure":
                        self._retract_event("eye_closure", self._cles_start)
                    self._cles_fired = "microsleep"
                    self._events.append(Event("microsleep", self._cles_start,
                                              self.cles_dur_ms))
            elif self.cles_dur_ms >= EYE_CLOSURE_MS:
                # 250ms <= CLES < 500ms: nhắm mắt dài, chưa phải microsleep.
                if self._cles_fired is None:
                    self._cles_fired = "eye_closure"
                    self._events.append(Event("eye_closure", self._cles_start,
                                              self.cles_dur_ms))
        else:
            # Mắt mở: kết thúc CLES (theo schema ds/vision/metrics = 0).
            self._cles_start = None
            self._cles_fired = None
            self.cles_dur_ms = 0.0

        # --- Yawn: MAR >= T_yawn, đếm khi kết thúc chu kỳ (mở->khép) ---
        yawn_now = self._last_face and self._last_mar >= self.t_yawn
        if yawn_now and not self._yawn_open:
            self._yawn_open = True
            self._yawn_start = t_ms
        elif yawn_now and self._yawn_open:
            # Vẫn còn ngáp: chốt ngay khi đủ 400ms (không phải chờ hạ miệng).
            dur = t_ms - self._yawn_start
            if dur >= YAWN_MS and not self._yawn_counted:
                self._yawn_counted = True
                self._yawn_count_window.append(t_ms)
                self._events.append(Event("yawn", self._yawn_start, dur))
        elif not yawn_now and self._yawn_open:
            # Hạ miệng: reset cờ đếm để chu kỳ ngáp sau lại đếm được.
            self._yawn_open = False
            self._yawn_start = None
            self._yawn_counted = False

        # --- Head drop: pitch < neutral - 15°, >= 0.8s ---
        # Spec §6.2 viết "pitch > neutral + 15°". Trong convention camera
        # (trục y xuống) đã kiểm chứng thực nghiệm ở VIS-03: nhìn thẳng ≈ 0,
        # CÚI ĐẦU → pitch âm (−28°), ngẩng đầu → dương. Tài xế ngủ gật là
        # cúi xuống, nên head_drop = pitch <= (neutral − 15°) (lấy cả điểm
        # biên, đúng dấu "≥ 15°" của spec).
        hd_now = self._last_face and (pitch <= self.pitch_neutral - HEAD_DROP_DEG)
        if not hd_now:
            self._hd_start = None
            self._hd_fired = False
        elif self._hd_start is None:
            self._hd_start = t_ms
            self._hd_fired = False
        elif not self._hd_fired and t_ms - self._hd_start >= HEAD_DROP_MS:
            # Vẫn đang cúi: chốt sự kiện ngay khi đủ 0.8s (không chờ ngẩng lên).
            self._hd_fired = True
            self._events.append(Event("head_drop", self._hd_start,
                                      t_ms - self._hd_start))

        # --- Face lost: không thấy mặt >= 2s ---
        # Chốt ngay khi đủ 2s (trạng thái có thể kéo dài đến hết phiên).
        if not self._last_face and not self._face_lost_fired:
            if t_ms - self._face_last_seen >= FACE_LOST_MS:
                self._events.append(Event("face_lost", self._face_last_seen,
                                          t_ms - self._face_last_seen))
                self._face_lost_fired = True

        # Lưu mẫu vào circular buffer.
        self._buffer.append(Sample(
            t_ms=t_ms, face=self._last_face, closed=closed,
            mar=self._last_mar, pitch=self._last_pitch,
            head_drop=hd_now,
        ))

    # ------------------------------------------------------------------
    # Truy vấn
    # ------------------------------------------------------------------
    def perclos(self) -> float:
        """PERCLOS(P78): tỷ lệ mẫu mắt khép trên cửa sổ 60s."""
        if not self._buffer:
            return 0.0
        n_closed = sum(1 for s in self._buffer if s.closed)
        return n_closed / len(self._buffer)

    def yawn_per_min(self) -> float:
        """Số lần ngáp trong 60s gần nhất, quy ra /phút."""
        if not self._buffer:
            return 0.0
        t_now = self._buffer[-1].t_ms
        t_start = t_now - 60000.0
        n = sum(1 for t in self._yawn_count_window if t >= t_start)
        return float(n)

    def snapshot(self) -> MetricsSnapshot:
        """Đặc tả các chỉ số tại thời điểm hiện tại (publish 1Hz)."""
        buf = list(self._buffer)
        last = buf[-1] if buf else None
        return MetricsSnapshot(
            ts=last.t_ms if last else 0.0,
            face=self._last_face,
            ear=self._last_ear,
            perclos_60s=self.perclos(),
            cles_dur_ms=self.cles_dur_ms if self._cles_start is not None else 0.0,
            mar=self._last_mar,
            yawn_per_min=self.yawn_per_min(),
            head_pitch_deg=self._last_pitch,
            head_drop=self._hd_start is not None,
            lux_mode="day",
        )

    def drain_events(self) -> list[Event]:
        """Lấy các sự kiện đã phát hiện (chuyển quyền sở hữu sang MQTT)."""
        out = list(self._events)
        self._events.clear()
        return out

    def set_thresholds(self, t_closed: float | None = None,
                       t_yawn: float | None = None,
                       pitch_neutral: float | None = None) -> None:
        """Cập nhật ngưỡng cá nhân hóa sau enroll (spec §6.3)."""
        if t_closed is not None:
            self.t_closed = t_closed
        if t_yawn is not None:
            self.t_yawn = t_yawn
        if pitch_neutral is not None:
            self.pitch_neutral = pitch_neutral

    @property
    def buffer_len(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        """Xoá toàn bộ lịch sử (bắt đầu phiên mới)."""
        self._buffer.clear()
        self._events.clear()
        self._yawn_count_window.clear()
        self._closed_run = 0
        self._closed_state = False
        self._eye_run_start = None
        self._cles_start = None
        self._cles_fired = None
        self._yawn_open = False
        self._yawn_start = None
        self._yawn_counted = False
        self._hd_start = None
        self._hd_fired = False
        # Đầu phiên: phiên được "theo dõi" từ giây 0 (xem __init__).
        self._face_last_seen = 0.0
        self._face_lost_fired = False
        self._last_sample_t = None
        self.cles_dur_ms = 0.0


def draw_temporal_hud(
    frame: np.ndarray,
    snap: MetricsSnapshot,
    recent_events: list[tuple[str, float]] | None = None,
) -> np.ndarray:
    """Vẽ bảng chỉ số thời gian lên góc trên-phải và alert banner lên frame."""
    h, w = frame.shape[:2]

    # Màu sắc PERCLOS theo mức độ rủi ro (xanh / cam / đỏ)
    if snap.perclos_60s >= 0.30:
        c_perclos = (0, 0, 255)
    elif snap.perclos_60s >= 0.15:
        c_perclos = (0, 165, 255)
    else:
        c_perclos = (0, 255, 0)

    # CLES duration
    if snap.cles_dur_ms >= MICROSLEEP_MS:
        c_cles = (0, 0, 255)
    elif snap.cles_dur_ms >= EYE_CLOSURE_MS:
        c_cles = (0, 165, 255)
    else:
        c_cles = (220, 220, 220)

    c_hd = (0, 0, 255) if snap.head_drop else (220, 220, 220)

    lines = [
        (f"PERCLOS: {snap.perclos_60s * 100:4.1f}%", c_perclos),
        (f"CLES: {snap.cles_dur_ms:4.0f} ms", c_cles),
        (f"YAWN/min: {snap.yawn_per_min:4.1f}", (255, 200, 0)),
        (f"HEAD-DROP: {'ON' if snap.head_drop else 'off'}", c_hd),
    ]

    y = 40
    for text, color in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = w - tw - 12
        cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 24

    # Vẽ alert banner ở cạnh dưới nếu có sự kiện gần đây (trong 2.5s)
    if recent_events:
        now = time.perf_counter()
        active = [ev for ev in recent_events if now - ev[1] < 2.5]
        if active:
            last_kind, _ = active[-1]
            banner_text = f"EVENT: {last_kind.upper()}"
            is_crit = last_kind in ("microsleep", "face_lost", "head_drop")
            color = (0, 0, 255) if is_crit else (0, 165, 255)
            (tw, _), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            bx = max(10, (w - tw) // 2)
            by = h - 50
            cv2.putText(frame, banner_text, (bx + 2, by + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, banner_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    return frame


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Temporal Metrics: PERCLOS, CLES, Yawn, Head-drop")
    p.add_argument("--src", choices=["webcam", "mjpeg"], default="webcam")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--url", default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model", default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--duration", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Import modules cùng thư mục vision
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

    model_path = args.model or str(MODEL_PATH)
    try:
        mesh = FaceMesh(model_path, running_mode="video")
    except (FileNotFoundError, ValueError) as e:
        print(f"[temporal] LOI: {e}", file=sys.stderr)
        return 1

    try:
        source = FrameSource(args.src, args.index, args.url, args.width, args.height)
        source.open()
    except (RuntimeError, ValueError) as e:
        print(f"[temporal] LOI: {e}", file=sys.stderr)
        mesh.close()
        return 1

    tm = TemporalMetrics()
    meter = FpsMeter()
    t_start = time.perf_counter()
    src_label = args.url if args.src == "mjpeg" else f"webcam[{args.index}]"
    recent_events: list[tuple[str, float]] = []
    last_log_t = t_start

    print("[temporal] Khoi dong thanh cong. Phim: q/ESC = thoat." if not args.headless else "[temporal] Headless mode")

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("[temporal] Mat frame tu camera — thoat", file=sys.stderr)
                return 2

            meter.tick()
            now = time.perf_counter()
            t_ms = (now - t_start) * 1000.0

            result = mesh.process(frame, int(t_ms))
            geo = compute_geo_metrics(result, (frame.shape[1], frame.shape[0]))
            tm.update(t_ms, result.present, geo.ear, geo.mar, geo.pitch_deg)

            # Lấy các sự kiện mới phát hiện
            events = tm.drain_events()
            for ev in events:
                recent_events.append((ev.kind, now))
                print(f"[temporal EVENT] {ev.kind.upper():<12} | bat dau: {ev.t_ms:.0f}ms | do dai: {ev.duration_ms:.0f}ms")

            snap = tm.snapshot()

            if args.headless:
                if now - last_log_t >= 1.0:
                    print(
                        f"[temporal 1Hz] FPS {meter.fps:4.1f} | "
                        f"EAR {snap.ear:.3f} | PERCLOS {snap.perclos_60s*100:4.1f}% | "
                        f"CLES {snap.cles_dur_ms:4.0f}ms | YAWN/min {snap.yawn_per_min:3.1f} | "
                        f"PITCH {snap.head_pitch_deg:+5.1f} deg | HD {snap.head_drop}"
                    )
                    last_log_t = now
            else:
                draw_landmarks(frame, result)
                draw_status(frame, result)
                draw_metrics(frame, result, geo)
                draw_temporal_hud(frame, snap, recent_events)
                frame = draw_hud(frame, meter.fps, src_label)
                cv2.imshow("DriverSafe-IoT | Temporal Metrics", frame)

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

    print("[temporal] Da dung tien trinh an toan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

