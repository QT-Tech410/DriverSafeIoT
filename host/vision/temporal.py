"""Lớp thời gian — PERCLOS/CLES/microsleep/yawn/head_drop/face_lost (VIS-04).

Được tối ưu hóa bởi Senior Computer Vision & DSP Engineer tuân thủ theo
tài liệu đặc tả 'docs/fix_errors/eye_tracking_fix_guide.md':
1. Đồng bộ hóa 100% logic xử lý mắt vào vùng tần số chuẩn hóa 10Hz (downsampled).
2. Bộ lọc trung bình động (Moving Average Filter) 3 mẫu triệt tiêu răng cưa EAR.
3. Cơ chế Cập nhật Tăng tiến Độ dài (Continuous Event Update) cho CLES và Microsleep.
4. Điều chỉnh tham số cấu hình sinh học: EYE_DEBOUNCE=6 mẫu, MICROSLEEP_MS=1200ms.
5. Bảo toàn nguyên vẹn kiến trúc OOP và các luồng xử lý độc lập (Yawn, Head-drop, Face-lost).
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

# --- Tần số lấy mẫu & kích thước buffer (spec §6.2) ---
SAMPLE_HZ = 10                       # chuẩn hóa xuống 10 Hz (spec §6.2)
WINDOW_S = 60                        # cửa sổ trượt 60 giây
BUFFER_LEN = SAMPLE_HZ * WINDOW_S    # 600 mẫu

# --- Ngưỡng thời gian sinh học (spec §6.2 & eye_tracking_fix_guide) ---
T_FRAME_MS = 1000.0 / SAMPLE_HZ      # 100ms mỗi mẫu
EYE_DEBOUNCE = 6                     # 6 mẫu (~200ms tại 30FPS / downsampled) vượt đỉnh chớp mắt tự nhiên
EYE_OPEN_DEBOUNCE = 2                # 2 mẫu (~200ms tại 10Hz) khử nhiễu mở mắt (Release Debounce)
EYE_CLOSURE_MS = 250.0               # CLES >= 250ms (nhắm mắt kéo dài)
MICROSLEEP_MS = 1200.0               # >= 1200ms (1.2s - chu kỳ sinh học giấc ngủ trắng microsleep)
YAWN_MS = 400.0                      # MAR >= T_yawn >= 400ms (1 chu kỳ mở->khép)
YAWN_CLOSE_DEBOUNCE = 2              # 2 mẫu (~200ms tại 10Hz) khử nhiễu khép miệng (Release Debounce)
YAWN_FACE_LOST_TOLERANCE_MS = 1500.0 # Dung thứ mất mặt khi ngáp (che tay / ngửa cổ) tối đa 1.5s
HEAD_DROP_MS = 800.0                 # |pitch - neutral| >= 15° >= 0.8s
FACE_LOST_MS = 2000.0                # không thấy mặt >= 2s

# --- Ngưỡng hình học mặc định ---
DEFAULT_T_CLOSED = 0.22              # Ngưỡng EAR nhắm mắt tối ưu cho người châu Á / mắt mí lót
DEFAULT_T_YAWN = 0.50
HEAD_DROP_DEG = 15.0


@dataclass
class Sample:
    """1 mẫu 10 Hz: trạng thái mắt/miệng/đầu tại một thời điểm."""

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
        return {
            "kind": self.kind,
            "t_ms": round(self.t_ms, 0),
            "duration_ms": round(self.duration_ms, 0),
        }


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
    """Xử lý tầng thời gian: Khử nhiễu lọc số, CLES, PERCLOS, Yawn, Head-drop, Face-lost.

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
        eye_debounce: int = EYE_DEBOUNCE,
        eye_open_debounce: int = EYE_OPEN_DEBOUNCE,
        microsleep_ms: float = MICROSLEEP_MS,
        eye_closure_ms: float = EYE_CLOSURE_MS,
    ) -> None:
        self.t_closed = t_closed
        self.t_yawn = t_yawn
        self.pitch_neutral = pitch_neutral
        self.eye_debounce = eye_debounce
        self.eye_open_debounce = eye_open_debounce
        self.microsleep_ms = microsleep_ms
        self.eye_closure_ms = eye_closure_ms
        self._sample_period_ms = 1000.0 / sample_hz

        # Circular buffer 60s: deque giới hạn độ dài tự loại mẫu cũ (10 Hz x 60s = 600 mẫu)
        self._buffer: deque[Sample] = deque(maxlen=int(sample_hz * window_s))

        # --- Bộ lọc trung bình động (Moving Average Filter) 3 mẫu tại 10Hz cho EAR & MAR ---
        self._ear_buffer: deque[float] = deque(maxlen=3)
        self._mar_buffer: deque[float] = deque(maxlen=3)

        # --- Khử nhiễu mắt: đếm mẫu liên tiếp có EAR < ngưỡng ---
        self._closed_run = 0
        self._closed_state = False
        self._eye_run_start: float | None = None   # Onset thực tế (trước debounce)

        # --- Theo dõi CLES (độ dài lần khép hiện tại) ---
        self._cles_start: float | None = None
        self._cles_fired: str | None = None       # "eye_closure" | "microsleep" | None
        self.cles_dur_ms: float = 0.0
        self._open_run: int = 0                   # Bộ đếm khử nhiễu mở mắt (Release Debounce)
        self.last_cles_dur_ms: float = 0.0        # Lưu độ dài trọn vẹn của lần nhắm mắt vừa kết thúc
        self.last_cles_end_t: float = 0.0         # Thời điểm kết thúc lần nhắm mắt vừa qua (ms)

        # --- Yawn: Chu kỳ MAR >= T_yawn (tối ưu theo yawn_logic_fix.md) ---
        self._yawn_open: bool = False
        self._yawn_start: float | None = None
        self._yawn_counted: bool = False
        self._yawn_close_run: int = 0             # Bộ đếm khử nhiễu khép miệng (Release Debounce)
        self._yawn_lost_start: float | None = None # Mốc thời gian mất mặt khi đang ngáp (dung thứ che tay)
        self.last_yawn_dur_ms: float = 0.0        # Lưu độ dài trọn vẹn của lần ngáp vừa kết thúc
        self.last_yawn_end_t: float = 0.0         # Thời điểm kết thúc lần ngáp vừa qua (ms)
        self._yawn_events: deque[float] = deque(maxlen=1200)
        self._yawn_count_window: deque[float] = deque(maxlen=600)

        # --- Head drop ---
        self._hd_start: float | None = None
        self._hd_fired = False

        # --- Face lost ---
        self._face_last_seen: float = 0.0
        self._face_lost_fired = False

        # Hàng đợi sự kiện chờ publish
        self._events: deque[Event] = deque()

        # Biến đệm lưu mẫu cuối & tần số update
        self._last_sample_t: float | None = None
        self._last_ear = 0.0
        self._last_mar = 0.0
        self._last_pitch = 0.0
        self._last_face = False

    # ------------------------------------------------------------------
    # Helper quản trị Event Queue
    # ------------------------------------------------------------------
    def _retract_event(self, kind: str, t_ms: float) -> None:
        """Rút lại sự kiện chưa publish khi được nâng cấp (eye_closure -> microsleep)."""
        for i, e in enumerate(self._events):
            if e.kind == kind and abs(e.t_ms - t_ms) < 1.0:
                del self._events[i]
                return

    def _update_event_duration(self, kind: str, t_ms: float, new_duration_ms: float) -> bool:
        """Giải pháp 3: Cập nhật tăng tiến duration_ms cho event đang có trong hàng đợi."""
        for e in self._events:
            if e.kind == kind and abs(e.t_ms - t_ms) < 1.0:
                e.duration_ms = round(new_duration_ms, 0)
                return True
        return False

    # ------------------------------------------------------------------
    # Cập nhật chu kỳ chính
    # ------------------------------------------------------------------
    def update(
        self,
        t_ms: float,
        face: bool,
        ear: float = 0.0,
        mar: float = 0.0,
        pitch: float = 0.0,
    ) -> None:
        """Nạp 1 mẫu từ Camera FPS (~30 FPS), downsample chuẩn xác xuống 10 Hz và xử lý."""

        # 1. Đo lường Downsampling: Kiểm tra xem đã đủ 1 chu kỳ 100ms (10 Hz) chưa
        is_sample = True
        if self._last_sample_t is not None:
            if t_ms - self._last_sample_t < self._sample_period_ms - 1.0:
                is_sample = False
        if is_sample:
            self._last_sample_t = t_ms

        # Cập nhật biến thô tức thời để tránh trễ khi đọc snapshot ngoài nhịp
        if face:
            self._last_ear = ear
            self._last_mar = mar
            self._last_pitch = pitch
            self._last_face = True
            self._face_last_seen = t_ms
            self._face_lost_fired = False
        else:
            self._last_face = False

        # --- Giải pháp 1: KHÔNG THỰC HIỆN BẤT KỲ THAY ĐỔI TRẠNG THÁI NÀO NGOÀI 10HZ ---
        # Ngăn chặn hoàn toàn xung đột tần số thô camera 30 FPS phá vỡ Release Debounce.
        if not is_sample:
            return

        # ==================================================================
        # VÙNG ĐỒNG BỘ 10 HZ CHUẨN XÁC (T_FRAME = 100ms)
        # ==================================================================

        # --- Áp dụng Bộ lọc Trung bình động (Moving Average Filter) 3 mẫu cho EAR & MAR ---
        if self._last_face:
            self._ear_buffer.append(self._last_ear)
            smoothed_ear = sum(self._ear_buffer) / len(self._ear_buffer)
            self._mar_buffer.append(self._last_mar)
            smoothed_mar = sum(self._mar_buffer) / len(self._mar_buffer)
        else:
            self._ear_buffer.clear()
            smoothed_ear = 0.0
            self._mar_buffer.clear()
            smoothed_mar = 0.0

        # --- Xử lý trạng thái mắt với EAR đã được làm mượt ---
        eye_closed_raw = self._last_face and (smoothed_ear < self.t_closed)

        if eye_closed_raw:
            self._open_run = 0
            if self._closed_run == 0:
                self._eye_run_start = t_ms
            self._closed_run += 1

            # Khử nhiễu khép mắt: Bắt buộc >= eye_debounce mẫu mới công nhận
            if self._closed_run >= self.eye_debounce:
                self._closed_state = True
                if self._cles_start is None:
                    self._cles_start = self._eye_run_start
                    self._cles_fired = None

                self.cles_dur_ms = t_ms - self._cles_start

                # --- Giải pháp 3: Phát hiện sự kiện & Cập nhật tăng tiến độ dài (Continuous Update) ---
                if self.cles_dur_ms >= self.microsleep_ms:
                    if self._cles_fired != "microsleep":
                        # Ưu tiên microsleep: rút lại eye_closure nếu trước đó đã phát
                        if self._cles_fired == "eye_closure":
                            self._retract_event("eye_closure", self._cles_start)
                        self._cles_fired = "microsleep"
                        self._events.append(Event("microsleep", self._cles_start, self.cles_dur_ms))
                    else:
                        # Tài xế tiếp tục nhắm mắt -> Liên tục cập nhật duration_ms theo thời gian thực
                        self._update_event_duration("microsleep", self._cles_start, self.cles_dur_ms)

                elif self.cles_dur_ms >= self.eye_closure_ms:
                    if self._cles_fired is None:
                        self._cles_fired = "eye_closure"
                        self._events.append(Event("eye_closure", self._cles_start, self.cles_dur_ms))
                    elif self._cles_fired == "eye_closure":
                        # Liên tục cập nhật duration_ms tăng tiến
                        self._update_event_duration("eye_closure", self._cles_start, self.cles_dur_ms)
            else:
                # Chưa tích lũy đủ eye_debounce mẫu -> vẫn coi là mắt chưa khép hoàn toàn
                self._closed_state = False
                self.cles_dur_ms = 0.0
        else:
            # Mắt có dấu hiệu mở hoặc mất mặt (smoothed_ear >= t_closed hoặc not face)
            self._closed_run = 0
            self._eye_run_start = None

            if self._cles_start is not None:
                self._open_run += 1

                # Cơ chế Khử nhiễu mở mắt (Release Debounce): Cần >= eye_open_debounce mẫu mở liên tiếp
                if self._open_run < self.eye_open_debounce:
                    # Coi như nhiễu chớp nháy / rung MediaPipe -> DUY TRÌ TRẠNG THÁI KHÉP MẮT
                    self._closed_state = True
                    self.cles_dur_ms = t_ms - self._cles_start
                    if self._cles_fired == "microsleep":
                        self._update_event_duration("microsleep", self._cles_start, self.cles_dur_ms)
                    elif self._cles_fired == "eye_closure":
                        self._update_event_duration("eye_closure", self._cles_start, self.cles_dur_ms)
                else:
                    # Đã mở mắt thực sự -> Chốt hạ giá trị thời lượng cuối cùng!
                    self._closed_state = False
                    final_dur = max(0.0, (t_ms - (self._open_run - 1) * self._sample_period_ms) - self._cles_start)
                    if self._cles_fired in ("microsleep", "eye_closure"):
                        updated = self._update_event_duration(self._cles_fired, self._cles_start, final_dur)
                        if not updated:
                            # Nếu event đã bị drain_events() lấy đi từ chu kỳ trước, phát event kết thúc với độ dài trọn vẹn
                            self._events.append(Event(f"{self._cles_fired}_end", self._cles_start, final_dur))

                    self.last_cles_dur_ms = final_dur
                    self.last_cles_end_t = t_ms
                    self._cles_start = None
                    self._cles_fired = None
                    self.cles_dur_ms = 0.0
                    self._open_run = 0
            else:
                self._open_run = 0
                self._closed_state = False
                self.cles_dur_ms = 0.0

        # --- Ghi 1 mẫu vào Circular Buffer 60s ---
        self._buffer.append(
            Sample(
                t_ms=t_ms,
                face=self._last_face,
                closed=self._closed_state,
                mar=self._last_mar,
                pitch=self._last_pitch,
                head_drop=self._hd_start is not None,
            )
        )

        # --- Yawn: MAR >= T_yawn, Chu kỳ Mở -> Khép với Continuous Update & Face-Lost Tolerance ---
        mouth_open_raw = self._last_face and (self._last_mar >= self.t_yawn)

        if mouth_open_raw:
            self._yawn_close_run = 0
            self._yawn_lost_start = None

            if not self._yawn_open:
                self._yawn_open = True
                self._yawn_start = t_ms
                self._yawn_counted = False
            else:
                dur = t_ms - self._yawn_start
                if dur >= YAWN_MS:
                    if not self._yawn_counted:
                        self._yawn_counted = True
                        self._yawn_count_window.append(t_ms)
                        self._events.append(Event("yawn", self._yawn_start, dur))
                    else:
                        # Giải pháp 1: Liên tục cập nhật độ dài tăng tiến theo thời gian thực (Continuous Update)
                        self._update_event_duration("yawn", self._yawn_start, dur)
        else:
            if self._yawn_open:
                if self._last_face:
                    # Trường hợp A: Tài xế chủ động khép miệng (Vẫn phát hiện khuôn mặt nhưng MAR < t_yawn)
                    self._yawn_lost_start = None
                    self._yawn_close_run += 1

                    if self._yawn_close_run < YAWN_CLOSE_DEBOUNCE:
                        # Khử nhiễu khép miệng: Duy trì chu kỳ ngáp để tránh jitter landmark môi
                        dur = t_ms - self._yawn_start
                        if self._yawn_counted:
                            self._update_event_duration("yawn", self._yawn_start, dur)
                    else:
                        # Đã thực sự khép miệng -> Chốt hạ độ dài trọn vẹn của hành vi ngáp
                        final_dur = max(0.0, (t_ms - (self._yawn_close_run - 1) * self._sample_period_ms) - self._yawn_start)
                        if self._yawn_counted:
                            updated = self._update_event_duration("yawn", self._yawn_start, final_dur)
                            if not updated:
                                self._events.append(Event("yawn_end", self._yawn_start, final_dur))
                            self.last_yawn_dur_ms = final_dur
                            self.last_yawn_end_t = t_ms

                        # Reset trạng thái ngáp sau khi kết thúc chu kỳ
                        self._yawn_open = False
                        self._yawn_start = None
                        self._yawn_counted = False
                        self._yawn_close_run = 0
                else:
                    # Trường hợp B: Mất dấu khuôn mặt khi đang ngáp (do che tay hoặc ngửa cổ)
                    self._yawn_close_run = 0
                    if self._yawn_lost_start is None:
                        self._yawn_lost_start = t_ms

                    lost_gap = t_ms - self._yawn_lost_start
                    if lost_gap < YAWN_FACE_LOST_TOLERANCE_MS:
                        # Trong ngưỡng dung thứ (1500ms): KHÔNG reset đột ngột cờ trạng thái ngáp
                        dur = self._yawn_lost_start - self._yawn_start
                        if self._yawn_counted:
                            self._update_event_duration("yawn", self._yawn_start, dur)
                    else:
                        # Vượt quá thời gian dung thứ: Chốt hạ thời lượng tại thời điểm trước khi mất mặt
                        final_dur = max(0.0, self._yawn_lost_start - self._yawn_start)
                        if self._yawn_counted:
                            updated = self._update_event_duration("yawn", self._yawn_start, final_dur)
                            if not updated:
                                self._events.append(Event("yawn_end", self._yawn_start, final_dur))
                            self.last_yawn_dur_ms = final_dur
                            self.last_yawn_end_t = t_ms

                        self._yawn_open = False
                        self._yawn_start = None
                        self._yawn_counted = False
                        self._yawn_close_run = 0
                        self._yawn_lost_start = None

        # --- Head drop: pitch <= neutral - 15°, >= 0.8s (Bảo toàn 100%) ---
        hd_now = self._last_face and (self._last_pitch <= self.pitch_neutral - HEAD_DROP_DEG)
        if not hd_now:
            self._hd_start = None
            self._hd_fired = False
        elif self._hd_start is None:
            self._hd_start = t_ms
            self._hd_fired = False
        elif not self._hd_fired and t_ms - self._hd_start >= HEAD_DROP_MS:
            self._hd_fired = True
            self._events.append(Event("head_drop", self._hd_start, t_ms - self._hd_start))

        # --- Face lost: không thấy mặt >= 2s (Bảo toàn 100%) ---
        if self._last_face:
            self._face_last_seen = t_ms
            self._face_lost_fired = False
        else:
            lost_dur = t_ms - self._face_last_seen
            if lost_dur >= FACE_LOST_MS and not self._face_lost_fired:
                self._face_lost_fired = True
                self._events.append(Event("face_lost", self._face_last_seen, lost_dur))

    # ------------------------------------------------------------------
    # Truy vấn & Thống kê
    # ------------------------------------------------------------------
    def perclos(self) -> float:
        """PERCLOS(P78): Tỷ lệ mẫu mắt khép trên cửa sổ 60s."""
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
        t_now = last.t_ms if last else 0.0

        # Nếu mắt đang nhắm: lấy thời gian nhắm hiện tại (tăng dần)
        # Nếu mắt vừa mở trong vòng 3.0s: giữ hiển thị thời lượng nhắm mắt của lần vừa qua
        if self._cles_start is not None:
            active_cles = self.cles_dur_ms
        elif (t_now - self.last_cles_end_t) <= 3000.0 and self.last_cles_dur_ms >= self.eye_closure_ms:
            active_cles = self.last_cles_dur_ms
        else:
            active_cles = 0.0

        return MetricsSnapshot(
            ts=t_now,
            face=self._last_face,
            ear=self._last_ear,
            perclos_60s=self.perclos(),
            cles_dur_ms=active_cles,
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

    def set_thresholds(
        self,
        t_closed: float | None = None,
        t_yawn: float | None = None,
        pitch_neutral: float | None = None,
    ) -> None:
        """Cập nhật ngưỡng cá nhân hóa sau enroll / auto-calibrate."""
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
        self._ear_buffer.clear()
        self._closed_run = 0
        self._closed_state = False
        self._eye_run_start = None
        self._cles_start = None
        self._cles_fired = None
        self._open_run = 0
        self.last_cles_dur_ms = 0.0
        self.last_cles_end_t = 0.0
        self._yawn_open = False
        self._yawn_start = None
        self._yawn_counted = False
        self._hd_start = None
        self._hd_fired = False
        self._face_last_seen = 0.0
        self._face_lost_fired = False
        self._last_sample_t = None
        self.cles_dur_ms = 0.0


def draw_temporal_hud(
    frame: np.ndarray,
    snap: MetricsSnapshot,
    recent_events: list[tuple[str, float]] | None = None,
) -> np.ndarray:
    """Vẽ overlay HUD hiển thị PERCLOS, CLES, Yawn, Head-drop trên khung hình camera."""
    h, w = frame.shape[:2]

    # Màu sắc PERCLOS
    if snap.perclos_60s >= 0.30:
        c_perclos = (0, 0, 255)       # Đỏ: CRITICAL
    elif snap.perclos_60s >= 0.15:
        c_perclos = (0, 165, 255)     # Cam: WARN
    else:
        c_perclos = (0, 255, 0)       # Xanh: SAFE

    # Màu sắc CLES
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
