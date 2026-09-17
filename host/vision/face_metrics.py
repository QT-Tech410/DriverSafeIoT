"""Face Mesh — bắt 468 điểm landmark khuôn mặt và vẽ overlay.

LƯU Ý QUAN TRỌNG VỀ PHIÊN BẢN
-----------------------------
MediaPipe 1.0.1 (bản trong venv) đã **xóa hẳn `mp.solutions`**. Mọi tutorial
dùng `mp.solutions.face_mesh.FaceMesh(...)` đều KHÔNG chạy được ở đây.
API thay thế là `mediapipe.tasks.python.vision.FaceLandmarker`, và nó cần một
file model `.task` tải riêng (khác bản 0.10.x vốn bundle sẵn model):
    python scripts/fetch_models.py

Chạy thử:
    python face_metrics.py                      # webcam, ve overlay
    python face_metrics.py --src mjpeg --url http://<ip>:81/stream
    python face_metrics.py --headless --duration 10
Phím: q/ESC thoát, t = bật/tắt tesselation (nặng), s = chụp ảnh
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# capture.py nằm cùng thư mục — cho phép chạy file này trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import FrameSource, FpsMeter, draw_hud, save_snapshot  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Model FaceLandmarker trả 478 điểm: 468 điểm lưới mặt + 10 điểm iris.
# Phần mềm cũ `refine_landmarks=False` chỉ lấy 468 (bỏ iris). Mọi chỉ số dưới
# đây dùng index < 468 nên chạy giống nhau ở cả hai chế độ.
FACE_LANDMARKS = 468
IRIS_LANDMARKS = 10

# --- Chỉ số landmark dùng cho các chỉ số hình học (spec §6.2) ---
# 6 điểm mỗi mắt, đặt tên P1..P6 theo đúng thứ tự công thức:
#   EAR = (|P2-P6| + |P3-P5|) / (2*|P1-P4|)
# P1/P4 là 2 góc mắt, P2/P3 mí trên, P5/P6 mí dưới.
RIGHT_EYE_P1_P6 = (33, 160, 158, 133, 153, 144)
LEFT_EYE_P1_P6 = (362, 385, 387, 263, 373, 380)

# Miệng cho MAR: cặp dọc (môi trên/dưới) và cặp ngang (2 góc miệng).
MOUTH_VERTICAL = (13, 14)
MOUTH_HORIZONTAL = (61, 291)

# 2 góc mắt ngoài — dùng chuẩn hóa EAR/MAR theo khoảng cách 2 mắt (spec §6.2).
EYE_OUTER_RIGHT = 33
EYE_OUTER_LEFT = 263

# --- Màu overlay (BGR) ---
C_OVAL = (0, 220, 0)
C_EYE = (255, 200, 0)
C_EAR_PT = (0, 255, 255)
C_LIPS = (200, 0, 200)
C_NOSE = (255, 120, 0)
C_TESS = (80, 80, 80)
C_BOX = (0, 255, 0)


@dataclass
class FaceResult:
    """Kết quả 1 frame: có mặt hay không + toạ độ landmark."""

    present: bool
    landmarks_px: np.ndarray | None = None    # (N, 2) float32 — toạ độ pixel
    landmarks_norm: np.ndarray | None = None  # (N, 3) float32 — chuẩn hóa 0..1

    @property
    def count(self) -> int:
        return 0 if self.landmarks_px is None else len(self.landmarks_px)

    def point(self, index: int) -> np.ndarray:
        """Toạ độ pixel của 1 landmark. Gọi khi present=True."""
        if self.landmarks_px is None:
            raise ValueError("khong co landmark (face absent)")
        return self.landmarks_px[index]


_UNSET = FaceResult(present=False)


class FaceMesh:
    """Bọc `FaceLandmarker` của MediaPipe tasks API.

    running_mode:
      - "video": cho webcam/MJPEG — có tracking giữa các frame, mượt hơn khi
        quay mặt. Cần timestamp_ms tăng dần.
      - "image": cho ảnh rời / test fixture, không cần timestamp.
    """

    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        running_mode: str = "video",
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Khong thay model: {model_path}\n"
                f"  Tai bang: python scripts/fetch_models.py\n"
                f"  Hoac curl -L -o \"{model_path}\" {MODEL_URL}"
            )

        mode = {
            "video": vision.RunningMode.VIDEO,
            "image": vision.RunningMode.IMAGE,
        }.get(running_mode)
        if mode is None:
            raise ValueError(f"running_mode khong hop le: {running_mode!r}")

        self.running_mode = running_mode
        self.num_faces = num_faces
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> FaceResult:
        """Chạy Face Mesh trên 1 frame BGR (OpenCV)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # mp.Image doi bo nho lien tuc; cvtColor thuong da tra ve mang moi lien tuc.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        if self.running_mode == "video":
            if timestamp_ms is None:
                raise ValueError("running_mode='video' can timestamp_ms")
            ts = int(timestamp_ms)
            # MediaPipe doi timestamp tang nghiem ngat — chan frame trung/giam.
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            result = self._landmarker.detect_for_video(mp_image, ts)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return FaceResult(present=False)

        # num_faces=1 -> luon lay khuon mat dau (va duy nhat).
        pts = result.face_landmarks[0]
        norm = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
        pixel = np.empty((len(pts), 2), dtype=np.float32)
        pixel[:, 0] = norm[:, 0] * w
        pixel[:, 1] = norm[:, 1] * h
        return FaceResult(present=True, landmarks_px=pixel, landmarks_norm=norm)

    def close(self) -> None:
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "FaceMesh":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _draw_connections(
    frame: np.ndarray,
    pts_i: np.ndarray,
    connections: list,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Vẽ từng đoạn thẳng của một nhóm connection lên frame."""
    for c in connections:
        cv2.line(frame, tuple(pts_i[c.start]), tuple(pts_i[c.end]), color, thickness, cv2.LINE_AA)


def draw_landmarks(
    frame: np.ndarray,
    result: FaceResult,
    tesselation: bool = False,
    metric_points: bool = True,
    bbox: bool = True,
) -> np.ndarray:
    """Vẽ overlay landmark lên frame (in-place). Trả về chính frame đó.

    metric_points=True sẽ tô đậm 6 điểm mắt + 4 điểm miệng dùng cho EAR/MAR —
    giúp nhìn bằng mắt thường biết thuật toán đang đọc đúng điểm nào.
    """
    if not result.present or result.landmarks_px is None:
        return frame

    pts_i = result.landmarks_px.astype(np.int32)
    conn = vision.FaceLandmarksConnections

    if tesselation:
        _draw_connections(frame, pts_i, conn.FACE_LANDMARKS_TESSELATION, C_TESS, 1)

    _draw_connections(frame, pts_i, conn.FACE_LANDMARKS_FACE_OVAL, C_OVAL, 2)
    _draw_connections(frame, pts_i, conn.FACE_LANDMARKS_LEFT_EYE, C_EYE, 1)
    _draw_connections(frame, pts_i, conn.FACE_LANDMARKS_RIGHT_EYE, C_EYE, 1)
    _draw_connections(frame, pts_i, conn.FACE_LANDMARKS_LIPS, C_LIPS, 1)
    _draw_connections(frame, pts_i, conn.FACE_LANDMARKS_NOSE, C_NOSE, 1)

    if metric_points:
        for idx in RIGHT_EYE_P1_P6 + LEFT_EYE_P1_P6:
            cv2.circle(frame, tuple(pts_i[idx]), 2, C_EAR_PT, -1, cv2.LINE_AA)
        for idx in MOUTH_VERTICAL + MOUTH_HORIZONTAL:
            cv2.circle(frame, tuple(pts_i[idx]), 3, C_EAR_PT, -1, cv2.LINE_AA)

    if bbox:
        x0, y0 = pts_i.min(axis=0)
        x1, y1 = pts_i.max(axis=0)
        cv2.rectangle(frame, (x0, y0), (x1, y1), C_BOX, 1, cv2.LINE_AA)

    return frame


def draw_status(frame: np.ndarray, result: FaceResult) -> None:
    """Góc dưới-trái: trạng thái bắt mặt + số landmark."""
    h = frame.shape[0]
    if result.present:
        text, color = f"FACE  {result.count} pts", (0, 255, 0)
    else:
        text, color = "NO FACE", (0, 0, 255)
    cv2.putText(frame, text, (12, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (12, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Face Mesh 468 diem + overlay")
    p.add_argument("--src", choices=["webcam", "mjpeg"], default="webcam")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--url", default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--headless", action="store_true")
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--tesselation", action="store_true", help="ve luoi mo (nang)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        mesh = FaceMesh(args.model, running_mode="video")
    except (FileNotFoundError, ValueError) as e:
        print(f"[face_metrics] LOI: {e}", file=sys.stderr)
        return 1

    try:
        source = FrameSource(args.src, args.index, args.url, args.width, args.height)
        source.open()
    except (RuntimeError, ValueError) as e:
        print(f"[face_metrics] LOI: {e}", file=sys.stderr)
        mesh.close()
        return 1

    src_label = args.url if args.src == "mjpeg" else f"webcam[{args.index}]"
    meter = FpsMeter()
    t_start = time.perf_counter()
    tesselation = args.tesselation
    frames_root = Path(__file__).resolve().parents[2] / "experiments"
    seen_face = frames_seen = 0
    n_landmarks = 0
    last_report = t_start

    print("[face_metrics] q/ESC thoat, t tesselation, s chup anh" if not args.headless else "[face_metrics] headless")
    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("[face_metrics] mat frame — thoat", file=sys.stderr)
                return 2
            meter.tick()
            frames_seen += 1

            ts_ms = int((time.perf_counter() - t_start) * 1000)
            result = mesh.process(frame, ts_ms)
            if result.present:
                seen_face += 1
                n_landmarks = result.count

            if args.headless:
                now = time.perf_counter()
                if now - last_report >= 1.0:
                    pct = 100.0 * seen_face / frames_seen if frames_seen else 0.0
                    print(f"[face_metrics] FPS {meter.fps:5.1f} | face {seen_face}/{frames_seen} "
                          f"({pct:4.0f}%) | pts={n_landmarks}")
                    last_report = now
            else:
                draw_landmarks(frame, result, tesselation=tesselation)
                draw_status(frame, result)
                cv2.imshow("DriverSafe-IoT | face mesh",
                           draw_hud(frame, meter.fps, src_label))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("t"):
                    tesselation = not tesselation
                if key == ord("s"):
                    print(f"[face_metrics] da luu {save_snapshot(frame, frames_root)}")

            if args.duration and time.perf_counter() - t_start >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        mesh.close()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    pct = 100.0 * seen_face / frames_seen if frames_seen else 0.0
    print(f"[face_metrics] ket thuc: {frames_seen} frame / {elapsed:.1f}s | "
          f"FPS {meter.fps_avg:.1f} | bat mat {pct:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())