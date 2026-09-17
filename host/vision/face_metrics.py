"""Face Mesh + chỉ số hình học — 468 điểm landmark, EAR/MAR/Pitch (VIS-02 + VIS-03).

LƯU Ý QUAN TRỌNG VỀ PHIÊN BẢN
-----------------------------
MediaPipe 1.0.1 (bản trong venv) đã **xóa hẳn `mp.solutions`**. Mọi tutorial
dùng `mp.solutions.face_mesh.FaceMesh(...)` đều KHÔNG chạy được ở đây.
API thay thế là `mediapipe.tasks.python.vision.FaceLandmarker`, và nó cần một
file model `.task` tải riêng (khác bản 0.10.x vốn bundle sẵn model):
    python scripts/fetch_models.py

Các chỉ số (spec §6.2):
  - EAR = (|P2-P6| + |P3-P5|) / (2*|P1-P4|), chuẩn hóa theo khoảng cách 2 mắt
  - MAR = |P_lips_ver| / |P_lips_hor| (chuẩn hóa tương tự)
  - Pitch  từ cv2.solvePnP với 5 điểm mặt 3D -> Euler pitch

Chạy thử:
    python face_metrics.py                      # webcam, ve overlay + chi so
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

# --- Head pose: 5 điểm mặt 3D mô hình hóa (spec §6.2) cho cv2.solvePnP ---
# Convention camera OpenCV: x sang phai, y XUONG, z ve phia camera.
# Dung convention nay (chu khong phai y-len nhu toa do 3D thong thuong) thi
# mat nhin thang cho R = don vi, cong thuc Euler pitch = atan2(R21, R22) = 0.
HEAD_POSE_3D = np.array([
    [-0.045, -0.025, -0.040],   # mat phai (xuat hien ben trai anh)
    [ 0.045, -0.025, -0.040],   # mat trai
    [ 0.000,  0.000,  0.035],   # chop mui (gan camera nhat)
    [-0.028,  0.030, -0.015],   # memp mieng phai
    [ 0.028,  0.030, -0.015],   # memp mieng trai
], dtype=np.float64)
HEAD_POSE_2D_IDX = (EYE_OUTER_RIGHT, EYE_OUTER_LEFT, 1, 61, 291)

# Một số đối số kỹ thuật: CV_CALIB_USE_INTRINSIC_GUESS phải đi cùng ma trận K + distortion.
# Camera 640x480 tiêu chuẩn: f ≈ fx = fy = 500 (góc nhìn ~57°), điểm chính giữa khung.
_DEFAULT_K = np.array([
    [500.0,   0.0, 320.0],
    [  0.0, 500.0, 240.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)
_DEFAULT_DIST = np.zeros(5, dtype=np.float64)


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


@dataclass
class GeoMetrics:
    """Các chỉ số hình học 1 frame (spec §6.2) — tất cả đều không thứ nguyên."""

    ear_right: float        # EAR mắt phải (tỷ lệ, chưa chuẩn hóa)
    ear_left: float
    ear: float              # trung bình 2 mắt — dùng cho EAR_norm
    ear_norm: float         # chuẩn hóa theo khoảng cách 2 mắt (spec §6.2)
    mar: float              # MAR đã chuẩn hóa
    pitch_deg: float        # Euler pitch từ solvePnP, độ
    head_ok: bool           # solvePnP hội tụ không

    def as_dict(self) -> dict:
        return {
            "ear": round(self.ear, 4),
            "ear_norm": round(self.ear_norm, 4),
            "mar": round(self.mar, 4),
            "pitch_deg": round(self.pitch_deg, 1),
            "head_ok": self.head_ok,
        }


_NO_FACE_GEO = GeoMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)


def _segment_length(pts: np.ndarray, i: int, j: int) -> float:
    """Khoảng cách Euclid giữa 2 landmark (dạng pixel float)."""
    return float(np.linalg.norm(pts[i] - pts[j]))


def compute_ear(pts: np.ndarray) -> tuple[float, float, float, float]:
    """EAR mắt phải / trái / trung bình, và cả bản chuẩn hóa (spec §6.2).

    Công thức:  EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
    P1..P6 theo RIGHT_EYE_P1_P6 / LEFT_EYE_P1_P6 (P1/P4 = góc mắt,
    P2/P3 = mí trên, P5/P6 = mí dưới).
    Trả về (ear_right, ear_left, ear_avg, ear_norm) — ear_norm = ear_avg / inter-eye.
    """
    inter_eye = _segment_length(pts, EYE_OUTER_RIGHT, EYE_OUTER_LEFT)

    def one_eye(idxs: tuple[int, ...]) -> float:
        p1, p2, p3, p4, p5, p6 = idxs
        vertical = _segment_length(pts, p2, p6) + _segment_length(pts, p3, p5)
        horizontal = 2.0 * _segment_length(pts, p1, p4)
        if horizontal < 1e-6:
            return 0.0
        return vertical / horizontal

    ear_r = one_eye(RIGHT_EYE_P1_P6)
    ear_l = one_eye(LEFT_EYE_P1_P6)
    ear = 0.5 * (ear_r + ear_l)
    ear_norm = ear / inter_eye if inter_eye > 1e-6 else 0.0
    return ear_r, ear_l, ear, ear_norm


def compute_mar(pts: np.ndarray) -> float:
    """MAR chuẩn hóa (spec §6.2): |P_lips_ver| / |P_lips_hor| / inter-eye.

    MAR thuần = |13-14| / |61-291| là một tỷ lệ không thứ nguyên nên đã độc lập
    kích thước mặt; thêm 1 lần chia inter-eye nữa theo đúng công thức spec.
    """
    inter_eye = _segment_length(pts, EYE_OUTER_RIGHT, EYE_OUTER_LEFT)
    ver = _segment_length(pts, MOUTH_VERTICAL[0], MOUTH_VERTICAL[1])
    hor = _segment_length(pts, MOUTH_HORIZONTAL[0], MOUTH_HORIZONTAL[1])
    if hor < 1e-6 or inter_eye < 1e-6:
        return 0.0
    return (ver / hor) / inter_eye


def compute_pitch(pts: np.ndarray, image_size: tuple[int, int],
                  camera_matrix: np.ndarray = _DEFAULT_K,
                  dist_coeffs: np.ndarray = _DEFAULT_DIST) -> tuple[float, bool]:
    """Euler pitch (độ) từ cv2.solvePnP với 5 điểm mặt 3D (spec §6.2).

    Trả về (pitch_deg, ok). ok=False khi solvePnP không hội tụ.
    Quy ước: pitch dương = ngửa đầu lên, âm = cúi xuống.
    """
    if pts.shape[0] <= max(HEAD_POSE_2D_IDX):
        return 0.0, False

    image_points = pts[list(HEAD_POSE_2D_IDX)].astype(np.float64)
    # Ma trận K phải khớp với kích thước ảnh thực (điểm chính giữa khung).
    k = camera_matrix.copy()
    k[0, 2] = image_size[0] / 2.0
    k[1, 2] = image_size[1] / 2.0

    # Spec §6.2 cho 5 điểm (2 mắt, mũi, 2 mép miệng) — các thuật toán SolvePnP
    # thông thường (ITERATIVE/DLT) cần >= 6 điểm. IPPE lại chỉ nhận đúng 4 điểm.
    # SQPNP: hỗ trợ 5+ điểm, ổn định với bộ điểm coplanar như của ta.
    ok, _rvec, _tvec = cv2.solvePnP(
        HEAD_POSE_3D, image_points, k, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
    )
    if not ok:
        return 0.0, False

    # Chuyển rotation vector -> rotation matrix -> Euler pitch (trục X).
    rmat, _ = cv2.Rodrigues(_rvec)
    # Chuẩn hóa để tránh ảnh hưởng bởi lỗi số.
    pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
    return float(np.degrees(pitch)), True


def compute_geo_metrics(result: FaceResult, image_size: tuple[int, int]) -> GeoMetrics:
    """Tính toàn bộ EAR/MAR/pitch từ 1 FaceResult. Trả về 0 khi không có mặt."""
    if not result.present or result.landmarks_px is None:
        return _NO_FACE_GEO
    pts = result.landmarks_px
    _er, _el, ear, ear_norm = compute_ear(pts)
    mar = compute_mar(pts)
    pitch, head_ok = compute_pitch(pts, image_size)
    return GeoMetrics(_er, _el, ear, ear_norm, mar, pitch, head_ok)


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


def draw_metrics(frame: np.ndarray, result: FaceResult, geo: GeoMetrics) -> np.ndarray:
    """Vẽ bảng chỉ số EAR/MAR/Pitch lên góc trái (overlay trong RAM)."""
    lines = [
        f"EAR    {geo.ear:.3f}  norm {geo.ear_norm:.3f}",
        f"MAR    {geo.mar:.3f}",
        f"PITCH  {geo.pitch_deg:+6.1f} deg" + ("" if geo.head_ok else " (PnP fail)"),
    ]
    y = 40
    for text in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (8, y - 4), (16 + tw, y + th + 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (12, y + th), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += th + 14
    if not result.present:
        text = "NO FACE — metrics = 0"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (8, y - 4), (16 + tw, y + th + 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (12, y + th), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 1, cv2.LINE_AA)
    return frame


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
    # Lưu 1 giá trị EAR_norm gần nhất để in ra dong ket thuc.
    last_ear = last_mar = 0.0

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
                last_ear = geo.ear
                last_mar = geo.mar
            geo = compute_geo_metrics(result, (frame.shape[1], frame.shape[0]))

            if args.headless:
                now = time.perf_counter()
                if now - last_report >= 1.0:
                    pct = 100.0 * seen_face / frames_seen if frames_seen else 0.0
                    print(f"[face_metrics] FPS {meter.fps:5.1f} | face {seen_face}/{frames_seen} "
                          f"({pct:4.0f}%) | pts={n_landmarks} | EAR {geo.ear:.3f} "
                          f"MAR {geo.mar:.3f} pitch {geo.pitch_deg:+.1f}")
                    last_report = now
            else:
                draw_landmarks(frame, result, tesselation=tesselation)
                draw_status(frame, result)
                draw_metrics(frame, result, geo)
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
          f"FPS {meter.fps_avg:.1f} | bat mat {pct:.0f}% | "
          f"EAR {last_ear:.3f} MAR {last_mar:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())