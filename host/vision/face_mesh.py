"""FaceLandmarker wrapper — MediaPipe tasks API (VIS-02).

LƯU Ý QUAN TRỌNG VỀ PHIÊN BẢN
-----------------------------
MediaPipe 1.0.1 (bản trong venv) đã **xóa hẳn `mp.solutions`**. Mọi tutorial
dùng `mp.solutions.face_mesh.FaceMesh(...)` đều KHÔNG chạy được ở đây.
API thay thế là `mediapipe.tasks.python.vision.FaceLandmarker`, và nó cần một
file model `.task` tải riêng (khác bản 0.10.x vốn bundle sẵn model):
    python scripts/fetch_models.py

Model FaceLandmarker trả 478 điểm: 468 điểm lưới mặt + 10 điểm iris.
Phần mềm cũ `refine_landmarks=False` chỉ lấy 468 (bỏ iris). Mọi chỉ số hình
học (geo_metrics.py) dùng index < 468 nên chạy giống nhau ở cả hai chế độ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Tổng số điểm landmark của model (468 mesh + 10 iris).
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
DEFAULT_K = np.array([
    [500.0,   0.0, 320.0],
    [  0.0, 500.0, 240.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)
DEFAULT_DIST = np.zeros(5, dtype=np.float64)


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
