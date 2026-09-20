"""Chỉ số hình học — EAR / MAR / Pitch (VIS-03).

Các chỉ số (spec §6.2):
  - EAR = (|P2-P6| + |P3-P5|) / (2*|P1-P4|), chuẩn hóa theo khoảng cách 2 mắt
  - MAR = |P_lips_ver| / |P_lips_hor| (chuẩn hóa tương tự)
  - Pitch  từ cv2.solvePnP với 5 điểm mặt 3D -> Euler pitch
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from face_mesh import (
    DEFAULT_DIST,
    DEFAULT_K,
    EYE_OUTER_LEFT,
    EYE_OUTER_RIGHT,
    FaceResult,
    HEAD_POSE_2D_IDX,
    HEAD_POSE_3D,
    LEFT_EYE_P1_P6,
    MOUTH_HORIZONTAL,
    MOUTH_VERTICAL,
    RIGHT_EYE_P1_P6,
)


@dataclass
class GeoMetrics:
    """Các chỉ số hình học 1 frame (spec §6.2) — tất cả đều không thứ nguyên.

    LƯU Ý VỀ "chuẩn hóa" (đã kiểm chứng thực nghiệm VIS-03/VIS-04):
    `_norm` = metric / inter-eye mang đơn vị px⁻¹ nên PHỤ THUỘC khoảng cách
    camera — đưa tay ra xa/lại gần là đổi giá trị, không thể so với 1 ngưỡng
    cố định. Còn bản không chia (ear, mar) vốn đã là tỷ lệ của 2 đoạn cùng
    tỉ lệ với khoảng cách camera → bất biến thứ nguyên, mới dùng để so ngưỡng.
    Do đó cả 2 bản đều được tính (bản `_norm` báo cáo cho đủ schema), nhưng
    lớp thời gian và các ngưỡng T_closed/T_yawn dùng bản không thứ nguyên.
    """

    ear_right: float        # EAR mắt phải (tỷ lệ, không thứ nguyên)
    ear_left: float
    ear: float              # trung bình 2 mắt — DÙNG CHO T_closed
    ear_norm: float         # chuẩn hóa theo khoảng cách 2 mắt (spec §6.2)
    mar: float              # MAR thuần (v/h) — DÙNG CHO T_yawn
    mar_norm: float         # chuẩn hóa theo khoảng cách 2 mắt (spec §6.2)
    pitch_deg: float        # Euler pitch từ solvePnP, độ
    head_ok: bool           # solvePnP hội tụ không

    def as_dict(self) -> dict:
        return {
            "ear": round(self.ear, 4),
            "ear_norm": round(self.ear_norm, 4),
            "mar": round(self.mar, 4),
            "mar_norm": round(self.mar_norm, 4),
            "pitch_deg": round(self.pitch_deg, 1),
            "head_ok": self.head_ok,
        }


NO_FACE_GEO = GeoMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)


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


def compute_mar(pts: np.ndarray) -> tuple[float, float]:
    """MAR (spec §6.2): |P_lips_ver| / |P_lips_hor|, kèm bản chuẩn hóa.

    Trả về (mar, mar_norm). mar_norm = mar / inter-eye (đúng công thức chữ
    của spec), nhưng mar mới là đầu vào so ngưỡng — xem GeoMetrics để biết lý do.
    """
    inter_eye = _segment_length(pts, EYE_OUTER_RIGHT, EYE_OUTER_LEFT)
    ver = _segment_length(pts, MOUTH_VERTICAL[0], MOUTH_VERTICAL[1])
    hor = _segment_length(pts, MOUTH_HORIZONTAL[0], MOUTH_HORIZONTAL[1])
    if hor < 1e-6:
        return 0.0, 0.0
    mar = ver / hor
    mar_norm = mar / inter_eye if inter_eye > 1e-6 else 0.0
    return mar, mar_norm


def compute_pitch(pts: np.ndarray, image_size: tuple[int, int],
                  camera_matrix: np.ndarray = DEFAULT_K,
                  dist_coeffs: np.ndarray = DEFAULT_DIST) -> tuple[float, bool]:
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

    # Chuyển rotation vector -> rotation matrix (3x3).
    rmat, _ = cv2.Rodrigues(_rvec)

    # Phân rã ma trận quay trực giao triệt tiêu nhiễu chéo (Cross-talk / Gimbal Lock)
    # Tính định thức hình chiếu sy = sqrt(R[0,0]^2 + R[1,0]^2) để tách biệt góc Yaw khỏi Pitch
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])

    # Đã kiểm chứng thực nghiệm (VIS-03): nhìn thẳng ≈ 0, cúi đầu → âm,
    # ngẩng đầu → dương. Khớp quy ước "pitch dương = ngửa đầu lên".
    return float(np.degrees(pitch)), True


def compute_geo_metrics(result: FaceResult, image_size: tuple[int, int]) -> GeoMetrics:
    """Tính toàn bộ EAR/MAR/pitch từ 1 FaceResult. Trả về 0 khi không có mặt."""
    if not result.present or result.landmarks_px is None:
        return NO_FACE_GEO
    pts = result.landmarks_px
    _er, _el, ear, ear_norm = compute_ear(pts)
    mar, mar_norm = compute_mar(pts)
    pitch, head_ok = compute_pitch(pts, image_size)
    return GeoMetrics(_er, _el, ear, ear_norm, mar, mar_norm, pitch, head_ok)
