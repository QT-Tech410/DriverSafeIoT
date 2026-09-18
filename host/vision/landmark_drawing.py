"""Vẽ overlay landmark + chỉ số lên frame camera (VIS-03).

Tất cả hàm vẽ đều modifier frame in-place và trả về chính frame đó
để compose tự nhiên: draw_landmarks(f); draw_status(f); draw_metrics(f, ...).
"""

from __future__ import annotations

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from face_mesh import (
    FaceResult,
    LEFT_EYE_P1_P6,
    MOUTH_HORIZONTAL,
    MOUTH_VERTICAL,
    RIGHT_EYE_P1_P6,
)
from geo_metrics import GeoMetrics

# --- Màu overlay (BGR) ---
C_OVAL = (0, 220, 0)
C_EYE = (255, 200, 0)
C_EAR_PT = (0, 255, 255)
C_LIPS = (200, 0, 200)
C_NOSE = (255, 120, 0)
C_TESS = (80, 80, 80)
C_BOX = (0, 255, 0)


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
        f"MAR    {geo.mar:.3f}  norm {geo.mar_norm:.4f}",
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
