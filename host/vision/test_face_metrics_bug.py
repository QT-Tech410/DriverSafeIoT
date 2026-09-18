"""Sanity check test cho face_metrics, geo_metrics và các hàm tính toán hình học."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

# Cho phép in ký tự UTF-8 trên Windows console
sys.stdout.reconfigure(encoding="utf-8")

# Cho phép import từ host/vision
sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_metrics import (
    EYE_OUTER_LEFT,
    EYE_OUTER_RIGHT,
    LEFT_EYE_P1_P6,
    MOUTH_HORIZONTAL,
    MOUTH_VERTICAL,
    RIGHT_EYE_P1_P6,
    FaceResult,
    compute_ear,
    compute_geo_metrics,
    compute_mar,
    compute_pitch,
)

# Tạo mảng 468 điểm landmark 2D
pts = np.zeros((468, 2), dtype=np.float32)

# Thiết lập mắt phải (P1..P6 = 33, 160, 158, 133, 153, 144)
pts[33] = [200.0, 200.0]    # P1 (góc ngoài mắt phải)
pts[160] = [230.0, 190.0]   # P2 (mí trên)
pts[158] = [270.0, 190.0]   # P3 (mí trên)
pts[133] = [300.0, 200.0]   # P4 (góc trong)
pts[153] = [270.0, 210.0]   # P5 (mí dưới)
pts[144] = [230.0, 210.0]   # P6 (mí dưới)

# Thiết lập mắt trái (362, 385, 387, 263, 373, 380)
pts[362] = [340.0, 200.0]   # P1
pts[385] = [370.0, 190.0]   # P2
pts[387] = [410.0, 190.0]   # P3
pts[263] = [440.0, 200.0]   # P4 (góc ngoài mắt trái)
pts[373] = [410.0, 210.0]   # P5
pts[380] = [370.0, 210.0]   # P6

# Thiết lập miệng (13, 14: dọc; 61, 291: ngang)
pts[13] = [320.0, 315.0]    # môi trên
pts[14] = [320.0, 325.0]    # môi dưới
pts[61] = [280.0, 320.0]    # khóe miệng phải
pts[291] = [360.0, 320.0]   # khóe miệng trái

# Thiết lập chóp mũi (điểm 1) cho solvePnP
pts[1] = [320.0, 250.0]

# 1. Test EAR
ear_r, ear_l, ear, ear_norm = compute_ear(pts)
print(f"EAR right={ear_r:.4f}, left={ear_l:.4f}, avg={ear:.4f}, norm={ear_norm:.6f}")
assert ear > 0.1, "EAR phai lon hon 0.1 voi mat mo"
assert ear_norm > 0, "EAR norm phai hop le"

# 2. Test MAR
mar, mar_norm = compute_mar(pts)
print(f"MAR={mar:.4f}, norm={mar_norm:.6f}")
assert mar > 0.05, "MAR phai hop le"

# 3. Test Pitch
pitch, ok = compute_pitch(pts, (640, 480))
print(f"Pitch={pitch:+.1f} deg, solvePnP_ok={ok}")

# 4. Test compute_geo_metrics
res = FaceResult(present=True, landmarks_px=pts)
geo = compute_geo_metrics(res, (640, 480))
print(f"GeoMetrics: EAR={geo.ear:.4f}, MAR={geo.mar:.4f}, Pitch={geo.pitch_deg:+.1f}")
assert geo.head_ok == ok

print("✓ Tat ca sanity checks cho face_metrics & geo_metrics deu PASS!")
