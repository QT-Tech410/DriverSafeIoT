"""Facade cho Face Mesh + chỉ số hình học (VIS-02 + VIS-03).

Module này re-export tất cả API công khai từ 3 module con để code cũ
(publisher.py, temporal.py, test scripts) không cần đổi import:

    from face_metrics import FaceMesh, MODEL_PATH, compute_geo_metrics, ...

Bên trong cấu trúc:
    face_mesh.py        — FaceLandmarker wrapper, FaceResult, hằng số landmark
    geo_metrics.py      — GeoMetrics, compute_ear/mar/pitch/geo_metrics
    landmark_drawing.py — draw_landmarks/metrics/status

Chạy thử (CLI demo):
    python face_metrics.py                      # webcam, ve overlay + chi so
    python face_metrics.py --src mjpeg --url http://<ip>:81/stream
    python face_metrics.py --headless --duration 10
Phím: q/ESC thoát, t = bật/tắt tesselation (nặng), s = chụp ảnh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# capture.py nằm cùng thư mục — cho phép chạy file này trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import FrameSource, FpsMeter, draw_hud, save_snapshot  # noqa: E402

# --- Re-export API công khai (backward compatibility) ---
from face_mesh import (  # noqa: E402,F401
    FACE_LANDMARKS,
    IRIS_LANDMARKS,
    MODEL_PATH,
    MODEL_URL,
    MODELS_DIR,
    DEFAULT_DIST,
    DEFAULT_K,
    EYE_OUTER_LEFT,
    EYE_OUTER_RIGHT,
    HEAD_POSE_2D_IDX,
    HEAD_POSE_3D,
    LEFT_EYE_P1_P6,
    MOUTH_HORIZONTAL,
    MOUTH_VERTICAL,
    RIGHT_EYE_P1_P6,
    FaceMesh,
    FaceResult,
)
from geo_metrics import (  # noqa: E402,F401
    NO_FACE_GEO,
    GeoMetrics,
    compute_ear,
    compute_geo_metrics,
    compute_mar,
    compute_pitch,
)
from landmark_drawing import (  # noqa: E402,F401
    C_BOX,
    C_EAR_PT,
    C_EYE,
    C_LIPS,
    C_NOSE,
    C_OVAL,
    C_TESS,
    draw_landmarks,
    draw_metrics,
    draw_status,
)

# Compat alias: tên private cũ trong code cũ.
_NO_FACE_GEO = NO_FACE_GEO
_default_K = DEFAULT_K
_default_DIST = DEFAULT_DIST


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
            # Tính chỉ số trước khi đọc: frame đầu có thể không có mặt.
            geo = compute_geo_metrics(result, (frame.shape[1], frame.shape[0]))
            if result.present:
                seen_face += 1
                n_landmarks = result.count
                last_ear = geo.ear
                last_mar = geo.mar

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
