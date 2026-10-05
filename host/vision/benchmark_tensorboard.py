# -*- coding: utf-8 -*-
"""
benchmark_tensorboard.py - Do dac sai so L1, MSE, NME va truc quan hoa tren TensorBoard
He thong: DriverSafeIoT (MediaPipe Face Mesh + Geometric Metrics)

Cac chi so duoc ghi nhan:
  1. Loss / Error:
     - L1 Loss (MAE Pixel): Do lech toa do pixel tuyet doi trung binh
     - MSE Loss (Pixel^2): Binh phuong sai so toa do pixel
     - NME (Normalized Mean Error %): Sai so chuan hoa theo khoang cach lien khoe mat (Inter-Ocular Distance)
     - EAR Target Loss: Sai so giua EAR thuc te va baseline
  2. Stability (Do on dinh qua thoi gian):
     - Landmark Temporal Jitter (MSE / L1 / NME drift giua cac frame)
     - Inter-Ocular Distance (px)
  3. Classification & Performance:
     - Accuracy, Precision, Recall, F1 theo thoi gian
     - Latency (ms) va FPS
  4. Visualization:
     - Anh khuon mat ve landmark truc tiep tren tab Images cua TensorBoard
     - Histogram phan bo do dich chuyen 468 diem moc

Cach chay:
  # Che do 1: Giam sat realtime tu webcam
  .venv/Scripts/python.exe host/vision/benchmark_tensorboard.py --mode realtime --duration 120

  # Che do 2: Danh gia tren thu muc dataset
  .venv/Scripts/python.exe host/vision/benchmark_tensorboard.py --mode dataset --data datasets/cew

Xem tren TensorBoard:
  .venv/Scripts/tensorboard.exe --logdir=runs
  Truy cap: http://localhost:6006
"""
from __future__ import annotations
import argparse
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tensorboardX import SummaryWriter

# Them thu muc vision de import
_VISION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_VISION_DIR))

try:
    from face_mesh import FaceMesh, MODEL_PATH
    from geo_metrics import compute_geo_metrics
    USE_PROJECT_MODULE = True
except Exception:
    import mediapipe as mp
    USE_PROJECT_MODULE = False

DEFAULT_T_CLOSED = 0.22  # Nguong EAR mac dinh trong temporal.py


# =====================================================================
# HAM TINH TOAN TOAN HOC: L1, MSE, NME
# =====================================================================
def calc_l1_error(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Mean Absolute Error (L1) theo pixel: (1/2N) * sum(|x1-x2| + |y1-y2|)."""
    return float(np.mean(np.abs(pts1 - pts2)))


def calc_mse_error(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Mean Squared Error (MSE) theo pixel^2: (1/N) * sum(||p1 - p2||_2^2)."""
    diff = pts1 - pts2
    return float(np.mean(np.sum(diff ** 2, axis=1)))


def calc_nme(pts1: np.ndarray, pts2: np.ndarray, d_inter: float) -> float:
    """
    Normalized Mean Error (NME %):
      NME = (1/N) * sum(||p1 - p2||_2 / d_inter) * 100%
    d_inter: Khoang cach chuan hoa lien khoe mat (Inter-Ocular Distance).
    """
    if d_inter < 1e-6:
        return 0.0
    euclidean = np.linalg.norm(pts1 - pts2, axis=1)
    return float(np.mean(euclidean) / d_inter * 100.0)


# =====================================================================
# TRICH XUAT TOA DO LANDMARKS TU FRAME
# =====================================================================
class LandmarkExtractor:
    def __init__(self):
        if USE_PROJECT_MODULE:
            self.mesh = FaceMesh(str(MODEL_PATH), running_mode="video")
            self._t0 = time.perf_counter()
        else:
            self.mp_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True
            )

    def process(self, frame_bgr: np.ndarray) -> tuple[bool, Optional[np.ndarray], float, float, float, float]:
        """
        Tra ve:
          (detected: bool, pts_px: np.ndarray(468, 2), ear: float, mar: float, pitch: float, d_inter: float)
        """
        h, w = frame_bgr.shape[:2]

        if USE_PROJECT_MODULE:
            t_ms = int((time.perf_counter() - self._t0) * 1000)
            res = self.mesh.process(frame_bgr, t_ms)
            if not res.present or res.landmarks_px is None:
                return False, None, 0.0, 0.0, 0.0, 0.0
            pts = res.landmarks_px
            geo = compute_geo_metrics(res, (w, h))
            # Khoang cach lien khoe mat (index 33 đuôi mắt phải, 263 đuôi mắt trái)
            d_inter = float(np.linalg.norm(pts[33] - pts[263]))
            return True, pts, geo.ear, geo.mar, geo.pitch_deg, d_inter
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.mp_mesh.process(rgb)
            if not res.multi_face_landmarks:
                return False, None, 0.0, 0.0, 0.0, 0.0
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

            def d(i, j): return float(np.linalg.norm(pts[i] - pts[j]))
            ear_r = (d(160, 144) + d(158, 153)) / (2.0 * d(33, 133) + 1e-9)
            ear_l = (d(385, 380) + d(387, 373)) / (2.0 * d(362, 263) + 1e-9)
            ear = (ear_r + ear_l) / 2.0
            mar = (d(13, 14) + d(312, 317)) / (2.0 * d(61, 291) + 1e-9)
            d_inter = d(33, 263)
            return True, pts, ear, mar, 0.0, d_inter

    def close(self):
        if USE_PROJECT_MODULE:
            try: self.mesh.close()
            except Exception: pass


# =====================================================================
# CHE DO 1: REALTIME LIVE MONITOR VAO TENSORBOARD
# =====================================================================
def run_realtime(duration_s: int = 120, cam_index: int = 0, log_dir: str = None):
    if not log_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/realtime_{timestamp}"

    writer = SummaryWriter(log_dir=log_dir)
    print("=" * 65)
    print(f"KHOI DONG TENSORBOARD LIVE RECORDER")
    print(f" - Log thu muc: {log_dir}")
    print(f" - Thoi luong : {duration_s} giay")
    print("Phim gan nhan:")
    print("  [O] Mat mo  |  [C] Mat nham  |  [Y] Ngap  |  [N] Binh thuong  |  [Q] Thoat")
    print("=" * 65)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Loi: Khong mo duoc camera!")
        return

    extractor = LandmarkExtractor()
    prev_pts: Optional[np.ndarray] = None
    step = 0
    t_start = time.time()

    # Ground truth runtime labels
    gt_closed = False
    gt_yawn = False

    # Cumulative counters
    cum_tp = cum_fp = cum_tn = cum_fn = 0

    while True:
        elapsed = time.time() - t_start
        if elapsed >= duration_s:
            break

        ret, frame = cap.read()
        if not ret: break

        t0 = time.perf_counter()
        detected, pts, ear, mar, pitch, d_inter = extractor.process(frame)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-3)

        step += 1

        # 1. Ghi nhan Performance
        writer.add_scalar("Performance/Latency_ms", latency_ms, step)
        writer.add_scalar("Performance/FPS", fps, step)

        if detected and pts is not None:
            # 2. Ghi nhan Metrics sinh hoc
            writer.add_scalar("Metrics/EAR", ear, step)
            writer.add_scalar("Metrics/MAR", mar, step)
            writer.add_scalar("Metrics/Pitch_Deg", pitch, step)
            writer.add_scalar("Stability/Inter_Ocular_Distance_px", d_inter, step)

            # 3. Tinh va ghi nhan Do rung lac (Temporal Stability & Jitter)
            if prev_pts is not None:
                jitter_l1 = calc_l1_error(pts, prev_pts)
                jitter_mse = calc_mse_error(pts, prev_pts)
                nme_drift = calc_nme(pts, prev_pts, d_inter)

                writer.add_scalar("Stability/Landmark_Jitter_L1_px", jitter_l1, step)
                writer.add_scalar("Stability/Landmark_Jitter_MSE_px2", jitter_mse, step)
                writer.add_scalar("Stability/NME_Drift_Percentage", nme_drift, step)

                # Histogram do dich chuyen cua 468 diem moc
                displacements = np.linalg.norm(pts - prev_pts, axis=1)
                if step % 15 == 0:
                    writer.add_histogram("Stability/Landmark_Displacement_Dist", displacements, step)

            prev_pts = pts.copy()

            # 4. Tinh Supervised Loss (neu co gan nhan)
            pred_closed = (ear < DEFAULT_T_CLOSED)
            target_ear = 0.12 if gt_closed else 0.32
            ear_l1_loss = abs(ear - target_ear)
            ear_mse_loss = (ear - target_ear) ** 2

            writer.add_scalar("Supervised_Loss/EAR_L1_Loss", ear_l1_loss, step)
            writer.add_scalar("Supervised_Loss/EAR_MSE_Loss", ear_mse_loss, step)

            # Cap nhat ma tran nham lan tich luy
            if pred_closed and gt_closed: cum_tp += 1
            elif pred_closed and not gt_closed: cum_fp += 1
            elif not pred_closed and gt_closed: cum_fn += 1
            else: cum_tn += 1

            total_labeled = cum_tp + cum_fp + cum_tn + cum_fn
            if total_labeled > 0:
                acc = (cum_tp + cum_tn) / total_labeled
                prec = cum_tp / max(cum_tp + cum_fp, 1)
                rec = cum_tp / max(cum_tp + cum_fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)

                writer.add_scalar("Classification/Cumulative_Accuracy", acc * 100, step)
                writer.add_scalar("Classification/Cumulative_Precision", prec * 100, step)
                writer.add_scalar("Classification/Cumulative_Recall", rec * 100, step)
                writer.add_scalar("Classification/Cumulative_F1", f1 * 100, step)

            # 5. Ghi hinh anh khuon mat len tab Images moi 30 frame (~1 giay)
            if step % 30 == 0:
                # Ve truc quan EAR len frame
                disp_img = frame.copy()
                for pt in pts[::10]:  # ve 1 so diem moc mau
                    cv2.circle(disp_img, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
                # Chuyen BGR sang RGB cho TensorBoard
                disp_rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
                disp_tensor = np.transpose(disp_rgb, (2, 0, 1))
                writer.add_image("Visualization/Annotated_Face", disp_tensor, step)

            # Ve HUD tren man hinh
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 80), (15, 15, 15), -1)
            cv2.putText(frame, f"EAR: {ear:.3f} | MAR: {mar:.3f} | Latency: {latency_ms:.1f}ms",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
            gt_text = f"GT: Mat={'NHAM' if gt_closed else 'MO'} | Ngap={'CO' if gt_yawn else 'KHONG'}"
            cv2.putText(frame, gt_text, (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 80), 2)
        else:
            cv2.putText(frame, "KHONG THAY MAT", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        rem = int(duration_s - elapsed)
        cv2.putText(frame, f"{rem}s", (frame.shape[1] - 80, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow("TensorBoard Monitor - DriverSafeIoT", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('o') or key == ord('O'):
            gt_closed = False
        elif key == ord('c') or key == ord('C'):
            gt_closed = True
        elif key == ord('y') or key == ord('Y'):
            gt_yawn = True
        elif key == ord('n') or key == ord('N'):
            gt_closed = gt_yawn = False

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    writer.close()

    print(f"\nDa ghi xong {step} steps vao TensorBoard tai thu muc: {log_dir}")
    print("Chay lenh sau de xem tren web:")
    print(f"  .venv\\Scripts\\tensorboard.exe --logdir=runs")


# =====================================================================
# CHE DO 2: OFFLINE DATASET BENCHMARK VAO TENSORBOARD
# =====================================================================
def run_dataset_benchmark(data_dir: str = "datasets/cew", log_dir: str = None):
    if not log_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/dataset_{timestamp}"

    data_path = Path(data_dir)
    open_dir = data_path / "open"
    closed_dir = data_path / "closed"

    if not open_dir.exists() or not closed_dir.exists():
        print(f"Loi: Thu muc {data_dir} phai chua 2 thu muc 'open' va 'closed'")
        return

    writer = SummaryWriter(log_dir=log_dir)
    extractor = LandmarkExtractor()

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    open_files = [(p, False) for p in open_dir.iterdir() if p.suffix.lower() in valid_exts]
    closed_files = [(p, True) for p in closed_dir.iterdir() if p.suffix.lower() in valid_exts]
    all_files = open_files + closed_files
    np.random.seed(42)
    np.random.shuffle(all_files)

    print("=" * 65)
    print(f"BAT DAU GHI DATASET BENCHMARK VAO TENSORBOARD")
    print(f" - So luong anh: {len(all_files)} (Open={len(open_files)}, Closed={len(closed_files)})")
    print(f" - Log thu muc : {log_dir}")
    print("=" * 65)

    step = 0
    open_ears = []
    closed_ears = []
    tp = fp = tn = fn = 0

    for path, gt_closed in all_files:
        img = cv2.imread(str(path))
        if img is None: continue
        step += 1

        t0 = time.perf_counter()
        detected, pts, ear, mar, pitch, d_inter = extractor.process(img)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        writer.add_scalar("Dataset/Inference_Latency_ms", latency_ms, step)

        if detected and pts is not None:
            # 1. Metric co ban
            writer.add_scalar("Dataset_Metrics/EAR", ear, step)
            writer.add_scalar("Dataset_Metrics/Inter_Ocular_Dist_px", d_inter, step)

            # Target Loss
            target_ear = 0.12 if gt_closed else 0.32
            ear_l1 = abs(ear - target_ear)
            ear_mse = (ear - target_ear) ** 2

            writer.add_scalar("Dataset_Loss/EAR_L1_Loss", ear_l1, step)
            writer.add_scalar("Dataset_Loss/EAR_MSE_Loss", ear_mse, step)

            if gt_closed:
                closed_ears.append(ear)
            else:
                open_ears.append(ear)

            # Phan loai voi nguong mac dinh
            pred_closed = (ear < DEFAULT_T_CLOSED)
            if pred_closed and gt_closed: tp += 1
            elif pred_closed and not gt_closed: fp += 1
            elif not pred_closed and gt_closed: fn += 1
            else: tn += 1

            total_seen = tp + fp + tn + fn
            acc = (tp + tn) / total_seen
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)

            writer.add_scalar("Dataset_Performance/Accuracy", acc * 100, step)
            writer.add_scalar("Dataset_Performance/Precision", prec * 100, step)
            writer.add_scalar("Dataset_Performance/Recall", rec * 100, step)
            writer.add_scalar("Dataset_Performance/F1_Score", f1 * 100, step)

            # Ghi anh vao TensorBoard moi 10 buoc
            if step % 10 == 0:
                annotated = img.copy()
                for pt in pts[::15]:
                    cv2.circle(annotated, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                color = (0, 0, 255) if pred_closed else (0, 255, 0)
                cv2.putText(annotated, f"EAR={ear:.3f} | GT={'CLOSED' if gt_closed else 'OPEN'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                writer.add_image(f"Samples/Sample_{step}", np.transpose(rgb_img, (2, 0, 1)), step)

    # Ghi Histogram phan bo EAR cua 2 tap Open vs Closed vao TensorBoard
    if open_ears:
        writer.add_histogram("Distributions/EAR_Open_Eyes", np.array(open_ears), step)
    if closed_ears:
        writer.add_histogram("Distributions/EAR_Closed_Eyes", np.array(closed_ears), step)

    # Ghi Precision-Recall Curve qua cac nguong
    if open_ears and closed_ears:
        for t_step in np.arange(0.15, 0.35, 0.01):
            cur_tp = sum(1 for e in closed_ears if e < t_step)
            cur_fp = sum(1 for e in open_ears if e < t_step)
            cur_fn = sum(1 for e in closed_ears if e >= t_step)
            p = cur_tp / max(cur_tp + cur_fp, 1)
            r = cur_tp / max(cur_tp + cur_fn, 1)
            writer.add_scalar("PR_Curve/Precision_vs_Threshold", p * 100, int(t_step * 100))
            writer.add_scalar("PR_Curve/Recall_vs_Threshold", r * 100, int(t_step * 100))

    extractor.close()
    writer.close()
    print("=" * 65)
    print(f"BENCHMARK HOAN TAT! Da ghi vao: {log_dir}")
    print(f" - Do chinh xac cuoi cung: {acc * 100:.2f}% (F1 = {f1 * 100:.2f}%)")
    print("Chay lenh sau de xem:")
    print("  .venv\\Scripts\\tensorboard.exe --logdir=runs")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="TensorBoard Benchmark Integration - DriverSafeIoT")
    parser.add_argument("--mode", choices=["realtime", "dataset"], default="realtime",
                        help="realtime: chay live webcam | dataset: chay offline tren anh")
    parser.add_argument("--data", default="datasets/cew", help="Duong dan dataset cho mode dataset")
    parser.add_argument("--duration", type=int, default=120, help="Thoi gian chay realtime (giay)")
    parser.add_argument("--cam", type=int, default=0, help="Index camera")
    parser.add_argument("--logdir", default=None, help="Thu muc ghi log TensorBoard")
    args = parser.parse_args()

    if args.mode == "realtime":
        run_realtime(duration_s=args.duration, cam_index=args.cam, log_dir=args.logdir)
    else:
        run_dataset_benchmark(data_dir=args.data, log_dir=args.logdir)


if __name__ == "__main__":
    main()
