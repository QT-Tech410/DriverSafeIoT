# -*- coding: utf-8 -*-
"""
benchmark_dataset.py - Danh gia tu dong MediaPipe tren Dataset chuan (CEW / Open-Closed Eyes)

Cach chay:
  .venv\Scripts\python.exe host\vision\benchmark_dataset.py --data datasets\cew
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Them thu muc vision de import
_VISION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_VISION_DIR))

try:
    from face_mesh import FaceMesh, MODEL_PATH
    from geo_metrics import compute_geo_metrics
    USE_PROJECT_MODULE = True
except Exception as e:
    import mediapipe as mp
    USE_PROJECT_MODULE = False

DEFAULT_T_CLOSED = 0.22  # Nguong EAR mac dinh trong he thong


def compute_ear_fallback(mesh, image_bgr):
    """Fallback tinh EAR khi khong co module du an."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    def pt(i): return np.array([lm[i].x * w, lm[i].y * h])
    def d(i, j): return float(np.linalg.norm(pt(i) - pt(j)))
    ear_r = (d(160, 144) + d(158, 153)) / (2.0 * d(33, 133) + 1e-9)
    ear_l = (d(385, 380) + d(387, 373)) / (2.0 * d(362, 263) + 1e-9)
    return (ear_r + ear_l) / 2.0


def evaluate_dataset(data_dir: str, thresh: float = DEFAULT_T_CLOSED):
    data_path = Path(data_dir)
    open_dir = data_path / "open"
    closed_dir = data_path / "closed"

    if not open_dir.exists() or not closed_dir.exists():
        print(f"LOI: Thu muc phai chua 2 thu muc con: 'open' va 'closed'")
        print(f"Vi tri kiem tra: {open_dir} va {closed_dir}")
        return

    # Khoi tao model
    if USE_PROJECT_MODULE:
        mesh = FaceMesh(str(MODEL_PATH), running_mode="image")
    else:
        mp_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        )

    # Danh sach file
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    open_files = [f for f in open_dir.iterdir() if f.suffix.lower() in valid_exts]
    closed_files = [f for f in closed_dir.iterdir() if f.suffix.lower() in valid_exts]

    print("=" * 65)
    print(f"BAT DAU BENCHMARK TREN DATASET: {data_dir}")
    print(f" - So anh mat MO   (Ground Truth = 0): {len(open_files)}")
    print(f" - So anh mat NHAM (Ground Truth = 1): {len(closed_files)}")
    print(f" - Nguong EAR kiem tra: {thresh}")
    print("=" * 65)

    all_samples = [] # list of (gt_closed: bool, ear_val: float, filename: str)
    no_face_count = 0
    t0 = time.time()

    # 1. Chay anh mat MO
    for p in open_files:
        img = cv2.imread(str(p))
        if img is None: continue
        ear = None
        if USE_PROJECT_MODULE:
            res = mesh.process(img, 0)
            if res.present:
                geo = compute_geo_metrics(res, (img.shape[1], img.shape[0]))
                ear = geo.ear
        else:
            ear = compute_ear_fallback(mp_mesh, img)

        if ear is None:
            no_face_count += 1
        else:
            all_samples.append((False, ear, p.name))

    # 2. Chay anh mat NHAM
    for p in closed_files:
        img = cv2.imread(str(p))
        if img is None: continue
        ear = None
        if USE_PROJECT_MODULE:
            res = mesh.process(img, 0)
            if res.present:
                geo = compute_geo_metrics(res, (img.shape[1], img.shape[0]))
                ear = geo.ear
        else:
            ear = compute_ear_fallback(mp_mesh, img)

        if ear is None:
            no_face_count += 1
        else:
            all_samples.append((True, ear, p.name))

    elapsed = time.time() - t0
    total_imgs = len(open_files) + len(closed_files)
    detected_imgs = len(all_samples)

    if detected_imgs == 0:
        print("Loi: Khong phat hien duoc khuon mat nao trong dataset!")
        return

    # 3. Tinh Confusion Matrix tai nguong thresh
    def calc_metrics(t):
        tp = fp = tn = fn = 0
        for gt_closed, ear, _ in all_samples:
            pred_closed = (ear < t)
            if pred_closed and gt_closed: tp += 1
            elif pred_closed and not gt_closed: fp += 1
            elif not pred_closed and gt_closed: fn += 1
            else: tn += 1
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
        return tp, fp, tn, fn, acc, prec, rec, f1

    tp, fp, tn, fn, acc, prec, rec, f1 = calc_metrics(thresh)

    print("\n--- KET QUA TAI NGUONG EAR HIEN TAI (thresh = {:.2f}) ---".format(thresh))
    print(f"Tong so anh xu ly        : {total_imgs} anh (thoi gian: {elapsed:.2f}s)")
    print(f"Phat hien khuon mat      : {detected_imgs}/{total_imgs} ({(detected_imgs/total_imgs)*100:.1f}%)")
    print(f"Khong phat hien mat (mat goc/toi): {no_face_count}")
    print("-" * 50)
    print(f"True Positive (Nham dung) : {tp}")
    print(f"True Negative (Mo dung)   : {tn}")
    print(f"False Positive (Bao nham) : {fp}")
    print(f"False Negative (Bo sot)   : {fn}")
    print("-" * 50)
    print(f"Do chinh xac (Accuracy)   : {acc * 100:.2f} %")
    print(f"Precision                 : {prec * 100:.2f} %")
    print(f"Recall (Do nhay phat hien): {rec * 100:.2f} %")
    print(f"F1-Score                  : {f1 * 100:.2f} %")
    print("=" * 65)

    # 4. Quet tim nguong EAR toi uu (Threshold Search de bao cao)
    print("\n[PHAN TICH NANG CAO] QUET TIM NGUONG EAR TOI UU:")
    print("Thresh | Accuracy | Precision |  Recall  | F1-Score |")
    print("-------|----------|-----------|----------|----------|")
    best_t = thresh
    best_f1 = 0.0
    for t_step in np.arange(0.16, 0.28, 0.01):
        _, _, _, _, a, p, r, f = calc_metrics(t_step)
        mark = " <-- Best" if f > best_f1 else ""
        if f > best_f1:
            best_f1 = f
            best_t = t_step
        print(f" {t_step:.2f}  |  {a*100:5.1f}%  |  {p*100:6.1f}%  |  {r*100:5.1f}%  |  {f*100:5.1f}%  |{mark}")

    print("-" * 55)
    print(f"==> Nguong EAR toi uu nhat theo tap du lieu la: {best_t:.2f} (F1 = {best_f1*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Dataset MediaPipe EAR")
    parser.add_argument("--data", default="datasets/cew", help="Duong dan thu muc dataset (chua open/ va closed/)")
    parser.add_argument("--thresh", type=float, default=DEFAULT_T_CLOSED, help="Nguong EAR kiem tra")
    args = parser.parse_args()
    evaluate_dataset(args.data, args.thresh)


if __name__ == "__main__":
    main()
