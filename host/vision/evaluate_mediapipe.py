# -*- coding: utf-8 -*-
"""
evaluate_mediapipe.py - Danh gia do chinh xac MediaPipe - DriverSafeIoT

Cach chay:
  cd D:\\\\Workspace\\\\DuanIoT
  python host\\vision\\evaluate_mediapipe.py --mode realtime --duration 120
  python host\\vision\\evaluate_mediapipe.py --mode video --input path\\video.mp4

Phim gan nhan ground truth:
  [O] Mat mo  [C] Mat nham  [Y] Ngap  [H] Guc dau  [N] Normal  [Q] Thoat+luu

Nguong dung chinh xac theo temporal.py:
  EAR < 0.22  -> nham mat (DEFAULT_T_CLOSED)
  MAR > 0.50  -> ngap     (DEFAULT_T_YAWN)
  Pitch < -15 -> guc dau  (HEAD_DROP_DEG)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# =====================================================================
# THEM PATH DU AN DE IMPORT MODULE THAT (face_mesh, geo_metrics)
# =====================================================================
_VISION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_VISION_DIR))

# =====================================================================
# NGUONG THUC TE - Lay chinh xac tu temporal.py cua du an
# =====================================================================
DEFAULT_T_CLOSED = 0.22    # temporal.py: DEFAULT_T_CLOSED = 0.22
DEFAULT_T_YAWN   = 0.50    # temporal.py: DEFAULT_T_YAWN   = 0.50
HEAD_DROP_DEG    = 15.0    # temporal.py: HEAD_DROP_DEG    = 15.0
MICROSLEEP_MS    = 2000.0  # temporal.py: MICROSLEEP_MS    = 2000.0
EYE_CLOSURE_MS   = 250.0   # temporal.py: EYE_CLOSURE_MS   = 250.0
EYE_DEBOUNCE     = 6       # temporal.py: EYE_DEBOUNCE     = 6


# =====================================================================
# THONG KE & BAO CAO
# =====================================================================
@dataclass
class EvalStats:
    total_frames:    int = 0
    detected_frames: int = 0
    ear_tp: int = 0
    ear_fp: int = 0
    ear_tn: int = 0
    ear_fn: int = 0
    mar_tp: int = 0
    mar_fp: int = 0
    mar_tn: int = 0
    mar_fn: int = 0
    head_tp: int = 0
    head_fp: int = 0
    head_tn: int = 0
    head_fn: int = 0
    latencies: list = field(default_factory=list)

    def _prf1(self, tp, fp, tn, fn):
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
        return round(prec*100, 2), round(rec*100, 2), round(f1*100, 2), round(acc*100, 2)

    def ear_m(self):
        return self._prf1(self.ear_tp, self.ear_fp, self.ear_tn, self.ear_fn)

    def mar_m(self):
        return self._prf1(self.mar_tp, self.mar_fp, self.mar_tn, self.mar_fn)

    def head_m(self):
        return self._prf1(self.head_tp, self.head_fp, self.head_tn, self.head_fn)

    def dr(self):
        return round(self.detected_frames / max(self.total_frames, 1) * 100, 2)

    def lat_avg(self):
        return round(float(np.mean(self.latencies)) if self.latencies else 0.0, 2)

    def lat_p95(self):
        return round(float(np.percentile(self.latencies, 95)) if self.latencies else 0.0, 2)

    def update_cm(self, kind: str, pred: bool, gt: bool):
        prefix = {"ear": "ear", "mar": "mar", "head": "head"}[kind]
        if pred and gt:
            setattr(self, f"{prefix}_tp", getattr(self, f"{prefix}_tp") + 1)
        elif pred and not gt:
            setattr(self, f"{prefix}_fp", getattr(self, f"{prefix}_fp") + 1)
        elif not pred and gt:
            setattr(self, f"{prefix}_fn", getattr(self, f"{prefix}_fn") + 1)
        else:
            setattr(self, f"{prefix}_tn", getattr(self, f"{prefix}_tn") + 1)

    def print_report(self):
        ep, er, ef1, ea = self.ear_m()
        mp_, mr, mf1, ma = self.mar_m()
        hp, hr, hf1, ha = self.head_m()
        sep = "=" * 65
        print(f"\n{sep}")
        print("   KET QUA DANH GIA DO CHINH XAC MEDIAPIPE - DriverSafeIoT")
        print(sep)
        print(f"  Tong frames phan tich      : {self.total_frames}")
        print(f"  Frame phat hien khuon mat  : {self.detected_frames}  ({self.dr()}%)")
        print(f"  Latency trung binh / P95   : {self.lat_avg()} ms / {self.lat_p95()} ms")
        print(f"\n  --- EAR Nham mat (nguong EAR < {DEFAULT_T_CLOSED}) ---")
        print(f"  TP={self.ear_tp}  FP={self.ear_fp}  TN={self.ear_tn}  FN={self.ear_fn}")
        print(f"  Accuracy:{ea}%  Precision:{ep}%  Recall:{er}%  F1:{ef1}%")
        print(f"\n  --- MAR Ngap (nguong MAR > {DEFAULT_T_YAWN}) ---")
        print(f"  TP={self.mar_tp}  FP={self.mar_fp}  TN={self.mar_tn}  FN={self.mar_fn}")
        print(f"  Accuracy:{ma}%  Precision:{mp_}%  Recall:{mr}%  F1:{mf1}%")
        print(f"\n  --- Head Drop (Pitch < -{HEAD_DROP_DEG} do) ---")
        print(f"  TP={self.head_tp}  FP={self.head_fp}  TN={self.head_tn}  FN={self.head_fn}")
        print(f"  Accuracy:{ha}%  Precision:{hp}%  Recall:{hr}%  F1:{hf1}%")
        print(sep)

    def save_json(self, path: str):
        ep, er, ef1, ea = self.ear_m()
        mp_, mr, mf1, ma = self.mar_m()
        hp, hr, hf1, ha = self.head_m()
        data = {
            "summary": {
                "total_frames": self.total_frames,
                "detected_frames": self.detected_frames,
                "detection_rate_pct": self.dr(),
                "avg_latency_ms": self.lat_avg(),
                "p95_latency_ms": self.lat_p95(),
            },
            "thresholds_used": {
                "ear_closed": DEFAULT_T_CLOSED,
                "mar_yawn": DEFAULT_T_YAWN,
                "head_drop_deg": HEAD_DROP_DEG,
                "microsleep_ms": MICROSLEEP_MS,
            },
            "ear":  {"tp": self.ear_tp, "fp": self.ear_fp, "tn": self.ear_tn,  "fn": self.ear_fn,
                     "accuracy_pct": ea, "precision_pct": ep, "recall_pct": er, "f1_pct": ef1},
            "mar":  {"tp": self.mar_tp, "fp": self.mar_fp, "tn": self.mar_tn,  "fn": self.mar_fn,
                     "accuracy_pct": ma, "precision_pct": mp_, "recall_pct": mr, "f1_pct": mf1},
            "head": {"tp": self.head_tp,"fp": self.head_fp,"tn": self.head_tn, "fn": self.head_fn,
                     "accuracy_pct": ha, "precision_pct": hp, "recall_pct": hr, "f1_pct": hf1},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Ket qua da luu: {path}")


# =====================================================================
# EVALUATOR CHINH
# =====================================================================
class Evaluator:
    """
    Thu import module that cua du an (face_mesh, geo_metrics).
    Neu khong co, dung fallback MediaPipe thuan tuy.
    """

    def __init__(self):
        self.stats = EvalStats()
        self.gt_closed = False
        self.gt_yawn   = False
        self.gt_head   = False
        self._use_project = False
        self._mesh_proj   = None
        self._compute_geo = None
        self._NO_FACE     = None
        self._mesh_raw    = None
        self._t0 = time.perf_counter()

        try:
            from face_mesh import FaceMesh, MODEL_PATH
            from geo_metrics import compute_geo_metrics, NO_FACE_GEO
            self._mesh_proj   = FaceMesh(str(MODEL_PATH), running_mode="video")
            self._compute_geo = compute_geo_metrics
            self._NO_FACE     = NO_FACE_GEO
            self._use_project = True
            print("[eval] Su dung module chinh thuc: face_mesh + geo_metrics")
        except Exception as e:
            import mediapipe as mp
            self._mesh_raw = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            print(f"[eval] Fallback MediaPipe thuan ({e})")

    def _fallback_metrics(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._mesh_raw.process(rgb)
        if not res.multi_face_landmarks:
            return False, 0.0, 0.0, 0.0
        lm = res.multi_face_landmarks[0].landmark

        def pt(i): return np.array([lm[i].x * w, lm[i].y * h])
        def d(i, j): return float(np.linalg.norm(pt(i) - pt(j)))

        # EAR - cong thuc chinh xac cho 2 mat
        ear_r = (d(160, 144) + d(158, 153)) / (2.0 * d(33, 133) + 1e-9)
        ear_l = (d(385, 380) + d(387, 373)) / (2.0 * d(362, 263) + 1e-9)
        ear   = (ear_r + ear_l) / 2.0
        # MAR
        mar_ver = d(13, 14) + d(312, 317)
        mar     = mar_ver / (2.0 * d(61, 291) + 1e-9)
        # Pitch xap xi (solvePnP don gian)
        nose_y = lm[1].y * h
        eye_y  = (lm[159].y + lm[386].y) * h / 2.0
        chin_y = lm[152].y * h
        face_h = chin_y - eye_y + 1e-9
        pitch  = (nose_y - eye_y) / face_h
        pitch_deg = (pitch - 0.35) * 90.0
        return True, ear, mar, pitch_deg

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        if self._use_project:
            ts_ms = int((time.perf_counter() - self._t0) * 1000)
            result = self._mesh_proj.process(frame, ts_ms)
            geo = self._compute_geo(result, (w, h))
            detected = result.present
            ear, mar, pitch = geo.ear, geo.mar, geo.pitch_deg
        else:
            detected, ear, mar, pitch = self._fallback_metrics(frame)

        dt = (time.perf_counter() - t0) * 1000
        self.stats.total_frames  += 1
        self.stats.latencies.append(dt)
        if detected:
            self.stats.detected_frames += 1

        out = frame.copy()
        if detected:
            p_eye  = ear   < DEFAULT_T_CLOSED
            p_yawn = mar   > DEFAULT_T_YAWN
            p_head = pitch < -HEAD_DROP_DEG

            for kind, pred, gt in [
                ("ear",  p_eye,  self.gt_closed),
                ("mar",  p_yawn, self.gt_yawn),
                ("head", p_head, self.gt_head),
            ]:
                self.stats.update_cm(kind, pred, gt)

            _, _, ef1, ea = self.stats.ear_m()
            c_e = (0, 0, 255) if p_eye  else (0, 220, 60)
            c_m = (0, 140, 255) if p_yawn else (0, 220, 60)
            c_h = (0, 0, 255) if p_head else (0, 220, 60)

            cv2.rectangle(out, (0, 0), (w, 125), (10, 10, 10), -1)
            cv2.putText(out,
                f"EAR={ear:.3f} (nguong {DEFAULT_T_CLOSED}) {'<< NHAM MAT' if p_eye else 'Mo mat'}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_e, 2)
            cv2.putText(out,
                f"MAR={mar:.3f} (nguong {DEFAULT_T_YAWN}) {'<< NGAP' if p_yawn else 'Binh thuong'}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_m, 2)
            cv2.putText(out,
                f"Pitch={pitch:.1f}d (nguong -{HEAD_DROP_DEG}) {'<< GUC DAU' if p_head else ''}",
                (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_h, 2)
            cv2.putText(out,
                f"Lat={dt:.1f}ms | Frame={self.stats.total_frames} | Det={self.stats.dr()}%",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)
        else:
            cv2.putText(out, "!! KHONG PHAT HIEN KHUON MAT !!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

        # Bottom info panel
        _, _, ef1, ea = self.stats.ear_m()
        cv2.rectangle(out, (0, h - 90), (w, h), (8, 8, 30), -1)
        gt_text = (
            f"[GT] Mat:{'NHAM' if self.gt_closed else 'Mo'} "
            f"Mieng:{'NGAP' if self.gt_yawn else 'Thuong'} "
            f"Dau:{'GUC' if self.gt_head else 'Thang'}"
        )
        cv2.putText(out, gt_text, (10, h - 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 80), 1)
        cv2.putText(out,
            f"Detection:{self.stats.dr()}%  |  EAR Acc:{ea}%  F1:{ef1}%",
            (10, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (80, 255, 200), 1)
        cv2.putText(out,
            "[O]Mo  [C]Nham  [Y]Ngap  [H]GucDau  [N]Normal  [Q]Thoat+LuuJSON",
            (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)
        return out

    def handle_key(self, key: int) -> bool:
        c = chr(key & 0xFF).lower()
        if c == "q": return True
        if c == "o": self.gt_closed = False
        if c == "c": self.gt_closed = True
        if c == "y": self.gt_yawn   = True
        if c == "h": self.gt_head   = True
        if c == "n": self.gt_closed = self.gt_yawn = self.gt_head = False
        return False

    def close(self):
        if self._mesh_proj:
            try: self._mesh_proj.close()
            except Exception: pass


# =====================================================================
# MAIN LOOP
# =====================================================================
def run_loop(cap: cv2.VideoCapture, ev: Evaluator, duration: int, out_json: str):
    t0 = time.time()
    print("Bat dau... Dung phim [O/C/Y/H/N] de gan nhan, [Q] de thoat va luu ket qua.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = ev.process(frame)
        if duration > 0:
            rem = int(duration - (time.time() - t0))
            cv2.putText(out, f"Con lai: {rem}s",
                        (out.shape[1] - 200, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            if rem <= 0:
                break
        cv2.imshow("MediaPipe Accuracy Evaluator - DriverSafeIoT", out)
        if ev.handle_key(cv2.waitKey(1)):
            break
    cap.release()
    cv2.destroyAllWindows()
    ev.close()
    ev.stats.print_report()
    ev.stats.save_json(out_json)


def main():
    parser = argparse.ArgumentParser(
        description="Danh gia do chinh xac MediaPipe pipeline - DriverSafeIoT")
    parser.add_argument("--mode", choices=["realtime", "video"], default="realtime",
                        help="realtime = tu webcam | video = tu file video")
    parser.add_argument("--input",    default=None,
                        help="Duong dan file video (chi dung khi --mode video)")
    parser.add_argument("--cam",      type=int, default=0,
                        help="Index camera (mac dinh 0)")
    parser.add_argument("--duration", type=int, default=120,
                        help="Thoi gian thu nghiem realtime (giay, mac dinh 120)")
    parser.add_argument("--out",      default="eval_result.json",
                        help="File JSON luu ket qua (mac dinh eval_result.json)")
    args = parser.parse_args()

    ev = Evaluator()

    if args.mode == "video":
        if not args.input:
            print("Loi: Can truyen --input <duong_dan_video> khi dung mode=video")
            sys.exit(1)
        cap = cv2.VideoCapture(args.input)
        dur = 0
    else:
        cap = cv2.VideoCapture(args.cam)
        dur = args.duration

    if not cap.isOpened():
        print("Loi: Khong mo duoc nguon video/camera!")
        sys.exit(1)

    run_loop(cap, ev, dur, args.out)


if __name__ == "__main__":
    main()
