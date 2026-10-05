# -*- coding: utf-8 -*-
"""
collect_dataset.py - Thu thap anh mau de benchmark EAR

Cach dung:
  .venv\Scripts\python.exe host\vision\collect_dataset.py

Phim:
  [O] - Chup 1 anh mat MO (luu vao datasets/cew/open)
  [C] - Chup 1 anh mat NHAM (luu vao datasets/cew/closed)
  [Q] - Ket thuc thu thap
"""
import cv2
import os
import time
from pathlib import Path

open_dir = Path("datasets/cew/open")
closed_dir = Path("datasets/cew/closed")
open_dir.mkdir(parents=True, exist_ok=True)
closed_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Khong mo duoc camera!")
    exit(1)

o_count = len(list(open_dir.glob("*.jpg")))
c_count = len(list(closed_dir.glob("*.jpg")))

print("=== CONG CU CHUP ANH DATASET TEST ===")
print(" [O] - Chup anh MAT MO")
print(" [C] - Chup anh MAT NHAM")
print(" [Q] - Thoat")
print("======================================")

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(frame, f"Da chup: OPEN={o_count} | CLOSED={c_count}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    cv2.putText(frame, "Bam [O]: Mo mat | [C]: Nham mat | [Q]: Xong", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("Collect Dataset - DriverSafeIoT", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('o') or key == ord('O'):
        o_count += 1
        fn = open_dir / f"open_{int(time.time()*1000)}.jpg"
        cv2.imwrite(str(fn), frame)
        print(f"[OK] Da luu anh MAT MO: {fn.name}")
    elif key == ord('c') or key == ord('C'):
        c_count += 1
        fn = closed_dir / f"closed_{int(time.time()*1000)}.jpg"
        cv2.imwrite(str(fn), frame)
        print(f"[OK] Da luu anh MAT NHAM: {fn.name}")

cap.release()
cv2.destroyAllWindows()
print(f"Xong! Tong cong da chup: {o_count} anh MO, {c_count} anh NHAM.")
