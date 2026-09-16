"""Nguồn frame cho Vision — webcam laptop hoặc MJPEG stream (ESP32-S3-CAM).

Tách riêng khỏi phần xử lý để `face_metrics` / `temporal` không phụ thuộc
nguồn ảnh: đổi nguồn chỉ là đổi tham số `--src`.

    python capture.py                      # webcam laptop, 640x480
    python capture.py --src mjpeg --url http://192.168.44.50:81/stream
    python capture.py --index 1            # webcam thứ 2
    python capture.py --headless --duration 60   # do FPS, khong mo cua so

Phím khi cửa sổ đang mở:  q/ESC = thoát,  s = chụp ảnh PNG vào experiments/
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

# Webcam laptop tren Windows: DSHOW nhanh hon MSMF dang ke (do thuc te 31 vs 13 FPS).
# MSMF chi dung lam phuong an du phong khi DSHOW khong mo duoc thiet bi.
_BACKENDS = [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)]

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
FPS_WINDOW = 30  # so frame dung de tinh FPS trung binh dong


@dataclass
class FpsMeter:
    """FPS trung binh dong tren cua so N frame gan nhat."""

    window: int = FPS_WINDOW
    _times: deque[float] = field(default_factory=lambda: deque(maxlen=FPS_WINDOW))
    _total: int = 0
    _t0: float = field(default_factory=time.perf_counter)

    def tick(self) -> None:
        self._times.append(time.perf_counter())
        self._total += 1

    @property
    def fps(self) -> float:
        """FPS tuc thoi (cua so truot)."""
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0

    @property
    def fps_avg(self) -> float:
        """FPS trung binh tu luc bat dau — dung de nghiem thu '>=20 FPS trong 5 phut'."""
        span = time.perf_counter() - self._t0
        return self._total / span if span > 0 else 0.0

    @property
    def frames(self) -> int:
        return self._total


class FrameSource:
    """Bọc `cv2.VideoCapture` cho cả webcam lẫn MJPEG, tự chọn backend & báo lỗi rõ ràng."""

    def __init__(
        self,
        src: str = "webcam",
        index: int = 0,
        url: str | None = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> None:
        self.src = src
        self.index = index
        self.url = url
        self.width = width
        self.height = height
        self.backend_name = "-"
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self.src == "mjpeg":
            self._open_mjpeg()
        else:
            self._open_webcam()

    def _open_mjpeg(self) -> None:
        if not self.url:
            raise ValueError("--src mjpeg can --url (vd http://<ip-esp32>:81/stream)")

        # cv2.VideoCapture tren URL khong toi duoc se BLOCK VO HAN (da kiem chung).
        # Kiem tra ket noi truoc bang socket de bao loi som thay vi treo im lang.
        self._check_reachable()

        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            raise RuntimeError(
                f"Khong mo duoc MJPEG stream: {self.url}\n"
                "  Kiem tra: ESP32 da chay chua? cung mang chua? dung IP chua?"
            )
        self.cap = cap
        self.backend_name = "mjpeg"
        # MJPEG giu nguyen do phan giai goc cua stream, khong ep kich thuoc.
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or "?"
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or "?"
        print(f"[capture] MJPEG {self.url} -> {w}x{h}")

    def _open_webcam(self) -> None:
        errors = []
        for name, backend in _BACKENDS:
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                errors.append(f"{name}: khong mo duoc thiet bi {self.index}")
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # MJPG: nhieu webcam chi cho >=30 FPS o dinh dang nen nay.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # giam do tre: luon lay frame moi nhat
            # Vai backend can 1-2 frame dau de settle truoc khi doc kich thuoc that.
            for _ in range(3):
                cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w == 0 or h == 0:
                errors.append(f"{name}: mo duoc nhung khong tra frame")
                cap.release()
                continue
            self.cap = cap
            self.backend_name = name
            print(f"[capture] webcam index={self.index} backend={name} -> {w}x{h}")
            return
        raise RuntimeError(
            "Khong mo duoc webcam nao.\n  " + "\n  ".join(errors) + "\n"
            "  Kiem tra: Settings > Privacy > Camera da bat? app khac dang giu camera?"
        )

    def _check_reachable(self, timeout: float = 3.0) -> None:
        """Mo TCP toi host:port cua URL truoc khi giao cho OpenCV."""
        parsed = urlparse(self.url)
        host = parsed.hostname
        if not host:
            raise ValueError(f"--url khong hop le: {self.url}")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except OSError as e:
            raise RuntimeError(
                f"Khong ket noi duoc toi {host}:{port} ({e}).\n"
                "  Kiem tra: ESP32 da bat chua? cung mang WiFi chua? dung IP chua?\n"
                "  Thu: ping " + host
            ) from e

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            raise RuntimeError("FrameSource chua duoc open()")
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def draw_hud(frame: np.ndarray, fps: float, src_label: str, paused: bool = False) -> np.ndarray:
    """Vẽ overlay thông số lên frame. Trả về chính frame đó (in-place)."""
    h, w = frame.shape[:2]
    text = f"FPS {fps:5.1f}"
    if paused:
        text += "  [PAUSE]"
    # Nen mo phia sau chu de doc duoc tren moi nen anh.
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (8, 8), (18 + tw, 20 + th), (0, 0, 0), -1)
    # Xanh khi dat moc >=20 FPS, do khi tut duoi.
    color = (0, 255, 0) if fps >= 20 else (0, 0, 255)
    cv2.putText(frame, text, (12, 16 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, src_label, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def save_snapshot(frame: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(path), frame)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nguon frame cho Vision (webcam / MJPEG)")
    p.add_argument("--src", choices=["webcam", "mjpeg"], default="webcam",
                   help="nguon frame (mac dinh: webcam)")
    p.add_argument("--index", type=int, default=0, help="chi so webcam (mac dinh: 0)")
    p.add_argument("--url", default=None, help="URL MJPEG khi --src mjpeg")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--headless", action="store_true",
                   help="khong mo cua so (dung de do FPS tu dong / chay tren may khong man hinh)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="tu thoat sau N giay (0 = chay mai)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        source = FrameSource(args.src, args.index, args.url, args.width, args.height)
        source.open()
    except (RuntimeError, ValueError) as e:
        print(f"[capture] LOI: {e}", file=sys.stderr)
        return 1

    src_label = args.url if args.src == "mjpeg" else f"webcam[{args.index}]"
    meter = FpsMeter()
    t_start = time.perf_counter()
    paused = False
    last_report = t_start
    frames_root = Path(__file__).resolve().parents[2] / "experiments"

    print("[capture] nhan q/ESC de thoat, s de chup anh" if not args.headless else "[capture] headless")
    try:
        while True:
            if not paused:
                ok, frame = source.read()
                if not ok or frame is None:
                    print("[capture] mat frame / khong doc duoc — thoat", file=sys.stderr)
                    return 2
                meter.tick()
            else:
                time.sleep(0.01)
                frame = last_frame

            last_frame = frame
            fps = meter.fps

            if args.headless:
                now = time.perf_counter()
                if now - last_report >= 1.0:
                    print(f"[capture] FPS {fps:5.1f} (avg {meter.fps_avg:5.1f}) "
                          f"frames={meter.frames} elapsed={now - t_start:.0f}s")
                    last_report = now
            else:
                cv2.imshow("DriverSafe-IoT | capture", draw_hud(frame.copy(), fps, src_label, paused))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    path = save_snapshot(frame, frames_root)
                    print(f"[capture] da luu {path}")
                if key == ord(" "):
                    paused = not paused

            if args.duration and time.perf_counter() - t_start >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(f"[capture] ket thuc: {meter.frames} frame / {elapsed:.1f}s "
          f"-> FPS trung binh {meter.fps_avg:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
