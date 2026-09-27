"""Driver Personalization & Calibration Workflow (EXP-01).

Thu thập phân bố sinh trắc học cá nhân của tài xế trong 60 giây tỉnh táo:
- Tính kỳ vọng μ và độ lệch chuẩn σ của EAR:
    + Ngưỡng ban ngày (day mode):   T_closed = μ - 1.5σ
    + Ngưỡng ban đêm (night mode): T_closed = μ - 2.0σ (chặt hơn theo spec §6.4)
- Tính góc nghiêng đầu trung tính neutral: pitch_neutral = μ_pitch
- Tính ngưỡng ngáp cá nhân: T_yawn = μ_mar + 1.5σ_mar
- Lưu hồ sơ cá nhân hóa ra file config/driver_profile.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

# Thêm path để import nội bộ
VISION_DIR = Path(__file__).resolve().parent
REPO_ROOT = VISION_DIR.parent.parent
sys.path.insert(0, str(VISION_DIR))
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROFILE_PATH = REPO_ROOT / "config" / "driver_profile.json"


@dataclass
class DriverProfile:
    """Hồ sơ thông số sinh trắc học và ngưỡng cá nhân hóa của tài xế."""

    driver_id: str = "default"
    calibrated_at: str = ""
    ear_mean: float = 0.28
    ear_std: float = 0.02
    t_closed_day: float = 0.25
    t_closed_night: float = 0.24
    pitch_neutral: float = 0.0
    t_yawn: float = 0.50
    num_samples: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, file_path: Path | str = DEFAULT_PROFILE_PATH) -> None:
        """Lưu hồ sơ ra file JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2, ensure_ascii=False)
        print(f"[calibrate] Da luu cau hinh lai xe vao: {path}")

    @classmethod
    def load(cls, file_path: Path | str = DEFAULT_PROFILE_PATH) -> DriverProfile | None:
        """Đọc hồ sơ từ file JSON. Trả None nếu file hỏng hoặc thiếu trường."""
        path = Path(file_path)
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[calibrate] File ho so bi hong (JSON loi): {path} -> {e}", file=sys.stderr)
            return None

        # Bỏ qua các field thừa từ phiên bản profile cũ để không crash TypeError
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        try:
            return cls(**filtered)
        except TypeError as e:
            print(f"[calibrate] File ho so thieu truong bat buoc: {path} -> {e}", file=sys.stderr)
            return None

    def get_t_closed(self, lux_mode: str = "day") -> float:
        """Lấy ngưỡng nhắm mắt thích ứng theo điều kiện ánh sáng (spec §6.4)."""
        if str(lux_mode).lower() in ("night", "dark"):
            return self.t_closed_night
        return self.t_closed_day


class DriverCalibrator:
    """Bộ thu thập dữ liệu và tính toán phân bố chuẩn sinh trắc học trong 60 giây."""

    def __init__(self, target_samples: int = 600) -> None:
        self.target_samples = target_samples  # 60 giây @ 10Hz = 600 mẫu
        self.ear_samples: list[float] = []
        self.mar_samples: list[float] = []
        self.pitch_samples: list[float] = []

    def add_sample(self, ear: float, mar: float, pitch: float, face: bool = True) -> bool:
        """Thêm một mẫu đo hợp lệ từ camera vào bộ đệm tích lũy."""
        if not face:
            return False

        # Chỉ lọc các giá trị bất thường (landmark bị nhiễu/mất, mặt chếch).
        # Lưu ý: MAR của miệng khép tự nhiên rất nhỏ (khoảng 0.001-0.02, đã
        # kiểm chứng bằng probe pixel gốc), nên KHÔNG được lọc theo ngưỡng MAR
        # — làm vậy sẽ vứt bỏ phần lớn mẫu "miệng khép" và làm μ_mar sai lệch.
        if ear <= 0.05:
            return False
        if mar <= 0.0:
            return False
        if not np.isfinite(ear) or not np.isfinite(mar) or not np.isfinite(pitch):
            return False

        self.ear_samples.append(float(ear))
        self.mar_samples.append(float(mar))
        self.pitch_samples.append(float(pitch))
        return True

    @property
    def sample_count(self) -> int:
        return len(self.ear_samples)

    @property
    def progress_pct(self) -> float:
        return min(100.0, (self.sample_count / max(1, self.target_samples)) * 100.0)

    def is_complete(self) -> bool:
        return self.sample_count >= self.target_samples

    def compute_profile(self, driver_id: str = "driver_01") -> DriverProfile:
        """Tính toán các tham số thống kê và ngưỡng tối ưu theo chuẩn Gaussian."""
        if not self.ear_samples:
            raise ValueError("Chua co du lieu mau de hieu chuan!")

        ears = np.array(self.ear_samples, dtype=np.float64)
        mars = np.array(self.mar_samples, dtype=np.float64)
        pitches = np.array(self.pitch_samples, dtype=np.float64)

        # 1. Phân bố EAR: EAR ~ N(μ, σ)
        mu_ear = float(np.mean(ears))
        sigma_ear = float(np.std(ears))
        # Giới hạn sigma [0.012, 0.045]: chống σ=0 khi ngồi bất động và chống nhiễu chớp mắt
        sigma_ear = max(0.012, min(0.045, sigma_ear))

        # Theo spec §6.3: T_closed = μ - 1.5σ (Ban ngày)
        t_closed_day = round(mu_ear - 1.5 * sigma_ear, 3)
        # Theo spec §6.4: T_closed = μ - 2.0σ (Ban đêm - chặt hơn)
        t_closed_night = round(mu_ear - 2.0 * sigma_ear, 3)

        # Giới hạn ngưỡng trong dải sinh lý [0.10, 0.32] — luôn dưới μ_ear để
        # mắt mở bình thường không bao giờ bị phân loại là "nhắm".
        # Lưu ý: KHÔNG kẹp t_closed_day về một cận dưới cố định, vì với người
        # mắt hẹp (μ_ear ≈ 0.15) một cận dưới 0.16 sẽ cao hơn cả mắt mở thật.
        lo, hi = 0.10, 0.32
        t_closed_day = max(lo, min(hi, t_closed_day))
        t_closed_night = max(lo, min(hi, t_closed_night))
        # Đảm bảo tính chất "đêm chặt hơn ngày" (spec §6.4) không bị phá vỡ
        if t_closed_night > t_closed_day:
            t_closed_night = t_closed_day
        # Bảo vệ cuối cùng: ngưỡng không được vượt quá mắt mở trung bình
        if t_closed_day >= mu_ear:
            t_closed_day = round(max(lo, mu_ear * 0.80), 3)
            t_closed_night = round(max(lo, mu_ear * 0.72), 3)

        # 2. Pitch neutral: Góc chúc đầu trung tính khi ngồi tự nhiên
        mu_pitch = float(np.mean(pitches))

        # 3. Phân bố MAR và ngưỡng ngáp: T_yawn
        mu_mar = float(np.mean(mars))
        sigma_mar = float(np.std(mars))
        # Ngưỡng ngáp cá nhân tối thiểu 0.48 (sigma tối thiểu 0.05 chống chia 0)
        t_yawn = round(max(0.48, mu_mar + 1.5 * max(0.05, sigma_mar)), 3)

        now_iso = datetime.now(timezone.utc).isoformat()
        profile = DriverProfile(
            driver_id=driver_id,
            calibrated_at=now_iso,
            ear_mean=round(mu_ear, 4),
            ear_std=round(sigma_ear, 4),
            t_closed_day=t_closed_day,
            t_closed_night=t_closed_night,
            pitch_neutral=round(mu_pitch, 1),
            t_yawn=t_yawn,
            num_samples=len(ears),
        )
        return profile


def run_calibration(
    duration_s: float = 60.0,
    driver_id: str = "driver_01",
    src: str = "webcam",
    cam_index: int = 0,
    cam_url: str | None = None,
    output_path: Path | str = DEFAULT_PROFILE_PATH,
    headless: bool = False,
) -> DriverProfile | None:
    """Chạy quy trình hiệu chuẩn từ camera thời gian thực."""
    import cv2
    from capture import FrameSource
    from face_mesh import FaceMesh
    from geo_metrics import compute_geo_metrics

    print("==================================================================")
    print(f"   DRIVER PERSONALIZATION CALIBRATION WORKFLOW (EXP-01)")
    print(f"   Tai xe: {driver_id} | Thoi luong: {duration_s:.0f}s")
    print("==================================================================")
    print(" Huong dan: Ngoi thang lung, mat mo tu nhien nhin ve phia truoc.")

    source = FrameSource(src=src, index=cam_index, url=cam_url)
    try:
        source.open()
    except (RuntimeError, ValueError) as e:
        print(f"[calibrate] Khong the mo camera {src}:{cam_index} -> {e}", file=sys.stderr)
        return None

    mesh = FaceMesh()
    target_samples = int(duration_s * 10)  # 10 Hz downsampled
    calibrator = DriverCalibrator(target_samples=target_samples)

    t_start = time.perf_counter()
    last_sample_t = 0.0
    last_progress_sec = -1
    cancelled = False

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                break

            now = time.perf_counter()
            elapsed_s = now - t_start
            t_ms = elapsed_s * 1000.0

            # Lấy mẫu tại 10 Hz (100ms)
            if now - last_sample_t >= 0.10:
                last_sample_t = now
                res = mesh.process(frame, int(t_ms))
                if res.present:
                    geo = compute_geo_metrics(res, (frame.shape[1], frame.shape[0]))
                    calibrator.add_sample(geo.ear, geo.mar, geo.pitch_deg, face=True)

            # In tiến độ đúng 1 lần mỗi 5 giây
            cur_sec = int(elapsed_s)
            if cur_sec != last_progress_sec and cur_sec % 5 == 0 and cur_sec > 0:
                last_progress_sec = cur_sec
                print(f"[calibrate] Tien do: {calibrator.progress_pct:4.1f}% ({calibrator.sample_count}/{target_samples} mau)")

            if not headless:
                progress_text = f"CALIBRATING: {calibrator.progress_pct:4.1f}% ({duration_s - elapsed_s:3.0f}s left)"
                cv2.putText(frame, progress_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.imshow("DriverSafe-IoT | Calibration", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[calibrate] Nguoi dung da huy bo qua trinh.")
                    cancelled = True
                    break
            else:
                # headless khong co waitKey: nho CPU giay cho camera
                time.sleep(0.01)

            if calibrator.is_complete() or elapsed_s >= duration_s:
                break

    except KeyboardInterrupt:
        print("\n[calibrate] Da dung theo yeu cau (Ctrl+C).")
        cancelled = True
    finally:
        source.release()
        mesh.close()
        if not headless:
            cv2.destroyAllWindows()

    if cancelled:
        return None

    if calibrator.sample_count < 100:
        print(f"[calibrate] Khong thu thap du so mau mat hop le ({calibrator.sample_count}/100)!", file=sys.stderr)
        return None

    profile = calibrator.compute_profile(driver_id=driver_id)
    profile.save(output_path)

    print("\n==================================================================")
    print(f" [V] HIEU CHUAN HOAN TAT!")
    print(f" - EAR trung binh (mu)    : {profile.ear_mean:.3f} (std={profile.ear_std:.3f})")
    print(f" - Nguong ban ngay (T_day) : {profile.t_closed_day:.3f}")
    print(f" - Nguong ban dem (T_night): {profile.t_closed_night:.3f}")
    print(f" - Pitch trung tinh       : {profile.pitch_neutral:+.1f} deg")
    print(f" - Nguong ngap (T_yawn)    : {profile.t_yawn:.3f}")
    print("==================================================================")
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Driver Personalization & Calibration (EXP-01)")
    parser.add_argument("--duration", type=float, default=60.0, help="Thoi gian thu thap (giay, mac dinh: 60s)")
    parser.add_argument("--driver-id", default="driver_01", help="ID dinh danh tai xe")
    parser.add_argument("--src", choices=["webcam", "mjpeg"], default="webcam")
    parser.add_argument("--index", type=int, default=0, help="Index webcam")
    parser.add_argument("--url", default=None, help="MJPEG stream URL (cho --src mjpeg)")
    parser.add_argument("--output", default=str(DEFAULT_PROFILE_PATH), help="Duong dan file output profile JSON")
    parser.add_argument("--headless", action="store_true", help="Chay khong mo cua so GUI")
    args = parser.parse_args(argv)

    prof = run_calibration(
        duration_s=args.duration,
        driver_id=args.driver_id,
        src=args.src,
        cam_index=args.index,
        cam_url=args.url,
        output_path=args.output,
        headless=args.headless,
    )
    return 0 if prof is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
