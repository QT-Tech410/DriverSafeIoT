"""Tai cac file model MediaPipe can thiet ve host/vision/models/.

MediaPipe >= 1.0 khong con bundle san model trong package, phai tai rieng.
Chay 1 lan sau khi clone repo:

    python scripts/fetch_models.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "host" / "vision" / "models"

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}


def fetch(name: str, url: str, force: bool = False) -> bool:
    dest = MODELS_DIR / name
    if dest.exists() and not force:
        print(f"  [bo qua] {name} da co ({dest.stat().st_size:,} bytes)")
        return True
    print(f"  [tai]    {name} ...")
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  [LOI]    {name}: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return False
    print(f"  [xong]   {name} ({dest.stat().st_size:,} bytes)")
    return True


def main() -> int:
    force = "--force" in sys.argv
    print(f"Thu muc model: {MODELS_DIR}")
    ok = all(fetch(n, u, force) for n, u in MODELS.items())
    if not ok:
        print("\nTai model that bai. Kiem tra ket noi mang roi chay lai.", file=sys.stderr)
        return 1
    print("\nSan sang. Model da o dung cho.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())