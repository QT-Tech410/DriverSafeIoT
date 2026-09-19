"""SQLite Database Engine for Telemetry & Events History (DASH-01).

Chịu trách nhiệm quản lý cơ sở dữ liệu SQLite:
- Khởi tạo schema 2 bảng: `telemetry` (lịch sử 1Hz) và `events` (nhật ký sự kiện rủi ro)
- Bảo toàn dữ liệu (persistence) khi server tắt/mở lại
- Tối ưu hiệu năng đa luồng với WAL mode (Write-Ahead Logging)
- Cung cấp API truy vấn lịch sử phục vụ biểu đồ Realtime và trang Analytics/Report
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from models import EventRecord, TelemetryRecord

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "driversafe.db"


class Database:
    """Quản lý kết nối và các thao tác CRUD với SQLite Database."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        # Đảm bảo thư mục cha tồn tại
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Tạo kết nối SQLite an toàn với row_factory."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Tạo bảng và các index nếu chưa tồn tại, bật chế độ WAL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Bật Write-Ahead Logging để đọc/ghi đồng thời không khóa DB
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")

            # 1. Bảng telemetry (1Hz)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    risk REAL DEFAULT 0.0,
                    band TEXT DEFAULT 'SAFE',
                    drivers TEXT DEFAULT '[]',
                    action TEXT DEFAULT 'none',
                    perclos_60s REAL DEFAULT 0.0,
                    ear REAL DEFAULT 0.0,
                    cles_dur_ms REAL DEFAULT 0.0,
                    mar REAL DEFAULT 0.0,
                    yawn_per_min REAL DEFAULT 0.0,
                    head_pitch_deg REAL DEFAULT 0.0,
                    head_drop INTEGER DEFAULT 0,
                    alcohol_g_l REAL DEFAULT 0.0,
                    alco_level INTEGER DEFAULT 0,
                    temp_c REAL DEFAULT 25.0,
                    lux_mode TEXT DEFAULT 'day',
                    ldr_pct INTEGER DEFAULT 50,
                    rssi INTEGER DEFAULT -60
                );
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry (ts);")

            # 2. Bảng events (sự kiện tức thời)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    risk REAL DEFAULT 0.0,
                    band TEXT DEFAULT 'SAFE',
                    action TEXT DEFAULT 'none',
                    details TEXT DEFAULT '{}'
                );
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_name ON events (event_name);")
            conn.commit()

    def insert_telemetry(self, record: TelemetryRecord) -> int:
        """Lưu một bản ghi telemetry 1Hz vào bảng telemetry."""
        drivers_json = json.dumps(record.drivers, ensure_ascii=False)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO telemetry (
                    ts, risk, band, drivers, action,
                    perclos_60s, ear, cles_dur_ms, mar, yawn_per_min,
                    head_pitch_deg, head_drop, alcohol_g_l, alco_level,
                    temp_c, lux_mode, ldr_pct, rssi
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(record.ts),
                    float(record.risk),
                    str(record.band),
                    drivers_json,
                    str(record.action),
                    float(record.perclos_60s),
                    float(record.ear),
                    float(record.cles_dur_ms),
                    float(record.mar),
                    float(record.yawn_per_min),
                    float(record.head_pitch_deg),
                    1 if record.head_drop else 0,
                    float(record.alcohol_g_l),
                    int(record.alco_level),
                    float(record.temp_c),
                    str(record.lux_mode),
                    int(record.ldr_pct),
                    int(record.rssi),
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def insert_event(self, record: EventRecord) -> int:
        """Lưu một bản ghi sự kiện rủi ro vào bảng events."""
        details_json = json.dumps(record.details, ensure_ascii=False)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO events (
                    ts, source, event_name, risk, band, action, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(record.ts),
                    str(record.source),
                    str(record.event_name),
                    float(record.risk),
                    str(record.band),
                    str(record.action),
                    details_json,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_recent_telemetry(self, limit: int = 60) -> list[TelemetryRecord]:
        """Lấy N bản ghi telemetry gần nhất (mặc định 60 điểm cho biểu đồ 1 phút)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM telemetry ORDER BY ts DESC LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        results: list[TelemetryRecord] = []
        for r in reversed(rows):  # Trả về theo thứ tự thời gian tăng dần
            results.append(
                TelemetryRecord(
                    id=r["id"],
                    ts=r["ts"],
                    risk=r["risk"],
                    band=r["band"],
                    drivers=json.loads(r["drivers"] or "[]"),
                    action=r["action"],
                    perclos_60s=r["perclos_60s"],
                    ear=r["ear"],
                    cles_dur_ms=r["cles_dur_ms"],
                    mar=r["mar"],
                    yawn_per_min=r["yawn_per_min"],
                    head_pitch_deg=r["head_pitch_deg"],
                    head_drop=bool(r["head_drop"]),
                    alcohol_g_l=r["alcohol_g_l"],
                    alco_level=r["alco_level"],
                    temp_c=r["temp_c"],
                    lux_mode=r["lux_mode"],
                    ldr_pct=r["ldr_pct"],
                    rssi=r["rssi"],
                )
            )
        return results

    def get_recent_events(self, limit: int = 50) -> list[EventRecord]:
        """Lấy danh sách N sự kiện rủi ro mới nhất."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM events ORDER BY ts DESC LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        results: list[EventRecord] = []
        for r in rows:
            results.append(
                EventRecord(
                    id=r["id"],
                    ts=r["ts"],
                    source=r["source"],
                    event_name=r["event_name"],
                    risk=r["risk"],
                    band=r["band"],
                    action=r["action"],
                    details=json.loads(r["details"] or "{}"),
                )
            )
        return results

    def get_counts(self) -> dict[str, int]:
        """Đếm số bản ghi trong các bảng để kiểm tra persistence."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry;")
            t_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM events;")
            e_count = cursor.fetchone()[0]
        return {"telemetry": t_count, "events": e_count}
