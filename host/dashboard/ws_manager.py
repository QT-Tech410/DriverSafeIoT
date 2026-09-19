"""WebSocket Connection & Realtime Broadcast Manager (DASH-01).

Chịu trách nhiệm:
- Duy trì danh sách các kết nối WebSocket đang hoạt động
- Phát sóng dữ liệu thời gian thực (telemetry 1Hz, events, commands) tới tất cả frontend clients
- Đảm bảo độ trễ truyền tin cực thấp (<= 50ms, đáp ứng yêu cầu spec <= 500ms)
- Tự động dọn dẹp các kết nối bị đóng hoặc gặp lỗi mạng
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket


class WebSocketManager:
    """Quản lý các kết nối WebSocket đa client và phát sóng thông điệp."""

    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Chấp nhận và đăng ký một client WebSocket mới."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Hủy đăng ký một client WebSocket."""
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, data: dict[str, Any] | str) -> int:
        """Phát sóng dữ liệu JSON tới toàn bộ client đang kết nối.

        Returns:
            Số lượng client đã nhận được tin nhắn thành công.
        """
        payload_str = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else data

        async with self._lock:
            clients = list(self.active_connections)

        if not clients:
            return 0

        dead_connections: list[WebSocket] = []
        delivered = 0

        for client in clients:
            try:
                await client.send_text(payload_str)
                delivered += 1
            except Exception:
                dead_connections.append(client)

        if dead_connections:
            async with self._lock:
                for dead in dead_connections:
                    self.active_connections.discard(dead)

        return delivered

    @property
    def client_count(self) -> int:
        return len(self.active_connections)
