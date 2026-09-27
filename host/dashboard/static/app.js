/**
 * Main Frontend Application Controller (DASH-02 / DASH-03).
 * Điều phối kết nối WebSocket, cập nhật trạng thái Node, Metric Cards và Điều khiển hệ thống.
 */

document.addEventListener("DOMContentLoaded", () => {
  // 1. Khởi tạo các Components
  const gauge = new window.RiskGauge("riskGaugeCanvas");
  const charts = new window.DashboardCharts("fatigueChartCanvas", "cabinChartCanvas");

  let ws = null;
  let reconnectTimer = null;

  // 2. Kết nối WebSocket với Auto-Reconnect
  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log(`[dashboard] Dang ket noi WebSocket toi ${wsUrl}...`);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("[dashboard] WebSocket ket noi thanh cong!");
      clearTimeout(reconnectTimer);
      updateNodeBadge("node-ws-status", "online", "WS ONLINE");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleWsMessage(msg);
      } catch (err) {
        console.error("[dashboard] Loi parse WS message:", err);
      }
    };

    ws.onclose = () => {
      console.warn("[dashboard] WebSocket da dong. Thu ket noi lai sau 2s...");
      updateNodeBadge("node-ws-status", "offline", "WS OFFLINE");
      reconnectTimer = setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
      console.error("[dashboard] WebSocket gap loi:", err);
      ws.close();
    };
  }

  // 3. Xử lý các loại thông điệp từ WebSocket
  function handleWsMessage(msg) {
    if (msg.type === "init") {
      // Nạp trạng thái ban đầu và lịch sử biểu đồ
      if (msg.status) updateSystemStatus(msg.status);
      if (msg.recent_telemetry && msg.recent_telemetry.length > 0) {
        charts.loadHistory(msg.recent_telemetry);
        const lastPoint = msg.recent_telemetry[msg.recent_telemetry.length - 1];
        updateMetricCards(lastPoint);
        gauge.setScore(lastPoint.risk, lastPoint.band);
      } else if (msg.latest_telemetry) {
        updateMetricCards(msg.latest_telemetry);
        gauge.setScore(msg.latest_telemetry.risk, msg.latest_telemetry.band);
      }
      if (msg.recent_events) renderEventsList(msg.recent_events);
    } else if (msg.type === "telemetry") {
      const data = msg.data;
      // Cập nhật đồng hồ Gauge
      gauge.setScore(data.risk, data.band);

      // Cập nhật biểu đồ
      charts.pushPoint(data);

      // Cập nhật các thẻ Metric Cards
      updateMetricCards(data);

      // Cập nhật trạng thái hệ thống
      if (msg.status) updateSystemStatus(msg.status);

      // Kiểm tra banner cảnh báo đỏ toàn màn hình
      toggleAlertBanner(data.band, data.drivers, data.action);
    } else if (msg.type === "event") {
      prependEventItem(msg.data);
    }
  }

  // 4. Cập nhật các chỉ số chi tiết trên Metric Cards
  function updateMetricCards(d) {
    setText("val-ear", d.ear.toFixed(3));
    setText("val-perclos", `${(d.perclos_60s * 100).toFixed(1)}%`);
    setText("val-yawn", d.yawn_per_min.toFixed(1));
    setText("val-cles", `${Math.round(d.cles_dur_ms)} ms`);
    setText("val-pitch", `${d.head_pitch_deg > 0 ? "+" : ""}${d.head_pitch_deg.toFixed(1)}°`);
    setText("val-brac", `${d.alcohol_g_l.toFixed(3)} g/L`);
    setText("val-temp", `${d.temp_c.toFixed(1)} °C`);
    setText("val-lux", `${d.lux_mode.toUpperCase()} (${d.ldr_pct}%)`);
    setText("val-rssi", `${d.rssi} dBm`);

    // Head drop badge
    const hdElem = document.getElementById("val-head-drop");
    if (hdElem) {
      if (d.head_drop) {
        hdElem.textContent = "CÚI ĐẦU";
        hdElem.className = "px-2 py-0.5 rounded text-xs font-bold bg-red-900/60 text-red-300 border border-red-500/50";
      } else {
        hdElem.textContent = "BÌNH THƯỜNG";
        hdElem.className = "px-2 py-0.5 rounded text-xs font-medium bg-gray-800 text-gray-400";
      }
    }
  }

  // 5. Cập nhật trạng thái các Node (Online / Degraded / Offline)
  function updateSystemStatus(s) {
    updateNodeBadge("node-vision-status", s.vision_online ? "online" : "offline", s.vision_online ? "ONLINE" : "OFFLINE");
    updateNodeBadge("node-esp32-status", s.esp32_online ? "online" : "offline", s.esp32_online ? "ONLINE" : "OFFLINE");
    updateNodeBadge("node-fusion-status", s.fusion_online ? "online" : "offline", s.fusion_online ? "ONLINE" : "OFFLINE");
    updateNodeBadge("node-mqtt-status", s.mqtt_online ? "online" : "offline", s.mqtt_online ? "CONNECTED" : "DISC");

    // Trạng thái khóa động cơ
    const lockElem = document.getElementById("engine-lock-status");
    if (lockElem) {
      if (s.engine_locked) {
        lockElem.textContent = "ĐÃ KHÓA";
        lockElem.className = "px-3 py-1 rounded-full text-xs font-bold bg-red-600/30 text-red-400 border border-red-500 animate-pulse";
      } else {
        lockElem.textContent = "HOẠT ĐỘNG";
        lockElem.className = "px-3 py-1 rounded-full text-xs font-bold bg-emerald-600/20 text-emerald-400 border border-emerald-500/40";
      }
    }

    // Drivers danh sách rủi ro hiện tại
    const driversElem = document.getElementById("active-drivers-list");
    if (driversElem) {
      if (s.active_drivers && s.active_drivers.length > 0) {
        driversElem.innerHTML = s.active_drivers
          .map(d => `<span class="px-2 py-0.5 rounded text-xs bg-gray-800 text-amber-300 border border-amber-500/30 font-mono">${d}</span>`)
          .join(" ");
      } else {
        driversElem.innerHTML = `<span class="text-xs text-gray-500">None (Normal)</span>`;
      }
    }
  }

  function updateNodeBadge(elemId, status, text) {
    const el = document.getElementById(elemId);
    if (!el) return;
    el.textContent = text;
    if (status === "online") {
      el.className = "px-2 py-0.5 rounded text-xs font-semibold badge-online";
    } else if (status === "degraded") {
      el.className = "px-2 py-0.5 rounded text-xs font-semibold badge-degraded";
    } else {
      el.className = "px-2 py-0.5 rounded text-xs font-semibold badge-offline";
    }
  }

  // 6. Quản lý Timeline sự kiện rủi ro
  function renderEventsList(events) {
    const listEl = document.getElementById("events-timeline-list");
    if (!listEl) return;
    listEl.innerHTML = "";
    events.forEach(e => listEl.appendChild(createEventElement(e)));
  }

  function prependEventItem(e) {
    const listEl = document.getElementById("events-timeline-list");
    if (!listEl) return;
    const item = createEventElement(e);
    listEl.insertBefore(item, listEl.firstChild);
    // Giới hạn hiển thị 30 sự kiện gần nhất
    while (listEl.children.length > 30) {
      listEl.removeChild(listEl.lastChild);
    }
  }

  function createEventElement(e) {
    const div = document.createElement("div");
    div.className = "flex items-center justify-between p-2.5 rounded bg-gray-900/60 border border-gray-800 text-xs hover:border-gray-700 transition";

    const timeStr = new Date(e.ts).toLocaleTimeString("vi-VN", { hour12: false });
    const isCritical = e.band === "CRITICAL" || e.action === "lock";
    const badgeColor = isCritical ? "text-red-400 bg-red-950/40 border-red-800" : "text-amber-400 bg-amber-950/40 border-amber-800";

    div.innerHTML = `
      <div class="flex items-center space-x-2">
        <span class="font-mono text-gray-500">${timeStr}</span>
        <span class="px-1.5 py-0.5 rounded border font-semibold ${badgeColor}">${e.event_name.toUpperCase()}</span>
        <span class="text-gray-300">${e.source.toUpperCase()}</span>
      </div>
      <div class="text-right">
        <span class="font-mono font-bold text-gray-200">Risk: ${Number(e.risk).toFixed(1)}</span>
        <span class="text-gray-500 ml-1">(${e.action})</span>
      </div>
    `;
    return div;
  }

  // 7. Banner cảnh báo khẩn cấp toàn màn hình
  let alertDismissedUntil = 0;

  window.dismissAlertBanner = function() {
    const banner = document.getElementById("critical-fullscreen-alert");
    if (banner) banner.classList.add("hidden");
    alertDismissedUntil = Date.now() + 15000; // Tạm tắt cảnh báo trong 15s
    showToast("Đã tạm tắt cảnh báo trong 15 giây", "info");
  };

  function toggleAlertBanner(band, drivers, action) {
    const banner = document.getElementById("critical-fullscreen-alert");
    if (!banner) return;

    if (Date.now() < alertDismissedUntil) {
      banner.classList.add("hidden");
      return;
    }

    if (band === "CRITICAL" || band === "ALARM") {
      banner.classList.remove("hidden");
      const title = document.getElementById("alert-banner-title");
      const sub = document.getElementById("alert-banner-sub");
      if (action === "lock") {
        if (title) title.textContent = "NGUY HIỂM CỒN CAO — ĐỘNG CƠ ĐÃ KHÓA!";
        if (sub) sub.textContent = "Nồng độ cồn vượt ngưỡng an toàn. Xe đã bị vô hiệu hóa khởi động.";
      } else {
        if (title) title.textContent = "CẢNH BÁO MỆT MỎI NGUY HIỂM — NGƯNG LÁI NGAY!";
        if (sub) sub.textContent = "Phát hiện buồn ngủ cực độ hoặc ngủ gật. Hãy tấp xe vào lề an toàn để nghỉ ngơi.";
      }
    } else {
      banner.classList.add("hidden");
    }
  }

  // 8. Quy trình Hiệu Chuẩn Cá Nhân Hóa Lái Xe 60 Giây (DASH-03 / EXP-01)
  let enrollTimer = null;
  let enrollRemainingSeconds = 60;

  window.startEnrollProcess = async function() {
    const modal = document.getElementById("enroll-modal");
    if (!modal) return;

    // Gửi lệnh enroll xuống thiết bị IoT
    await window.sendSystemCommand("enroll");

    enrollRemainingSeconds = 60;
    updateEnrollUi(60);
    modal.classList.remove("hidden");

    if (enrollTimer) clearInterval(enrollTimer);

    enrollTimer = setInterval(() => {
      enrollRemainingSeconds--;
      updateEnrollUi(enrollRemainingSeconds);

      if (enrollRemainingSeconds <= 0) {
        clearInterval(enrollTimer);
        enrollTimer = null;
        modal.classList.add("hidden");
        showToast("Hiệu chuẩn lái xe 60s hoàn tất thành công!", "success");
      }
    }, 1000);
  };

  window.cancelEnrollProcess = function() {
    const modal = document.getElementById("enroll-modal");
    if (modal) modal.classList.add("hidden");
    if (enrollTimer) {
      clearInterval(enrollTimer);
      enrollTimer = null;
    }
    showToast("Đã hủy bỏ quy trình hiệu chuẩn", "info");
  };

  function updateEnrollUi(remainingSec) {
    const countdownEl = document.getElementById("enroll-countdown");
    const progressEl = document.getElementById("enroll-progress-bar");
    if (countdownEl) countdownEl.textContent = `${remainingSec}s`;
    if (progressEl) {
      const pct = Math.round(((60 - remainingSec) / 60) * 100);
      progressEl.style.width = `${pct}%`;
    }
  }

  // 9. Tương tác gửi lệnh điều khiển (DASH-03 Ready)
  window.sendSystemCommand = async function(cmd, extra = {}) {
    try {
      const resp = await fetch("/api/cmd", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd, ...extra }),
      });
      const result = await resp.json();
      if (resp.ok) {
        if (cmd === "unlock") {
          // Cập nhật ngay trạng thái mở khóa trên UI
          const lockElem = document.getElementById("engine-lock-status");
          if (lockElem) {
            lockElem.textContent = "HOẠT ĐỘNG";
            lockElem.className = "px-3 py-1 rounded-full text-xs font-bold bg-emerald-600/20 text-emerald-400 border border-emerald-500/40";
          }
          showToast("Đã mở khóa động cơ thành công!", "success");
        } else {
          showToast(`Đã gửi lệnh: ${cmd.toUpperCase()}`, "success");
        }
      } else {
        showToast(`Lỗi: ${result.detail || "Không thể gửi lệnh"}`, "error");
      }
    } catch (err) {
      showToast(`Lỗi mạng: ${err.message}`, "error");
    }
  };

  function showToast(msg, type = "info") {
    const toast = document.createElement("div");
    const bg = type === "success" ? "bg-emerald-600" : (type === "error" ? "bg-red-600" : "bg-blue-600");
    toast.className = `fixed bottom-4 right-4 ${bg} text-white px-4 py-2 rounded shadow-lg text-sm font-semibold z-50 transition-all duration-300 opacity-0 transform translate-y-2`;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => { toast.classList.remove("opacity-0", "translate-y-2"); }, 50);
    setTimeout(() => {
      toast.classList.add("opacity-0", "translate-y-2");
      setTimeout(() => toast.remove(), 300);
    }, 2500);
  }

  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  // Khởi động kết nối
  connectWebSocket();
});
