# CÔNG VIỆC — PHẦN B: Python / AI Vision / Fusion / Dashboard

> **Người phụ trách:** B (Python/AI) · **Chạy trên:** laptop demo · **Ngôn ngữ:** Python 3.10+
> **Spec gốc:** `docs/superpowers/specs/2026-09-15-driver-safety-iot-design.md` (§3, 5, 6, 8, 9)
> **File này là "hợp đồng việc làm" của bạn — tick dần từng ô, mỗi ô có tiêu chí "xong" rõ ràng.**

---

## 0. Bạn sở hữu những gì trong hệ thống

```
┌──────────────────── PHẦN B (Python, trên laptop) ────────────────────┐
│  ① VISION      camera → MediaPipe → EAR/PERCLOS/ngáp/gật → metrics  │
│  ② BROKER      mosquitto (cài đặt + chạy, không code)               │
│  ③ FUSION      nhận metrics + telemetry → Mamdani → RISK → lệnh     │
│  ④ DASHBOARD   FastAPI + WebSocket + SQLite + HTML/Chart.js          │
│  ⑤ SIM/TOOLS   sim_sensors.py, fixture test, scripts thực nghiệm     │
└──────────────────────────────────────────────────────────────────────┘
        ▲ MQTT: ds/esp32/sensors, ds/esp32/events          │ MQTT: ds/fusion/level, ds/esp32/cmd
        │                                                  ▼
┌──────────────────── PHẦN A (C++ trên ESP32-S3) ─────────────────────┐
│  đọc MQ-3/NTC/LDR · buzzer/LED/OLED/servo · FSM cục bộ · WiFi MQTT  │
└──────────────────────────────────────────────────────────────────────┘
```

**Nguyên tắc vàng:** 2 phần **không gọi code nhau**, chỉ trao đổi JSON qua MQTT theo schema §5. B không cần biết chân ADC, A không cần biết MediaPipe.

---

## PHẦN 1 — VISION (`host/vision/`) — tuần 1 → 2

| # | Việc | File | Tiêu chí "xong" | Spec |
|---|---|---|---|---|
| B0.1 | Môi trường: venv, `pip install mediapipe opencv-python paho-mqtt numpy scipy pyyaml fastapi uvicorn`, cài mosquitto, tạo repo/nhánh | `requirements.txt` | `python capture.py` mở được webcam | — |
| B1.1 | Capture webcam (640×480, đo FPS in overlay) | `capture.py` | ≥20 FPS ổn định 5 phút | §6.1 |
| B1.2 | Face Mesh 468 điểm, max 1 face, vẽ overlay | `face_metrics.py` | Landmark bám mặt khi quay ±20° | §6.1 |
| B1.3 | Tính **EAR, MAR** chuẩn hóa theo khoảng cách 2 mắt | `face_metrics.py` | Log ra số; nhắm mắt → EAR tụt thấy rõ | §6.2 |
| B1.4 | **Head pose** bằng `solvePnP` (5 điểm 3D) → pitch/neutral 5s | `face_metrics.py` | Gật đầu → pitch đổi >15° | §6.2 |
| B1.5 | Lớp thời gian: buffer tròn 60s mẫu 10Hz → **PERCLOS, blink/CLES ≥250ms, microsleep ≥500ms, yawn/min, head_drop ≥0.8s**, chống nhiễu 3 frame, `face_lost` ≥2s | `temporal.py` | Tự dàn cảnh "nhắm mắt 3s" trong video → log ra đúng 1 CLES event | §6.2 |
| B1.6 | `publisher.py` — gom metrics 1Hz → publish `ds/vision/metrics` đúng schema | `publisher.py` | `mosquitto_sub -t ds/vision/metrics` thấy JSON chảy đều | §5 |
| B1.7 | Test không cần camera: chạy pipeline từ file video/ảnh fixture | `tests/` + fixture | CI-local: pytest xanh, PERCLOS khớp nhãn thủ công ±5% | §12.1 |
| B1.8 | *(dự phòng, nếu A có S3-CAM)* reader MJPEG từ ESP32 làm nguồn frame thay webcam | `capture.py --src=mjpeg` | Switch source bằng 1 tham số | §3 |

## PHẦN 2 — FUSION (`host/fusion/`) — tuần 2 → 3

| # | Việc | File | Tiêu chí "xong" | Spec |
|---|---|---|---|---|
| B3.1 | Subscriber: nhận `ds/esp32/sensors` + `ds/esp32/events` + `ds/vision/metrics`; cache mới nhất + timestamp | `fusion.py` | In ra đủ 2 luồng khi test | §3 |
| B3.2 | **Fusion tạm rule-weighted** (chưa Mamdani): risk = w₁·PERCLOS + w₂·blink_events + w₃·ALCO + w₄·CABIN, clamp 0–100 | `policy.py` | Đủ để chạy E2E cuối tuần 2 | §11 |
| B3.3 | Publish `ds/fusion/level` (QoS1 retain) + `ds/esp32/cmd` theo bảng band; cooldown giữa các đợt | `fusion.py` | Sub thấy risk đổi trong ≤1s từ khi đầu vào đổi | §5, §8 |
| B3.4 | **Mamdani**: mf tam giác 6 inputs/1 tập ra, 12 luật, min-implication, defuzz centroid; luật + mf đọc từ **YAML** | `fuzzy.py`, `rules.yaml` | Test fixture: bộ đầu vào cho trước → risk nằm khoảng mong đợi; in được `drivers` (luật fired) | §8 |
| B3.5 | Chính sách hành động: phân biệt driver=fatigue (không khóa) vs driver=alcohol (khóa khi ALCO=2 xác nhận 2/3 lần); fallback `vision_offline` >2s | `policy.py` | Unit test từng nhánh chính sách | §8 |
| B3.6 | Log sự kiện → `ds/events` + đẩy vào DB | `fusion.py` | Mỗi cảnh báo có bản ghi: ts, risk, drivers, kèm bộ metrics tại thời điểm fires | §5, §9 |

## PHẦN 3 — DASHBOARD (`host/dashboard/`) — tuần 2 khung → tuần 3 đủ

| # | Việc | File | Tiêu chí "xong" | Spec |
|---|---|---|---|---|
| B4.1 | FastAPI + SQLite schema (events, telemetry 1Hz sample); WS bridge từ paho | `server.py`, `db.py` | 1 tiến trình, restart không mất DB | §9 |
| B4.2 | **Realtime panel**: gauge risk 4 band, chỉ số vision/cabin, trạng thái node (online/degraded/offline + RSSI), video overlay MJPEG/MJPEG-in-WS hoặc ảnh JPEG 5Hz | `static/index.html` | Cập nhật ≤500ms; node offline hiện rõ | §9, §12.7 |
| B4.3 | **Event timeline** + banner đỏ toàn màn hình khi ALARM+ | `static/` | Click sự kiện → xem metrics kèm theo | §9 |
| B4.4 | **History/analytics**: chart 5′ + theo ca (Chart.js), trang Report, **export CSV** | `static/` + `/api/history` | Xuất được CSV của 1 ca demo | §9 |
| B4.5 | Nút vận hành: **Enroll driver** (gọi flow calibrate), **Unlock engine**, **Reset FSM** → publish cmd tương ứng | `server.py` | Bấm Unlock → ESP32 của A nhận `{"cmd":"unlock"}` | §9 |

## PHẦN 4 — CÁ NHÂN HÓA & THỰC NGHIỆM (`host/vision/calibrate.py`, `experiments/`) — tuần 3 → 4

| # | Việc | Tiêu chí "xong" | Spec |
|---|---|---|---|
| B5.1 | **Enroll 60s**: thu phân bố EAR/MAR khi tỉnh (+1 lần há miệng) → `T_closed = μ−kσ` (ngày) / `k=2.0σ` (đêm, theo `lux_mode`); lưu JSON per-driver; neutral head-pitch 5s | Flow bấm nút → enroll → metrics đổi threshold ngay; 60s tỉnh táo không phát sinh "mắt-khép" giả | §6.3, §6.4 |
| B5.2 | `lux_mode` từ telemetry A điều chỉnh ngưỡng Vision | Test che sáng → ngưỡng đổi, log ghi rõ | §6.4 |
| B5.3 | Thực nghiệm báo cáo: confusion matrix mở/khép; PERCLOS cá nhân vs cố định; bảng latency frame→cmd→(buzzer A); surface Risk 2D PERCLOS×ALCO (matplotlib); bảng CSV xuất từ dashboard | Đủ số liệu dán vào chương Thuật toán của báo cáo | §15 |

## PHẦN 5 — HOÀN THIỆN — tuần 4

- [ ] `scripts/run_all.bat` (hoặc tài liệu 4 cmd): broker → vision → fusion → dashboard, chống thứ tự sai
- [ ] Chạy thử demo 8 bước (§13) ít nhất 3 lần, bấm giờ
- [ ] Video quay màn hình demo phòng sự cố
- [ ] Đóng góp chương kiến trúc + thuật toán vào báo cáo, slide phần B

---

# LIÊN KẾT HAI PHẦN — Góc nhìn từ PHẦN B

## L1. Hiệp ước giao tiếp: schema MQTT (đọc spec §5 cho chi tiết từng trường)

Broker mosquitto chạy trên **laptop của B** (`mqtt/mosquitto.conf`, port 1883, `allow_anonymous true` — mạng demo nội bộ). A connect vào IP laptop này qua WiFi.

| Topic | A → B / B → A | B dùng để làm gì |
|---|---|---|
| `ds/esp32/sensors` (1Hz) | A→B | Input fusion: `alcohol_g_l, temp_c, ldr_pct, lux_mode, rssi, degraded` |
| `ds/esp32/events` | A→B | Sự kiện cồn tức thời → log + event timeline |
| `ds/vision/metrics` (1Hz) | B→B | Output của Vision, input của Fusion (nội bộ, cùng máy nhưng vẫn qua MQTT cho "thật" hệ thống) |
| `ds/fusion/level` (retain) | B→mọi | Dashboard hiển thị; A không bắt buộc phải dùng (FSM A tự chạy) |
| `ds/esp32/cmd` | **B→A** | **Bạn gửi, A thực thi**: `lock`/`unlock`/`beep`/`oled`/`reset` — xem L3 |

**Quy tắc hiệp ước:**
1. **Đóng băng schema chiều thứ 6 tuần 1.** Sau đó muốn đổi field → PR có review chéo, không "đột xuất sửa lúc 11h đêm trước demo".
2. Mọi message là JSON UTF-8, có `ts` (ms, clock laptop B; A gửi kèm `uptime_s` để suy offset); field số dùng số học thuần, không chuỗi hóa.
3. Field thừa thì bỏ qua, field thiếu thì dùng default + log warning — code phía B phải `try/except KeyError` ở subscriber, **không crash vì 1 message lỗi**.
4. QoS: telemetry `QoS0`, event/cmd `QoS1`.

## L2. Cách B test khi A CHƯA xong (không bao giờ bị block)

- `scripts/sim_sensors.py`: tiến trình giả node A — publish `ds/esp32/sensors` 1Hz, phím `1/2/3` mô phỏng nồng độ cồn level 0/1/2, phím `t` đổi temp, `l` đổi lux, `d` bật `degraded_mode`. → **Từ tuần 1 bạn đã test fusion + dashboard với "node tưởng tượng" này.**
- Chiều ngược lại cho A: `scripts/test_pub.py` gửi cmd giả, A test chấp hành mà chưa cần fusion thật.
- Fixture video: thu 3 clip (tỉnh/lim dim/nhắm 3s) ngay tuần 1, lưu `experiments/` — mọi lần chỉnh thuật toán chạy lại từ clip, không cần người diễn lại.

## L3. Hợp đồng hành vi khi hệ cùng chạy (A phải tuân theo những gì B gửi)

| Tình huống | Hành động mong đợi của hệ thống | Ai quyết | Ai thực thi |
|---|---|---|---|
| Cồn ALCO=2 xác nhận 2/3 lần | `{"cmd":"lock"}` | Fusion B | Servo A; unlock **chỉ** khi người bấm dashboard → B publish `{"cmd":"unlock"}` |
| Buồn ngủ CRITICAL (risk>75, driver=fatigue) | `{"cmd":"beep","n":3}` + `{"cmd":"oled","msg":"NGUng LÁI"}` | Fusion B | Buzzer/OLED A. **KHÔNG bao giờ lock** |
| Mất WiFi (A `degraded_mode=true` trong telemetry) | Dashboard hiện cảnh báo vàng "chế độ tự chủ" | B hiển thị | A tự FSM cục bộ |
| Mất metrics vision >2s | Fusion dùng nhánh thuần cồn+cabin, gắn cờ `vision_offline` | B | Không cần A làm gì |
| Host/laptop tắt | — | — | A **vẫn** kêu còi vì cồn (FSM cục bộ) — đây là tính năng, không phải bug |

## L4. Các lần HẸP GẶP thật sự giữa 2 người (chỉ 3 lần, xếp lịch trước)

1. **Thứ 6 tuần 1 — chốt schema (30–60′):** 2 máy cùng mạng, B chạy broker + sim_sensors, A connect thật, ping-pong 1 message mỗi chiều theo §5. *Xong buổi này là "ly hôn mềm": mỗi người tự làm 2 tuần tiếp.*
2. **Cuối tuần 2 — RAP E2E (nửa ngày, mốc quan trọng nhất dự án):** A flash firmware thật, tắt sim → thổi cồn thật: đo (A) → metrics (B) → fusion Mamdani tạm (B) → cmd (B) → còi+LED (A) → dashboard đỏ (B). Checklist 7 tiêu chí §12 bắt đầu đo từ đây.
3. **Tuần 4 — ráp demo + đo latency liên thông (nửa ngày):** đồng hồ bấm từ "frame nhắm mắt" → "tiếng còi"; chạy full kịch bản §13 ≥3 lần.

Ngoài 3 lần đó: commit/push hằng ngày lên **cùng 1 repo git**, mỗi nhánh tự chủ (`firmware/*` vs `host/*`), đụng file của nhau thì qua PR.

## L5. Ranh giới trách nhiệm — những việc KHÔNG được làm hộ/đùn cho nhau

- ❌ B không viết thuật toán đọc cảm biến vào firmware (chỉ góp ý ngưỡng qua `config` YAML khi test chung).
- ❌ A không nhân bản luật fusion sang FSM cục bộ (FSM A chỉ có nhánh cồn/nhiệt đơn giản để sống-sót-mất-kết-nối; logic quyết định nằm ở B).
- ✅ Chung duy nhất: file `mqtt/` + schema §5 — ai đổi cũng phải cả 2 đồng ý.

---

## Checklist nhanh "tối nay bắt đầu thế nào"

1. Tạo project PyCharm tại `D:\Workspace\DuanIoT\host\`, dán `requirements.txt` từ §10 spec vào, `pip install -r`.
2. Cài mosquitto, chạy `mosquitto -c ..\mqtt\mosquitto.conf -v`.
3. Viết `capture.py` ~30 dòng (B1.1+B1.2) — mục tiêu tối nay: **thấy 468 điểm trên mặt mình trong webcam**.
4. Nhắc bạn A: đặt hàng linh kiện HÔM NAY (lead time là rủi ro R5).
