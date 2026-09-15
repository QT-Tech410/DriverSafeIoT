# CÔNG VIỆC — PHẦN A: Embedded / ESP32-S3 Firmware

> **Người phụ trách:** A (Embedded/C++) · **Chạy trên:** board ESP32-S3 · **Ngôn ngữ:** C++ (Arduino framework)
> **Spec gốc:** `docs/superpowers/specs/2026-09-15-driver-safety-iot-design.md` (§4, 5, 7)
> **File này là "hợp đồng việc làm" của bạn — tick dần từng ô, mỗi ô có tiêu chí "xong" rõ ràng.**

---

## 0. Bạn sở hữu những gì trong hệ thống

```
┌──────────────── PHẦN A (C++, flash lên ESP32-S3) ────────────────────┐
│  CẢM BIẾN        MQ-3 (cồn) · NTC (nhiệt) · LDR (ánh sáng)           │
│  CHẤP HÀNH       buzzer · LED RGB · OLED I2C · servo SG90 (khóa máy) │
│  FSM CỤC BỘ      SAFE→WARN→ALARM→LOCKED — tự kêu còi khi MẤT WiFi    │
│  COMM            WiFi station + MQTT client (publish/subscribe)       │
└──────────────────────────────────────────────────────────────────────┘
        │ MQTT: ds/esp32/sensors, ds/esp32/events        ▲ MQTT: ds/esp32/cmd
        ▼                                                │
┌──────────────── PHẦN B (Python, trên laptop) ────────────────────────┐
│  vision (EAR/PERCLOS) · fusion (Mamdani → RISK) · dashboard · broker │
└──────────────────────────────────────────────────────────────────────┘
```

**Nguyên tắc vàng:** 2 phần **không gọi code nhau**, chỉ trao đổi JSON qua MQTT theo schema §5. A không cần biết MediaPipe là gì; B không cần biết chân ADC số mấy.

**Việc đặc quyền của A (B không làm, cũng không quyết được):** *đo cho đúng* và *chấp hành cho nhanh*. Mọi "chuyện gì xảy ra khi cồn cao" mà người ta demo được tại chỗ — đều đi qua tay bạn.

---

## PHẦN 1 — NỀN TẢNG (tuần 1)

| # | Việc | Tiêu chí "xong" | Spec |
|---|---|---|---|
| A0.1 | **Đặt hàng linh kiện HÔM NAY** (BOM §4: ESP32-S3-DevKitC-1, MQ-3, NTC10K+R, LDR+R, OLED SSD1306, buzzer, LED, SG90, breadboard ~330k) | Có mã vận đơn, tin ship 1–2 ngày | §4 |
| A0.2 | PlatformIO project tại `firmware/esp32-node/` (board esp32-s3-devkitc-1, framework arduino), git branch `firmware/`, build/flash blink | `pio run -t upload` blink LED onboard; **Arduino IDE cũng chấp nhận được** nếu bạn không quen PlatformIO — chỉ cần cấu trúc module §7.5 | §7.5 |
| A0.3 | WiFi station + reconnect backoff, log IP | Board vào được WiFi nhà/hotspot, IP in ra Serial | §3 |
| A0.4 | MQTT client (PubSubClient) — connect broker IP-laptop-B, publish `hello`, subscribe `ds/esp32/cmd` echo lại Serial | B chạy `mosquitto_sub` thấy `hello` 1Hz | §5 |

## PHẦN 2 — CẢM BIẾN & ADC (tuần 1–2)

| # | Việc | Tiêu chí "xong" | Spec |
|---|---|---|---|
| A1.1 | Cấu hình ADC1 attenuated 11dB; **hiệu chuẩn ADC** bằng multimeter 5 điểm (0.5/1.2/2.0/2.7/3.2V), fit đa thức bậc 2 vào `adc_cal` | Serial đọc áp sai số ≤±0.05V | §7.1.1 |
| A1.2 | **MQ-3**: mạch chia áp AO về ≤3.3V, đọc 10Hz, IIR `y=0.8y+0.2x`; tính `Rs=(V_H/V_out−1)·R_L` | V_ao, Rs hiển thị khi hà hơi | §7.1.1–2 |
| A1.3 | **Baseline tự động**: minimum-tracking 30′ → `ratio=Rs/R0_running` + log raw+ratio | 10′ sau khi hà hơi, R0 tự hạ dần về "không khí" | §7.1.3 |
| A1.4 | **Bù nhiệt bằng NTC chính nó**: `ratio_corr=ratio·(1+k_t(T−25))` (chỉ bật nếu test §7.1.4 chứng minh lệch >15% — test xong quyết định, ghi cả kết quả âm) | Có số liệu test 2 mức nhiệt trong `calibration/` | §7.1.4 |
| A1.5 | **NTC** công thức B-parameter (B=3950, T0=298.15), phân tầng LẠNH/OK/NÓNG; kiểm chứng cốc nước đá + nước ấm | Sai số ≤±1.5°C so nhiệt kế đối chứng | §7.2 |
| A1.6 | **LDR** divider 10K, đọc %ADC, phân loại SÁNG/ĐỦ/TỐI (úp tay= TỐI) | `lux_mode` đổi đúng khi che sáng | §7.3 |
| A1.7 | **Đường chuẩn MQ-3 thực nghiệm**: mẫu ethanol ~40° pha loãng headspace 2–3 nồng độ → bảng `ratio→ppm→g/L` (0.004×ppm) — file `calibration/mq3_curve.csv` + ảnh setup | Vẽ được curve; có ngưỡng ALCO 0/1/2 chốt bằng số | §7.1.5–6 |

## PHẦN 3 — CHẤP HÀNH & FSM (tuần 2)

| # | Việc | Tiêu chí "xong" | Spec |
|---|---|---|---|
| A2.1 | Buzzer (pattern 1 nhát / 3 nhịp / 5s liên tục), LED RGB (vàng/đỏ), Servo 0↔90° | Test tuần tự từng thiết bị | §7.4 |
| A2.2 | **OLED SSD1306**: trạng thái band + alcohol/temp/lux + "DEGRADED" khi mất mạng (chỉ refresh khi đổi, tránh tearing) | Đọc được trạng thái từ 2m | §7.4 |
| A2.3 | **FSM 4 trạng thái** SAFE→WARN→ALARM→LOCKED: nhánh cục bộ (ratio cồn, temp) + nhánh nhận risk từ host; cooldown 90s; **quy tắc bất di bất dịch: FSM cục bộ không bị host ghi đè khi đang ALARM/LOCKED do ngưỡng cục bộ** | Ngắt WiFi, thổi cồn → vẫn ALARM ≤3s; host không "tắt còi" được | §7.4 |
| A2.4 | Xử lý `ds/esp32/cmd`: `lock` (chỉ khi B xác nhận — A cứ tin cmd, logic điều kiện ở B), `unlock` phải kèm `sig`, `beep`, `oled`, `reset` | B bấm dashboard → servo quay; unlock → quay về | §5, §8 |
| A2.5 | Watchdog: MQTT mất >5s → telemetry gắn `degraded_mode:true`, tự reconnect backoff, không block task cảm biến | Cắm/rút mạng demo: tự hồi phục ≤10s | §3 |

## PHẦN 4 — ĐÓNG GÓI & TỐI ƯU (tuần 3–4)

| # | Việc | Tiêu chí "xong" | Spec |
|---|---|---|---|
| A3.1 | Chia module đúng §7.5 (`sensors_mq3/ntc/ldr`, `fsm`, `buzzer`, `oled`, `mqtt_client`, `adc_cal`), mỗi file <200 dòng | Đọc code 1 người hiểu được từng module | §7.5 |
| A3.2 | Unit-test logic quy đổi trên PC (fixture ADC dựng sẵn: bảng giá trị giả → so output) | `pio test` xanh | §7.5 |
| A3.3 | **Mạch đồng/mô hình cabin**: cắm chắc (không breadboard lỏng khi demo), dây gọn, nguồn 5V≥500mA độc lập, công tắc nguồn | Rung nhẹ bàn không đứt kết nối | §11 |
| A3.4 | *(tùy chọn nếu mua ESP32-S3-CAM)* stream MJPEG trên cổng 81 để B dùng làm nguồn vision dự phòng | B mở `http://<ip-esp>:81` thấy ảnh | §3, B1.8 |
| A3.5 | Đo đạc cho báo cáo: thời gian warm-up MQ-3, bảng độ lặp 5 lần cùng mẫu chuẩn (≤±15%), ảnh chụp setup | Số vào chương Thực nghiệm | §12.3, §15 |
| A3.6 | Đóng góp chương Kiến trúc phần cứng + cảm biến vào báo cáo; chuẩn bị phần demo "cabin" (úp tay che LDR, hơ NTC) | Kịch bản §13 chạy suôn | §13 |

---

# LIÊN KẾT HAI PHẦN — Góc nhìn từ PHẦN A

## L1. Hiệp ước giao tiếp: schema MQTT (chi tiết từng trường ở spec §5)

**Broker nằm ở laptop B** (`mosquitto`, IP laptop — hardcode trong `config.h`, đổi được khi test). A là **WiFi station** cùng mạng, KHÔNG tự chạy access point.

| Topic | Chiều | A dùng để làm gì |
|---|---|---|
| `ds/esp32/sensors` (QoS0, **1Hz**) | A→B | Bạn publish JSON: `{"ts","mq3_ao_v","rs_r0","alcohol_g_l","temp_c","ldr_pct","lux_mode","rssi","uptime_s","degraded"}` (`"degraded":true` khi mất mạng, `"lux_mode"` phân loại §7.3) |
| `ds/esp32/events` (QoS1, tức thời) | A→B | Khi FSM cục bộ đổi band: `{"ts","event":"alcohol_level2",...}` |
| `ds/fusion/level` (QoS1, retain) | B→mọi | Bạn **subscribe để hiển thị risk lên OLED** + làm input WARN/ALARM (nhưng không bắt buộc — mất B vẫn chạy) |
| `ds/esp32/cmd` (QoS1) | **B→A** | Lệnh chấp hành: `lock`/`unlock`/`beep`/`oled`/`reset` — xem L3 |

**Quy tắc hiệp ước:**
1. **Đóng băng schema thứ 6 tuần 1.** Muốn thêm field (vd `degraded_mode`) → nói B, 2 người đồng ý rồi đổi cả 2 phía trong cùng PR.
2. JSON UTF-8, có `ts` (ms) + `uptime_s` để B suy lệch clock. **Số học thuần, không chuỗi hóa**, không dùng số NaN (nếu ADC lỗi → gửi `-1` + flag).
3. Parser phía A phải bền: `ArduinoJson` kiểm tra `containsKey` trước khi đọc; msg lạ → bỏ qua, log Serial, **không panic/reboot**.
4. Publish fail (mất mạng) → xếp hàng tối đa 5 msg gần nhất, không phình heap.

## L2. Cách A test khi B chưa xong (không bao giờ bị block)

- Tuần 1: test mọi thứ bằng **Serial monitor** — chưa cần MQTT cũng thấy V_ao/Rs/T/ratio.
- Test lệnh chấp hành mà chưa cần fusion của B: dùng `mosquitto_pub -t ds/esp32/cmd -m '{"cmd":"lock","sig":"1"}'` trên laptop (hoặc script `scripts/test_pub.py` bên B) — servo phải quay.
- `mosquitto_sub -v -t 'ds/#'` trên laptop: thấy telemetry/event của mình chảy = phần comm của bạn đã xong, bất kể dashboard B có đẹp chưa.
- Chiều B dùng "node giả" riêng (`sim_sensors.py`) — **không đụng** firmware của bạn. Bạn rảnh tay hàn, hiệu chuẩn, không bị giục "dashboard chưa có số!".

## L3. Hợp đồng hành vi — bạn PHẢI giữ, vì B (và ban giám khảo) tin vào đó

| Tình huống | Trách nhiệm của A |
|---|---|
| Nhận `{"cmd":"lock"}` | Quay servo 0→90° ngay ≤200ms, trạng thái LOCKED vào event; `unlock` **chỉ** khi `sig` khớp + log. A không tự phán "có nên lock không" — điều kiện nằm ở fusion B |
| Cồn cao do FSM cục bộ (mất B) | Tự WARN→ALARM, còi+LED+OLED; **bất chấp mọi cmd từ host** trong lúc đang ALARM/LOCKED cục bộ |
| Nhận risk từ host | Đủ để đổi band hiển thị (OLED/vàng/đỏ) nhưng **không bao giờ khóa servo** vì risk fatigue — logic "cồn mới khóa" do B quyết |
| Mất MQTT >5s | Vào degraded mode, telemetry gắn `"degraded":true`, tự reconnect; cấp mạng → dashboard B tự hồi phục (A không cần làm gì thêm) |
| Còi đang kêu khi demo | Có nút vật lý/`{"cmd":"reset"}` để tắt sau ACK — lịch sự với giảng viên 😄 |

## L4. Ba lần HẸP GẶP giữa 2 người (giống hệ chiếu bên file B)

1. **Thứ 6 tuần 1 — chốt schema:** 2 máy cùng mạng, ping-pong message theo §5 (A thấy `hello`, B thấy JSON hợp lệ). Sau đó mỗi người tự chạy 2 tuần.
2. **Cuối tuần 2 — RÁP E2E (nửa ngày, mốc quyết định dự án):** thổi cồn THẬT: A đo → MQTT → B fusion → cmd → A servo+buzzer → B dashboard đỏ. Bắt đầu đo 7 tiêu chí nghiệm thu §12 (đặc biệt #5 edge autonomy, #6 khóa động cơ).
3. **Tuần 4 — demo thử + đo latency liên thông:** bấm giờ "thổi → kêu"; chạy kịch bản §13 ×3; chốt firmware final (không đổi sau mốc này).

Ngoài 3 lần đó: cùng **1 repo git** (branch `firmware/*`), commit/push hằng ngày; file chung duy nhất là `mqtt/` + schema — đổi là phải 2 bên đồng ý.

## L5. Ranh giới trách nhiệm — phần của A KHÔNG làm hộ B

- ❌ Không viết luật fusion (weight, Mamdani) vào firmware — FSM cục bộ chỉ là nhánh cảm biến đơn giản để tự chủ.
- ❌ Không "dịch" code Python sang C để tự chạy AI (vision thuộc B).
- ✅ Chỉ đảm bảo: **số đo đáng tin + chấp hành lệnh nhanh + sống sót khi mất kết nối** — đúng 3 điều spec §7 & §12 đòi hỏi.

---

## Checklist nhanh "hôm nay bắt đầu thế nào"

1. Đặt hàng BOM §4 ngay (ship 1–2 ngày là rủi ro R5 của cả team).
2. Trong lúc chờ hàng: cài PlatformIO (hoặc Arduino IDE + core ESP32), kéo repo, làm A0.2–A0.4 bằng LED onboard — board chưa có cảm biến vẫn test được WiFi/MQTT.
3. Cảm biến về: A1.1 hiệu chuẩn ADC với multimeter trước tiên (mọi thứ sau phụ thuộc nó).
4. Gửi bạn B: "IP ESP32 của tôi là …, broker của bạn chạy chưa, cho tôi xin topic test" — lần hẹp gặp 1 sẵn sàng bất cứ lúc nào tuần 1.
