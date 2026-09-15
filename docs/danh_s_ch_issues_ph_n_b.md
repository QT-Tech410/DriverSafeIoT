# DANH SÁCH ISSUES / TASKS — PHẦN B (PYTHON / AI / FUSION / DASHBOARD)

> **Dự án:** DriverSafe-IoT — Giám sát an toàn tài xế  
> **Người phụ trách:** B (Python/AI/Backend)  
> **Tham chiếu spec:** `docs/superpowers/specs/2026-09-15-driver-safety-iot-design.md` & `CONG-VIEC-B-PYTHON-AI.md`

---

## [INFRA-01] Project Bootstrap & Environment Setup
Labels: `INFRA` | `enhancement` | `MVP` | `High`  
Type: Task  
Sprint: Sprint 1  
Branch: `feat/project-bootstrap`  
Depends: --  

### Mô tả:
Khởi tạo cấu hình môi trường phát triển ban đầu cho Phần B (Python), bao gồm file `requirements.txt`, thiết lập MQTT Mosquitto broker local, tạo cấu trúc thư mục repository chuẩn theo spec và viết script khởi động nhanh cho các dịch vụ.

### Acceptance Criteria:
- [ ] Cấu trúc thư mục: `host/vision/`, `host/fusion/`, `host/dashboard/`, `mqtt/`, `experiments/`, `scripts/`, `requirements.txt`
- [ ] `requirements.txt` chứa đủ các thư viện: `mediapipe`, `opencv-python`, `paho-mqtt`, `numpy`, `scipy`, `pyyaml`, `fastapi`, `uvicorn`
- [ ] Chạy `mosquitto -c mqtt/mosquitto.conf` khởi động broker thành công ở port 1883 (`allow_anonymous true`)
- [ ] Môi trường virtualenv cài đặt hoàn tất không phát sinh lỗi dependency

---

## [VIS-01] Webcam Capture & Basic Overlay Page
Labels: `VISION` | `feature` | `MVP` | `High`  
Type: Task  
Sprint: Sprint 1  
Branch: `feat/vision-capture`  
Depends: `INFRA-01`  

### Mô tả:
Xây dựng module capture webcam thời gian thực với độ phân giải 640×480, đo và hiển thị chỉ số FPS trực tiếp lên frame overlay.

### Acceptance Criteria:
- [ ] File: `host/vision/capture.py`
- [ ] Khởi tạo webcam capture ở độ phân giải $640 \times 480$, duy trì $\ge 20$ FPS ổn định liên tục trong 5 phút
- [ ] Hiển thị thông số FPS lên góc màn hình video stream
- [ ] Hỗ trợ tham số dòng lệnh `--src=mjpeg` hoặc URL stream (dự phòng cho ESP32-S3-CAM)

---

## [VIS-02] MediaPipe Face Mesh & Geometric Landmarks Extraction
Labels: `VISION` | `feature` | `MVP` | `High`  
Type: Task  
Sprint: Sprint 1  
Branch: `feat/face-mesh-landmarks`  
Depends: `VIS-01`  

### Mô tả:
Tích hợp MediaPipe Face Mesh (468 điểm landmark 3D) để bắt cấu trúc khuôn mặt tài xế và vẽ đường viền landmark overlay lên frame.

### Acceptance Criteria:
- [ ] File: `host/vision/face_metrics.py`
- [ ] Tải mô hình MediaPipe Face Mesh với cấu hình `max_num_faces=1`, `refine_landmarks=False`
- [ ] Bắt chính xác 468 điểm landmark, duy trì tracking mượt mà khi quay mặt góc $\pm 20^\circ$
- [ ] Khả năng vẽ overlay khung mặt và các điểm mắt/miệng trực tiếp trên hình ảnh

---

## [VIS-03] Core Facial Metrics Computation (EAR, MAR, Pitch)
Labels: `VISION` | `algorithm` | `MVP` | `High`  
Type: Task  
Sprint: Sprint 1  
Branch: `feat/facial-metrics`  
Depends: `VIS-02`  

### Mô tả:
Cài đặt thuật toán tính toán các chỉ số hình học EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio) đã chuẩn hóa theo khoảng cách 2 mắt, và ước lượng pitch nghiêng đầu bằng `cv2.solvePnP`.

### Acceptance Criteria:
- [ ] File: `host/vision/face_metrics.py`
- [ ] $EAR = \frac{|p_2 - p_6| + |p_3 - p_5|}{2 \cdot |p_1 - p_4|}$, chuẩn hóa $EAR_{norm} = \frac{EAR}{|p_{left\_eye} - p_{right\_eye}|}$
- [ ] $MAR = \frac{|p_{lips\_ver}|}{|p_{lips\_hor}|}$ (chuẩn hóa tương tự)
- [ ] Giải bài toán `cv2.solvePnP` với 5 điểm mặt 3D chuẩn để suy ra góc Euler Pitch (độ)
- [ ] Log chỉ số out ra console: nhắm mắt làm $EAR_{norm}$ giảm rõ rệt, ngáp làm $MAR$ vọt cao, gật đầu làm pitch đổi $> 15^\circ$

---

## [VIS-04] Temporal Metrics Processing & Event Detection
Labels: `VISION` | `algorithm` | `MVP` | `High`  
Type: Task  
Sprint: Sprint 1 → 2  
Branch: `feat/temporal-metrics`  
Depends: `VIS-03`  

### Mô tả:
Xây dựng lớp xử lý thời gian (circular buffer 60s mẫu 10Hz) để tính toán các chỉ số PERCLOS(P78), CLES duration, Microsleep, Yawn/min, Head-drop và phát hiện sự kiện Mất mặt (Face lost).

### Acceptance Criteria:
- [ ] File: `host/vision/temporal.py`
- [ ] Lọc nhiễu trạng thái mắt khép: yêu cầu $\ge 3$ frame liên tiếp ($\approx 100\text{ms}$)
- [ ] Bắt chính xác sự kiện: Eye-closure ($\ge 250\text{ms}$), Microsleep ($\ge 500\text{ms}$), Head-drop (pitch vượt neutral $+ 15^\circ$ duy trì $\ge 0.8\text{s}$), Ngáp (MAR vọt duy trì $\ge 400\text{ms}$)
- [ ] Tính PERCLOS(P78) chuẩn xác trên cửa sổ trượt 60 giây
- [ ] Phát hiện sự kiện `face_lost` khi không tìm thấy mặt $\ge 2\text{s}$

---

## [VIS-05] Vision Metrics MQTT Publisher
Labels: `VISION` | `MQTT` | `MVP` | `High`  
Type: Task  
Sprint: Sprint 2  
Branch: `feat/vision-mqtt-pub`  
Depends: `VIS-04`  

### Mô tả:
Tổng hợp các chỉ số Vision ở tầng thời gian và publish dữ liệu định kỳ 1Hz lên topic MQTT `ds/vision/metrics` đúng JSON schema đã thống nhất với Phần A.

### Acceptance Criteria:
- [ ] File: `host/vision/publisher.py`
- [ ] Kế thừa Paho MQTT Client, kết nối tới Mosquitto Broker local
- [ ] Publish gói tin 1Hz lên `ds/vision/metrics` với QoS 0
- [ ] Payload JSON chứa đủ các trường: `ts`, `face`, `ear`, `perclos_60s`, `cles_dur_ms`, `mar`, `yawn_per_min`, `head_pitch_deg`, `head_drop`, `lux_mode`

---

## [TEST-01] Simulated Sensors & MQTT Test Utility
Labels: `TOOLS` | `testing` | `MVP` | `High`  
Type: Task  
Sprint: Sprint 1  
Branch: `feat/sim-sensors`  
Depends: `INFRA-01`  

### Mô tả:
Viết script giả lập các thông số từ ESP32-S3 (Phần A) để giúp Phần B test độc lập logic Fusion & Dashboard trước khi có phần cứng thật.

### Acceptance Criteria:
- [ ] File: `scripts/sim_sensors.py`
- [ ] Continuously publish `ds/esp32/sensors` với tần số 1Hz
- [ ] Cho phép bấm phím điều khiển nhanh: `1/2/3` (đổi mức cồn 0/1/2), `t` (thay đổi nhiệt độ NTC), `l` (thay đổi ánh sáng LDR), `d` (bật/tắt cờ `degraded_mode`)

---

## [FUS-01] Telemetry Ingestion & Data Caching
Labels: `FUSION` | `backend` | `MVP` | `High`  
Type: Task  
Sprint: Sprint 2  
Branch: `feat/fusion-ingestion`  
Depends: `INFRA-01`, `TEST-01`  

### Mô tả:
Xây dựng module Fusion Subscriber lắng nghe đồng thời 3 luồng MQTT (`ds/esp32/sensors`, `ds/esp32/events`, `ds/vision/metrics`), quản lý cache bộ nhớ đệm kèm timestamp cho từng nguồn.

### Acceptance Criteria:
- [ ] File: `host/fusion/fusion.py`
- [ ] Lắng nghe ổn định các topic, tự khôi phục kết nối khi mất kết nối MQTT
- [ ] Cache các thông số telemetry mới nhất từ cả Vision và ESP32
- [ ] Xử lý an toàn (`try/except KeyError`): không bị crash khi gói tin JSON bị lỗi hoặc thiếu trường

---

## [FUS-02] Rule-Weighted Temporary Fusion Engine
Labels: `FUSION` | `algorithm` | `MVP` | `High`  
Type: Task  
Sprint: Sprint 2  
Branch: `feat/fusion-rule-weighted`  
Depends: `FUS-01`  

### Mô tả:
Cài đặt bộ suy luận Fusion dạng Rule-Weighted đơn giản (chưa dùng Fuzzy) để tính Risk Score 0–100 phục vụ việc test End-to-End hệ thống ở cuối Tuần 2.

### Acceptance Criteria:
- [ ] File: `host/fusion/policy.py`
- [ ] Công thức Risk tạm: $Risk = w_1 \cdot PERCLOS + w_2 \cdot blink\_events + w_3 \cdot ALCO + w_4 \cdot CABIN$ (clamp giá trị từ 0–100)
- [ ] Phân dải Risk thành 4 band chính xác: SAFE ($<25$), WARN ($25\text{--}50$), ALARM ($50\text{--}75$), CRITICAL ($>75$)
- [ ] Publish kết quả lên `ds/fusion/level` (QoS 1, retain) và phát lệnh `ds/esp32/cmd` tương ứng

---

## [FUS-03] Mamdani Fuzzy Inference Engine Integration
Labels: `FUSION` | `algorithm` | `High`  
Type: Task  
Sprint: Sprint 3  
Branch: `feat/fusion-mamdani`  
Depends: `FUS-02`  

### Mô tả:
Phát triển động cơ suy luận mờ Mamdani hoàn chỉnh (6 inputs, 12 luật mờ, giải mờ Centroid), hỗ trợ load cấu hình luật và tập thuộc tính linh hoạt từ file YAML.

### Acceptance Criteria:
- [ ] File: `host/fusion/fuzzy.py`, `host/fusion/rules.yaml`
- [ ] Đọc cấu hình Tập thuộc (Membership functions) & Ngân hàng luật từ `rules.yaml`
- [ ] Thực hiện giải mờ Mamdani min-implication và defuzzification bằng phương pháp trọng tâm (Centroid)
- [ ] Đạt unit test fixture: dữ liệu đầu vào cho trước trả về điểm Risk chính xác và trả về mảng các luật đã kích hoạt (`drivers`)

---

## [FUS-04] Action Policy & Safety Enforcement Logic
Labels: `FUSION` | `architecture` | `High`  
Type: Task  
Sprint: Sprint 3  
Branch: `feat/fusion-action-policy`  
Depends: `FUS-03`  

### Mô tả:
Cài đặt chính sách hành động phân tầng: phân biệt nguyên nhân rủi ro do Mệt mỏi (Fatigue) vs Cồn (Alcohol), quản lý thời gian cooldown giữa các cảnh báo và cơ chế xử lý dự phòng `vision_offline`.

### Acceptance Criteria:
- [ ] File: `host/fusion/policy.py`
- [ ] Quy tắc khóa động cơ: Chỉ gửi lệnh `{"cmd":"lock"}` khi `ALCO=2` được xác nhận $\ge 2/3$ lần đo liên tiếp.
- [ ] Khi rủi ro do Fatigue (dù ở mức CRITICAL): Tuyệt đối KHÔNG gửi lệnh lock động cơ, chỉ gửi lệnh beep & cảnh báo OLED
- [ ] Khi mất tín hiệu Vision quá 2s: Tự động chuyển sang nhánh suy luận thuần cảm biến và bật cờ `vision_offline`

---

## [DASH-01] FastAPI Server & SQLite Database Setup
Labels: `DASHBOARD` | `backend` | `MVP` | `High`  
Type: Task  
Sprint: Sprint 2 → 3  
Branch: `feat/dashboard-backend`  
Depends: `FUS-01`  

### Mô tả:
Khởi tạo web server FastAPI, tích hợp SQLite database để lưu trữ lịch sử telemetry 1Hz và các sự kiện rủi ro, cùng WebSocket bridge để stream dữ liệu xuống giao diện web.

### Acceptance Criteria:
- [ ] File: `host/dashboard/server.py`, `host/dashboard/db.py`
- [ ] Khởi tạo SQLite DB lưu bảng `telemetry` và bảng `events`
- [ ] WebSocket server gửi dữ liệu realtime cho frontend với độ trễ $\le 500\text{ms}$
- [ ] Server khởi động lại không làm mất dữ liệu SQLite cũ

---

## [DASH-02] Realtime Dashboard Interface & Gauge Widgets
Labels: `DASHBOARD` | `frontend` | `MVP` | `High`  
Type: Task  
Sprint: Sprint 2 → 3  
Branch: `feat/dashboard-realtime-ui`  
Depends: `DASH-01`  

### Mô tả:
Thiết kế giao diện Realtime Panel dạng 1 trang (sử dụng HTML, Chart.js, Tailwind CDN) hiển thị Risk Gauge, thông số Vision, Cabin, trạng thái kết nối node và video overlay stream.

### Acceptance Criteria:
- [ ] File: `host/dashboard/static/index.html`
- [ ] Hiển thị đồng hồ Risk gauge đổi màu linh hoạt theo 4 band (Xanh/Vàng/Cam/Đỏ)
- [ ] Hiển thị các chỉ số chi tiết: EAR, PERCLOS, Yawn, BrAC, Temp, Lux, RSSI
- [ ] Tích hợp khung xem video feed có vẽ overlay đường viền landmark
- [ ] Trạng thái Node (Online / Degraded / Offline) được cập nhật chính xác theo thời gian thực

---

## [DASH-03] Event Timeline, Alerts & Control Actions
Labels: `DASHBOARD` | `frontend` | `High`  
Type: Task  
Sprint: Sprint 3  
Branch: `feat/dashboard-alerts-controls`  
Depends: `DASH-02`  

### Mô tả:
Xây dựng giao diện danh sách sự kiện rủi ro, banner cảnh báo toàn màn hình khi chạm ngưỡng ALARM+, và các nút điều khiển hệ thống (Enroll, Unlock, Reset).

### Acceptance Criteria:
- [ ] File: `host/dashboard/static/index.html`, `host/dashboard/server.py`
- [ ] Banner cảnh báo đỏ phủ toàn màn hình tự động bật khi Risk chạm mức ALARM trở lên
- [ ] Bấm nút "Unlock Engine" trên UI $\rightarrow$ Server gửi payload `{"cmd":"unlock"}` qua MQTT topic `ds/esp32/cmd`
- [ ] Bấm nút "Enroll Driver" kích hoạt quy trình hiệu chuẩn cá nhân 60 giây

---

## [DASH-04] History Analytics & CSV Export
Labels: `DASHBOARD` | `feature` | `Medium`  
Type: Task  
Sprint: Sprint 3  
Branch: `feat/dashboard-history-export`  
Depends: `DASH-03`  

### Mô tả:
Tạo biểu đồ lịch sử theo dõi biến thiên chỉ số theo thời gian (Chart.js) và API xuất dữ liệu báo cáo ra file CSV.

### Acceptance Criteria:
- [ ] File: `host/dashboard/static/index.html`, `host/dashboard/server.py`
- [ ] Biểu đồ Chart.js hiển thị biến thiên PERCLOS và Risk Score trong 5 phút gần nhất
- [ ] Endpoint GET `/api/history/export` hỗ trợ tải xuống dữ liệu lịch sử ca lái dưới dạng `.csv`

---

## [EXP-01] Driver Personalization & Calibration Workflow
Labels: `ALGORITHM` | `feature` | `High`  
Type: Task  
Sprint: Sprint 3 → 4  
Branch: `feat/driver-calibration`  
Depends: `VIS-04`  

### Mô tả:
Phát triển module hiệu chuẩn cá nhân hóa 60s để thu thập phân bố EAR/MAR của từng tài xế, tự động tính ngưỡng $T_{closed} = \mu - k\cdot\sigma$ và thích ứng góc nghiêng đầu neutral.

### Acceptance Criteria:
- [ ] File: `host/vision/calibrate.py`
- [ ] Thu thập dữ liệu trạng thái tỉnh táo 60s $\rightarrow$ Lưu cấu hình cá nhân ra file `config/driver_profile.json`
- [ ] Áp dụng thích ứng theo `lux_mode`: khi tối (night mode) đổi hệ số $k = 2.0\sigma$
- [ ] Thực nghiệm kiểm chứng: 60s ngồi bình thường không phát sinh tình trạng mắt-khép giả

---

## [EXP-02] Experimental Evaluation Scripts & Report Data Generation
Labels: `RESEARCH` | `testing` | `Medium`  
Type: Task  
Sprint: Sprint 4  
Branch: `feat/experimental-benchmarks`  
Depends: `FUS-04`, `EXP-01`  

### Mô tả:
Viết các script đo đạc và sinh số liệu thực nghiệm phục vụ viết báo cáo: Confusion matrix mắt mở/khép, so sánh PERCLOS cố định vs cá nhân hóa, đo Latency End-to-End, và vẽ đồ thị Risk Surface 2D.

### Acceptance Criteria:
- [ ] Thư mục: `experiments/`
- [ ] Xuất bảng Confusion Matrix và độ chính xác của ngưỡng cá nhân hóa trên tập video thử nghiệm
- [ ] Đo đạc và ghi nhận bảng Latency End-to-End (từ lúc nhắm mắt trên camera $\rightarrow$ còi còi kêu $\le 1.2\text{s}$)
- [ ] Script Python (`matplotlib`) sinh đồ thị 3D/2D Risk Surface theo biến PERCLOS và ALCO cho báo cáo

---

## [SYS-01] One-Click Execution Script & System Integration
Labels: `INTEGRATION` | `enhancement` | `High`  
Type: Task  
Sprint: Sprint 4  
Branch: `feat/system-launch-scripts`  
Depends: `VIS-05`, `FUS-04`, `DASH-04`  

### Mô tả:
Tạo script khởi động toàn bộ các tiến trình của Phần B theo đúng thứ tự (Broker $\rightarrow$ Vision $\rightarrow$ Fusion $\rightarrow$ Dashboard) và chuẩn bị kịch bản demo 8 bước.

### Acceptance Criteria:
- [ ] File: `scripts/run_all.bat` (hoặc shell script)
- [ ] Khởi động mượt mà toàn bộ hệ thống bằng 1 thao tác duy nhất
- [ ] Tiến hành diễn tập thành công 8 bước kịch bản demo (trong §13 spec) ít nhất 3 lần liên tiếp không lỗi crash