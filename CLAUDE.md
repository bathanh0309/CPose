> **ĐỌC FILE NÀY TRƯỚC KHI SINH BẤT KỲ CODE NÀO.**
> Đây là tài liệu ràng buộc bắt buộc cho Codex / Claude / Copilot / GPT khi làm việc với dự án **CPose**.
> Khi có mâu thuẫn giữa các tài liệu cũ, **ưu tiên file này**.
> Không tự ý đổi kiến trúc lớn, không thêm dependency ngoài danh sách cho phép, không đổi mục tiêu từ demo tuần tự sang hệ thống production phức tạp nếu người dùng chưa yêu cầu.

---

# 1. Mục tiêu hiện tại của dự án

## 1.1 Bối cảnh

**CPose** là đồ án tốt nghiệp kỹ sư Điện tử Viễn thông & Kỹ thuật Máy tính chất lượng cao, Đại học Bách Khoa — Đại học Đà Nẵng.

Mục tiêu hiện tại là làm **app demo chạy được, ổn định, dễ giải thích**, phục vụ:
- thu thập clip test từ RTSP
- chạy offline auto-label người
- chạy pose + ADL
- demo xử lý không gian–thời gian tuần tự trên clip test thủ công nhiều camera
- demo trước với **3 người**, sau đó mới mở rộng

## 1.2 Phạm vi chính thức của phiên bản hiện tại

Phiên bản Codex phải bám theo **2 luồng rõ ràng**:

### Luồng A — CPose gốc 3 phase
- **Phase 1**: RTSP → YOLOv8n detect người → ghi short MP4
- **Phase 2**: offline YOLOv11n → PNG + `labels.txt`
- **Phase 3**: offline YOLO11n-pose → `keypoints.txt` + `adl.txt` + overlay PNG

### Luồng B — Demo không gian–thời gian tuần tự trên clip test thủ công
- input từ `data/multicam/`
- video tên dạng `camxx-ss-mm-hh-dd-MM-yyyy.mp4`
- **xử lý theo thời gian trước, theo camera sau**
- pose + ADL + ReID phải chạy **tuần tự**, không parallel
- hiển thị **4 bóng đèn camera từ trái sang phải**
- cố gắng giữ **ít Global ID nhất có thể**

## 1.3 Điều không được hiểu sai

- `data/multicam/*.mp4` là **test/demo only**, không phải train/val.
- Các dataset như NTU, ETRI, Charades, Toyota Smarthome chỉ dùng cho **research / train / val / benchmark**, không phải dữ liệu runtime.
- Mục tiêu hiện tại là **demo thực dụng**, không phải triển khai production cloud hay hệ thống enterprise.

---

# 2. Quy tắc ưu tiên khi Codex ra quyết định

Khi có mâu thuẫn, thứ tự ưu tiên là:

1. **Yêu cầu mới nhất của người dùng trong chat**
2. **File `CLAUDE.md` này**
3. Tinh thần của các tài liệu mới:
   - `14-04-spatio-temporal.md`
   - `14-04-fixbug-pipeline.md`
   - `13-04-research-ADL.md`
   - `Pipeline(1).md`
4. Tài liệu CLAUDE cũ
5. Suy luận của Codex

Codex **không được tự đoán** khi đã có ràng buộc trong file này.

---

# 3. Kiến trúc mục tiêu

## 3.1 Tư tưởng chung

Giữ kiến trúc **đơn giản, chạy được, dễ sửa**, ưu tiên:
- code rõ ràng
- pipeline tuần tự
- không tạo thêm complexity không cần thiết
- không thêm stack mới nếu chưa thật cần

## 3.2 Chế độ hoạt động cần hỗ trợ

### Mode 1 — RTSP Recording + Offline CPose
Luồng chuẩn:

`resources.txt -> Phase1 raw_videos -> Phase2 output_labels -> Phase3 output_pose`

### Mode 2 — Sequential Multicam Demo
Luồng chuẩn:

`data/multicam/*.mp4 -> parse timestamp -> global sort by time -> sequential processing -> output_pose`

Mode 2 là trọng tâm mới cho demo hiện tại.

## 3.3 Không được làm gì lúc này

- Không chuyển toàn bộ dự án sang microservices
- Không thêm Docker / Redis / Celery
- Không chuyển sang React / Vue / Next.js
- Không tự động mở rộng thành distributed multi-camera production system
- Không ép mọi thứ chạy realtime song song nếu người dùng đang ưu tiên demo tuần tự

---

# 4. Cấu trúc thư mục — nguyên tắc thực tế

## 4.1 Nguyên tắc cứng

- **Tôn trọng cấu trúc repo hiện có**.
- Không đổi tên thư mục gốc.
- Không di chuyển hàng loạt file nếu chưa được yêu cầu.
- Không tạo thêm cấu trúc framework rườm rà kiểu `src/`, `controllers/`, `modules/`, `feature_xxx/`.
- Nếu cần thêm file mới, phải đặt ở nơi hợp lý theo logic hiện có.

## 4.2 Những thư mục dữ liệu phải tồn tại

- `data/config/`
- `data/raw_videos/`
- `data/output_labels/`
- `data/output_pose/`
- `data/multicam/`

## 4.3 Ý nghĩa

- `data/raw_videos/`: clip sinh từ Phase 1 RTSP recorder
- `data/output_labels/`: kết quả Phase 2
- `data/output_pose/`: kết quả Phase 3 và kết quả demo tuần tự multicam
- `data/multicam/`: nơi người dùng copy video test thủ công

---

# 5. Dependency policy

## 5.1 Ưu tiên giữ stack nhẹ

Cho phép ưu tiên các thư viện sau nếu repo đã hoặc sẽ dùng hợp lý:
- `flask`
- `flask-socketio`
- `flask-cors`
- `eventlet`
- `ultralytics`
- `opencv-python`
- `Pillow`
- `numpy`
- `PyYAML`
- `psutil`
- `python-dotenv`

## 5.2 Không được tự ý thêm

- FastAPI
- Django
- React / Vue / Angular
- Tailwind / Bootstrap / jQuery
- Redis / Celery
- Docker bắt buộc
- framework pose khác nếu chưa được yêu cầu

## 5.3 Quy tắc bổ sung dependency

Chỉ được thêm dependency mới khi:
1. người dùng yêu cầu rõ, hoặc
2. không có cách hợp lý nào khác để hoàn thành chức năng cốt lõi

Nếu thêm mới, phải giải thích ngắn trong comment hoặc README.

---

# 6. Chính sách dữ liệu train/val và test

## 6.1 Train / Val

Dùng tài liệu research ADL làm nền tham khảo:
- ST-GCN, MS-G3D, CTR-GCN, BlockGCN
- SkateFormer
- RTMPose, RTMO
- TSM
- NTU RGB+D 60/120
- ETRI Activity3D
- Toyota Smarthome
- Charades

Train/val là **ngoài runtime app**.

## 6.2 Test / Demo runtime

Các file trong:
- `data/multicam/`
- `data/raw_videos/`

chỉ là:
- input test
- input demo
- input đánh giá hành vi / Pose / ReID

Không dùng chúng như train set nội bộ runtime.

---

# 7. Quy ước tên file

## 7.1 Phase 1 clip từ RTSP

Format ưu tiên:

```text
YYYYMMDD_HHMMSS_camXX.mp4
```

Ví dụ:

```text
20240315_143022_cam01.mp4
```

## 7.2 Video test thủ công trong `data/multicam/`

Format bắt buộc cho demo tuần tự:

```text
camxx-ss-mm-hh-dd-MM-yyyy.mp4
```

Ví dụ:

```text
cam01-25-26-16-29-01-2026.mp4
cam02-40-26-16-29-01-2026.mp4
cam03-15-58-15-28-01-2026.mp4
cam04-27-46-16-29-01-2026.mp4
```

Codex phải parse filename này đúng.

---

# 8. Quy tắc sort và xử lý clip demo multicam

## 8.1 Tuyệt đối không sort theo string filename đơn thuần

Phải parse thành:
- year
- month
- day
- hour
- minute
- second
- cam_index

## 8.2 Sort key chính thức

```python
sort_key = (year, month, day, hour, minute, second, cam_index)
```

## 8.3 Nguyên tắc vàng

**Ưu tiên thời gian trước, camera sau.**

Nếu 2 clip cùng timestamp thì mới dùng `cam01 < cam02 < cam03 < cam04`.

---

# 9. Thuật toán demo không gian–thời gian tuần tự

## 9.1 Tên thuật toán nội bộ

`TFCS-PAR` = `Time-First Cross-Camera Sequential Pose–ADL–ReID`

## 9.2 Flow bắt buộc

```text
manual test clips
-> parse filename timestamp
-> global sort by (time, cam)
-> sequential scheduler
-> open one clip at a time
-> local person tracking
-> pose inference
-> ADL inference
-> cross-camera ReID update
-> save outputs
-> move to next clip
```

## 9.3 Không được chạy parallel trong mode demo này

- Không multi-thread per clip
- Không process nhiều clip cùng lúc
- Không chạy pose và ADL theo hai pipeline song song khác nhau

**Demo tuần tự phải đúng trước, nhanh sau.**

---

# 10. Topology camera và logic không gian

## 10.1 Topology chuẩn cho demo

```text
cam01 -> cam02 -> cam03 -> elevator -> cam04
cam04 -> room_upstairs -> cam04
cam03 -> quay lại / xuống lại -> cam03 hoặc cam02
```

## 10.2 Ý nghĩa

- `cam01`: đầu tuyến
- `cam02`: trung gian tầng dưới
- `cam03`: gần thang máy / điểm chuyển tầng
- `cam04`: tầng trên
- `elevator`: vùng mù giữa cam03 và cam04
- `room_upstairs`: phòng kín ở cam04

## 10.3 Chính sách ID

Không tạo ID mới quá sớm.

Khi người:
- biến mất ở `cam03` gần elevator
- xuất hiện sau đó ở `cam04` trong time window hợp lý

thì **ưu tiên giữ nguyên Global ID**.

---

# 11. Trạng thái hệ thống cần có cho demo tuần tự

Codex nên ưu tiên thiết kế các state sau:

## 11.1 `ClipQueue`
Danh sách clip đã parse và sort.

## 11.2 `GlobalPersonTable`
Lưu Global ID xuyên camera.

Trạng thái gợi ý:
- `ACTIVE`
- `PENDING_TRANSFER`
- `IN_ROOM`
- `DORMANT`
- `CLOSED`

## 11.3 `PendingTransitionBuffer`
Dùng khi một người vừa biến mất ở vùng chuyển tiếp như elevator.

## 11.4 `RoomHoldBuffer`
Dùng khi người vào phòng kín ở cam04.

## 11.5 `LampState`
Dùng cho mapping 4 bóng đèn camera.

---

# 12. Mapping 4 bóng đèn camera

UI demo phải hỗ trợ hiển thị trực quan:

```text
Cam01 -> Cam02 -> Cam03 -> Cam04
```

Trạng thái màu:
- `IDLE`
- `ACTIVE`
- `DONE`
- `ALERT`

Gợi ý hiển thị:
- `⚪` IDLE
- `🟡` ACTIVE
- `🟢` DONE
- `🔴` ALERT

Khi clip camera nào đang xử lý thì bóng đèn đó sáng.

---

# 13. Global ID policy — càng ít ID càng tốt

## 13.1 Nguyên tắc chung

**Ưu tiên giữ ID cũ nếu còn hợp lý**, chỉ tạo ID mới khi thực sự cần.

## 13.2 Bằng chứng ghép ID

### Mạnh
- face match mạnh
- temporal continuity mạnh
- topology phù hợp và chỉ có 1 candidate rõ ràng

### Trung bình
- body appearance gần giống
- chiều cao tương đối gần giống
- pose / gait signature tương đồng
- ADL continuity hợp lý

### Bổ trợ
- biến mất ở cửa ra camera trước
- xuất hiện ở cửa vào camera sau
- thời gian di chuyển hợp lý

## 13.3 Khi vào phòng hoặc đi thang máy

- Không tạo ID mới ngay
- Đưa vào pending buffer / room buffer
- Chờ clip kế tiếp hợp lệ theo thời gian–topology

## 13.4 Khi thay áo màu khác

Nếu người vào `room_upstairs` rồi đi ra với áo khác màu:
- giảm trọng số body color
- tăng trọng số face, height, gait, temporal topology
- nếu chỉ có 1 candidate hợp lý trong room buffer, **ưu tiên giữ ID cũ**

## 13.5 Chỉ tạo ID mới khi

- không có candidate hợp lệ theo thời gian–không gian
- face / body / pose đều không khớp đủ tin cậy
- có xung đột nhiều người cùng thời điểm

---

# 14. Time window gợi ý cho chuyển camera

Giá trị demo ban đầu:

- `cam01 -> cam02`: `0–60s`
- `cam02 -> cam03`: `0–60s`
- `cam03 -> cam04` qua thang máy: `20–180s`
- `cam04 -> cam03` đi xuống: `20–180s`
- `cam04 -> room -> cam04`: `5–300s`
- `cam03 -> quay lại cam02`: `10–120s`

Các giá trị này phải để trong config hoặc constant dễ chỉnh.

---

# 15. Phase 1 — Quy tắc recorder

## 15.1 Logic

- đọc `resources.txt`
- tạo worker theo camera RTSP
- detect người bằng YOLOv8n
- event-based clip recording
- pre-buffer + post-buffer
- clip quá ngắn thì bỏ
- enforce storage limit

## 15.2 Tối ưu bắt buộc

**Preview chỉ bật khi cần xem**, không encode/emit liên tục nếu người dùng không xem.

Mục tiêu là tránh buffer và giảm tải.

---

# 16. Phase 2 — Quy tắc offline analysis

- input: thư mục clip mp4
- process lần lượt từng clip
- YOLOv11n detect người
- chỉ lưu PNG frame khi có người
- ghi `labels.txt` với tọa độ pixel tuyệt đối
- không thêm tracking phức tạp vào phase này nếu chưa cần

---

# 17. Phase 3 — Quy tắc pose và ADL

## 17.1 Model hiện tại

- ưu tiên `YOLO11n-pose`
- output 17 COCO keypoints

## 17.2 ADL hiện tại

- dùng **một** `rule_based_adl()` thống nhất
- không được để tồn tại 2 bộ classifier song song mâu thuẫn nhau

## 17.3 ADL classes chuẩn

```text
standing
sitting
walking
lying_down
falling
reaching
bending
unknown
```

## 17.4 Sliding window

- mặc định `WINDOW_SIZE = 30`
- đủ window mới classify
- mode demo tuần tự phải giữ logic nhất quán

---

# 18. Bug fix priority — Codex phải ưu tiên sửa đúng thứ tự

## P0 — phải sửa trước

### Bug A — Chỉ giữ 1 ADL classifier duy nhất
Không được để `pose_utils.py` và một classifier khác đưa ra kết quả mâu thuẫn.

### Bug B — `person_id` không ổn định trong Phase 3
Không dùng `enumerate` theo thứ tự detection để làm ID dài hạn.
Phải dùng **stable local track id** trong phạm vi clip.

### Bug C — Demo multicam phải sort theo thời gian thật
Không sort theo alphabet đơn thuần.

### Bug D — Mode demo multicam phải chạy tuần tự
Không parallel pose / ADL / ReID giữa nhiều clip.

## P1 — nên sửa tiếp theo

### Bug E — Preview / stream chỉ render khi có subscriber
Giảm buffering.

### Bug F — Quy tắc ít ID nhất
Không sinh ID mới quá sớm khi có pending transfer hoặc room hold hợp lệ.

### Bug G — Tách train/val và test rõ trong code / docs / UI

## P2 — để sau

- nâng cấp learned ADL model
- nâng cấp face/body ReID tốt hơn
- mở rộng >3 người
- realtime multi-camera production-grade

---

# 19. ReID policy cho bản demo hiện tại

## 19.1 Trạng thái implementation

ReID hiện tại có thể ở mức heuristic / simple appearance matching.
Codex **không được quảng cáo quá mức** như thể đã là SOTA.

## 19.2 Tư tưởng đúng

- bản demo hiện tại ưu tiên **temporal-topology consistency**
- ReID là lớp hỗ trợ cho việc giữ Global ID ít nhất có thể
- face là tín hiệu mạnh nhất nếu có
- body appearance là tín hiệu trung bình
- pose/gait/height là tín hiệu bổ trợ rất hữu ích khi đổi áo hoặc đi qua vùng mù

## 19.3 Nếu implementation chưa hoàn chỉnh

Codex phải nói rõ trong comment/doc:
- đây là heuristic demo
- chưa phải final research-grade ReID engine

---

# 20. File output và format

## 20.1 Phase 2

`labels.txt`

Format:

```text
frame_id x_min y_min x_max y_max
```

## 20.2 Phase 3

`keypoints.txt`

Format:

```text
frame_id person_id kp0_x kp0_y kp0_conf ... kp16_x kp16_y kp16_conf
```

`adl.txt`

Format:

```text
frame_id person_id adl_label confidence
```

## 20.3 Output folder

Kết quả pose/ADL luôn phải về:

```text
data/output_pose/<clip_stem>/
```

Kết quả demo multicam cũng phải tuân theo tinh thần này.

---

# 21. Frontend policy

## 21.1 Giữ UI đơn giản

- SPA đơn giản
- ưu tiên Vanilla JS
- không ép thêm framework nặng

## 21.2 Những gì frontend nên hiển thị cho demo mới

- danh sách clip multicam theo timeline
- camera nào đang xử lý
- 4 bóng đèn camera
- progress clip hiện tại
- ADL summary theo clip
- trạng thái Global ID / pending transfer / room hold nếu có

## 21.3 Không cần làm quá nhiều dashboard phức tạp

Mục tiêu là **demo rõ thuật toán**, không phải BI dashboard.

---

# 22. Coding style

## 22.1 Nguyên tắc

- code rõ ràng
- tách hàm vừa đủ
- tên biến dễ hiểu
- ưu tiên `pathlib.Path`
- tránh hard-code path string lộn xộn
- thêm comment ở nơi có logic thuật toán quan trọng

## 22.2 Không được

- tạo code thừa để “trông chuyên nghiệp” nhưng không cần thiết
- viết logic quá trừu tượng khó debug
- âm thầm đổi API cũ khi chưa cập nhật frontend

---

# 23. Cách Codex phải làm việc khi được giao task

Trước khi sửa code, Codex phải tự kiểm tra:

1. Task này thuộc mode nào?
   - Phase 1/2/3 gốc?
   - Sequential multicam demo?
2. Có ảnh hưởng tới file naming / output format không?
3. Có làm sai nguyên tắc time-first không?
4. Có vô tình sinh quá nhiều Global ID không?
5. Có làm pose/ADL chạy song song trái yêu cầu không?
6. Có thêm dependency trái policy không?
7. Có phá cấu trúc repo không?

Nếu có, phải dừng và chỉnh lại thiết kế.

---

# 24. Acceptance checklist trước khi kết thúc một task

Codex chỉ được coi task là hoàn tất nếu thỏa:

- [ ] Không phá Phase 1/2/3 hiện có
- [ ] Không trộn train/val vào runtime test
- [ ] `data/multicam` được xử lý theo **thời gian trước, camera sau**
- [ ] Pose + ADL chạy **tuần tự**
- [ ] Có thể hiển thị mapping 4 bóng đèn camera
- [ ] Kết quả lưu về `data/output_pose`
- [ ] Logic ID ưu tiên **ít ID nhất có thể**
- [ ] Có xử lý hợp lý cho `cam03 -> elevator -> cam04`
- [ ] Có xử lý hợp lý cho `cam04 -> room -> cam04`
- [ ] Không tự ý thêm framework / dependency lớn
- [ ] Có comment / doc ngắn giải thích phần logic mới

---

# 25. Non-goals của phiên bản hiện tại

Những thứ **không phải mục tiêu bắt buộc ngay bây giờ**:
- đạt SOTA ADL benchmark trong app demo
- production-grade person ReID
- multi-user web platform
- cloud-native architecture
- realtime fully parallel orchestration
- nhận diện hàng chục người đồng thời xuyên nhiều camera phức tạp

---

# 26. Kết luận cho Codex

Nếu phải chọn giữa:
- giải pháp hoành tráng nhưng khó chạy
- giải pháp đơn giản nhưng đúng yêu cầu demo

=> luôn chọn **giải pháp đơn giản, tuần tự, dễ demo, đúng file này**.

**Triết lý chính thức của phiên bản hiện tại:**

> `Train/Val ở ngoài runtime.`
> `Runtime test xử lý theo thời gian trước.`
> `Pose và ADL xử lý tuần tự.`
> `Global ID phải càng ít càng tốt.`
> `UI phải trực quan, đặc biệt với 4 camera và timeline.`
