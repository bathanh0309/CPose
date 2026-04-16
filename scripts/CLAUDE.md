> **ĐỌC FILE NÀY TRƯỚC KHI SINH BẤT KỲ CODE NÀO.**
> Đây là tài liệu ràng buộc bắt buộc cho Claude Code / Codex / Copilot khi làm việc với **CPose**.
> Khi có mâu thuẫn giữa các tài liệu cũ, **ưu tiên file này**.
> Không tự ý đổi kiến trúc lớn, không thêm dependency ngoài danh sách cho phép.

---

# CLAUDE.md — CPose Project Binding Reference

## 1. Bối Cảnh và Mục Tiêu
CPose là đồ án tốt nghiệp kỹ sư Điện tử Viễn thông & Kỹ thuật Máy tính, Đại học Bách Khoa — Đại học Đà Nẵng.

Mục tiêu hiện tại: Demo app chạy được, ổn định, dễ giải thích, phục vụ:

Thu thập clip test từ RTSP

Chạy offline auto-label người

Chạy pose + ADL

Demo xử lý không gian–thời gian tuần tự trên clip test thủ công nhiều camera

Demo trước với 3 người, sau mới mở rộng

QUY ĐỊNH CHO LLM (Claude Code / Copilot / Codex):

Mọi thay đổi code phải phục vụ trực tiếp cho các mục tiêu trên.

Không được tự ý thêm tính năng lớn (realtime nhiều user, microservice, v.v.) nếu user không yêu cầu.

### Tài liệu kiến trúc & phân tích

- `PIPELINE.md`: mô tả đầy đủ 3 phase (Phase1/2/3), ReID layer, tracking, FAISS, cùng danh sách bug & lỗ hổng kỹ thuật cần sửa.

**QUY ĐỊNH CHO LLM:**
- Khi sửa các file `recorder.py`, `analyzer.py`, `recognizer.py`, `reid.py`, `global_id.py`, `persistence.py`, phải đảm bảo không làm sai các thiết kế đã mô tả trong `PIPELINE.md`.
- Danh sách bug trong `PIPELINE.md` hỗ trợ giải thích, nhưng thứ tự ưu tiên sửa bug vẫn theo mục 17 của `CLAUDE.md`.

### Tài liệu thuật toán multicam TFCS-PAR

- `TFCS-PAR.md`: đặc tả chi tiết thuật toán **Time‑First Cross‑Camera Sequential Pose–ADL–ReID** cho demo 3 người từ `data/multicam/`.

Bao gồm:
- Định dạng tên file input, ClipQueue, GlobalPersonTable, PendingTransitionBuffer, RoomHoldBuffer, LampState.
- Topology cam01→cam04, elevator, room, cùng các test case 1–15 cho P1/P2/P3.
- Chính sách giảm ID mới, xử lý đổi áo, elevator, room.

**QUY ĐỊNH CHO LLM:**
- Khi chỉnh `recognizer.py` hoặc logic multicam, phải tuân theo TFCS‑PAR trong `TFCS-PAR.md`.
- Hành vi đúng được xác nhận bằng cách pass các test case 1–15 trong `TFCS-PAR.md` (không swap ID, không sinh thêm ID không cần thiết).

1. Mapping UI (static) ↔ API (app/api)
Trong static/index.html và static/js/app.js hiện có các vùng UI:

Workspace control: nút Start / Stop, chọn Multicam / Raw Video.

Pane “Đăng ký face/body”.

Video viewer: Original / Processed.

4 lamp Cam01–Cam04.

Video Queue + Pending Results (Preview / Show / Save).

Logs panel.

Ứng với đó, trong app/api/routes.py nên ràng buộc:

/api/workspace/start_multicam

Gọi service phase3_recognizer ở chế độ multicam TFCS-PAR.

/api/workspace/stop

Dừng scheduler multicam.

/api/pose/status

Trả về:

LampState (4 cam)

clip đang xử lý

tiến độ frame, fps

danh sách pending_results rút gọn.

/api/pose/pending_results

Dùng cho panel “VIDEO QUEUE / PROCESSED RESULTS”.

/api/pose/save_result

Gắn với nút Save trên card kết quả.

/ws/ui (flask-socketio)

Push event REGISTER_FACE_REQUEST, REGISTER_FACE_DONE, log mới, cập nhật LampState realtime.

Rule thêm vào CLAUDE.md:

Mọi API phục vụ UI phải nằm trong app/api/routes.py, không viết API rời rạc nơi khác.

Khi sửa static/js/app.js, nếu cần API mới, phải định nghĩa trong routes.py và đặt tên /api/... rõ ràng.

2. Mapping UI ↔ services (app/services)
phase1_recorder.py: phục vụ tab Raw Video / RTSP recorder (nếu sau này có).

phase2_analyzer.py: phục vụ luồng offline labeling (không dính đến multicam demo).

phase3_recognizer.py:

Chế độ single clip: chạy trên 1 video do user chọn (Raw Video mode).

Chế độ multicam TFCS-PAR: chạy theo data/multicam, sort time-first, theo đúng TFCS-PAR.md.

Rule:

Khi UI bật “Multicam” → routes gọi đúng mode multicam trong phase3_recognizer, không được dùng path tắt nào khác.

phase3_recognizer không được tự khởi động từ nơi khác ngoài routes.py (để dễ trace).

3. Mapping logic ADL / ReID ↔ core (app/core)
app/core/tracking.py: cung cấp tracker (YOLO track/DeepSORT) cho phase3_recognizer.

app/core/global_id.py: giữ Global ID xuyên camera, tuân TFCS-PAR & PIPELINE.

app/core/reid.py: nếu còn dùng, phải được gọi qua một lớp thống nhất (vd từ global_id.py) chứ không gọi trực tiếp từ UI.

app/core/adl.py: không dùng trong runtime (đã khóa ở CLAUDE.md).

Rule:

static/UI không bao giờ gọi trực tiếp vào app/core/*. Chỉ services và routes gọi core.

Khi chỉnh ReID/Global ID cho multicam demo, làm trong global_id.py + phase3_recognizer.py, không nhảy qua API riêng khác.

4. Mapping lưu trữ ↔ storage (app/storage)
app/storage/persistence.py: lưu GID, embedding, history.

app/storage/vector_db.py: FAISS index.

Bất kỳ thao tác “save final output clip” (khi user bấm Save) đều đi qua 1 hàm trong storage/services, không làm trực tiếp trong routes.py.

Rule:

Các API /api/pose/save_result chỉ gọi hàm “save_pending_result” đã định nghĩa (ở service/storage), không tự shutil.move trong routes.

UI không biết đường dẫn thật; mọi path/TLFS quản lý trong storage + services.

## 2. Hai Luồng Hoạt Động Chính
Luồng A — CPose gốc 3 phase
text
resources.txt
  → Phase 1 (YOLOv8n RTSP recorder) → data/raw_videos/
  → Phase 2 (YOLOv11n offline)      → data/output_labels/
  → Phase 3 (YOLO11n-pose)          → data/output_pose/
Luồng B — Sequential Multicam Demo (TRỌNG TÂM HIỆN TẠI)
text
data/multicam/*.mp4
  → parse timestamp từ filename
  → sort toàn bộ theo thời gian (time-first)
  → xử lý tuần tự từng clip (1 clip tại 1 thời điểm)
  → pose + ADL + ReID cập nhật cross-camera
  → lưu kết quả → data/output_pose/<clip_stem>/
QUY ĐỊNH CHO LLM:

Khi user nói về “multicam”, “spatio-temporal”, “demo 3 người”, bắt buộc dùng Luồng B.

Không được đổi flow Luồng B sang parallel processing, multi-thread hay multi-process. Mọi clip phải chạy tuần tự, theo danh sách đã sort.

Luồng A vẫn phải chạy bình thường; không phá code Phase 1/2/3 khi chỉnh cho multicam.

## 3. Quy Tắc Ưu Tiên Khi Có Mâu Thuẫn
Yêu cầu mới nhất của người dùng trong chat

File CLAUDE.md này

Tài liệu cũ khác

Suy luận của Claude Code

QUY ĐỊNH CHO LLM:

Nếu tài liệu cũ hoặc code hiện tại mâu thuẫn với CLAUDE.md, ưu tiên CLAUDE.md, trừ khi user nói rõ là muốn đổi luật.

Khi không chắc, hỏi lại user thay vì tự thiết kế kiến trúc mới.

## 4. Cấu Trúc Thư Mục — Nguyên Tắc Cứng
text
Capstone_Project/
├── app/
│   ├── api/routes.py           ← API layer chính
│   ├── core/                   ← reid.py, tracking.py, global_id.py, adl.py
│   ├── services/               ← phase1_recorder.py, phase2_analyzer.py, phase3_recognizer.py
│   ├── storage/                ← vector_db.py, persistence.py
│   └── utils/                  ← pose_utils.py, file_handler.py
├── configs/
│   └── config.yaml
├── data/
│   ├── config/
│   ├── multicam/               ← video test thủ công (READ ONLY — chỉ đọc)
│   ├── output_labels/          ← kết quả Phase 2
│   ├── output_pose/            ← kết quả Phase 3 và multicam demo
│   ├── output_process/         ← temporary processed results (chờ Save)
│   └── raw_videos/             ← clip từ Phase 1 RTSP
├── static/
│   ├── index.html
│   ├── js/app.js
│   └── css/style.css
├── .github/agents/
│   └── CPose-dev.agent.md
└── CLAUDE.md                   ← file này
Không được:

Đổi tên thư mục gốc Capstone_Project/

Tạo cấu trúc mới kiểu src/, controllers/, modules/, feature_xxx/

Di chuyển hàng loạt file khi chưa được yêu cầu rõ

QUY ĐỊNH CHO LLM:

Chỉ được tạo file mới bên trong các thư mục đã liệt kê, và chỉ khi user yêu cầu (ví dụ “tạo thêm file utils X”).

Không được “refactor kiến trúc” bằng cách gom file vào package mới hoặc đổi layout project.

Khi cần import, ưu tiên dùng đường dẫn tương thích với cấu trúc này (ví dụ from app.utils.pose_utils import ...).

## 5. Format Tên File Video Multicam — QUAN TRỌNG
✅ Format thực tế trong repo (dùng format này):
text
camXX_YYYY-MM-DD_HH-mm-ss.mp4
Ví dụ thực tế từ repo:

text
cam01_2026-01-29_16-26-25.mp4
cam02_2026-01-28_15-57-54.mp4
cam02_2026-01-29_16-26-40.mp4
cam03_2026-01-28-15-58-16.mp4
cam03_2026-01-29_16-26-49.mp4
cam04_2026-01-28-15-59-10.mp4
cam04_2026-01-29-16-27-33.mp4
cam04_2026-01-29-16-42-20.mp4
cam04_2026-01-29-16-44-29.mp4
cam04_2026-01-29-16-46-27.mp4
Parse logic đúng:
python
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ClipMeta:
    path: Path           # đường dẫn đầy đủ tới file
    cam_id: str          # "cam01", "cam02", ...
    cam_index: int       # 1, 2, 3, 4
    clip_dt: datetime    # thời gian bắt đầu clip


def parse_multicam_name(path: Path) -> ClipMeta:
    """
    Parse: camXX_YYYY-MM-DD_HH-mm-ss.mp4
    Example: cam01_2026-01-29_16-26-25.mp4
    """
    stem = path.stem              # "cam01_2026-01-29_16-26-25"
    parts = stem.split("_")
    # parts[0] = "cam01"
    # parts[1] = "2026-01-29"
    # parts[2] = "16-26-25"

    cam_id = parts[0]
    cam_index = int(cam_id.replace("cam", ""))

    date_str = parts[1]           # "2026-01-29"
    time_str = parts[2]           # "16-26-25" (dùng dấu -)

    hh, mm, ss = time_str.split("-")
    yyyy, mo, dd = date_str.split("-")

    clip_dt = datetime(int(yyyy), int(mo), int(dd),
                       int(hh), int(mm), int(ss))

    return ClipMeta(
        path=path,
        cam_id=cam_id,
        cam_index=cam_index,
        clip_dt=clip_dt,
    )
Sort key chính thức:
python
def sort_key(meta: ClipMeta):
    return (
        meta.clip_dt.year,
        meta.clip_dt.month,
        meta.clip_dt.day,
        meta.clip_dt.hour,
        meta.clip_dt.minute,
        meta.clip_dt.second,
        meta.cam_index,
    )


clips_sorted = sorted(all_metas, key=sort_key)
Nguyên tắc vàng: Ưu tiên thời gian trước, camera sau.

QUY ĐỊNH CHO LLM:

Khi xử lý data/multicam, bắt buộc:

Duyệt file bằng Path(...).glob("cam*.mp4") hoặc tương tự.

Parse từng file bằng parse_multicam_name.

Luôn sort bằng clips_sorted = sorted(all_metas, key=sort_key) trước khi loop.

Cấm sort theo tên string (alphabet) hoặc dựa vào os.listdir không sort.

Trong mọi vòng lặp xử lý clip multicam, chỉ duyệt theo clips_sorted đã nói ở trên.

## 6. Thuật Toán Demo Multicam — TFCS-PAR
TFCS-PAR = Time-First Cross-Camera Sequential Pose–ADL–ReID.

Flow bắt buộc cho mỗi clip
text
parse timestamp → global sort → mở clip_i
  → bật lamp(cam_i) = ACTIVE
  → local person tracking (stable track_id per clip)
  → pose inference (YOLO11n-pose, tuần tự)
  → ADL inference (sliding window, tuần tự)
  → cross-camera ReID update
  → ghi output_pose/<clip_stem>/
  → tắt lamp(cam_i) = DONE
  → chuyển clip tiếp theo
Giải thích ngắn cho runtime:

Lấy danh sách clips_sorted đã sort time-first bằng sort_key.

Với mỗi clip_i trong clips_sorted:

Cập nhật LampState[cam_i] = ACTIVE và gửi trạng thái lên UI.

Chạy local tracking trên clip_i để tạo stable_track_id trong phạm vi clip.

Với từng track, chạy pose (YOLO11n-pose) tuần tự, sau đó ADL (rule-based, sliding window) tuần tự.

Gọi module ReID để cập nhật Global ID theo logic spatio‑temporal.

Ghi output vào data/output_pose/<clip_stem>/.

Đặt LampState[cam_i] = DONE (hoặc ALERT nếu có anomaly) rồi đi sang clip tiếp theo.

Tuyệt đối KHÔNG làm trong mode này
Không multi-thread pose/ADL giữa các clip.

Không xử lý nhiều clip song song.

Không chạy song song pose và ADL.

Không sort theo tên file thuần túy (alphabet).

QUY ĐỊNH CHO LLM:

Mọi code mới cho multicam demo phải tuân theo thứ tự:
tracking → pose → ADL → ReID → save trong một vòng lặp duy nhất trên clip.

Cấm tạo worker pool / thread pool / asyncio để xử lý nhiều clip hoặc nhiều camera cùng lúc.

Nếu cần tối ưu hiệu năng, chỉ được tối ưu bên trong từng bước (ví dụ dùng model nhỏ hơn), không được đổi flow tuần tự.

## 7. Topology Camera và Logic Không Gian
text
cam01 → cam02 → cam03 → [thang máy / elevator] → cam04
                     ↑                           ↓
                     └─────── [quay lại] ────────┘

cam04 → [room_upstairs] → cam04
Camera	Ý nghĩa
cam01	Đầu tuyến, MASTER — được phép tạo Global ID mới
cam02	Trung gian tầng dưới, SLAVE
cam03	Gần thang máy / điểm chuyển tầng, SLAVE
cam04	Tầng trên, SLAVE — STRICT (báo ALERT nếu UNKNOWN / conflict)
Topology này dùng để giới hạn candidate ReID và giảm số lượng ID mới.

Cửa sổ thời gian chuyển camera (để trong config)
python
TRANSITION_WINDOWS = {
    ("cam01", "cam02"): (0, 60),      # giây
    ("cam02", "cam03"): (0, 60),
    ("cam03", "cam04"): (20, 180),    # elevator
    ("cam04", "cam03"): (20, 180),
    ("cam04", "room"): (5, 300),
    ("room", "cam04"): (5, 300),
    ("cam03", "cam02"): (10, 120),    # quay đầu
}
QUY ĐỊNH CHO LLM:

Khi viết hàm ReID / candidate selection, chỉ xem xét các cặp camera có trong TRANSITION_WINDOWS.

Nếu một GID ở cam01 thì candidate tiếp theo hợp lệ chỉ là cam02 trong khoảng 0–60 giây.

cam03 ↔ cam04 phải đi qua cửa sổ elevator với khoảng 20–180 giây, không được match ngoài range.

Không được tự thêm topology mới (ví dụ cam01→cam03 trực tiếp) nếu user không sửa TRANSITION_WINDOWS.

cam04 có chế độ STRICT: nếu không match được Global ID theo topology + thời gian + feature thì để UNKNOWN và bật ALERT, không tự tạo ID mới ở cam04.

## 8. Chính Sách Global ID — Càng Ít Càng Tốt
Global ID là ID xuyên camera; mục tiêu là giảm tối đa số lượng Global ID, chỉ tạo khi thật cần thiết.

Khi nào ĐƯỢC tạo ID mới
Chỉ được tạo Global ID mới nếu đồng thời thỏa các điều kiện:

Không có candidate hợp lệ theo thời gian–topology (theo TRANSITION_WINDOWS).

Không có face/body/gait match đủ tin cậy (điểm thấp ở tất cả kênh).

Không có pending transfer / room hold hợp lý trong PendingTransitionBuffer hoặc RoomHoldBuffer.

Tổng điểm confidence 
Stotal < 0.60 S total
​
 <0.60.

Nếu bất kỳ điều kiện nào ở trên chưa được kiểm tra thì không được tạo ID mới (phải kiểm tra đủ 4 nhóm bằng chứng trước).

Công thức điểm ghép ID
python
 Trường hợp bình thường:
S_total = (
    0.35 * S_face
    + 0.25 * S_body
    + 0.15 * S_pose_gait
    + 0.10 * S_height
    + 0.15 * S_time_topology
)

 Trường hợp nghi thay quần áo (room / elevator):
S_total = (
    0.40 * S_face
    + 0.10 * S_body
    + 0.20 * S_pose_gait
    + 0.10 * S_height
    + 0.20 * S_time_topology
)

 Quyết định:
 S_total >= 0.75        → giữ ID cũ (match mạnh)
 0.60 <= S_total < 0.75 → giữ ID cũ + gắn cờ SOFT_MATCH
 S_total < 0.60         → CÓ THỂ tạo ID mới
                          (chỉ khi không có pending candidate duy nhất)
QUY ĐỊNH CHO LLM:

Khi viết hàm match ReID, phải dùng một biến tương đương S_total với trọng số trên; không được đổi trọng số nếu user không yêu cầu.

Nếu tồn tại chính xác 1 candidate trong PendingTransitionBuffer/RoomHoldBuffer phù hợp topology + thời gian, thì ưu tiên giữ ID cũ, kể cả khi S_total < 0.60 (gắn SOFT_MATCH nếu cần) → không tạo ID mới.

Nếu không có candidate hợp lệ, mới xét tới việc tạo Global ID mới (và vẫn phải tôn trọng can_assign_new_id).

Chính sách theo camera (cam03/cam04 – ID assignment)
python
ID_ASSIGN_CAMERAS = {"cam01"}      # chỉ cam01 được phép tạo ID mới
ID_CONFIRM_CAMERAS = {"cam02"}     # cam02 có thể confirm (không tạo)
                                   # cam03, cam04 chỉ được match ID cũ
def can_assign_new_id(cam_id: str) -> bool:
    return cam_id in ID_ASSIGN_CAMERAS
cam01: MASTER, được phép tạo Global ID mới nếu thỏa điều kiện ở trên.

cam02: chỉ xác nhận/giữ/soft-match ID cũ; không tạo ID mới.

cam03, cam04: chỉ được match existing ID; nếu không match → gán UNKNOWN và có thể ALERT (đặc biệt ở cam04).

QUY ĐỊNH CHO LLM:

Trong code ReID, mọi chỗ tạo ID mới phải đi qua can_assign_new_id(cam_id) (hoặc logic tương đương); không được tạo ID mới trực tiếp ở cam02/03/04.

Nếu đang ở cam03/cam04 và không match được, kết quả đúng là UNKNOWN + cập nhật trạng thái ALERT/log, không tạo ID mới.

## 9. Trạng Thái Hệ Thống Cần Duy Trì
Runtime cần giữ một số cấu trúc trạng thái trung tâm.

GlobalPersonTable — trạng thái Global ID xuyên camera
python
 GlobalPersonTable — trạng thái ID xuyên camera
GlobalPersonTable = {
    "GID-001": {
        "status": "ACTIVE | PENDING_TRANSFER | IN_ROOM | DORMANT | CLOSED",
        "last_cam": "cam03",
        "last_time": datetime(...),
        "last_adl": "walking",
        "last_face_emb": ...,
        "last_body_emb": ...,
        "last_height_ratio": ...,
    },
    # ...
}
status: trạng thái vòng đời ID (đang thấy, đang chờ chuyển camera, trong phòng, ngủ, đã đóng).

Các embedding và thông tin chiều cao dùng để tính S_face, S_body, S_pose_gait, S_height trong công thức trên.

PendingTransitionBuffer — người vừa rời camera ở vùng chuyển tiếp
python
PendingTransitionBuffer = {
    "GID-001": {
        "from_cam": "cam03",
        "expected_next": ["cam04", "cam03"],
        "reason": "elevator_zone",
        "ttl_sec": 180,
        "created_at": datetime(...),
    },
    # ...
}
Dùng cho các case như cam03 → cam04, cam03 → cam02; giữ ID trong trạng thái PENDING_TRANSFER trong khoảng thời gian hợp lệ.

RoomHoldBuffer — người đang ở phòng kín cam04
python
RoomHoldBuffer = {
    "GID-002": {
        "cam": "cam04",
        "zone": "room_upstairs",
        "enter_time": datetime(...),
        "ttl_sec": 300,
        "allow_clothes_change": True,
    },
    # ...
}
Dùng khi người vào phòng ở cam04; cho phép thay quần áo nhưng cố gắng giữ cùng ID khi ra, dựa trên face/pose/height/time-topology.

LampState — 4 bóng đèn camera
python
LampState = {
    "cam01": "IDLE",   # ⚪ IDLE | 🟡 ACTIVE | 🟢 DONE | 🔴 ALERT
    "cam02": "IDLE",
    "cam03": "IDLE",
    "cam04": "IDLE",
}
Dùng để hiển thị trên UI: camera nào đang được xử lý, đã xong, hay đang có cảnh báo.

QUY ĐỊNH CHO LLM:

Không được tạo thêm một “Global ID table” khác; mọi logic ReID phải dùng GlobalPersonTable, PendingTransitionBuffer, RoomHoldBuffer hiện có (hoặc tên tương đương nếu đã implement).

Khi clip/track kết thúc, phải cập nhật status và dọn các entry hết ttl_sec để tránh memory leak, nhưng không thay đổi format cấu trúc này.

Mọi cập nhật UI đèn camera phải thông qua LampState (hoặc wrapper), không tự tạo biến trạng thái song song.


## 10. ADL — Một Classifier Duy Nhất
Bắt buộc: chỉ dùng pose_utils.rule_based_adl() trong Phase 3
Phase 3 (pose + ADL) chỉ được dùng hàm pose_utils.rule_based_adl() để suy luận ADL từ pose.

Không được tạo hoặc gọi thêm bất kỳ ADL classifier nào khác trong runtime (model skeleton, LSTM, v.v. nếu có thì để ở nhánh research riêng, ngoài pipeline demo).

QUY ĐỊNH CHO LLM:

Không viết mới file app/core/adl.py hoặc các class ADLClassifier, AdlModel, … trong runtime.

Nếu code hiện tại vẫn còn app/core/adl.py, phải vô hiệu hóa: không được import / gọi từ bất kỳ chỗ nào trong pipeline Phase 3 và multicam.

ADL classes chuẩn (không thêm/bớt)
python
ADL_CLASSES = [
    "standing",
    "sitting",
    "walking",
    "lying_down",
    "falling",
    "reaching",
    "bending",
    "unknown",
]
Đây là danh sách nhãn ADL duy nhất được phép dùng trong output (file *_adl.txt, overlay, timeline).

QUY ĐỊNH CHO LLM:

Không thêm, bớt, hoặc đổi tên class trong ADL_CLASSES trừ khi user yêu cầu trực tiếp.

Nếu muốn map ra text khác cho UI, phải mapping ở UI, không đổi nhãn gốc trong file output.

Sliding window
python
WINDOW_SIZE = 30  # frames, đặt trong config — không hardcode
Kích thước cửa sổ thời gian cho ADL phải lấy từ config (ví dụ config.yaml), không được hardcode số 30 ở nhiều nơi.

QUY ĐỊNH CHO LLM:

Khi cần dùng WINDOW_SIZE trong code, phải đọc từ config hoặc hằng số tập trung, không được viết trực tiếp 30 trong logic ADL.

## 11. Temporal Stability — Bắt Buộc Fix
Mục tiêu: giảm flicker của pose và ADL giữa các frame, giữ nhãn ổn định theo track.

A. Stable pose per track (chống flicker)
python
 Đặt trong phase3_recognizer.py, scope per-clip
track_pose_memory = {}
POSE_TTL = 5  # frames


def merge_pose_with_memory(track_id, current_keypoints):
    memory = track_pose_memory.get(track_id)
    if memory is None:
        track_pose_memory[track_id] = {
            "kps": current_keypoints,
            "ttl": POSE_TTL,
        }
        return current_keypoints

    merged = []
    for cur, prev in zip(current_keypoints, memory["kps"]):
        cx, cy, cc = cur
        px, py, pc = prev

        if cc >= 0.35:
            mx = 0.7 * cx + 0.3 * px
            my = 0.7 * cy + 0.3 * py
            mc = cc
        elif memory["ttl"] > 0 and pc >= 0.35:
            mx, my, mc = px, py, pc * 0.9
        else:
            mx, my, mc = cx, cy, cc

        merged.append((mx, my, mc))

    memory["kps"] = merged
    memory["ttl"] = (
        POSE_TTL
        if any(k[2] >= 0.35 for k in current_keypoints)
        else max(memory["ttl"] - 1, 0)
    )
    return merged
track_pose_memory phải được reset theo clip, không dùng chung giữa các clip.

QUY ĐỊNH CHO LLM:

Không biến track_pose_memory thành biến global toàn hệ thống; phải giữ scope per‑clip để tránh lẫn track giữa clip khác nhau.

Nếu refactor, vẫn phải giữ ý tưởng: dùng key track_id + TTL để smooth pose.

B. Stable ADL label per track (chống flicker)
python
track_adl_state = {}


def smooth_adl(track_id, raw_label, raw_conf,
               hold_frames=8, switch_margin=0.08):
    state = track_adl_state.get(track_id)
    if state is None:
        track_adl_state[track_id] = {
            "label": raw_label,
            "conf": raw_conf,
            "hold": hold_frames,
        }
        return raw_label, raw_conf

    if raw_label == state["label"]:
        state["conf"] = max(state["conf"], raw_conf)
        state["hold"] = hold_frames
        return state["label"], state["conf"]

    if raw_conf >= state["conf"] + switch_margin or state["hold"] <= 0:
        state["label"] = raw_label
        state["conf"] = raw_conf
        state["hold"] = hold_frames
    else:
        state["hold"] -= 1

    return state["label"], state["conf"]
smooth_adl dùng để giữ nhãn ADL ổn định cho từng track_id trong clip.

QUY ĐỊNH CHO LLM:

Không bỏ qua smooth_adl khi ghi adl.txt hoặc overlay; mọi nhãn ADL hiển thị ra ngoài phải là nhãn đã smooth.

Tương tự như pose, track_adl_state nên reset per‑clip (hoặc ít nhất per‑session) để tránh track_id cũ ảnh hưởng clip mới.

C. Stable local track_id per clip
Không dùng enumerate() để làm person_id.
Bắt buộc dùng YOLO track mode hoặc một tracker (DeepSORT/ByteTrack) để có stable_track_id.

python
Sai — tuyệt đối không làm:
for person_id, kp in enumerate(keypoints_in_frame):   # ❌


Đúng — dùng tracker:
results = model.track(frame, persist=True, classes=[0])
for box in results[0].boxes:
    track_id = int(box.id)    # ✅ stable across frames in clip

## 12. Save Flow — Manual Save (Không Auto-Save)
Kết quả Phase 3 / multicam không được auto-save thẳng vào output_pose/. Mọi clip sau khi xử lý xong phải đi qua bước “pending + user click Save”.

Backend flow (sau khi xử lý xong 1 clip)
python
 Sau khi xử lý xong 1 clip:
self._pending_results[clip.stem] = {
    "clip_stem": clip.stem,
    "preview_video": str(preview_path),
    "temp_dir": str(clip_output_dir),  # nằm trong data/output_process/
    "keypoint_rows": clip_keypoints,
    "adl_rows": clip_adl,
    "saved": False,
}
 KHÔNG copy sang output_pose/ ngay — chờ user click Save trên UI
clip_output_dir ở đây là thư mục tạm trong data/output_process/clip_stem/ chứa video processed và các file txt/json tương ứng.

Khi user click Save trên UI
python
 Move từ output_process/ sang output_pose/<clip_stem>/
shutil.move(str(temp_dir), str(final_output_dir))
temp_dir là đường dẫn trong data/output_process/clip_stem/.

final_output_dir là data/output_pose/clip_stem/.

API endpoints cần có
python
@bp.post("/api/pose/save_result")
def pose_save_result():
    payload = request.get_json(silent=True) or {}
    clip_stem = payload.get("clip_stem")
    result = _pose.save_pending_result(clip_stem)
    return jsonify({"ok": True, "result": result})


@bp.get("/api/pose/pending_results")
def pose_pending_results():
    return jsonify(_json_safe(_pose.pending_results()))
QUY ĐỊNH CHO LLM:

Không viết bất kỳ đoạn code nào tự động gọi save_pending_result hoặc shutil.move ngay sau khi clip xử lý xong.

Nếu thêm endpoint mới, vẫn phải tuân theo flow: xử lý → pending_results → user click Save → move sang output_pose/.

Viewer trên UI khi clip xong chỉ cập nhật preview/pending list, không trigger auto-save.

## 13. Overlay Frame — Thông Tin Bắt Buộc
Mỗi frame processed phải vẽ đủ 3 loại thông tin: bbox + label người + header frame.

python
 Label text per person:
label_parts = [f"T{track_id}"]          # track ID trong clip
if global_id is not None:
    label_parts.append(f"G{global_id}") # Global ID xuyên camera
else:
    label_parts.append("UNKNOWN")
if adl_label:
    label_parts.append(adl_label)       # standing / walking / ...

label_text = " | ".join(label_parts)

 Frame header:
header = f"{cam_id}  frame {frame_id}/{total_frames}  {fps:.1f}fps"

 Vẽ bbox, skeleton, label lên frame
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
cv2.putText(
    frame,
    label_text,
    (x1, max(y1 - 10, 20)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
)
cv2.putText(
    frame,
    header,
    (20, 28),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
)
QUY ĐỊNH CHO LLM:

Không xoá bỏ hoặc giản lược overlay này trong phiên bản demo; nếu thêm overlay khác (ví dụ score), phải giữ nguyên format tối thiểu T/G/ADL + header.

track_id phải là ID từ tracker, global_id là GID, không được dùng index frame‑local.

Header phải chứa cam_id + số frame hiện tại / tổng số + fps, để dễ debug timeline.

## 14. Output Format
Output của Phase 2 & 3 phải theo đúng format text dưới đây; mọi tool phân tích hậu kỳ sẽ dựa vào đó.

Phase 2 — labels.txt
text
frame_id x_min y_min x_max y_max
Một dòng cho mỗi bbox người.

Tọa độ pixel theo hệ OpenCV (x theo chiều ngang, y theo chiều dọc).

Phase 3 — keypoints.txt
text
frame_id person_id kp0_x kp0_y kp0_conf ... kp16_x kp16_y kp16_conf
person_id = track_id ổn định trong clip.

Mỗi keypoint có 3 giá trị: x, y, confidence (COCO 17 điểm).

Phase 3 — adl.txt
text
frame_id person_id adl_label confidence
adl_label ∈ ADL_CLASSES ở mục 10.

confidence là độ tin cậy sau khi đã smooth (từ smooth_adl).

Output folder
text
data/output_pose/<clip_stem>/
  <clip_stem>_processed.mp4
  <clip_stem>_keypoints.txt
  <clip_stem>_adl.txt
  <clip_stem>_tracks.json
  <clip_stem>_timeline.json
<clip_stem> giống như tên file input không đuôi, ví dụ cam01_2026-01-29_16-26-25.

tracks.json: mapping giữa track_id và global_id + meta track.

timeline.json: event-level log (enter/exit camera, elevator, room, ReID decision, alert…).

QUY ĐỊNH CHO LLM:

Không đổi tên file output hoặc thêm phần mở rộng khác trong output_pose/ cho các file chính; nếu cần file phụ, đặt tên khác nhưng không phá cấu trúc trên.

Không đổi format cột/field trong labels.txt, keypoints.txt, adl.txt (chỉ được thêm field mới ở cuối nếu thực sự cần và được user yêu cầu rõ).

Khi di chuyển từ output_process/ sang output_pose/, phải giữ nguyên cấu trúc folder và tên file.

## 15— routes.py — Bắt Buộc Fix
B1. Sanitize pose status (quanh line 334)
Để tránh lỗi TypeError: Object of type 'bytes' is not JSON serializable khi jsonify() trả về status có chứa bytes, mọi giá trị trả ra API phải đi qua _json_safe.

python
def _json_safe(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


@bp.get("/api/pose/status")
def pose_status():
    return jsonify(_json_safe(_pose.status()))
QUY ĐỊNH CHO LLM:

Không sửa signature _pose.status() nếu không cần; chỉ đảm bảo mọi dữ liệu trả ra API đi qua _json_safe.

Khi tạo API mới trả về dict phức tạp, ưu tiên dùng _json_safe(...) trước jsonify(...) để tránh lỗi bytes tương tự.

## 16. UI — Yêu Cầu Refactor
Các yêu cầu này áp dụng cho static/index.html, static/js/app.js, static/css/style.css.

Sidebar
css
.sidebar { width: 220px; }
.sidebar.compact { width: 72px; }
Cho phép thu gọn sidebar để dành chỗ cho viewer.

Viewer (không reset khi xong)
javascript
const state = {
  currentOriginalUrl: null,
  currentProcessedUrl: null,
  lastCompletedClip: null,
  pendingResults: [],
};
// Khi clip xong → cập nhật state, KHÔNG blank viewer
Sau khi một clip xử lý xong, viewer vẫn giữ nguyên video processed cuối cùng, không chuyển sang màn hình “đen/empty”.

Result card — 3 nút
xml
<div class="result-actions">
  <button class="preview-btn">Preview</button>
  <button class="show-btn">Show</button>
  <button class="save-btn">Save</button>
</div>
Preview: mở video processed của clip đó trong viewer.

Show: hiển thị thêm chi tiết (timeline, tracks, log).

Save: gọi API /api/pose/save_result để chuyển từ output_process sang output_pose.

Logs drawer (không chiếm full column)
css
.logs-drawer { width: 320px; max-width: 30vw; }
Logs hiển thị dạng ngăn kéo bên phải hoặc trái, không chiếm trọn chiều ngang.

Queue card hiển thị
Mỗi item trong hàng đợi video (VIDEO QUEUE) phải hiển thị:

Camera ID (badge)

Clip name (stem)

Clip datetime (parse từ tên file)

State: pending / active / done / error

4 Camera Lamp (bắt buộc)
text
⚪ Cam01 ─── ⚪ Cam02 ─── ⚪ Cam03 ─── ⚪ Cam04
⚪ IDLE

🟡 ACTIVE

🟢 DONE

🔴 ALERT

Các lamp này phải cập nhật theo LampState từ backend.

QUY ĐỊNH CHO LLM:

Không thiết kế lại toàn bộ UI; chỉ chỉnh CSS/JS để đáp ứng các yêu cầu trên.

Khi thêm logic JS, phải kết nối đúng với API hiện có (/api/pose/status, /api/pose/pending_results, /api/pose/save_result).

Viewer không được tự reset về “Waiting for start” mỗi khi clip xong hoặc pending list thay đổi.

## 17. Bug Priority — Phải Fix Theo Thứ Tự
Mức ưu tiên sửa bug:

P0 – blocking correctness (ảnh hưởng đúng–sai của kết quả, demo có thể fail).

P1 – quan trọng nhưng không làm sai kết quả (hiệu năng, memory, UX).

P2 – cải tiến / research để sau.

P0 — Fix Ngay (blocking correctness)
Bug	File	Mô tả
A	pose_utils.py / adl.py	Chỉ giữ 1 ADL classifier — xóa hoặc vô hiệu hóa adl.py
B	phase3_recognizer.py	person_id không stable — phải dùng tracker, không dùng enumerate
C	phase3_recognizer.py	Sort multicam theo thời gian thật — không sort alphabet
D	phase3_recognizer.py	Mode demo multicam phải chạy tuần tự — không parallel
E	routes.py ~line 334	Sanitize bytes trước jsonify() — đã có TypeError
P1 — Fix Tiếp Theo
Bug	Mô tả
F	Preview/stream chỉ render khi có subscriber — giảm buffering
G	_pending_ids / track_to_global không dọn khi track expire → leak
H	_update_vector_index() đang pass → FAISS / vector DB không update
I	Overlay PNG lưu mỗi frame → đầy disk — chỉ save mỗi 30 frames
J	_expand_embeddings() cần gc.collect() trước shutil.move() (Windows)
K	_get_valid_candidates() trả về tất cả GID — cần spatio‑temporal gating
L	Manual save flow — không auto-save, giữ result trên UI, chờ user Save
P2 — Để Sau
Nâng cấp ADL từ rule-based lên CTR-GCN / BlockGCN

ReID mạnh hơn (face embedding)

Scale > 3 người

Realtime production, multi-camera song song

QUY ĐỊNH CHO LLM:

Khi user yêu cầu “sửa bug” mà không nói rõ bug nào, ưu tiên xử lý từ P0 → P1 → P2.

Không implement các tính năng P2 (deep ADL, ReID SOTA, realtime lớn) nếu P0/P1 chưa được giải quyết.
## 19. Coing Style
python
# ✅ Dùng pathlib.Path cho path
from pathlib import Path
output_dir = Path("data/output_pose") / clip.stem

# ✅ Tên biến rõ ràng, comment ở logic quan trọng
track_pose_memory: dict[int, dict] = {}  # stable pose per track_id within clip

# ✅ Tách hàm vừa đủ — không quá trừu tượng
def merge_pose_with_memory(track_id: int, current_kps: list) -> list:
    ...
python
# ❌ Không hardcode path string
output = "D:\\Capstone_Project\\data\\output_pose"  # SAI

# ❌ Không thêm abstraction không cần thiết
class PoseMemoryManagerFactoryBuilderSingleton: ...  # SAI
QUY ĐỊNH CHO LLM:

Khi xử lý path, luôn ưu tiên Path(...) thay vì string thuần (os.path.join vẫn dùng được nếu code cũ, nhưng không thêm string hardcode kiểu Windows).

Không tự nghĩ thêm pattern phức tạp (factory, singleton, manager lồng nhau…) nếu không cần thiết cho bug hiện tại.

Comment ngắn gọn chỉ ở những chỗ có logic quan trọng (ReID, ADL, spatio‑temporal), tránh spam comment thừa.

## 20. Checklist Trước Khi Sửa Code
Trước khi sửa bất kỳ đoạn code nào, luôn tự hỏi:

Task này ảnh hưởng mode nào? 

Có thay đổi file naming hoặc output format không?

Có vi phạm nguyên tắc time-first trong xử lý multicam không?

Có vô tình sinh thêm Global ID không (nhất là ở cam02/03/04)?

Có làm pose/ADL chạy song song trái yêu cầu tuần tự không?

Có thêm dependency Python ngoài danh sách cho phép không?

Có phá cấu trúc repo (mục 4) không?

Có để 2 ADL classifier song song (ngoài pose_utils.rule_based_adl) không?

Có auto-save kết quả thay vì manual-save như mục 12 không?

Nếu có bất kỳ câu “Có” nào → dừng và thiết kế lại trước.

QUY ĐỊNH CHO LLM:

Khi user yêu cầu “refactor / optimize”, phải chạy checklist này trong đầu: nếu giải pháp đề xuất làm sai bất kỳ điều kiện nào, không được implement, chỉ được mô tả rủi ro hoặc đề xuất hướng khác.

## 21. Acceptance Checklist Trước Khi Coi Task Hoàn Tất
Một task liên quan đến multicam / Phase 3 chỉ được coi là xong khi tất cả mục sau đều đúng:

Train/val logic nằm hoàn toàn ngoài runtime (không train trong app demo).

data/multicam được xử lý thời gian trước, camera sau (dùng clips_sorted + sort_key).

Pose + ADL chạy tuần tự — không parallel giữa clip/camera.

person_id dùng stable track_id từ tracker (không phải enumerate index).

Chỉ có 1 ADL classifier (pose_utils.rule_based_adl).

4 camera lamp mapping hoạt động đúng theo LampState.

Kết quả lưu đúng data/output_pose/<clip_stem>/ với cấu trúc file đã định nghĩa.

Global ID logic ưu tiên ít ID nhất có thể (tôn trọng S_total, pending buffer, room buffer).

cam03 → elevator → cam04: giữ ID cũ, không tạo ID mới ngay.

cam04 → room → cam04: giữ ID cũ, kể cả khi đổi áo (trừ case conflict rõ ràng).

Chỉ cam01 được tạo Global ID mới.

cam03, cam04 không tạo ID mới — chỉ match existing hoặc UNKNOWN.

jsonify() trong routes.py không bị TypeError với bytes (dùng _json_safe).

Processed result không auto-save — chỉ save khi user click Save trong UI.

Viewer không bị blank sau khi clip xong (giữ video processed cuối cùng).

Sidebar gọn hơn, logs không chiếm cả column (theo mục 16).

Logic quan trọng có comment giải thích ngắn, dễ hiểu.

Không thêm dependency ngoài danh sách cho phép.

QUY ĐỊNH CHO LLM:

Khi user bảo “generate code hoàn chỉnh cho task X”, output code phải thỏa checklist này. Nếu còn mục chưa đảm bảo được (ví dụ không thấy phần Global ID), cần nhắc user còn việc phải làm thêm.

## 22. Non-Goals Phiên Bản Hiện Tại
Những thứ không phải mục tiêu bắt buộc ngay bây giờ (chỉ nên đề xuất ý tưởng, không implement nếu user không yêu cầu rõ):

Đạt SOTA ADL benchmark.

Production-grade person ReID.

Multi-user web platform (account, auth, permission).

Cloud-native / microservices, autoscaling, Kubernetes, v.v.

Realtime fully parallel orchestration trên nhiều GPU / nhiều server.

Nhận diện hàng chục người đồng thời trong mọi camera.

QUY ĐỊNH CHO LLM:

Khi user chỉ yêu cầu “sửa bug / hoàn thiện demo”, không được tự thêm kiến trúc phức tạp thuộc nhóm Non-Goals (microservice, multi-user, realtime scale lớn).

Các ý tưởng liên quan Non-Goals chỉ nên xuất hiện trong phần “gợi ý tương lai” / comment, không push vào code hiện tại.

## 23. Triết Lý Chính Thức
Train/Val nằm ngoài runtime.
Runtime test xử lý theo thời gian trước.
Pose và ADL xử lý tuần tự.
Global ID phải càng ít càng tốt.
Demo đúng trước, nhanh sau.
Đơn giản + chạy được > phức tạp + ấn tượng.

QUY ĐỊNH CHO LLM:

Nếu có lựa chọn giữa giải pháp đơn giản, dễ debug và giải pháp “ngầu nhưng phức tạp”, ưu tiên giải pháp đơn giản miễn là thỏa checklist mục 21.

Bất cứ đề xuất nào làm runtime xử lý không còn “time-first + sequential pose/ADL” hoặc làm bùng nổ số Global ID đều trái với triết lý này và không nên implement nếu user không yêu cầu trực tiếp.