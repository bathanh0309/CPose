# 1. Bắt đầu từ đâu, và vì sao?

Điểm xuất phát là một câu hỏi tưởng như nhỏ:

Làm sao tối ưu vector DB (FAISS) cho ReID xuyên camera,

rồi trượt sang: tích hợp ADL skeleton (CTR‑GCN) vào CPose,

rồi thực tế hơn: làm sao để cả cái hệ thống CPose chạy ổn định, cài được libs, không crash.

Vì bạn đang ở giai đoạn gần tốt nghiệp, nên ưu tiên được chọn là:

Hệ thống phải chạy end‑to‑end trước (product),

phần research (ADL deep, ReID, mmcv/pyskl) sẽ cắm dần vào các Phase.

Nên cách tiếp cận hôm nay là:

Nhìn lại kiến trúc hiện có của CPose (Phase1–2–3, data folders, scripts).

Chuẩn hóa entrypoints (file .bat), để phân biệt rõ:

thứ user bấm được, thứ bạn dùng để nghiên cứu.

Sau đó mới tính tới tích hợp CTR‑GCN, mmcv, pyskl, faiss – và tất nhiên, đụng ngay lỗi environment.

## 1. Những hướng khác đã nghĩ đến nhưng bỏ

Có hai con đường “nghe hay” nhưng mình cố tình không đẩy bạn vào:

Đưa toàn bộ research (CTR‑GCN, PYSKL, mmcv, FAISS) vào requirements chính ngay từ đầu.

Hậu quả: như bạn thấy rồi, mmcv build fail trên Windows, pkg_resources thiếu, Visual Studio Build Tools chưa sẵn, tất cả kẹt ở bước pip install.

Nếu cố, bạn sẽ mất thời gian sửa môi trường nhiều hơn là làm CPose.

Dùng .env làm nguồn config chính cho RTSP/IP cam.

Có vẻ “hiện đại” (12‑factor app, v.v.), nhưng repo của bạn đã có resources.txt + runtimeconfig, và bạn còn phải tránh commit IP/RTSP thật.

Trộn cả .env và resources.txt sẽ làm user lẫn lộn: sửa ở đâu, ưu tiên cái nào?

Chốt lại, hai con đường này bị bỏ vì:

Chúng tăng complexity mà không tăng giá trị ngay cho đồ án của bạn.

Trong bối cảnh bảo vệ tốt nghiệp, điều bạn cần nhất là:

demo chạy mượt,

kiến trúc pipeline rõ ràng,

research có chỗ để “móc” vào, nhưng không bắt buộc phải cấu hình siêu phức tạp.

## 1. Các mảnh ghép kết nối với nhau thế nào?

Nếu nhìn cả buổi nay như một kiến trúc:

Tầng vận hành (product & dev)

run-product.bat: người dùng bấm → venv → main.py → localhost.

run-dev.bat: bạn bấm → venv + LOG_LEVEL=DEBUG → main.py với config dev.

Tầng pipeline offline

Phase 1: run-phase1-recorder.bat
→ đọc resources.txt
→ RTSP → YOLO → mp4 vào data/raw_videos/.

Phase 2: run-phase2-analyzer.bat
→ đọc raw_videos → YOLO → PNG + labels.txt vào data/output_labels/.

Phase 3: run-phase3-adl.bat
→ đọc raw_videos
→ YOLO pose → keypoints
→ ADL (rule hoặc CTR‑GCN)
→ staging ở output_process → confirm → output_pose.

Tầng dữ liệu (data/ folder)

config/ (resources.txt),

multicam/ (video demo đa camera),

raw_videos/ (input cho Phase 2–3),

output_labels/, output_process/, output_pose/.

Tầng research

Từ output_pose/keypoints.txt + adl.txt, bạn train:

CTR‑GCN / BlockGCN (qua PYSKL),

FAISS ReID với Pose2ID.

Mỗi phần không đứng một mình; nó là bước trong pipeline:

RTSP → clip → bbox → pose → ADL → research.

## 1. Tools & frameworks đã chọn (và tại sao)

Những thứ bạn đã chọn (hoặc định chọn):

Flask + Flask-SocketIO + eventlet:

Đủ nhẹ cho dashboard, realtime preview.

SocketIO giúp đẩy log / snapshot / progress Phase 1–3 lên UI.

Ultralytics YOLO / YOLO‑pose:

One‑stop cho detection + pose, dễ dùng, tốc độ đủ.

Giảm thời gian “chế backbone” để bạn dành thời gian cho ADL/ReID.

OpenCV, NumPy, PyYAML:

Nền tảng xử lý video, mảng, và config.

mmcv, mmengine, pyskl (dự kiến):

Cần cho skeleton action recognition kiểu CTR‑GCN, BlockGCN.

Lý do chọn: PYSKL đã wrap sẵn các model SOTA skeleton; bạn chỉ cần chuyển keypoints sang đúng format.

faiss-cpu (dự kiến):

Cho vector DB phục vụ ReID xuyên camera (Pose2ID,…).

Điều thay đổi so với “vẽ ra ban đầu” là:

mmcv/pyskl/faiss không được nhét vội vào requirements.txt cho product,

mà dự kiến sẽ cài riêng cho môi trường research (hoặc máy khác với Linux/GPU phù hợp).

## 1. Tradeoff: bạn ưu tiên gì, chấp nhận mất gì?

Một vài tradeoff rõ nhất:

Ưu tiên hệ thống chạy được trước

Ưu tiên:

run-product.bat chạy OK.

Phase1–3 với rule‑based ADL chạy OK.

Chấp nhận:

CTR‑GCN/BlockGCN chưa chạy ngay trên máy Windows vì vướng mmcv.

ReID/FAISS cũng chưa tích hợp ngay.

Một nguồn config camera duy nhất (resources.txt)

Ưu tiên:

User không phải nhớ thêm .env.

An toàn hơn với sensitive data (resources.txt thể ignore git).

Chấp nhận:

Bỏ bớt “flex” của .env (nhưng đổi lại, docs đơn giản, dễ trình bày cho hội đồng hơn).

Comment mmcv/pyskl khỏi requirements

Ưu tiên:

Dự án cài được trên Windows, không kẹt ở pip install.

Chấp nhận:

ADL deep (CTR‑GCN) chưa thể demo ngay trên máy này.

Bạn phải giải thích phần research dưới dạng kiến trúc + code skeleton, và có thể demo trên môi trường khác (Linux) nếu cần.

## 1. Sai lầm, ngõ cụt và cách thoát

Một số “hố” hôm nay:

Cố cài mmcv trực tiếp từ requirements.txt trên Windows

Đụng:

ModuleNotFoundError: pkg_resources

Failed to build 'mmcv'

Fix tức thời:

Cài/upgrade setuptools để có pkg_resources.

bash
python -m pip install --upgrade pip setuptools
Nhưng vẫn thấy trước:

mmcv trên Windows thường còn vướng compiler/CUDA, nên không nhét nó vào core pipeline.

Giữ .env song song với resources.txt

Dễ dẫn tới:

Một số code đọc .env, một số đọc resources.txt, user bị loạn.

Cách giải:

Quy định rõ: RTSP chỉ đến từ resources.txt.

.env bị coi là legacy / xóa khỏi root.

Khả năng circular import giữa API và PoseADLRecognizer

Nguy cơ:

API import recognizer, recognizer lại import API hoặc module đã lôi API vào → vòng.

Hướng sửa (chưa code chi tiết nhưng đã định hướng):

Tách service xử lý (PoseADLRecognizer / RecognizerService) vào app/core/....

API layer chỉ gọi service, không import ngược.

## 1. “I wish someone told me this earlier”

Một vài lời “giá mà có người nói trước”:

Trên Windows, mmcv/pyskl là hard‑mode.

Đừng trộn nó vào requirements của web app chính.

Hãy làm mọi thứ chạy với PyTorch + Ultralytics trước, rồi giành một buổi riêng cho OpenMMLab trên môi trường phù hợp.

Luôn có một nơi duy nhất khai báo camera.

Dùng resources.txt (hoặc 1 YAML duy nhất).

Mỗi layer khác nhau tự đọc từ đó, đừng thêm một kênh config mới nếu không thật sự cần.

Tách product vs research ngay từ entrypoint.

Một file .bat cho user, vài file .bat cho dev/offline.

Giúp bạn không lỡ sửa nhầm config prod khi đang thử nghiệm research.

## 1. Điều “người có kinh nghiệm” sẽ nhận ra còn người mới dễ bỏ qua

Người mới thường tập trung xem “mô hình nào SOTA, paper nào mới”,
người đã triển khai vài hệ thống sẽ nhìn vào:

Dòng chảy dữ liệu:

raw_videos → output_labels → output_pose – cái này bạn đã define khá rõ.

Ranh giới module:

Camera & recorder (Phase 1), analyzer (Phase 2), recognizer + ADL (Phase 3), API/UI.

Khả năng thay thế từng phần:

Hôm nay rule‑based ADL, mai CTR‑GCN – interface không đổi (vẫn ghi adl.txt format cũ).

Quản lý cấu hình:

Một source cho RTSP, một nơi cho runtime config.

Đó là những dấu hiệu của “kiến trúc có suy nghĩ”, không chỉ là “chạy được”.

## 1. Bài học áp dụng được cho mọi project khác

Những gì bạn học được hôm nay không chỉ dừng ở CPose:

Luôn chia rõ product / dev / research

Dù là web, IoT, hay AI:

Một entrypoint cho người dùng,

Một entrypoint cho bạn test,

Một entrypoint cho các job offline.

Khi tích hợp model nặng (mmcv, faiss, …):

Đừng đưa thẳng vào “main requirements” trước.

Hãy làm một môi trường riêng, chứng minh cài được – rồi mới merge.

Không tản mát: dùng 1–2 file config có cấu trúc rõ (resources.txt, YAML).

Các module đọc cùng một nguồn → giảm bug ẩn.
