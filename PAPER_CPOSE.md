# CPose: Khung suy luận không gian–thời gian theo thứ tự thời gian cho nhận diện tư thế, hoạt động và định danh xuyên camera trong hệ thống giám sát đa camera

**Tác giả:** Nguyễn Bá Thành, [Tên thành viên 2], [Tên thành viên 3], [Tên thành viên 4]  
**Đơn vị:** Khoa Điện tử - Viễn thông, Trường Đại học Bách khoa - Đại học Đà Nẵng  
**GVHD:** [Tên GVHD 1], [Tên GVHD 2]  
**Email:** [email sinh viên], [email GVHD]

---

## Tóm tắt

Giám sát hoạt động con người trong môi trường nhiều camera là một bài toán quan trọng trong các hệ thống nhà thông minh, chăm sóc sức khỏe, giám sát an ninh và phân tích hành vi. Tuy nhiên, các hệ thống hiện tại thường gặp khó khăn khi người dùng di chuyển qua nhiều vùng quan sát khác nhau, đi qua vùng mù như thang máy hoặc phòng kín, bị che khuất, quay lưng, mất khuôn mặt hoặc thay đổi ngoại hình. Các phương pháp theo dõi truyền thống như DeepSORT, ByteTrack hoặc BoTSORT chủ yếu duy trì định danh trong một luồng camera đơn lẻ, trong khi các phương pháp ReID dựa trên ngoại hình dễ suy giảm khi người dùng thay áo hoặc điều kiện ánh sáng thay đổi.

Bài báo này đề xuất **CPose**, một khung xử lý tuần tự theo thời gian cho bài toán nhận diện tư thế, nhận diện hoạt động sinh hoạt hằng ngày và định danh con người xuyên camera. CPose tích hợp các thành phần phát hiện người, ước lượng tư thế, nhận diện hoạt động, nhận diện khuôn mặt, đặc trưng cơ thể, cơ sở dữ liệu vector và cơ chế suy luận Global ID dựa trên ràng buộc không gian–thời gian. Khác với cách xử lý song song nhiều camera, CPose ưu tiên sắp xếp và xử lý các clip theo timestamp, sau đó dùng topology camera, cửa sổ thời gian chuyển tiếp, pose/gait signature, chiều cao tương đối, face embedding và body appearance để quyết định giữ định danh cũ hay tạo định danh mới.

Hệ thống được đánh giá trên bộ dữ liệu đa camera tự thu thập gồm các kịch bản đi qua camera liên tiếp, vào/ra phòng, đi qua thang máy, quay lại camera cũ, che khuất một phần và thay đổi trang phục. Các chỉ số đánh giá bao gồm Global ID Accuracy, Cross-camera IDF1, ID Switch, Fragmentation Rate, Transfer Success Rate, Blind-zone Recovery Rate, Clothing-change ID Preservation, ADL Macro-F1 và FPS toàn hệ thống. Kết quả thực nghiệm dự kiến cho thấy CPose giúp giảm hiện tượng tách ID và đổi ID so với các baseline chỉ dùng local tracking, face ReID hoặc body appearance ReID.

**Từ khóa:** CPose, pose estimation, activity recognition, ADL, person ReID, multi-camera tracking, cross-camera identity association, spatio-temporal reasoning, computer vision.

---

## Abstract

Human activity monitoring in multi-camera environments is important for smart homes, healthcare monitoring, security surveillance, and behavior analysis. However, maintaining consistent identities across cameras remains challenging due to blind zones, asynchronous camera views, occlusion, missing facial cues, appearance changes, and clothing changes. Conventional tracking methods such as DeepSORT, ByteTrack, and BoTSORT mainly preserve identities within a single video stream, while appearance-based ReID methods are sensitive to clothing and illumination changes.

This paper proposes **CPose**, a time-first cross-camera Pose–ADL–ReID framework for sequential multi-camera human activity monitoring. CPose integrates person detection, pose estimation, activity recognition, face recognition, body appearance features, vector-based retrieval, and spatio-temporal Global ID reasoning. Instead of processing all cameras in parallel, CPose sorts input clips by timestamp and performs sequential inference. It then combines camera topology, transition time windows, pose/gait signatures, relative body height, face embeddings, and body appearance cues to determine whether a person should keep an existing Global ID or be assigned a new identity.

The proposed framework is evaluated on a self-collected multi-camera dataset containing normal camera transitions, elevator blind-zone transfers, room re-entry, occlusion, no-face scenarios, and clothing-change cases. The evaluation metrics include Global ID Accuracy, Cross-camera IDF1, ID Switch, Fragmentation Rate, Transfer Success Rate, Blind-zone Recovery Rate, Clothing-change ID Preservation, ADL Macro-F1, and full-pipeline FPS. Experimental results are expected to demonstrate that CPose reduces ID fragmentation and ID switches compared with local tracking only, face-only ReID, and appearance-only ReID baselines.

**Keywords:** CPose, pose estimation, activity recognition, ADL, person ReID, multi-camera tracking, cross-camera identity association, spatio-temporal reasoning, computer vision.

---

## 1. Đặt vấn đề

Trong những năm gần đây, các hệ thống thị giác máy tính ứng dụng trong giám sát con người đã phát triển mạnh nhờ sự tiến bộ của học sâu, đặc biệt là các mô hình phát hiện đối tượng, ước lượng tư thế người và nhận diện hành động. Các ứng dụng như nhà thông minh, chăm sóc người cao tuổi, theo dõi bệnh nhân, giám sát an ninh và phân tích hành vi đều yêu cầu hệ thống không chỉ phát hiện sự xuất hiện của con người, mà còn phải hiểu được người đó đang làm gì và duy trì được danh tính của họ trong suốt quá trình di chuyển.

Trong môi trường thực tế, một camera đơn lẻ thường không đủ bao phủ toàn bộ không gian. Vì vậy, hệ thống giám sát nhiều camera được sử dụng để mở rộng vùng quan sát. Tuy nhiên, việc giám sát nhiều camera đặt ra nhiều thách thức mới. Khi một người rời khỏi camera này và xuất hiện ở camera khác, hệ thống cần xác định đó có phải là cùng một người hay không. Vấn đề này trở nên khó hơn khi giữa hai camera tồn tại vùng mù như thang máy, hành lang không có camera, phòng kín hoặc khu vực bị che khuất. Ngoài ra, người dùng có thể quay lưng, bị che mặt, thay đổi tư thế, thay đổi ánh sáng hoặc thậm chí thay áo sau khi đi vào phòng.

Các thuật toán theo dõi đối tượng hiện đại như DeepSORT, ByteTrack, OCSORT, BoTSORT và StrongSORT đã chứng minh hiệu quả trong bài toán theo dõi nhiều đối tượng trong một video. Tuy nhiên, các thuật toán này chủ yếu giải quyết bài toán duy trì ID trong cùng một luồng camera. Khi người biến mất khỏi một camera và xuất hiện lại ở camera khác sau một khoảng thời gian, các tracker này không đủ thông tin để duy trì định danh toàn cục. Trong khi đó, các phương pháp Person ReID dựa trên ngoại hình hoặc màu sắc trang phục dễ bị sai khi người thay áo, bị che khuất hoặc điều kiện ánh sáng thay đổi.

Để giải quyết các hạn chế trên, bài báo này đề xuất **CPose**, một khung xử lý tuần tự theo thời gian cho bài toán nhận diện tư thế, nhận diện hoạt động và định danh xuyên camera. CPose không chỉ dùng một nguồn bằng chứng duy nhất như khuôn mặt hoặc màu áo, mà kết hợp nhiều tín hiệu gồm khuôn mặt, ngoại hình cơ thể, tư thế, dáng đi, chiều cao tương đối, nhãn hoạt động, thời gian di chuyển và topology camera. Trọng tâm của CPose là cơ chế suy luận không gian–thời gian để giảm việc tạo ID mới không cần thiết khi người đi qua vùng mù hoặc thay đổi ngoại hình.

Các đóng góp chính của bài báo gồm:

1. Đề xuất một framework end-to-end cho giám sát hoạt động con người đa camera, tích hợp phát hiện người, ước lượng tư thế, nhận diện ADL, nhận diện khuôn mặt, ReID cơ thể và cơ sở dữ liệu vector.
2. Đề xuất cơ chế **Time-First Cross-Camera Sequential Pose–ADL–ReID**, trong đó các clip được xử lý theo thứ tự thời gian trước, sau đó dùng ràng buộc camera topology và transition window để giữ Global ID xuyên camera.
3. Đề xuất chiến lược gán Global ID kết hợp nhiều bằng chứng gồm face similarity, body appearance, pose/gait signature, height ratio, time-window gating và camera topology gating.
4. Xây dựng bộ benchmark đa kịch bản cho CPose, gồm normal transfer, elevator transfer, room re-entry, clothing change, no-face scenario và occlusion.
5. Đánh giá hệ thống bằng các metric định lượng gồm Global ID Accuracy, Cross-camera IDF1, ID Switch, Fragmentation Rate, Transfer Success Rate, Blind-zone Recovery Rate, Clothing-change ID Preservation, ADL Macro-F1 và full-pipeline FPS.

Phần còn lại của bài báo được tổ chức như sau. Mục 2 trình bày các nghiên cứu liên quan. Mục 3 mô tả kiến trúc tổng thể của CPose. Mục 4 trình bày thuật toán suy luận Global ID không gian–thời gian. Mục 5 mô tả thiết lập thực nghiệm và bộ dữ liệu. Mục 6 trình bày kết quả đánh giá và so sánh. Mục 7 thảo luận các trường hợp lỗi và hạn chế. Mục 8 đưa ra kết luận và hướng phát triển tiếp theo.

---

## 2. Các nghiên cứu liên quan

### 2.1. Phát hiện người trong ảnh và video

Phát hiện người là bước đầu tiên trong nhiều hệ thống giám sát thị giác máy tính. Các phương pháp truyền thống thường dựa trên đặc trưng thủ công như HOG kết hợp SVM. Tuy nhiên, các phương pháp này khó hoạt động ổn định trong môi trường thực tế có nhiều thay đổi về ánh sáng, góc nhìn, che khuất và mật độ người.

Các mô hình học sâu một giai đoạn như YOLO, SSD và RetinaNet đã cải thiện đáng kể tốc độ phát hiện. Trong đó, họ mô hình YOLO được sử dụng rộng rãi trong các hệ thống thời gian thực nhờ khả năng cân bằng giữa tốc độ và độ chính xác. Các phiên bản nhẹ như YOLOv8n hoặc YOLO11n phù hợp với hệ thống cần xử lý nhanh trên CPU/GPU phổ thông hoặc thiết bị edge. Trong CPose, detector được sử dụng để phát hiện vùng chứa người trước khi đưa vào module pose estimation và ReID.

### 2.2. Ước lượng tư thế người

Ước lượng tư thế người nhằm xác định vị trí các điểm khớp chính trên cơ thể, ví dụ mũi, vai, khuỷu tay, cổ tay, hông, gối và cổ chân. Các mô hình pose estimation hiện đại có thể được chia thành hai nhóm: top-down và bottom-up. Phương pháp top-down phát hiện người trước, sau đó ước lượng keypoint cho từng người. Phương pháp bottom-up phát hiện tất cả keypoint trong ảnh rồi ghép chúng thành từng người.

Trong CPose, mô hình YOLO-Pose được sử dụng để trích xuất bộ 17 keypoints theo định dạng COCO. Các keypoint này không chỉ dùng để vẽ skeleton mà còn được dùng để nhận diện hoạt động và hỗ trợ định danh xuyên camera thông qua pose/gait signature.

### 2.3. Nhận diện hoạt động sinh hoạt hằng ngày

Nhận diện hoạt động sinh hoạt hằng ngày, hay ADL recognition, là bài toán phân loại trạng thái hoặc hành động của con người như đứng, ngồi, đi bộ, cúi người, với tay, nằm hoặc té ngã. Có ba hướng tiếp cận phổ biến: dựa trên ảnh RGB, dựa trên chuỗi skeleton và dựa trên cảm biến.

Các phương pháp RGB/video có thể tận dụng nhiều thông tin về ngữ cảnh, nhưng thường yêu cầu tài nguyên tính toán lớn và dễ bị ảnh hưởng bởi background. Các phương pháp skeleton-based như ST-GCN, MS-G3D, CTR-GCN hoặc các mô hình Transformer trên skeleton phù hợp với hệ thống giám sát vì dữ liệu keypoint gọn nhẹ, ít phụ thuộc vào màu sắc và background. Trong phiên bản hiện tại, CPose sử dụng rule-based ADL như một baseline ổn định, dựa trên góc khớp, góc thân, vận tốc cổ chân và tỉ lệ hình học cơ thể. Trong các phiên bản tiếp theo, module này có thể được thay bằng mô hình skeleton-based học sâu.

### 2.4. Theo dõi nhiều đối tượng trong một camera

Multiple Object Tracking là bài toán phát hiện và duy trì ID của nhiều đối tượng qua các frame trong một video. Các phương pháp như DeepSORT, ByteTrack, OCSORT, DeepOCSORT, BoTSORT và StrongSORT đã đạt kết quả tốt trong nhiều benchmark. Những phương pháp này thường kết hợp detector với mô hình chuyển động, đặc trưng ngoại hình và thuật toán association.

Tuy nhiên, các tracker này chủ yếu giải quyết bài toán trong một camera. Khi đối tượng biến mất khỏi camera này và xuất hiện ở camera khác, đặc biệt sau một khoảng thời gian hoặc qua vùng mù, các tracker truyền thống không còn đủ thông tin để duy trì ID. CPose sử dụng local tracker để giữ ID trong từng clip, sau đó dùng Global ID reasoning để hợp nhất các local track thành định danh xuyên camera.

### 2.5. Person ReID và định danh xuyên camera

Person ReID nhằm so khớp danh tính của một người giữa các camera khác nhau. Các phương pháp ReID truyền thống dựa nhiều vào ngoại hình, màu sắc trang phục và đặc trưng học sâu từ ảnh crop người. Tuy nhiên, trong các kịch bản thực tế như nhà thông minh hoặc giám sát dài hạn, người dùng có thể thay áo, bị che khuất hoặc không có khuôn mặt rõ ràng. Khi đó, ReID chỉ dựa trên ngoại hình dễ tạo ra ID mới không cần thiết hoặc gán nhầm người.

CPose tiếp cận bài toán ReID theo hướng kết hợp nhiều bằng chứng. Ngoài face embedding và body appearance, CPose sử dụng pose/gait signature, chiều cao tương đối, nhãn ADL, thời gian di chuyển hợp lý và topology camera. Cách tiếp cận này giúp giảm phụ thuộc vào một tín hiệu duy nhất.

### 2.6. Suy luận không gian–thời gian trong hệ thống giám sát

Suy luận không gian–thời gian là hướng tiếp cận sử dụng thông tin về vị trí, thời gian và quan hệ giữa các vùng quan sát để cải thiện quyết định của hệ thống. Trong bài toán đa camera, không phải mọi chuyển tiếp giữa hai camera đều hợp lý. Ví dụ, nếu topology thực tế là cam01 → cam02 → cam03 → cam04, thì một người vừa biến mất ở cam01 khó có thể xuất hiện ở cam04 chỉ sau vài giây nếu không đi qua các camera trung gian.

CPose khai thác ý tưởng này bằng cách xây dựng transition window cho từng cặp camera. Nếu một candidate xuất hiện ở camera mới trong khoảng thời gian hợp lý và phù hợp với topology, hệ thống ưu tiên giữ Global ID cũ thay vì tạo ID mới. Đây là điểm khác biệt chính giữa CPose và các phương pháp tracking/ReID chỉ dựa trên appearance.

---

## 3. Kiến trúc tổng thể của CPose

### 3.1. Tổng quan hệ thống

CPose được thiết kế theo ba tầng chính:

1. **AI Core Layer:** chứa các module phát hiện người, ước lượng tư thế, nhận diện khuôn mặt, ReID cơ thể, vector database và các hàm xử lý pose.
2. **Pipeline Layer:** gồm recorder, analyzer và recognizer để xử lý dữ liệu theo từng giai đoạn.
3. **Application Layer:** dashboard Flask/Socket.IO để hiển thị trạng thái camera, skeleton, ADL, Global ID và kết quả xử lý.

**Hình 1. Kiến trúc tổng thể của CPose.**  
`[CHÈN HÌNH Figure_1_CPose_System_Architecture.png]`

### 3.2. Luồng xử lý chính

Luồng xử lý của CPose gồm các bước:

1. Đọc danh sách video hoặc RTSP camera.
2. Parse timestamp từ tên file hoặc metadata.
3. Sắp xếp clip theo thứ tự thời gian.
4. Xử lý từng clip tuần tự.
5. Phát hiện người trong từng frame.
6. Duy trì local track ID trong từng clip.
7. Ước lượng pose cho từng người.
8. Nhận diện ADL bằng sliding window.
9. Trích xuất đặc trưng face/body/pose.
10. Gán hoặc cập nhật Global ID.
11. Lưu kết quả gồm video overlay, keypoints, ADL labels, local-to-global mapping và timeline JSON.
12. Cập nhật dashboard qua Socket.IO.

**Hình 2. Pipeline xử lý tuần tự theo thời gian của CPose.**  
`[CHÈN HÌNH Figure_2_Time_First_Sequential_Pipeline.png]`

### 3.3. Topology camera

Trong kịch bản demo mặc định, hệ thống sử dụng bốn camera:

```text
cam01  ->  cam02  ->  cam03  ->  [elevator]  ->  cam04
                             \                     /
                              \-> [return path] <-/

cam04 -> [room] -> cam04
```

Ý nghĩa từng camera:

| Camera | Vai trò |
|---|---|
| cam01 | Điểm vào đầu tuyến |
| cam02 | Camera trung gian |
| cam03 | Gần thang máy hoặc vùng chuyển tầng |
| cam04 | Tầng trên hoặc khu vực phòng |
| room@cam04 | Phòng kín/vùng mù |
| elevator | Vùng mù giữa cam03 và cam04 |

**Hình 3. Topology camera trong hệ thống CPose.**  
`[CHÈN HÌNH Figure_3_Camera_Topology.png]`

### 3.4. Cấu trúc output

Mỗi clip sau khi xử lý sinh ra các file:

| File | Nội dung |
|---|---|
| `_processed.mp4` | Video có overlay bbox, skeleton, ADL và Global ID |
| `_keypoints.txt` | Keypoints theo frame và track ID |
| `_adl.txt` | Nhãn ADL theo frame và track ID |
| `_tracks.json` | Mapping local track sang Global ID |
| `_timeline.json` | Sự kiện vào/ra camera, room, elevator và quyết định ReID |

---

## 4. Phương pháp đề xuất

### 4.1. Time-First Sequential Processing

Khác với các hệ thống xử lý nhiều camera song song, CPose xử lý clip theo thứ tự thời gian. Với mỗi clip, hệ thống parse timestamp từ tên file, sau đó sắp xếp theo khóa:

```text
sort_key = (year, month, day, hour, minute, second, cam_index)
```

Nếu hai clip có timestamp khác nhau, clip cũ hơn luôn được xử lý trước, bất kể camera nào. Nếu timestamp trùng nhau, hệ thống mới xét đến thứ tự camera.

Cách xử lý này giúp Global ID manager có thể duy trì trạng thái xuyên suốt timeline, tránh lỗi race condition và dễ phân tích các chuyển tiếp giữa camera.

### 4.2. Local Tracking trong từng clip

Trong mỗi clip, CPose sử dụng local tracker để duy trì ID tạm thời. Mỗi detection người được biểu diễn bởi bbox:

```math
B_i = (x_1, y_1, x_2, y_2)
```

Tracker gán local track ID dựa trên độ tương đồng giữa bbox hiện tại và track trước đó. Một độ đo cơ bản là IoU:

```math
IoU = \frac{Area(B_{pred} \cap B_{gt})}{Area(B_{pred} \cup B_{gt})}
```

Nếu IoU hoặc khoảng cách tâm bbox đủ gần, detection mới được gán vào track cũ. Nếu không, hệ thống tạo local track mới.

### 4.3. Pose Estimation

Với mỗi người được phát hiện, CPose trích xuất 17 keypoints theo định dạng COCO:

| ID | Keypoint |
|---:|---|
| 0 | Nose |
| 1 | Left eye |
| 2 | Right eye |
| 3 | Left ear |
| 4 | Right ear |
| 5 | Left shoulder |
| 6 | Right shoulder |
| 7 | Left elbow |
| 8 | Right elbow |
| 9 | Left wrist |
| 10 | Right wrist |
| 11 | Left hip |
| 12 | Right hip |
| 13 | Left knee |
| 14 | Right knee |
| 15 | Left ankle |
| 16 | Right ankle |

Mỗi keypoint được biểu diễn bởi:

```math
K_i = (x_i, y_i, c_i)
```

trong đó `c_i` là confidence score. Một keypoint được xem là visible nếu:

```math
c_i \geq \tau_{kp}
```

Với cấu hình hiện tại:

```math
\tau_{kp}=0.30
```

### 4.4. Rule-based ADL Recognition

Trong phiên bản baseline, CPose sử dụng rule-based ADL dựa trên các đặc trưng hình học từ skeleton. Với sliding window kích thước 30 frame, hệ thống tính các đặc trưng:

- góc thân người;
- góc gối trái/phải;
- vận tốc cổ chân;
- tỉ lệ rộng/cao của người;
- vị trí cổ tay so với vai;
- số keypoint visible.

Góc tại một khớp được tính bằng:

```math
\theta = \cos^{-1}\left(\frac{(p_1-v)\cdot(p_2-v)}{||p_1-v||||p_2-v||}\right)
```

Trong đó `v` là điểm khớp trung tâm.

Các rule chính gồm:

| ADL | Điều kiện minh họa |
|---|---|
| falling | torso angle lớn và vận tốc lớn |
| lying_down | torso angle lớn và aspect ratio lớn |
| sitting | góc gối nhỏ và vận tốc thấp |
| bending | torso angle lớn và vận tốc thấp |
| reaching | cổ tay cao hơn vai |
| walking | vận tốc cổ chân vượt ngưỡng |
| standing | không thuộc các nhóm trên |
| unknown | không đủ keypoint visible |

### 4.5. Global ID Matching Score

Đây là lõi của CPose. Khi một local track xuất hiện ở camera mới, hệ thống không tạo Global ID mới ngay. Thay vào đó, nó tìm các candidate hợp lệ trong bảng Global ID dựa trên time-window và topology camera.

Điểm matching tổng được đề xuất:

```math
S_{total}=w_fS_{face}+w_bS_{body}+w_pS_{pose}+w_hS_{height}+w_tS_{time}+w_cS_{camera}
```

Trong đó:

| Thành phần | Ý nghĩa |
|---|---|
| `S_face` | Similarity giữa face embeddings |
| `S_body` | Similarity ngoại hình/body appearance |
| `S_pose` | Similarity pose/gait signature |
| `S_height` | Tương đồng chiều cao/tỉ lệ cơ thể |
| `S_time` | Điểm hợp lệ theo transition time window |
| `S_camera` | Điểm hợp lệ theo camera topology |

Quyết định cuối cùng:

```math
ID =
\begin{cases}
GID_{old}, & S_{total} \geq \tau_{strong} \\
GID_{old}^{soft}, & \tau_{weak} \leq S_{total}<\tau_{strong} \\
UNK, & S_{total}<\tau_{weak}
\end{cases}
```

Với cấu hình hiện tại:

```math
\tau_{strong}=0.65, \quad \tau_{weak}=0.45
```

**Hình 4. Sơ đồ quyết định Global ID trong CPose.**  
`[CHÈN HÌNH Figure_4_Global_ID_Decision_Flow.png]`

### 4.6. Time-window Gating

CPose kiểm tra xem một người có thể di chuyển hợp lý từ camera trước sang camera hiện tại hay không. Với hai camera `c_i` và `c_j`, time-window gating được định nghĩa:

```math
G_{time}(i,j)=
\begin{cases}
1, & \Delta t_{ij}\in[T_{min}^{c_i\rightarrow c_j},T_{max}^{c_i\rightarrow c_j}] \\
0, & otherwise
\end{cases}
```

Các transition window mặc định:

| Chuyển camera | Window |
|---|---:|
| cam01 → cam02 | 0–60 s |
| cam02 → cam03 | 0–60 s |
| cam03 → cam02 | 10–120 s |
| cam03 → cam04 | 20–180 s |
| cam04 → cam03 | 20–180 s |
| cam04 → cam04 | 5–300 s |

### 4.7. Camera Topology Gating

Không phải mọi cặp camera đều có thể chuyển tiếp trực tiếp. CPose định nghĩa topology graph:

```math
G_{camera}(c_i,c_j)=
\begin{cases}
1, & (c_i,c_j)\in E_{topology} \\
0, & otherwise
\end{cases}
```

Trong đó `E_topology` là tập cạnh hợp lệ giữa các camera.

### 4.8. Temporal Voting

Để tránh gán ID sai do một frame nhiễu, CPose sử dụng xác nhận qua nhiều frame:

```math
V(GID_k)=\sum_{t=n-m+1}^{n}\mathbf{1}(ID_t=GID_k)
```

```math
ID_{final}=GID_k \quad \text{if} \quad V(GID_k)\geq N_{confirm}
```

Với cấu hình hiện tại:

```math
N_{confirm}=3
```

### 4.9. Xử lý clothing-change

Khi người đi vào phòng và thay áo, body appearance có thể thay đổi mạnh. Vì vậy, CPose giảm trọng số appearance và tăng trọng số cho face, pose/gait, height, time và topology:

```math
S_{change}=w_fS_{face}+w_b'S_{body}+w_p'S_{pose}+w_hS_{height}+w_t'S_{time}+w_c'S_{camera}
```

với:

```math
w_b' < w_b
```

Nếu trong RoomHoldBuffer chỉ có một candidate hợp lệ, hệ thống ưu tiên giữ ID cũ ở dạng soft match thay vì tạo ID mới.

### 4.10. Xử lý blind-zone

Các vùng mù gồm:

| Vùng mù | Ví dụ | Cách xử lý |
|---|---|---|
| Elevator | cam03 → cam04 | PENDING_TRANSFER |
| Room | cam04 → room → cam04 | IN_ROOM / RoomHoldBuffer |
| Door blind zone | vừa khuất cửa | TTL ngắn |

Khi một Global ID biến mất ở vùng ra hợp lệ, hệ thống không đóng ID ngay mà đưa vào buffer chờ:

```text
ACTIVE -> PENDING_TRANSFER -> ACTIVE
ACTIVE -> IN_ROOM -> ACTIVE
ACTIVE -> DORMANT -> CLOSED
```

---

## 5. Thiết lập thực nghiệm

### 5.1. Bộ dữ liệu

Bộ dữ liệu được thu thập từ hệ thống nhiều camera trong môi trường thực tế gồm hành lang, vùng gần thang máy và phòng kín. Các camera được bố trí theo topology trong Hình 3. Video được ghi ở độ phân giải `[TODO: resolution]`, tốc độ `[TODO: FPS]` và tổng thời lượng `[TODO: duration]`.

**Bảng 1. Thống kê bộ dữ liệu CPose.**

| Scenario | Cameras | Subjects | Clips | Frames | ADL labels | Special case |
|---|---:|---:|---:|---:|---:|---|
| Normal transfer | TODO | TODO | TODO | TODO | TODO | cam01→cam04 |
| Elevator transfer | TODO | TODO | TODO | TODO | TODO | cam03→cam04 |
| Room re-entry | TODO | TODO | TODO | TODO | TODO | cam04→cam04 |
| Clothing change | TODO | TODO | TODO | TODO | TODO | room@cam04 |
| Occlusion | TODO | TODO | TODO | TODO | TODO | partial body |
| No face | TODO | TODO | TODO | TODO | TODO | back view |

### 5.2. Ground truth annotation

Các nhãn ground truth gồm:

1. **Global ID ground truth:** định danh thật của từng người.
2. **Local track ground truth:** track trong từng camera nếu có.
3. **ADL ground truth:** nhãn hành động theo frame hoặc theo đoạn.
4. **Event ground truth:** vào/ra camera, vào/ra phòng, vào/ra thang máy, thay áo, mất mặt, occlusion.

Các file annotation đề xuất:

| File | Nội dung |
|---|---|
| `dataset_summary.csv` | Thống kê clip/frame/camera |
| `global_id_gt.csv` | Ground truth Global ID |
| `adl_gt.csv` | Ground truth ADL |
| `events_gt.csv` | Ground truth event |
| `camera_topology.yaml` | Topology và transition window |

### 5.3. Cấu hình thực nghiệm

**Bảng 2. Hyperparameter Configuration.**

| Module | Parameter | Value |
|---|---|---:|
| Detector | `conf_threshold` | 0.35 / 0.45 |
| Pose | `keypoint_conf_min` | 0.30 |
| ADL | `window_size` | 30 |
| ADL | `min_visible_keypoints` | 8 |
| Global ID | `strong_threshold` | 0.65 |
| Global ID | `weak_threshold` | 0.45 |
| Global ID | `confirm_frames` | 3 |
| ReID | `threshold` | 0.65 |
| VectorDB | `search_top_k` | 20 |

### 5.4. Môi trường phần cứng và phần mềm

**Bảng 3. Cấu hình phần cứng và phần mềm.**

| Thành phần | Giá trị |
|---|---|
| CPU | TODO |
| GPU | TODO |
| RAM | TODO |
| OS | TODO |
| Python | TODO |
| PyTorch | TODO |
| OpenCV | TODO |
| Ultralytics | TODO |
| FAISS | TODO |
| Backend | Flask + Socket.IO |
| Input source | MP4 / RTSP / Webcam |

---

## 6. Chỉ số đánh giá

### 6.1. Detection metrics

Precision:

```math
Precision = \frac{TP}{TP+FP}
```

Recall:

```math
Recall = \frac{TP}{TP+FN}
```

F1-score:

```math
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
```

Mean Average Precision:

```math
mAP = \frac{1}{N}\sum_{i=1}^{N} AP_i
```

### 6.2. Pose metrics

PCK:

```math
PCK@\alpha = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}
\left(
\frac{||p_i-\hat{p}_i||_2}{s} < \alpha
\right)
```

Nếu không có ground-truth keypoint, hệ thống báo cáo:

- visible keypoint ratio;
- missing keypoint rate;
- pose failure rate;
- mean keypoint confidence.

### 6.3. ADL metrics

ADL được đánh giá bằng:

- Accuracy;
- Macro-F1;
- per-class Precision;
- per-class Recall;
- per-class F1;
- confusion matrix.

### 6.4. Tracking và ReID metrics

IDF1:

```math
IDF1 = \frac{2IDTP}{2IDTP+IDFP+IDFN}
```

MOTA:

```math
MOTA = 1 - \frac{\sum_t(FN_t+FP_t+IDSW_t)}{\sum_t GT_t}
```

Fragmentation Rate:

```math
FragRate = \frac{N_{predicted\_GID}-N_{true\_ID}}{N_{true\_ID}}
```

Transfer Success Rate:

```math
TSR = \frac{N_{correct\_transfers}}{N_{total\_transfers}}
```

False Merge Rate:

```math
FMR = \frac{N_{false\_merge}}{N_{all\_merge}}
```

False Split Rate:

```math
FSR = \frac{N_{false\_split}}{N_{true\_ID}}
```

Unknown Rate:

```math
UNKRate = \frac{N_{UNK}}{N_{all\_assignments}}
```

Blind-zone Recovery Rate:

```math
BZRR = \frac{N_{correct\_blind\_zone\_recoveries}}{N_{blind\_zone\_events}}
```

Clothing-change ID Preservation Rate:

```math
CCIPR = \frac{N_{correct\_ID\_after\_clothing\_change}}{N_{clothing\_change\_events}}
```

### 6.5. Runtime metrics

End-to-end latency:

```math
Latency_{total}=T_{detect}+T_{pose}+T_{ADL}+T_{ReID}+T_{render}
```

FPS:

```math
FPS = \frac{N_{frames}}{T_{processing}}
```

---

## 7. Kết quả thực nghiệm

> Ghi chú: Các bảng dưới đây đã đặt tên sẵn. Khi có kết quả thật, thay các giá trị `TODO` bằng số liệu đo được.

### 7.1. Kết quả phát hiện người

**Bảng 4. Detection Benchmark.**

| Model | Precision | Recall | F1 | mAP@50 | mAP@50–95 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n pretrained | TODO | TODO | TODO | TODO | TODO | TODO |
| YOLOv8n fine-tuned | TODO | TODO | TODO | TODO | TODO | TODO |
| YOLO11n | TODO | TODO | TODO | TODO | TODO | TODO |

**Nhận xét dự kiến:**  
Mô hình nhẹ như YOLOv8n hoặc YOLO11n phù hợp với yêu cầu thời gian thực. Nếu fine-tune trên dữ liệu nội bộ, precision và recall dự kiến tăng so với pretrained COCO do bối cảnh camera cố định và class person được tối ưu cho môi trường triển khai.

### 7.2. Kết quả ước lượng tư thế

**Bảng 5. Pose Benchmark.**

| Pose model | PCK@0.05 | PCK@0.1 | OKS-mAP | Missing keypoint rate | FPS |
|---|---:|---:|---:|---:|---:|
| YOLOv8n-pose | TODO | TODO | TODO | TODO | TODO |
| YOLO11n-pose | TODO | TODO | TODO | TODO | TODO |
| RTMPose | TODO | TODO | TODO | TODO | TODO |

**Hình 5. Ví dụ kết quả pose estimation trong các điều kiện khác nhau.**  
`[CHÈN HÌNH Figure_5_Pose_Qualitative_Results.png]`

**Nhận xét dự kiến:**  
Trong các frame người nhìn rõ toàn thân, keypoints vai, hông, gối và cổ chân ổn định. Các lỗi thường xuất hiện khi người bị che khuất một phần, camera nhìn từ phía sau hoặc người ở quá xa camera.

### 7.3. Kết quả nhận diện ADL

**Bảng 6. ADL Benchmark.**

| Method | Accuracy | Macro-F1 | Standing F1 | Sitting F1 | Walking F1 | Falling F1 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based ADL | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Rule-based + smoothing | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| ST-GCN / CTR-GCN | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| CPose full | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**Hình 6. Confusion matrix cho ADL recognition.**  
`[CHÈN HÌNH Figure_6_ADL_Confusion_Matrix.png]`

**Nhận xét dự kiến:**  
Các class như standing và walking thường dễ nhận diện hơn vì có đặc trưng hình học rõ. Các class sitting, bending và falling dễ bị nhầm nếu chỉ dùng rule-based ADL, đặc biệt khi camera đặt ở góc nghiêng hoặc keypoint phần chân bị thiếu. Việc dùng temporal smoothing giúp giảm hiện tượng nhãn hành động bị nhấp nháy giữa các frame liên tiếp.

### 7.4. Kết quả định danh xuyên camera

**Bảng 7. Cross-camera Global ID Benchmark.**

| Method | Global ID Acc ↑ | IDF1 ↑ | IDSW ↓ | Fragmentation ↓ | False Merge ↓ | False Split ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Local tracker only | TODO | TODO | TODO | TODO | TODO | TODO |
| Face ReID only | TODO | TODO | TODO | TODO | TODO | TODO |
| Body ReID only | TODO | TODO | TODO | TODO | TODO | TODO |
| Pose/height/time only | TODO | TODO | TODO | TODO | TODO | TODO |
| CPose full | TODO | TODO | TODO | TODO | TODO | TODO |

**Nhận xét dự kiến:**  
Local tracker only có fragmentation cao vì mỗi camera tạo ID riêng. Face ReID hoạt động tốt khi khuôn mặt rõ, nhưng suy giảm khi người quay lưng hoặc bị che mặt. Body ReID hoạt động tốt trong trường hợp cùng trang phục, nhưng dễ sai khi thay áo. CPose full dự kiến đạt kết quả tốt nhất vì kết hợp nhiều nguồn bằng chứng và ràng buộc không gian–thời gian.

### 7.5. Ablation study

**Bảng 8. Ablation Study.**

| Variant | Global ID Acc | IDF1 | IDSW | Fragmentation | Clothing-change success | Blind-zone success |
|---|---:|---:|---:|---:|---:|---:|
| CPose full | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o temporal gating | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o camera topology | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o face | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o body appearance | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o pose/gait | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o ADL continuity | TODO | TODO | TODO | TODO | TODO | TODO |

**Nhận xét dự kiến:**  
Khi bỏ temporal gating hoặc camera topology, số lượng false merge và false split dự kiến tăng. Khi bỏ face, hiệu năng giảm trong các trường hợp nhìn rõ mặt. Khi bỏ body appearance, hiệu năng giảm ở trường hợp cùng trang phục nhưng không thấy mặt. Khi bỏ pose/gait, hệ thống yếu hơn trong kịch bản thay áo hoặc không nhìn rõ khuôn mặt. CPose full cân bằng tốt nhất giữa các nguồn bằng chứng.

### 7.6. Ảnh hưởng của transition window

**Bảng 9. Transition Window Ablation.**

| Window setting | Global ID Acc | IDSW | False Merge | False Split |
|---|---:|---:|---:|---:|
| No window | TODO | TODO | TODO | TODO |
| Strict window | TODO | TODO | TODO | TODO |
| Default CPose window | TODO | TODO | TODO | TODO |
| Loose window | TODO | TODO | TODO | TODO |

**Nhận xét dự kiến:**  
Không dùng time window làm tăng false merge vì candidate không hợp lý vẫn có thể được xét. Window quá chặt làm tăng false split vì người đi chậm hoặc dừng lại có thể bị loại. Window quá lỏng làm tăng false merge. Cấu hình mặc định cân bằng giữa khả năng phục hồi ID và tránh gán nhầm.

### 7.7. Kịch bản thay áo

**Bảng 10. Clothing Change Benchmark.**

| Method | Same clothes | Changed clothes | No face | Changed clothes + no face |
|---|---:|---:|---:|---:|
| Body ReID only | TODO | TODO | TODO | TODO |
| Face ReID only | TODO | TODO | TODO | TODO |
| Time-topology only | TODO | TODO | TODO | TODO |
| CPose full | TODO | TODO | TODO | TODO |

**Hình 7. Minh họa kịch bản thay áo trong phòng và Global ID được giữ lại.**  
`[CHÈN HÌNH Figure_7_Clothing_Change_Case.png]`

**Nhận xét dự kiến:**  
Body ReID only giảm mạnh khi thay áo. Face ReID only phụ thuộc vào việc khuôn mặt có nhìn rõ hay không. Time-topology only có thể giữ ID trong trường hợp đơn giản nhưng dễ sai khi nhiều người cùng vào/ra phòng. CPose full dự kiến ổn định hơn nhờ kết hợp nhiều bằng chứng.

### 7.8. Kịch bản vùng mù

**Bảng 11. Blind-zone Recovery Benchmark.**

| Scenario | Events | Correct Recovery | Recovery Rate | Common Error |
|---|---:|---:|---:|---|
| cam03→elevator→cam04 | TODO | TODO | TODO | TODO |
| cam04→elevator→cam03 | TODO | TODO | TODO | TODO |
| cam04→room→cam04 | TODO | TODO | TODO | TODO |
| cam03→return→cam03 | TODO | TODO | TODO | TODO |

**Hình 8. Minh họa xử lý vùng mù thang máy/phòng.**  
`[CHÈN HÌNH Figure_8_Blind_Zone_Recovery.png]`

### 7.9. Runtime performance

**Bảng 12. Runtime Comparison.**

| System | Detector FPS | Pose FPS | ADL FPS | ReID FPS | Full FPS | Latency |
|---|---:|---:|---:|---:|---:|---:|
| CPU only | TODO | TODO | TODO | TODO | TODO | TODO |
| GPU RTX/T4 | TODO | TODO | TODO | TODO | TODO | TODO |
| Jetson Nano / edge | TODO | TODO | TODO | TODO | TODO | TODO |

**Bảng 13. Module Latency Breakdown.**

| Module | Latency/frame or window | Percentage |
|---|---:|---:|
| Detection | TODO | TODO |
| Pose estimation | TODO | TODO |
| ADL recognition | TODO | TODO |
| Face recognition | TODO | TODO |
| Body ReID | TODO | TODO |
| Vector search | TODO | TODO |
| Rendering/dashboard | TODO | TODO |
| Full pipeline | TODO | 100% |

---

## 8. Thảo luận

### 8.1. Vì sao CPose giảm fragmentation?

Các baseline như local tracker only thường tạo ID mới ở mỗi camera. Khi một người đi qua bốn camera, hệ thống có thể tạo bốn ID khác nhau cho cùng một người. CPose giảm fragmentation bằng cách dùng transition window và topology camera để duy trì candidate Global ID trong một khoảng thời gian hợp lý. Nếu người xuất hiện ở camera tiếp theo đúng hướng và đúng thời gian, hệ thống ưu tiên giữ ID cũ.

### 8.2. Vì sao CPose tốt hơn body ReID trong kịch bản thay áo?

Body ReID phụ thuộc nhiều vào màu sắc và texture trang phục. Khi người thay áo, similarity body giảm mạnh, dẫn đến false split. CPose xử lý trường hợp này bằng cách giảm trọng số body appearance trong room-change mode và tăng trọng số cho face, pose/gait, height ratio, time và camera topology. Nếu chỉ có một candidate hợp lệ trong RoomHoldBuffer, CPose ưu tiên giữ ID cũ ở dạng soft match.

### 8.3. Vì sao CPose cần ADL?

ADL không chỉ phục vụ nhận diện hành động, mà còn hỗ trợ tính liên tục hành vi. Ví dụ, một người đang walking ở cam03 và xuất hiện ở cam04 sau 40 giây với trạng thái walking/standing có tính hợp lý cao hơn một candidate có hành vi không liên quan. Trong các phiên bản nâng cao, ADL sequence có thể được dùng như một đặc trưng bổ sung cho ReID.

### 8.4. Các trường hợp lỗi thường gặp

Các lỗi dự kiến gồm:

1. **False split:** một người bị tạo nhiều Global ID khi face không rõ, body thay đổi và pose bị mất keypoint.
2. **False merge:** hai người khác nhau bị gộp ID khi xuất hiện gần nhau trong cùng time window và có ngoại hình tương tự.
3. **ADL confusion:** sitting nhầm bending, bending nhầm falling hoặc standing nhầm walking do keypoint chân không ổn định.
4. **Pose failure:** người ở xa camera, bị che khuất hoặc thiếu sáng khiến keypoint confidence thấp.
5. **Topology mismatch:** người đi đường không đúng topology cấu hình làm hệ thống loại candidate đúng.

**Hình 9. Các failure cases điển hình của CPose.**  
`[CHÈN HÌNH Figure_9_Failure_Cases.png]`

---

## 9. Hạn chế

Mặc dù CPose có khả năng kết hợp nhiều nguồn bằng chứng, hệ thống vẫn còn một số hạn chế:

1. Phiên bản hiện tại vẫn phụ thuộc vào chất lượng pose estimation. Khi keypoint bị mất nhiều, ADL và pose/gait signature suy giảm.
2. Rule-based ADL phù hợp để làm baseline nhưng chưa đủ mạnh cho các hành động phức tạp.
3. Các transition window hiện tại được cấu hình thủ công, cần được học hoặc hiệu chỉnh tự động theo dữ liệu thực tế.
4. Body ReID đơn giản dựa trên đặc trưng màu sắc/hình học có thể chưa đủ mạnh trong môi trường đông người.
5. Hệ thống xử lý tuần tự giúp dễ kiểm soát timeline nhưng có thể chưa tối ưu cho triển khai real-time nhiều camera đồng thời.
6. Dataset nội bộ cần được mở rộng thêm về số người, số camera, điều kiện ánh sáng, occlusion và hành vi phức tạp.

---

## 10. Hướng phát triển tiếp theo

Các hướng phát triển tiếp theo gồm:

1. Thay rule-based ADL bằng mô hình skeleton-based như ST-GCN, CTR-GCN, MS-G3D hoặc Transformer-based skeleton model.
2. Fine-tune pose model trên dữ liệu camera nội bộ để giảm missing keypoint.
3. Bổ sung deep person ReID model thay cho body appearance đơn giản.
4. Học tự động transition window dựa trên thống kê thời gian di chuyển thực tế.
5. Tối ưu pipeline để chạy real-time song song nhiều camera nhưng vẫn giữ Global ID manager nhất quán.
6. Triển khai trên edge device như Jetson Nano, Jetson Orin hoặc mini PC.
7. Xây dựng CPose-Bench làm bộ benchmark chuẩn cho multi-camera Pose–ADL–ReID trong môi trường trong nhà.
8. Bổ sung event-level evaluation thay vì chỉ frame-level evaluation.

---

## 11. Kết luận

Bài báo đã đề xuất CPose, một khung xử lý tuần tự theo thời gian cho bài toán nhận diện tư thế, nhận diện hoạt động và định danh xuyên camera. Khác với các phương pháp theo dõi hoặc ReID chỉ dựa vào một nguồn thông tin, CPose kết hợp nhiều bằng chứng gồm face embedding, body appearance, pose/gait signature, chiều cao tương đối, ADL continuity, transition window và camera topology để duy trì Global ID ổn định hơn trong môi trường nhiều camera.

Cơ chế time-first sequential processing giúp hệ thống kiểm soát timeline rõ ràng, giảm xung đột trạng thái và phù hợp với các kịch bản demo/đánh giá có timestamp. Các chiến lược như PendingTransitionBuffer, RoomHoldBuffer, temporal voting và clothing-change matching giúp CPose xử lý tốt hơn các trường hợp vùng mù, vào/ra phòng, đi qua thang máy và thay đổi ngoại hình.

Kết quả thực nghiệm sẽ được báo cáo bằng các chỉ số Global ID Accuracy, Cross-camera IDF1, ID Switch, Fragmentation Rate, Transfer Success Rate, Blind-zone Recovery Rate, Clothing-change ID Preservation, ADL Macro-F1 và FPS toàn hệ thống. Với đầy đủ benchmark và ablation study, CPose có thể được trình bày như một framework nghiên cứu nghiêm túc cho bài toán multi-camera human activity monitoring thay vì chỉ là một ứng dụng demo.

---

## Tài liệu tham khảo

> Ghi chú: Danh sách dưới đây là khung tham khảo ban đầu. Khi viết paper chính thức, cần chuẩn hóa lại theo IEEE/APA và kiểm tra đầy đủ thông tin xuất bản.

[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” in CVPR, 2016.

[2] N. Wojke, A. Bewley, and D. Paulus, “Simple Online and Realtime Tracking with a Deep Association Metric,” in ICIP, 2017.

[3] Y. Zhang et al., “ByteTrack: Multi-Object Tracking by Associating Every Detection Box,” in ECCV, 2022.

[4] J. Cao, J. Pang, X. Weng, R. Khirodkar, and K. Kitani, “Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking,” in CVPR, 2023.

[5] N. Aharon, R. Orfaig, and B. Z. Bobrovsky, “BoT-SORT: Robust Associations Multi-Pedestrian Tracking,” arXiv preprint arXiv:2206.14651, 2022.

[6] Y. Du et al., “StrongSORT: Make DeepSORT Great Again,” IEEE Transactions on Multimedia, 2023.

[7] S. Yan, Y. Xiong, and D. Lin, “Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition,” in AAAI, 2018.

[8] Z. Liu et al., “MS-G3D: Multi-Scale 3D Graph Convolution Network for Skeleton-Based Action Recognition,” in CVPR, 2020.

[9] Y. Chen et al., “Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition,” in ICCV, 2021.

[10] K. Sun et al., “Deep High-Resolution Representation Learning for Human Pose Estimation,” in CVPR, 2019.

[11] M. Contributors, “OpenMMLab Pose Estimation Toolbox and Benchmark,” OpenMMLab, 2020–2024.

[12] Ultralytics, “Ultralytics YOLO Documentation and Models,” 2023–2025.

[13] J. Deng et al., “ArcFace: Additive Angular Margin Loss for Deep Face Recognition,” in CVPR, 2019.

[14] J. Johnson, M. Douze, and H. Jégou, “Billion-scale Similarity Search with GPUs,” IEEE Transactions on Big Data, 2019.

[15] J. Luiten et al., “HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking,” International Journal of Computer Vision, 2021.

---

## Phụ lục A. Danh sách hình cần chèn

| Hình | Tên file đề xuất | Nội dung |
|---|---|---|
| Figure 1 | `Figure_1_CPose_System_Architecture.png` | Kiến trúc tổng thể CPose |
| Figure 2 | `Figure_2_Time_First_Sequential_Pipeline.png` | Pipeline xử lý tuần tự |
| Figure 3 | `Figure_3_Camera_Topology.png` | Topology camera |
| Figure 4 | `Figure_4_Global_ID_Decision_Flow.png` | Luồng quyết định Global ID |
| Figure 5 | `Figure_5_Pose_Qualitative_Results.png` | Kết quả pose estimation |
| Figure 6 | `Figure_6_ADL_Confusion_Matrix.png` | Confusion matrix ADL |
| Figure 7 | `Figure_7_Clothing_Change_Case.png` | Case thay áo |
| Figure 8 | `Figure_8_Blind_Zone_Recovery.png` | Case vùng mù |
| Figure 9 | `Figure_9_Failure_Cases.png` | Các failure cases |

---

## Phụ lục B. Danh sách bảng cần hoàn thiện

| Bảng | Tên | Trạng thái |
|---|---|---|
| Bảng 1 | Dataset Summary | TODO |
| Bảng 2 | Hyperparameter Configuration | Có sẵn giá trị ban đầu |
| Bảng 3 | Hardware and Software Setup | TODO |
| Bảng 4 | Detection Benchmark | TODO |
| Bảng 5 | Pose Benchmark | TODO |
| Bảng 6 | ADL Benchmark | TODO |
| Bảng 7 | Cross-camera Global ID Benchmark | TODO |
| Bảng 8 | Ablation Study | TODO |
| Bảng 9 | Transition Window Ablation | TODO |
| Bảng 10 | Clothing Change Benchmark | TODO |
| Bảng 11 | Blind-zone Recovery Benchmark | TODO |
| Bảng 12 | Runtime Comparison | TODO |
| Bảng 13 | Module Latency Breakdown | TODO |
