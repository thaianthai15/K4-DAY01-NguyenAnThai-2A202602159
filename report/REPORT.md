# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 
    - Record hạng 1 (`468`, `cab`, `1`, `0.510915`, `ImageNet-1K`)
- Record này mô tả toàn ảnh như thế nào?
    - Model dự đoán toàn ảnh thuộc lớp "cab", đứng hạng 1 với score là 0.510915
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    - ImageNet-1K
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    - `class_id` để định danh lớp, `class_name` để dễ đọc, `taxonomy_name` để xác định lớp thuộc hệ phân loại nào.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    - Quy định rõ thứ tự ưu tiên chủ thể chính và quy tắc gán nhãn đơn hay đa nhãn để người gán nhãn không bối rối
- Vì sao model score không phải ground truth?
    - score chỉ là mức độ model tin vào prediction; ground truth phải đến từ label/annotation chuẩn, không phải từ output của model.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    - (`bowl`, `0.719063`, `[32.65, 342.12, 100.16, 384.93]`,`78.9`)
- Diễn giải vị trí box bằng lời:
    - Box nằm ở góc dưới bên trái của ảnh, bao quanh một chiếc bowl.
- So sánh số prediction ở hai threshold:
    - Threshold 0.35: 11 predictions, Threshold 0.50: 6 predictions
    - Khi tăng threshold từ 0.35 → 0.50, số prediction giảm 5 object vì các prediction có score dưới 0.50 bị loại.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    - Tăng threshold làm giảm số prediction và workload của reviewer nhưng có thể làm giảm độ bao phủ do loại bỏ các object có confidence thấp. 
    - Threshold thấp giúp tăng coverage nhưng cần reviewer kiểm tra nhiều prediction hơn.
- Đề xuất một quy tắc box chặt:
    - Box phải bao sát phần object nhìn thấy, không chứa background hoặc object khác nếu không cần thiết.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    - Cần quy định box theo phần object nhìn thấy hay phần toàn bộ object được ước đoán; trường hợp không rõ thì escalate để reviewer quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    - (`kitchen-003`, `bowl`, `0.698698`, `70`, [
        84.0,
        289.0
      ],
      [
        83.0,
        290.0
      ],
      [
        81.0,
        290.0
      ],
      [
        80.0,
        291.0
      ],
      [
        77.0,
        291.0
      ],)
- Polygon bổ sung chi tiết gì so với box?
    - Bounding box chỉ mô tả một hình chữ nhật bao quanh toàn bộ object (xyxy), nên có thể chứa cả phần background.
    - Polygon mô tả đường biên/hình dạng thực tế của object theo từng điểm (x, y), vì vậy chính xác hơn khi object có hình dạng bất quy tắc, cong hoặc không chiếm toàn bộ bounding box. Evidence cho thấy polygon của object có thể chứa rất nhiều điểm.
- `instance_id` dùng để làm gì và không phải loại ID nào?
    - Dùng để phân biệt từng object/instance riêng biệt trong cùng một ảnh, kể cả khi chúng cùng class_name.
    - Ví dụ: `kitchen-007` là một instance của class `oven`.
    instance_id không phải class_id và không phải ID của loại object. class_id xác định class (oven, bowl, person...), còn instance_id xác định một object cụ thể.
- Đề xuất một quy tắc biên mask:
    - Polygon nên bám sát đường biên nhìn thấy của object, không mở rộng sang background.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    - Cần guideline rõ về đâu là biên của object khi biên không nhìn thấy rõ.
    - Vùng mờ: xác định mức độ mờ nào vẫn đủ bằng chứng để annotate; nếu không xác định được biên thì không nên tự đoán.
    - Tiếp xúc/chồng lấn: cần quy định object nào sở hữu vùng pixel tại phần tiếp xúc và cách tách hai polygon.
    - Che khuất: cần thống nhất annotate phần nhìn thấy hay toàn bộ hình dạng được suy đoán.
    - Nếu không thể xác định một cách nhất quán → escalate để người phụ trách annotation đưa ra quyết định và bổ sung guideline, tránh mỗi annotator xử lý một kiểu.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Image-level label: mỗi ảnh có một class_id / class_name tương ứng với nhãn của ảnh. | Ảnh khó phân loại, nhiều đối tượng/lớp cùng xuất hiện, ảnh mờ hoặc không đủ thông tin để xác định class. | Gán đúng class theo guideline; đánh dấu/escalate các ảnh không đủ thông tin hoặc class không rõ. | Kiểm tra class có đúng với nội dung ảnh không; kiểm tra các trường hợp borderline và ảnh bị gán nhầm class. |
| Phát hiện vật thể | Bounding box dạng bbox_xyxy + class_id / class_name cho từng object. | Box quá rộng/hẹp, bỏ sót object, box bao gồm quá nhiều background, nhầm class, object bị che khuất. | Vẽ box bao quanh object theo guideline; gán class; xử lý từng object riêng biệt; escalate nếu không xác định được ranh giới hoặc class. | Kiểm tra đủ object chưa, class đúng chưa, box có bao sát object không, có box thừa/trùng không và các trường hợp occlusion.|
| Instance segmentation | Polygon/mask cho từng instance + instance_id, class_id, class_name, score; polygon được biểu diễn bằng các điểm polygon_xy. | Biên object khó xác định, vùng mờ, object tiếp xúc nhau, bị che khuất, polygon quá thô hoặc lấn sang background. | Vẽ polygon bám theo biên thực tế của từng instance; mỗi object có instance riêng; không tự suy diễn phần bị che nếu guideline không cho phép; escalate trường hợp không thể xác định nhất quán. | Kiểm tra polygon có bám biên không, có bỏ sót/lấn vùng không, instance_id có phân biệt đúng từng object không, và xử lý các trường hợp contact/occlusion có nhất quán không. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
    - Chỉ sử dụng, lưu trữ và chia sẻ ảnh/dữ liệu đúng phạm vi được phân công; không tự ý tải dữ liệu ra ngoài hệ thống, chia sẻ cho người không có quyền truy cập hoặc sử dụng dữ liệu cho mục đích khác.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
    - Reviewer/QA hoặc người phụ trách dự án (Project Lead/Data Manager) để được xác nhận và hướng dẫn xử lý; không tự ý tiếp tục annotation hoặc chia sẻ dữ liệu.

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
