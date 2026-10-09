# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** DOLPHIN · **Thành viên:** Nguyễn Văn Diện · **MSV:** 02615

Nhóm có một thành viên. Detector cố định: `yolo26n.pt`, ảnh 640 px, lớp
người, Re-ID `osnet_x0_25_msmt17.pt`. Các lượt chạy dùng GPU trong môi
trường conda `cv_robotics_lab21`; chỉ thay tracker và ngưỡng detector.

## 1. Cấu hình đã chọn

Các bản nộp được chạy đủ ảnh nguồn, không đặt `--max-frames`. So sánh
ban đầu dùng 150 frame; mỗi video đã thử một tracker chuyển động và
một tracker có Re-ID. Khi quét ngưỡng, mỗi lượt chỉ đổi một số.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---:|---:|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | BoT-SORT (`botsort`) | 0.15 | 0.4 | ID 2 và 3 có hộp cả 150 frame đầu; người áo tím giữ ID 3 tới gần cuối. Vẫn bỏ sót người nhỏ phía xa. Bản đủ 600 frame không có cặp hộp IoU trên 0.85. | BoT-SORT, conf=0.15, iou=0.5: hai ID có hộp gần như trùng trên người áo tím ở nhiều mốc. |
| video_2 (phố đêm, tĩnh, rất đông) | BoT-SORT (`botsort`) | 0.15 | 0.5 | Người áo trắng giữ ID 8 cả 150 frame đầu; nhóm gần cửa hàng còn nhiều người không có hộp. Cột đèn và giỏ hoa gây che khuất, vẫn có đổi ID. | BoT-SORT, conf=0.15, iou=0.7: người áo trắng đổi từ ID 8 sang 13 trong đoạn thử. |
| video_3 (camera di động, ảnh nhỏ) | OC-SORT (`ocsort`) | 0.3 | 0.5 | Hai người chính giữ ID 1 và 2 cả 150 frame đầu; đoạn sau còn hộp chồng lấp và track ngắn ở người xa. OC-SORT chạy nhanh hơn trong lượt đối chiếu. | BoT-SORT, conf=0.3, iou=0.5: chưa thấy lợi ích rõ ở hai người chính, khoảng 7.5 FPS so với 20.0 FPS của OC-SORT trong đoạn thử. |
| video_4 (trong nhà, camera di chuyển) | ByteTrack (`bytetrack`) | 0.15 | 0.5 | Người phụ nữ phía xa có cùng ID 7 tại frame 75 và 150. Bản đủ 900 frame không có cặp hộp IoU trên 0.85; người áo trắng vẫn đổi ID 5 sang 38 giữa frame 300 và 450. | BoT-SORT, conf=0.15, iou=0.5: bản đủ frame có hộp gần như trùng trên cùng người, ví dụ ID 22 và 34 ở frame 300. |
| video_5 (trên xe bus, giao lộ đông) | OC-SORT (`ocsort`) | 0.3 | 0.5 | Người áo hồng giữ ID 2 cả 144 frame đầu rồi ra mép phải. Còn bỏ sót người ở xa và người chạy qua đường; frame 673 và 684 không có track xuất ra. | OC-SORT, conf=0.15, iou=0.5: giảm mất hộp của người cầm túi đỏ nhưng có hộp trên vùng xe máy đỗ ở frame 150 và nhiều track ngắn hơn. |

Thư mục `runs/nop_bai/` có đủ `video_1.txt` … `video_5.txt`, tương ứng
600 / 1.050 / 837 / 900 / 750 frame nguồn. Video xem thử của cả năm bản
đã giải mã đủ số frame. Với video_5, file có track ở 748 frame; hai frame
không có track vẫn được xử lý, không thêm dòng giả để lấp frame.

Ghi chú các lượt thử và kiểm tra nằm trong `runs/thu_video_1/` …
`runs/thu_video_5/`. Số dòng và số ID chỉ là thống kê, không thay thế
đánh giá chất lượng.

## 2. Số liệu video_1

Cấu hình chấm: BoT-SORT, conf=0.15, iou=0.4, đủ 600 frame; lần chấm
`dolphin_video1`. Các số dưới được TrackEval in ở thang phần trăm.

| HOTA | MOTA | IDF1 |
|---:|---:|---:|
| 29.123% | 20.478% | 30.018% |

Trích nguyên bảng của `scripts/evaluate_practice.py`:

```text
HOTA: dolphin_video1-pedestrian    HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)
video_1                            29.123    18.523    46.017    19.037    79.099    48.62     84.175    83.809    29.556    35.461    79.371    28.146

CLEAR: dolphin_video1-pedestrian   MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag
video_1                            20.478    81.43     20.602    22.335    92.8      11.29     22.581    66.129    16.33     4150      14431     322       23        7         14        41        65

Identity: dolphin_video1-pedestrianIDF1      IDR       IDP       IDTP      IDFN      IDFP
video_1                            30.018    18.621    77.37     3460      15121     1012
```

MOTA còn thấp chủ yếu do bỏ sót: 14.431 FN so với 322 FP và 23 lần đổi
ID, trên 18.581 hộp nhãn người hợp lệ. Recall của CLEAR là 22.335%, phù
hợp với quan sát nhiều người nhỏ phía xa không có hộp. IDF1=30.018%
cho thấy việc giữ đúng danh tính trên toàn bộ nhãn còn hạn chế, dù một
số người gần camera giữ ID khá lâu. HOTA=29.123% phản ánh hạn chế chung
của phát hiện và liên kết; không đánh giá toàn video chỉ từ vài track rõ.

Gói nhận được thiếu `video_1/eval_config.json`. Để chạy chấm, tạo bản
cục bộ `runs/buoc_f/du_lieu_cham/video_1/` với tên trung tính `LAB/train`,
giữ nguyên nội dung `gt.txt` (đã đối chiếu SHA-256), giữ các thông số
chuỗi ảnh và đổi tên chuỗi trong bản sao thành `video_1`. TrackEval dùng
lớp `pedestrian`, tiền xử lý nhãn mặc định bật (`DO_PREPROC=True`) và
ngưỡng CLEAR/Identity mặc định 0.5. Đây là cấu hình chấm cục bộ, không
phải file cấu hình do giảng viên phát; nếu nhận cấu hình khác, cần chấm
lại để đối chiếu.

```powershell
python scripts/evaluate_practice.py --trackeval-root TrackEval --lab-data-root runs/buoc_f/du_lieu_cham --submission runs/nop_bai/video_1.txt --run-name dolphin_video1
```

Script được sửa để áp dụng alias NumPy trong chính tiến trình TrackEval;
kiểm tra hồi quy và các kiểm tra sẵn có đều đạt. Nhật ký chấm và bảng
kết quả được lưu trong `runs/buoc_f/`.

`video_2` đến `video_5` không có nhãn trong gói lab; chỉ ghi quan sát,
không điền HOTA / MOTA / IDF1 cho các video này.

## 3. Phân tích

### Video_1

Camera tĩnh, ban ngày và mật độ vừa giúp ngoại hình người ở gần khá rõ,
nên chọn BoT-SORT để hỗ trợ liên kết bằng Re-ID. Với conf=0.15 và iou=0.4,
ID 2 và 3 có hộp trong cả 150 frame đầu, đồng thời không còn hộp gần như
trùng trên người áo tím như cấu hình iou=0.5. Giảm conf giúp giữ thêm
người phía xa, nhưng ngưỡng thấp chưa khắc phục được tất cả bỏ sót.
Điểm chấm xác nhận hạn chế này: FN lớn hơn nhiều FP và IDSW. Quan sát
thuộc toàn bộ cấu hình đã chạy, chưa tách riêng đóng góp của Re-ID hay
chứng minh BoT-SORT tốt nhất trong mọi cảnh camera tĩnh.

### Video_4

Camera tiến tới trong nhà làm hộp người thay đổi kích thước, còn kính
và sàn phản chiếu khiến việc nhận diện khó hơn. BoT-SORT trông ổn ở
150 frame đầu nhưng bản đủ video xuất hiện hai ID bám gần như cùng hộp,
ví dụ ID 22 và 34 trên người áo đen tại frame 300. ByteTrack được chọn
sau đối chiếu đủ video vì không có cặp hộp IoU trên 0.85, và người áo
hồng giữ ID ban đầu lâu hơn trong các bản đã chạy. Với conf=0.15, người
phụ nữ phía xa giữ cùng ID tại frame 75 và 150, tốt hơn các mức conf
0.3 và 0.5 trong đoạn thử. Tracker chuyển động vẫn đổi ID của người áo
trắng khi có che khuất, nên kết quả này không có nghĩa mọi danh tính
hoặc mọi phản chiếu đều đã xử lý đúng.

## 4. Nếu có thêm thời gian

Sẽ xem kỹ các frame có che khuất và quét conf mịn hơn quanh cấu hình đã
chọn để cân bằng bỏ sót với hộp giả, đặc biệt ở video_2 và video_5.
Giữ detector, kích thước ảnh và Re-ID cố định theo luật của bài nộp chính.
