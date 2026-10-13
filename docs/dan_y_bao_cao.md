# Dàn ý báo cáo: Nhận diện biển báo giao thông Việt Nam với YOLO

Mỗi mục ghi rõ **lấy số liệu và hình từ đâu** trong dự án.

## 1. Giới thiệu
- Bài toán: phát hiện và phân loại biển báo trong ảnh hoặc video quay từ xe ở Việt Nam.
- Ý nghĩa: hỗ trợ lái xe, xe tự hành, số hóa hạ tầng giao thông.
- Thách thức: biển rất nhỏ và ở xa, nhiều lớp giống nhau (các biển tốc độ 40/50/60, biển cấm rẽ trái và rẽ phải), ánh sáng thay đổi, bị che khuất.
- Đóng góp của nhóm: gộp hai dataset vào một bộ 85 lớp theo QCVN 41, xây pipeline train và đánh giá, làm app demo cho video kèm nhật ký hành trình.

## 2. Các nghiên cứu liên quan
- Phương pháp truyền thống: phân đoạn theo màu (HSV), Hough Circle, HOG kết hợp SVM.
- Deep learning: detector hai giai đoạn (Faster R-CNN) và một giai đoạn (SSD, họ YOLO).
- Các dataset: GTSRB/GTSDB (Đức), TT100K (Trung Quốc), Mapillary; Zalo AI 2020 và các bộ biển báo Việt Nam.

## 3. Dữ liệu
- Nguồn và license: bảng trong README, mục Dataset. **Phải ghi nguồn hai dataset.**
- Cách gộp: map tên lớp của từng nguồn về 85 lớp theo QCVN (`configs/mappings/`), bỏ ảnh trùng, chia train/val/test theo nhóm ảnh.
- Thống kê, lấy từ `scripts/dataset_stats.py`:
  - Hình `reports/figures/class_distribution.png` (mất cân bằng lớp)
  - Hình `reports/figures/box_sizes.png` (biển rất nhỏ)
  - Bảng số ảnh và số box của train/val/test
- Hạn chế: hai bộ lớp không giống nhau, nên có biển không được gán nhãn ở ảnh của bộ kia; một số mã QCVN cần đối chiếu lại.

## 4. Phương pháp
- YOLO11 (backbone, neck, head, anchor-free) và lý do chọn: chạy realtime, có sẵn trọng số pretrained trên COCO.
- Transfer learning từ trọng số COCO.
- Augmentation, xem `configs/train.yaml`: **tắt lật ngang** vì lật làm đổi nghĩa biển trái/phải; giảm biến đổi màu `hsv_h` vì màu là đặc trưng của nhóm biển; dùng mosaic, scale, translate.
- Kích thước ảnh vào 960, vì biển nhỏ.
- Suy luận trên video: tracking ByteTrack, gộp mỗi biển thành một dòng log, bỏ track xuất hiện dưới 3 frame.
- Suy luận cắt ô (sliced inference, tương tự SAHI) cho biển ở xa.

## 5. Thí nghiệm và kết quả
- Môi trường: RTX 4060 Laptop 8GB và Colab Pro; số epoch, batch size.
- Chỉ số đánh giá: Precision, Recall, mAP@0.5, mAP@0.5:0.95, tốc độ (ms/ảnh).
- **Bảng so sánh**: `reports/experiments.md` (sinh bởi `scripts/compare_runs.py`)
  - kích thước model n / s / m
  - imgsz 640 / 960 / 1280
  - có và không có `fliplr` (dự đoán: bật lật ngang làm tăng nhầm lẫn trái/phải)
- Hình từ thư mục `runs/<name>/`: `results.png` (đường loss và mAP), `confusion_matrix_normalized.png`, `BoxPR_curve.png`, `val_batch*_pred.jpg`.

## 6. Phân tích lỗi
Lấy từ `reports/error_analysis/<name>/summary.md` và `per_class.csv`:
- Các cặp lớp hay bị nhầm (ví dụ biển tốc độ 50 và 60) và giải thích.
- Bỏ sót: bao nhiêu phần trăm là biển nhỏ dưới 20 px; so sánh kết quả có và không có chế độ cắt ô.
- Báo nhầm: biển quảng cáo, đèn, vật thể tròn màu đỏ.
- Chọn 4–6 ảnh ví dụ trong các thư mục `missed/`, `wrong_class/`, `false_positive/`.

## 7. Demo
- Ảnh chụp màn hình app Streamlit (tab Ảnh và tab Video).
- Ví dụ file `signs_log.csv` của một video quay thực tế.
- Tốc độ xử lý video (FPS) trên RTX 4060.

## 8. Kết luận và hướng phát triển
- Tóm tắt kết quả tốt nhất.
- Hướng phát triển: bổ sung dữ liệu tự quay (ban đêm, trời mưa), gán nhãn lại cho thống nhất giữa hai nguồn, dùng classifier hai tầng cho các lớp giống nhau, đọc số trên biển tốc độ, chạy trên thiết bị nhúng (ONNX, TensorRT).

## Tài liệu tham khảo
- QCVN 41:2024/BGTVT – Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ.
- Ultralytics YOLO11 docs: https://docs.ultralytics.com
- ByteTrack (Zhang et al., ECCV 2022).
- SAHI (Akyon et al., ICIP 2022).
- Hai dataset Hugging Face (xem README).
