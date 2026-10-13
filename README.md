# 🚦 Nhận diện biển báo giao thông Việt Nam với YOLO (CPV301)

Đầu vào là **ảnh** hoặc **video quay từ xe** (điện thoại, dashcam). Hệ thống phát hiện vị trí biển báo, phân loại theo mã QCVN 41 và hiển thị tên, ý nghĩa bằng tiếng Việt. Với video, hệ thống xuất **nhật ký hành trình** gồm thời điểm, loại biển và ảnh crop của từng biển đã gặp.

## Cấu trúc thư mục

```
configs/
  classes.yaml          # DANH SÁCH LỚP (mã QCVN, tên, ý nghĩa), là nguồn chuẩn duy nhất
  train.yaml            # tham số train (đã tắt lật ngang, giảm hsv_h,...)
  mappings/             # map tên lớp của dataset ngoài -> key của mình
scripts/
  extract_frames.py     # video -> frame để gán nhãn
  import_dataset.py     # dataset YOLO ngoài (Roboflow...) -> data/pool/<tên>/
  split_dataset.py      # data/pool -> dataset/{train,val,test} + data.yaml (chia theo nhóm, tránh rò rỉ)
  dataset_stats.py      # EDA + biểu đồ cho báo cáo
  train.py              # train + đánh giá trên test, ghi test_summary.json
  predict.py            # chạy trên ảnh / thư mục / video (tracking + CSV)
src/vnsign/             # code dùng chung (đọc classes, inference, vẽ nhãn tiếng Việt, tracking log)
app/app.py              # demo Streamlit
notebooks/train_colab.ipynb
models/                 # đặt file best.pt để app tìm thấy (không commit)
data/, dataset/, runs/, outputs/   # KHÔNG commit (đã có trong .gitignore)
```

## Cài đặt (Windows, GPU NVIDIA)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # phải in ra True
```

## Workflow

### 1. Chốt danh sách lớp
Cả nhóm cùng sửa `configs/classes.yaml` và đối chiếu mã biển với **QCVN 41:2024/BGTVT**. Sau khi đã bắt đầu gán nhãn thì **không đổi thứ tự** các lớp; muốn thêm lớp thì thêm vào cuối danh sách.

### 2. Thu thập dữ liệu
- **Dataset có sẵn:** tìm "vietnam traffic sign" trên [Roboflow Universe](https://universe.roboflow.com), chọn Export → *YOLOv8* và giải nén vào `downloads/<tên>/`. Có thể dùng thêm Zalo AI 2020 Traffic Sign.
- **Tự quay:** gắn điện thoại lên xe, quay 1080p ban ngày, ban đêm, trời mưa. Chép video vào `data/raw_videos/`:
  ```bash
  python scripts/extract_frames.py --videos data/raw_videos --out data/frames --every 0.5
  ```
  Upload `data/frames` lên một project Roboflow chung của nhóm, **đặt tên lớp đúng bằng `key`** trong classes.yaml (`p102`, `p127`...), gán nhãn rồi export YOLOv8.

### 3. Gộp và chia dữ liệu
```bash
# dataset ngoài: cần file mapping (xem configs/mappings/example.yaml)
python scripts/import_dataset.py --src downloads/rf_vn --name rf_vn --mapping configs/mappings/rf_vn.yaml --unmapped other
# dữ liệu nhóm tự gán nhãn (tên lớp đã trùng key nên không cần mapping)
python scripts/import_dataset.py --src downloads/team_labels --name team
python scripts/split_dataset.py --force
python scripts/dataset_stats.py
```
Nên tạo thêm một **tập test khó** gồm ảnh ban đêm, trời mưa, ngược sáng và biển ở xa, để phân tích lỗi trong báo cáo.

### 4. Train
Thí nghiệm nhỏ chạy trên máy (RTX 4060 8GB), bản train lớn chạy trên Colab Pro (`notebooks/train_colab.ipynb`).
```bash
python scripts/train.py --model yolo11n.pt --name n_640 --imgsz 640      # baseline
python scripts/train.py --model yolo11s.pt --name s_960                  # mặc định imgsz 960
python scripts/train.py --model yolo11s.pt --name s_960_flip --fliplr 0.5   # chứng minh lật ngang gây hại
```
Mỗi lần chạy sinh ra `runs/<name>/` (gồm best.pt, results.png, confusion_matrix.png) và `test_summary.json`. Ghi kết quả vào bảng so sánh trong báo cáo.

**Gợi ý thí nghiệm cho báo cáo:**
- so sánh kích thước model n/s/m,
- so sánh imgsz 640/960/1280,
- so sánh có và không có `fliplr`,
- so sánh chỉ dùng dataset ngoài với dataset ngoài cộng dữ liệu tự quay.

### 5. Chạy thử và demo
```bash
python scripts/predict.py --weights runs/s_960/weights/best.pt --source path/to/video.mp4
copy runs\s_960\weights\best.pt models\best.pt
streamlit run app/app.py
```

## Output
| Đầu vào | Kết quả |
|---|---|
| Ảnh | Ảnh có bbox và nhãn tiếng Việt; `detections.json` (mã, tên, độ tin cậy, bbox) |
| Video | Video có bbox; `signs_log.csv` (mỗi biển một dòng: thời điểm, mã, tên, ý nghĩa, độ tin cậy); thư mục `crops/` |

## Phân công (3 người, khoảng 5–6 tuần)
| Tuần | A – Data | B – Model | C – App và báo cáo |
|---|---|---|---|
| 1 | Chốt classes.yaml, tải và gộp dataset ngoài | Cài môi trường, train baseline | Đi quay video, chạy thử app |
| 2 | Gán nhãn dữ liệu tự quay, EDA | Thí nghiệm imgsz và kích thước model | Cùng gán nhãn |
| 3 | Bổ sung dữ liệu cho lớp yếu | Tinh chỉnh augmentation | Hoàn thiện xử lý video và CSV |
| 4 | Tạo tập test khó | Train bản cuối trên Colab | Tích hợp model cuối vào app |
| 5–6 | Viết phần Dataset | Viết phần Method và Results | Làm slide và video demo |

## Quy ước Git
- `main` luôn chạy được. Mỗi người làm trên nhánh riêng (`data/...`, `model/...`, `app/...`) rồi mở Pull Request để merge.
- **Không commit** ảnh, video, dataset hay file `.pt`. Dữ liệu chia sẻ qua Google Drive hoặc Roboflow; model tốt nhất để trên Drive.
- Khi sửa `classes.yaml`, phải báo cả nhóm vì thay đổi này ảnh hưởng đến toàn bộ nhãn.
