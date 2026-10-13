# 🚦 Nhận diện biển báo giao thông Việt Nam với YOLO (CPV301)

Đầu vào là **ảnh** hoặc **video quay từ xe** (điện thoại, dashcam). Hệ thống phát hiện vị trí biển báo, phân loại theo mã QCVN 41 (85 lớp) và hiển thị tên, ý nghĩa bằng tiếng Việt. Với video, hệ thống xuất **nhật ký hành trình** gồm thời điểm, loại biển và ảnh crop của từng biển đã gặp.

---

## ⚡ Chạy từ đầu (Windows, GPU NVIDIA)

Mở PowerShell tại thư mục dự án.

**1. Cài môi trường** (một lần, tải khoảng 3 GB):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Lệnh cuối phải in `True` kèm tên GPU. Những lần mở terminal sau chỉ cần chạy `.venv\Scripts\activate`.

**2. Chạy toàn bộ pipeline bằng một lệnh**, gồm: tải dataset, chia tập, thống kê, train, đánh giá, phân tích lỗi và chép model sang `models/`:
```bash
python scripts/run_pipeline.py --model yolo11n.pt --imgsz 640 --epochs 30 --name n_640_quick
```
Lệnh này để chạy thử nhanh (khoảng 1 giờ trên RTX 4060). Bản train chính để có kết quả tốt hơn (vài giờ, nên chạy qua đêm hoặc trên Colab):
```bash
python scripts/run_pipeline.py --skip-data --model yolo11s.pt --imgsz 960 --epochs 100 --name s_960
```

**3. Mở app demo:**
```bash
streamlit run app/app.py
```
Trình duyệt sẽ mở `http://localhost:8501`. Chọn model ở thanh bên trái, rồi upload ảnh hoặc video.

---

## Dataset

Script `download_datasets.py` tự tải hai dataset công khai trên Hugging Face (không cần tài khoản):

| Tên | Nguồn | Ảnh | Lớp gốc | License |
|---|---|---|---|---|
| hf_thanhhiepvos | [thanhhiepvos/vietnam_traffic_sign](https://huggingface.co/datasets/thanhhiepvos/vietnam_traffic_sign) | 3.216 | 52 | CC BY-SA 3.0 |
| hf_star092304 | [star092304/Traffic-sign-detection-VietNam](https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam) | 10.157 | 82 | CC BY 4.0 |

- Tên lớp của từng nguồn được map sang 85 lớp chung trong [configs/classes.yaml](configs/classes.yaml), xem [configs/mappings/](configs/mappings/). Hai lớp "đèn xanh" và "đèn đỏ" bị bỏ vì là đèn giao thông, không phải biển báo.
- Khi viết báo cáo, nhớ **ghi nguồn hai dataset** (yêu cầu của license).
- Hạn chế cần nêu trong báo cáo: hai dataset có bộ lớp không giống nhau. Một biển có trong bộ này nhưng không có trong bộ kia sẽ không được gán nhãn ở ảnh của bộ kia.
- Thêm dataset khác: khai báo trong [configs/datasets.yaml](configs/datasets.yaml). Hỗ trợ nguồn Hugging Face, Roboflow (cần `ROBOFLOW_API_KEY` trong `.env`, xem `.env.example`) và link file zip. Lần chạy đầu với một dataset mới, script tạo file mapping nháp để nhóm điền.
- Tải bị gián đoạn (mạng chậm, lỗi 429 của Hugging Face): chỉ cần chạy lại lệnh, script tự tải tiếp.

## Cấu trúc thư mục

```
configs/
  classes.yaml          # DANH SÁCH LỚP (mã QCVN, tên, ý nghĩa), là nguồn chuẩn duy nhất
  datasets.yaml         # các dataset sẽ tải
  mappings/             # map tên lớp của từng dataset -> key trong classes.yaml
  train.yaml            # tham số train (đã tắt lật ngang, giảm hsv_h,...)
scripts/
  run_pipeline.py       # chạy tất cả các bước bên dưới bằng một lệnh
  download_datasets.py  # tải dataset (Hugging Face / Roboflow / zip) + import
  import_dataset.py     # dataset định dạng YOLO -> data/pool/<tên>/ (đổi tên lớp theo mapping)
  split_dataset.py      # data/pool -> dataset/{train,val,test} + data.yaml (chia theo nhóm, bỏ ảnh trùng)
  dataset_stats.py      # thống kê + biểu đồ cho báo cáo (reports/figures/)
  train.py              # train + đánh giá trên tập test -> runs/<name>/test_summary.json
  error_analysis.py     # phân tích lỗi: bỏ sót / nhầm lớp / báo nhầm + ảnh ví dụ
  compare_runs.py       # bảng so sánh các lần train -> reports/experiments.md
  predict.py            # chạy trên ảnh / thư mục / video (tracking + CSV), --tiled cho biển nhỏ
  extract_frames.py     # video tự quay -> frame để gán nhãn
src/vnsign/             # code dùng chung (đọc classes, inference, vẽ nhãn tiếng Việt, log tracking)
app/app.py              # demo Streamlit
notebooks/train_colab.ipynb
models/                 # model dùng cho app (không commit)
downloads/, data/, dataset/, runs/, outputs/, reports/   # sinh ra khi chạy, KHÔNG commit
```

## Từng bước (khi không dùng run_pipeline)

```bash
python scripts/download_datasets.py              # -> downloads/, data/pool/
python scripts/split_dataset.py --force          # -> dataset/ + dataset/data.yaml
python scripts/dataset_stats.py                  # -> reports/figures/
python scripts/train.py --model yolo11s.pt --name s_960 --imgsz 960
python scripts/error_analysis.py --weights runs/s_960/weights/best.pt --imgsz 960
python scripts/compare_runs.py                   # -> reports/experiments.md
python scripts/predict.py --weights runs/s_960/weights/best.pt --source video.mp4
```

**Train trên Colab Pro:** push code lên GitHub, mở `notebooks/train_colab.ipynb` trên Colab, sửa `REPO_URL` rồi chạy lần lượt các cell. Notebook tự tải dataset; kết quả train được lưu vào `MyDrive/vnsign/runs/`.

**Gợi ý thí nghiệm cho báo cáo** (mỗi lần dùng một `--name` khác nhau, sau đó chạy `compare_runs.py`):
- so sánh kích thước model n/s/m: `--model yolo11n.pt / yolo11s.pt / yolo11m.pt`,
- so sánh thế hệ model YOLO11 và YOLO26 (bản mới hơn của Ultralytics): `--model yolo26s.pt`,
- so sánh imgsz 640/960/1280,
- bật lật ngang để chứng minh nó gây hại cho biển trái/phải: `--fliplr 0.5`,
- so sánh có và không có dữ liệu tự quay.

## Output
| Đầu vào | Kết quả |
|---|---|
| Ảnh | Ảnh có bbox và nhãn tiếng Việt; `detections.json` (mã, tên, độ tin cậy, bbox) |
| Video | Video có bbox; `signs_log.csv` (mỗi biển một dòng: thời điểm, mã, tên, ý nghĩa, độ tin cậy); thư mục `crops/` |
| Đánh giá | `runs/<name>/` (results.png, confusion_matrix.png, PR_curve.png), `reports/experiments.md`, `reports/error_analysis/<name>/summary.md` |

## Thu thập thêm dữ liệu tự quay
Gắn điện thoại lên xe, quay 1080p cả ban ngày, ban đêm và trời mưa. Chép video vào `data/raw_videos/`, rồi chạy:
```bash
python scripts/extract_frames.py --videos data/raw_videos --out data/frames --every 0.5
```
Upload `data/frames` lên một project Roboflow chung của nhóm, **đặt tên lớp đúng bằng `key`** trong classes.yaml (`p102`, `p127_50`...), gán nhãn rồi bật mục `team` trong `datasets.yaml`.

## Phân công (3 người, khoảng 5–6 tuần)
| Tuần | A – Data | B – Model | C – App và báo cáo |
|---|---|---|---|
| 1 | Kiểm tra mapping, đối chiếu mã QCVN (các dòng `# ?`) | Chạy pipeline, train baseline | Đi quay video, chạy thử app |
| 2 | Gán nhãn dữ liệu tự quay | Thí nghiệm imgsz và kích thước model | Cùng gán nhãn |
| 3 | Bổ sung dữ liệu cho lớp yếu (xem error_analysis) | Tinh chỉnh augmentation | Hoàn thiện demo video |
| 4 | Tạo tập test khó (đêm, mưa, xa) | Train bản cuối trên Colab | Tích hợp model cuối vào app |
| 5–6 | Viết phần Dataset | Viết phần Method và Results | Làm slide và video demo |

## Quy ước Git
- `main` luôn chạy được. Mỗi người làm trên nhánh riêng (`data/...`, `model/...`, `app/...`) rồi mở Pull Request để merge.
- **Không commit** ảnh, video, dataset hay file `.pt`. Dữ liệu tải lại được bằng script; model tốt nhất chia sẻ qua Google Drive.
- Khi sửa `classes.yaml`, phải báo cả nhóm và train lại, vì thứ tự lớp chính là class_id.
