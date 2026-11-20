# Phân công công việc chi tiết — 4 thành viên (7 tuần)

> Nguyên tắc: mỗi người = **1 họ mô hình** (phần Deep Learning) + **1 mảng hệ thống** (phần Application).
> Mọi người dùng chung `configs/base.yaml`, `src/data`, `src/utils/metrics.py`, `src/evaluate.py` → kết quả so sánh được.
> Đánh dấu `[x]` khi xong. Mỗi task ghi rõ **đầu ra** để kiểm tra.

| | Mô hình | Mảng hệ thống | File chính |
|---|---|---|---|
| **TV1** | M1 R(2+1)D-18 + B0 MobileNetV3-GRU | Dữ liệu & tiền xử lý | `scripts/prepare_data.py`, `src/data/*`, `src/models/r2plus1d.py`, `mobilenet_gru.py` |
| **TV2** | M2 I3D-R50 | Đánh giá & benchmark | `src/utils/metrics.py`, `src/evaluate.py`, `src/benchmark.py`, `src/visualize.py`, `scripts/run_all_benchmarks.py` |
| **TV3** | M3 Video Swin-T | Training framework & MIL | `src/train.py`, `src/train_mil.py`, `src/extract_features.py`, `src/models/heads.py`, `src/losses/` |
| **TV4** | M4 X3D-S (/MoViNet-A0) | Ứng dụng & triển khai | `app/*`, `src/export_onnx.py`, `Dockerfile` |

---

## Giai đoạn 0 — Khởi động (Tuần 1, nửa đầu) — CẢ NHÓM
- [ ] Tạo repo GitHub từ thư mục này, mời 4 thành viên; quy ước branch `feat/<tên>-<việc>`, merge qua Pull Request (1 người review)
- [ ] Đọc PLAN.md, thống nhất **giao thức thí nghiệm** (`configs/base.yaml`) — không đổi sau tuần 1
- [ ] Chốt môi trường: Python 3.10, `pip install -r requirements.txt`; chọn nơi chạy GPU (máy cá nhân / Colab / Kaggle)
- [ ] Tạo wandb project `ucf-crime-vad`, mời cả nhóm
- [ ] Lịch họp: 1 buổi/tuần (30 phút) + nhóm chat; TV2 giữ bảng tiến độ

---

## TV1 — Dữ liệu + R(2+1)D-18 + MobileNetV3-GRU

**Tuần 1 — Dữ liệu**
- [x] Tải Kaggle UCF-Crime vào `data/raw/ucf_crime/`; tải file nhãn thời gian vào `data/annotations/`
- [x] `scripts/prepare_data.py`: nhóm ảnh theo video, kiểm tra thiếu/lỗi, tạo split train/val (90/10 theo video) /test → `data/processed/splits/`
- [x] Map ảnh test ↔ nhãn frame (`src/data/annotations.py`), lưu `test_frame_labels.npz`; kiểm tra khớp 290 video test
- [x] `notebooks/01_eda.ipynb`: phân bố lớp, độ dài video, tỉ lệ frame bất thường, ảnh mẫu → **hình cho báo cáo**

**Tuần 2 — Dataloader + baseline**
- [x] `src/data/ucf_crime.py` (clip 16 frame, stride 8), `transforms.py` (train/eval), sampler cân bằng → **bàn giao cho cả nhóm đầu tuần 2**
- [ ] Fine-tune **M1 R(2+1)D-18** chế độ cls (dùng `train.py` của TV3) → `weights/r2plus1d_18_cls.pt`
- [ ] Chạy `evaluate.py` cho M1 → kết quả baseline đầu tiên

**Tuần 3–4 — Mô hình**
- [ ] Train **B0 MobileNetV3+GRU** chế độ cls
- [ ] M1 chế độ MIL (trích feature + `train_mil.py`)
- [ ] Ablation trên M1: input 112 vs 160; clip 8 vs 16 frame

**Tuần 5–6 — Phân tích**
- [ ] `notebooks/03_error_analysis.ipynb`: AUC theo lớp, lớp khó (Shoplifting, Stealing…), ví dụ dự đoán sai
- [ ] Vẽ score theo thời gian vs ground truth cho 6–8 video tiêu biểu
- [ ] Thử mô hình tốt nhất trên 5–10 video ngoài tập (CCTV công khai)

**Tuần 7 — Báo cáo:** Mục *Dữ liệu & tiền xử lý*, *Mô hình M1/B0*, slide tương ứng

---

## TV2 — Đánh giá & benchmark + I3D-R50

**Tuần 1**
- [ ] Hoàn thiện `src/utils/metrics.py` (frame AUC, AUC theo lớp, FPR@0.5, FPR@TPR90) + `tests/test_metrics.py`
- [ ] Tải & chuẩn bị **RWF-2000** vào `data/external/` (cross-dataset)

**Tuần 2**
- [ ] `src/evaluate.py`: chạy 290 video test, xuất JSON vào `results/tables/`, điểm frame vào `results/predictions/`
- [ ] `src/benchmark.py`: params, file size, VRAM đỉnh, RAM, latency mean/p95, FPS (batch=1, warm-up)
- [ ] Đánh giá baseline M1 của TV1 → kiểm tra pipeline đúng

**Tuần 3–4 — Mô hình**
- [ ] Fine-tune **M2 I3D-R50** chế độ cls; rồi chế độ MIL
- [ ] Ablation trên M2: frozen vs fine-tune vài block cuối vs full

**Tuần 5**
- [ ] `scripts/run_all_benchmarks.py`: 5 model × {cls, mil} × {FP32, FP16, ONNX} → `results/tables/summary.csv`
- [ ] `src/visualize.py`: ROC, **Pareto AUC-vs-FPS** (size = VRAM), bar AUC theo lớp
- [ ] Bootstrap theo video (1000 lần) → khoảng tin cậy AUC, kiểm tra khác biệt có ý nghĩa

**Tuần 6**
- [ ] Đánh giá cross-dataset trên RWF-2000 (`notebooks/04_cross_dataset.ipynb`)
- [ ] `notebooks/02_results_analysis.ipynb`: bảng & hình cuối cùng

**Tuần 7 — Báo cáo:** Mục *Thiết lập đánh giá*, *Kết quả & so sánh*; ghép bảng tổng hợp; slide kết quả

---

## TV3 — Training framework & MIL + Video Swin-T

**Tuần 1**
- [x] `src/utils/config.py` (đã có), `src/train.py`: vòng train AMP, AdamW + cosine, early stopping theo val AUC, lưu best checkpoint, log wandb
- [x] `src/models/heads.py` (ClsHead, MILHead) — kiểm tra với mọi backbone trong `src/models/`

**Tuần 2**
- [x] `src/extract_features.py`: trích feature snippet → `data/features/<model>/`
- [ ] `src/data/mil_dataset.py` + `src/train_mil.py` + `src/losses/mil_loss.py`; test trên feature M1 *(code + test dữ liệu giả xong; chờ chạy trên GPU)*
- [x] Viết hướng dẫn chạy train/MIL trong README cho cả nhóm

**Tuần 3–4 — Mô hình**
- [ ] Fine-tune **M3 Video Swin-T** chế độ cls (gradient checkpointing / batch nhỏ nếu thiếu VRAM)
- [ ] M3 chế độ MIL
- [ ] Ablation: thêm temporal conv/top-k (kiểu RTFM) vào MIL head trên model tốt nhất

**Tuần 5**
- [ ] Tinh chỉnh model tốt nhất (lr, augmentation, số segment 32 vs 64)
- [ ] Chọn **ngưỡng cảnh báo** theo FPR mục tiêu trên tập val → bàn giao TV4

**Tuần 6**
- [ ] Giải thích mô hình: Grad-CAM (3D-CNN) / attention map (Swin) cho các ca đúng & sai

**Tuần 7 — Báo cáo:** Mục *Phương pháp* (fine-tune, MIL loss), *Mô hình M3*, *Giải thích mô hình*; slide phương pháp

---

## TV4 — Ứng dụng & triển khai + X3D-S

**Tuần 1**
- [ ] Wireframe giao diện (video + biểu đồ score + bảng sự kiện)
- [ ] `app/video_source.py`: đọc file/webcam/RTSP trong thread riêng

**Tuần 2**
- [ ] `app/inference.py`: bộ đệm trượt, lấy mẫu 1/10 frame, dùng **cùng transform** của TV1; chạy thử với M1
- [ ] Pipeline CLI: video vào → file CSV score theo thời gian

**Tuần 3–4 — Mô hình + app v1**
- [ ] Fine-tune **M4 X3D-S** chế độ cls, rồi chế độ MIL (nếu kịp: MoViNet-A0)
- [ ] `app/streamlit_app.py` v1: upload video, biểu đồ score, cảnh báo cơ bản

**Tuần 5 — Triển khai**
- [ ] `src/export_onnx.py`: FP32/FP16 ONNX + INT8 quantization cho M1/M4; kiểm tra sai lệch output (< 1e-3)
- [ ] `app/alerts.py`: làm mượt, hysteresis, lưu ảnh/clip sự kiện, SQLite log, Telegram (tuỳ chọn)
- [ ] App v2: webcam/RTSP, chọn model GPU/CPU, chỉnh ngưỡng

**Tuần 6**
- [ ] Đo **độ trễ cảnh báo** và **số báo động giả/giờ** trên video Normal dài
- [ ] `Dockerfile`, README hướng dẫn chạy app; tích hợp model cuối cùng

**Tuần 7 — Báo cáo:** Mục *Thiết kế hệ thống & triển khai*, *Mô hình M4*; **quay video demo** 3–5 phút; slide demo

---

## Mốc kiểm tra chung
| Mốc | Hạn | Điều kiện đạt |
|---|---|---|
| M0 — Giao thức | Cuối tuần 1 | Split + config chốt, `pytest` pass |
| M1 — Baseline end-to-end | Cuối tuần 2 | R(2+1)D có AUC/FPR/FPS trên test |
| M2 — Đủ mô hình | Cuối tuần 4 | 5 model chế độ cls + ≥ 3 model chế độ MIL; bảng kết quả v1 |
| M3 — Chọn model triển khai | Cuối tuần 5 | Pareto + benchmark ONNX; app v2 chạy |
| M4 — Code freeze | Cuối tuần 6 | Docker chạy, cross-dataset xong |
| M5 — Nộp bài | Tuần 7 | Báo cáo + slide + code + video demo |

## Sản phẩm nộp & người chịu trách nhiệm
| Sản phẩm | Tổng hợp | Đóng góp |
|---|---|---|
| Báo cáo (`reports/report/`) | TV2 (ghép) | Mỗi TV viết mục của mình |
| Slide (`reports/slides/`) | TV3 | Mỗi TV 3–5 slide |
| Mã nguồn + README | TV4 (review cuối) | Cả nhóm |
| Video demo (`reports/demo/`) | TV4 | TV1 chọn video thử |
