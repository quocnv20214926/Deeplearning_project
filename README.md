# Phát hiện bạo lực & bất thường từ camera giám sát (UCF-Crime)
Đồ án môn Introduction to Deep Learning — Track 1 (Application), Đề tài 7.

- **Kế hoạch:** [PLAN.md](PLAN.md) · **Phân công:** [TASKS.md](TASKS.md)
- **Mô hình:** R(2+1)D-18, I3D-R50, Video Swin-T, X3D-S, MobileNetV3-GRU — mỗi mô hình chạy 2 chế độ: phân loại clip (fine-tune) và MIL
- **Metric:** Frame-level ROC-AUC, FPR, FPS, latency, memory footprint

## Cài đặt
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
pytest tests/
```

## Quy trình
```bash
python scripts/prepare_data.py                                   # TV1
python -m src.train --config configs/m1_r2plus1d.yaml            # fine-tune (cls)
python -m src.extract_features --config configs/m1_r2plus1d.yaml # feature cho MIL
python -m src.train_mil --config configs/m1_r2plus1d.yaml        # MIL
python -m src.evaluate --config configs/m1_r2plus1d.yaml --mode cls
python -m src.benchmark --config configs/m1_r2plus1d.yaml
python -m src.export_onnx --config configs/m4_x3d.yaml
streamlit run app/streamlit_app.py
```

## Cấu trúc
```
configs/   cấu hình chung + từng mô hình
data/      dữ liệu (không commit) — xem data/README.md
notebooks/ EDA, phân tích kết quả, phân tích lỗi
scripts/   chuẩn bị dữ liệu, chạy benchmark hàng loạt
src/       data · models · losses · utils · train · evaluate · benchmark · export
app/       ứng dụng Streamlit realtime + cảnh báo
weights/   checkpoint (không commit)
results/   bảng, hình, log, dự đoán
reports/   báo cáo, slide, video demo
tests/     unit test
```

## Train & MIL (TV3) — hướng dẫn cho cả nhóm
Mọi lệnh đọc `configs/<model>.yaml` (kế thừa `base.yaml`). Ghi đè không cần sửa file:
`--set train.batch_size=8 train.lr=5e-5`, hoặc biến môi trường `CFG_OVERRIDES="khoa=gia_tri;..."` (áp dụng cho
**mọi** script kể cả `evaluate`, `prepare_data` — tiện trên Kaggle/Colab).

```bash
# 0. chạy thử nhanh (vài video mỗi lớp, 1 epoch) trước khi chạy thật
python -m src.train --config configs/m1_r2plus1d.yaml --limit_videos 2 --epochs 1 --no_wandb
# 1. fine-tune chế độ cls -> weights/<model>_cls.pt (best) + <model>_cls_last.pt (resume)
python -m src.train --config configs/m1_r2plus1d.yaml
python -m src.train --config configs/m1_r2plus1d.yaml --resume weights/r2plus1d_18_cls_last.pt   # tiếp tục
# 2. trích feature snippet 16 frame -> data/features/<model>/{train,val,test}/<video>.npy (bỏ qua video đã có)
python -m src.extract_features --config configs/m1_r2plus1d.yaml --ckpt weights/r2plus1d_18_cls.pt
# 3. train MIL head -> weights/<model>_mil.pt, tự chấm test từ feature -> results/tables/<model>_mil.json
python -m src.train_mil --config configs/m1_r2plus1d.yaml
# 4. đánh giá end-to-end (TV2) — đọc được cả 2 loại checkpoint
python -m src.evaluate --config configs/m1_r2plus1d.yaml --ckpt weights/r2plus1d_18_cls.pt --mode cls
python -m src.evaluate --config configs/m1_r2plus1d.yaml --ckpt weights/r2plus1d_18_mil.pt --mode mil
```

**Val metric.** Tập val tách từ Train nên **không có nhãn frame**. `train.py` báo `val_clip_auc` (AUC nhị phân mức clip,
score = 1 − softmax[Normal], nhãn clip = nhãn video — dùng chọn best, đổi bằng `train.monitor`), `val_video_auc`
(top-3 clip), `val_acc14`. `train_mil.py` chọn best theo `val_video_auc` (max segment). Số báo cáo chính thức là
**frame AUC trên test** (`evaluate.py`).

**Checkpoint** (`src/utils/model_io.py`): `load_model(cfg, mode, ckpt, device)` trả model end-to-end nhận clip
(B,3,T,H,W) chuẩn hoá Kinetics; cls → logits 14 lớp (Normal = lớp 0), mil → logit điểm bất thường
(`sigmoid` = điểm MIL; dùng mặc định `--mil_output logit` của evaluate). `load_backbone(cfg, ckpt)` để trích feature.

**Khoá cấu hình thêm** (có mặc định): `train.grad_accum` 1 · `train.warmup_epochs` 1 · `train.patience` 5 ·
`train.samples_per_epoch` (= số clip train ~40k) · `train.val_max_clips` 16 · `train.label_smoothing` 0 ·
`train.freeze` none|partial|backbone · `model.grad_checkpointing` (Swin) · `mil.iterations` 3000 · `mil.batch_size` 60 ·
`mil.optimizer` adagrad|adamw · `mil.lr` 1e-3.

**Thiếu VRAM:** giảm `train.batch_size` và tăng `train.grad_accum` (giữ tích = 16), hoặc `model.grad_checkpointing=true`
(Swin), hoặc `train.freeze=partial`. Hướng dẫn chạy trên Kaggle: [docs/kaggle.md](docs/kaggle.md).

**Lưu ý backbone:** I3D/X3D (PyTorchVideo) gốc dùng AvgPool kernel cố định cho 8×224 / 13×160 → lỗi với clip
16×112; đã thay bằng `AdaptiveAvgPool3d` (`src/models/i3d.py`, `x3d.py`). Swin3D K400 chuẩn hoá theo ImageNet → tự đổi
chuẩn hoá bên trong `src/models/swin3d.py`. Test: `pytest tests/test_tv3_training.py`.
