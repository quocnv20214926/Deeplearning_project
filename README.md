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
