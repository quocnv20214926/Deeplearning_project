# Đề tài 7 — Hệ thống cảnh báo bạo lực & bất thường từ camera giám sát
*Đồ án môn Deep Learning — **Application Track***

**Bài toán:** Spatio-temporal video classification / Video Anomaly Detection
**Mô hình:** 3D-CNN (I3D, R(2+1)D) hoặc Video Swin Transformer
**Dữ liệu:** UCF-Crime (Kaggle: odins0n/ucf-crime-dataset)
**Metric:** Frame-level ROC-AUC, False Positive Rate (FPR), FPS

> **Định hướng application track:** sản phẩm cuối là **một ứng dụng chạy được end-to-end** (nhận video/stream → phát hiện → cảnh báo), không chạy đua SOTA. Mô hình cần *đủ chính xác* (AUC hợp lý, FPR thấp để không báo động giả) và *đủ nhanh* (real-time). Đóng góp nằm ở: pipeline hoàn chỉnh, đánh đổi tốc độ/độ chính xác, trải nghiệm người dùng, demo thuyết phục.

---

## 0. Yêu cầu của môn học (theo *Final Project Guidelines*)
Track 1 — Application Track yêu cầu nhóm:
1. **Đặt bài toán thực tế** → giám sát CCTV, giảm tải cho nhân viên an ninh.
2. **Tải & tiền xử lý một bộ dữ liệu mở cụ thể** → UCF-Crime (Kaggle); *được phép bổ sung dataset khác phù hợp*.
3. **Adapt & fine-tune kiến trúc DL** → fine-tune 3D-CNN (R(2+1)D / I3D) và Video Swin pretrained Kinetics, không chỉ dùng model có sẵn.
4. **Đánh giá metric thực tiễn**, gồm **inference latency**, **memory footprint** và **độ chính xác theo miền** → Frame-level ROC-AUC, FPR, FPS + latency (ms), VRAM/RAM, kích thước model.
5. **Chứng minh giá trị thực tế** bằng dataset đại diện và metric chuẩn → demo ứng dụng + thử trên video ngoài tập.

**Nộp bài:** báo cáo, slide, mã nguồn (link nộp sẽ thông báo sau). Đề không quy định demo app hay hạn cụ thể → app là phần *minh hoạ giá trị thực tế*, giữ ở mức gọn (Streamlit), không đầu tư quá mức so với phần mô hình và đánh giá.

---

## 1. Mục tiêu sản phẩm (user story)
*Nhân viên an ninh theo dõi nhiều camera, không thể nhìn liên tục.* Hệ thống:
1. Nhận **video upload**, **webcam** hoặc **luồng RTSP/IP camera**.
2. Chấm **điểm bất thường theo thời gian thực** cho từng đoạn ~1–2 giây.
3. Khi điểm vượt ngưỡng liên tục N giây → **cảnh báo** (popup/âm thanh, tuỳ chọn gửi Telegram), kèm **loại sự kiện dự đoán** (Fighting, Robbery, Explosion…).
4. Lưu **nhật ký sự kiện**: thời điểm, loại, độ tin cậy, ảnh chụp + đoạn clip ngắn.
5. Dashboard: video đang chạy + biểu đồ điểm theo thời gian + danh sách sự kiện.

**Tiêu chí thành công (tham khảo)**
| Tiêu chí | Mục tiêu |
|---|---|
| Frame-level ROC-AUC (test UCF-Crime) | ≥ 75–80% |
| FPR trên video Normal | ≤ 5–10% |
| Tốc độ | ≥ 25 FPS (real-time) trên GPU phổ thông; báo cả FPS trên CPU |
| Latency mỗi snippet | báo ms (mean / p95) |
| Memory footprint | VRAM đỉnh khi inference, RAM, kích thước file model (MB) |
| Độ trễ cảnh báo | ≤ 2–3 giây từ lúc sự kiện bắt đầu |

---

## 2. Khảo sát (tóm tắt)

### 2.1. Dữ liệu — lưu ý quan trọng
| | Bản Kaggle (odins0n) | Bản gốc UCF-Crime |
|---|---|---|
| Dạng | Ảnh PNG **64×64**, **10 frame lấy 1** (~3 fps) | 1.900 video, 320×240, 30 fps (~100 GB) |
| Số lượng | 1.266.345 ảnh train, 111.308 ảnh test (~11.8 GB) | 1.610 train / 290 test |
| Lớp | 13 loại bất thường + Normal | như trên |
| Nhãn | Theo video (thư mục) | Test có **nhãn thời gian** (frame bắt đầu/kết thúc) |

- Tính frame-level AUC cần file `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` từ bản gốc → map ảnh thứ *k* ↔ frame ≈ *10k*.
- **Lệch miền khi triển khai:** model học trên 64×64 @3fps, còn app nhận video thật 30fps độ phân giải cao → trong app phải **tiền xử lý giống lúc train** (lấy mẫu 1/10 frame, resize về độ phân giải train rồi upsample lên kích thước đầu vào). Đây là điểm cần nêu rõ trong báo cáo.

### 2.2. Các hướng mô hình
| Hướng | Ưu | Nhược | Dùng cho app? |
|---|---|---|---|
| **Fine-tune 3D-CNN/Swin phân loại clip** (nhãn theo video) | Đáp ứng trực tiếp yêu cầu *fine-tune*; cho luôn **loại sự kiện** | Nhãn nhiễu (video bất thường có nhiều đoạn bình thường) | ✔ **Mô hình chính #1** + phân loại loại sự kiện |
| **Weakly-supervised MIL** (feature từ backbone đã fine-tune → head chấm điểm) | Chuẩn UCF-Crime, AUC tốt, head rất nhẹ | Chỉ cho điểm bất thường | ✔ **Mô hình chính #2** — so sánh với #1 |
| CLIP / VLM (VadCLIP…) | AUC cao nhất | Ngoài phạm vi đề, nặng | ✘ (chỉ nhắc trong survey) |

**Mốc tham chiếu AUC (I3D, paper):** MIL ~80%, RTFM ~84%, UR-DMU ~87% → với dữ liệu 64×64, kỳ vọng thực tế **75–82%**.

### 2.3. Các mô hình sẽ đánh giá (nhóm 4 người)
Mỗi thành viên **sở hữu một họ mô hình**, train theo **cùng giao thức** (cùng split, cùng tiền xử lý, cùng `eval.py`) để so sánh công bằng.

| # | Mô hình | Họ | Tham số | Vai trò trong so sánh | Nguồn pretrained |
|---|---|---|---|---|---|
| M1 | **R(2+1)D-18** | 3D-CNN tách không gian/thời gian | ~31M | Baseline 3D-CNN, nhẹ | `torchvision` (K400) |
| M2 | **I3D-R50** | 3D-CNN inflated | ~28M | Backbone chuẩn của paper VAD | PyTorchVideo (K400) |
| M3 | **Video Swin-T** | Video Transformer | ~28M (FLOPs cao) | Kỳ vọng chính xác nhất | `torchvision swin3d_t` (K400) |
| M4 | **X3D-S** / **MoViNet-A0** | 3D-CNN hiệu quả cho mobile/edge | ~2–4M | Mô hình nhẹ cho triển khai realtime/CPU | PyTorchVideo / TF-Hub→PyTorch |
| (B0) | MobileNetV3 + GRU (2D CNN + RNN) | Baseline không 3D | ~3–5M | Chứng minh lợi ích của mô hình không-thời gian 3D | `torchvision` (ImageNet) |

**Mỗi mô hình chạy 2 chế độ** (đáp ứng yêu cầu fine-tune + bài toán VAD):
- **(a) Fine-tune phân loại clip**: Normal vs Anomaly + 14 lớp → điểm bất thường = 1 − P(Normal).
- **(b) MIL head** trên feature của backbone đã fine-tune (32 segment/video, MIL ranking loss).

**Thí nghiệm chung (mỗi người làm trên mô hình của mình, kết quả gộp lại):**
1. Độ phân giải đầu vào: 112 vs 160/224.
2. Độ dài clip: 8 vs 16 frame.
3. Frozen backbone vs fine-tune toàn bộ vs fine-tune vài block cuối.
4. Tối ưu triển khai: FP32 → FP16 → ONNX Runtime → (INT8 quantization cho M1/M4).

→ Kết quả trung tâm của báo cáo: **biểu đồ Pareto AUC vs FPS** (kích thước điểm = VRAM) và bảng tổng hợp 5 mô hình × 2 chế độ → chọn mô hình triển khai cho GPU và cho CPU/edge.

---|---|---|
| R(2+1)D-18 (torchvision) | ~31M | Nhẹ, nhanh, dễ fine-tune |
| I3D-R50 (PyTorchVideo) | ~28M | Backbone chuẩn của paper VAD |
| Video Swin-T (torchvision `swin3d_t`) | ~28M, FLOPs cao hơn | Chính xác nhất, chậm nhất |
→ Một phần quan trọng của báo cáo là **biểu đồ AUC vs FPS** để chọn backbone triển khai.

---

## 3. Kiến trúc hệ thống

```
[Nguồn: file / webcam / RTSP]
        │  OpenCV đọc frame (thread riêng)
        ▼
[Bộ đệm trượt] lấy mẫu 1/10 frame, gom 16 frame → 1 snippet (stride 8)
        ▼
[Backbone 3D (frozen, FP16/ONNX)] → feature
        ├─► [MIL head]  → điểm bất thường s ∈ [0,1]
        └─► [Classifier head] → loại sự kiện (top-1)
        ▼
[Hậu xử lý] làm mượt (moving average) + ngưỡng + hysteresis (≥N snippet liên tiếp)
        ▼
[Cảnh báo & Log]  popup / Telegram, lưu ảnh + clip, SQLite/CSV
        ▼
[UI Streamlit/Gradio]  video + biểu đồ score realtime + bảng sự kiện
```

**Công nghệ:** PyTorch, torchvision/PyTorchVideo, OpenCV, ONNX Runtime (tuỳ chọn TensorRT), Streamlit (hoặc Gradio), SQLite, Docker để đóng gói.

---

## 4. Đánh giá
**Đánh giá mô hình (offline, trên test UCF-Crime):**
- Frame-level ROC-AUC (gộp) + AUC theo từng lớp; vẽ ROC.
- FPR tại ngưỡng dùng trong app, tính trên các video Normal; FPR @ TPR=90%.
- Accuracy / confusion matrix cho classifier loại sự kiện.

**Đánh giá hệ thống (điểm nhấn của application track):**
- **Memory footprint:** số tham số, kích thước file (MB), VRAM đỉnh (`torch.cuda.max_memory_allocated`), RAM tiến trình — cho từng backbone và từng định dạng (FP32/FP16/ONNX/INT8 nếu có).
- **Inference latency** mỗi snippet (ms, mean/p95).
- **FPS end-to-end** (đọc frame → cảnh báo), batch=1, có warm-up; ghi rõ GPU/CPU. So sánh PyTorch FP32 vs FP16 vs ONNX.
- **Độ trễ cảnh báo** (giây từ lúc bắt đầu sự kiện thật đến lúc báo).
- **Số báo động giả mỗi giờ** khi chạy video Normal dài → chọn ngưỡng.
- Thử trên **video ngoài tập dữ liệu** để đánh giá khả năng tổng quát hoá, phân tích ca thất bại. Đề cho phép dùng thêm dataset: gợi ý **RWF-2000** (đánh nhau trên CCTV) hoặc một phần **XD-Violence** làm tập kiểm tra cross-dataset, hoặc video gốc UCF-Crime độ phân giải đầy đủ.
- **Bảng tổng hợp cuối:** Backbone | AUC | FPR | FPS | Latency (ms) | VRAM (MB) | Size (MB) → chọn model triển khai.

---

## 5. Kế hoạch thực hiện & phân công (4 người, ~7 tuần)

> Danh sách việc chi tiết từng người, mốc kiểm tra và người chịu trách nhiệm sản phẩm nộp: xem **[TASKS.md](TASKS.md)**.

### 5.1. Vai trò
Mỗi người = **1 họ mô hình** (phần DL, chia đều) + **1 mảng hệ thống** (phần application).

| Thành viên | Mô hình phụ trách | Mảng hệ thống phụ trách |
|---|---|---|
| **TV1** | M1 R(2+1)D-18 + B0 (MobileNetV3+GRU) | **Dữ liệu**: tải, map nhãn thời gian, `dataset.py`, dataloader chung, EDA |
| **TV2** | M2 I3D-R50 | **Đánh giá & benchmark**: `eval.py` (AUC, FPR, AUC theo lớp), đo FPS/latency/VRAM, bảng & biểu đồ Pareto, test cross-dataset (RWF-2000) |
| **TV3** | M3 Video Swin-T | **Training framework & MIL**: `train.py` chung (config yaml, logging wandb), MIL head + loss, quản lý checkpoint |
| **TV4** | M4 X3D-S / MoViNet-A0 | **Ứng dụng & triển khai**: inference realtime, export ONNX/FP16/INT8, Streamlit app, cảnh báo, Docker |

Việc chung: mỗi người viết phần báo cáo về mô hình + mảng của mình; slide chia theo cùng cấu trúc; TV2 tổng hợp kết quả, TV4 dựng video demo. Họp tuần ngắn để chốt giao thức thí nghiệm (không đổi split/tiền xử lý giữa chừng).

### 5.2. Lịch theo tuần
| Tuần | TV1 (Data + M1/B0) | TV2 (Eval + M2) | TV3 (Train/MIL + M3) | TV4 (App + M4) | Mốc chung |
|---|---|---|---|---|---|
| 1 | Tải dữ liệu, map nhãn, EDA, `dataset.py` | Thiết kế `eval.py`, tìm & chuẩn bị RWF-2000 | Khung `train.py` + config, wandb | Wireframe UI, đọc video/RTSP + bộ đệm trượt | **Chốt giao thức & repo** |
| 2 | Fine-tune M1 (clip cls) | Hoàn thiện `eval.py`, tool đo FPS/VRAM; chạy eval cho M1 | Cài MIL head + loss, test trên feature M1 | Pipeline inference CLI với M1 | **Baseline M1 end-to-end** |
| 3 | Train B0; M1 chế độ MIL | Fine-tune M2 (clip cls) | Fine-tune M3 (clip cls) | Fine-tune M4 (clip cls) | Tất cả mô hình chạy chế độ (a) |
| 4 | Ablation độ phân giải/độ dài clip (M1) | M2 chế độ MIL; ablation | M3 chế độ MIL; ablation frozen vs full | M4 chế độ MIL; Streamlit app v1 | **Bảng kết quả v1** |
| 5 | Phân tích lỗi theo lớp, trực quan score theo thời gian | Benchmark FP32/FP16/ONNX cho cả 5 mô hình; Pareto | Tinh chỉnh model tốt nhất; chọn ngưỡng theo FPR | Export ONNX/INT8, app v2 (webcam/RTSP, log, cảnh báo) | **Chọn model triển khai** |
| 6 | Test video ngoài tập | Đánh giá cross-dataset RWF-2000 | Grad-CAM/attention map giải thích dự đoán | Docker, README, tích hợp model cuối | **Code freeze** |
| 7 | Báo cáo: dữ liệu | Báo cáo: đánh giá & so sánh | Báo cáo: phương pháp & MIL | Báo cáo: hệ thống; video demo | **Nộp báo cáo, slide, code** |

## 6. Cấu trúc repo (đã tạo sẵn)
```
Deeplearning_project/
├── README.md · PLAN.md · TASKS.md · requirements.txt · Dockerfile · .gitignore
├── configs/        base.yaml (giao thức chung) + m1_r2plus1d · m2_i3d · m3_swin3d · m4_x3d · b0_mobilenet_gru
├── data/           raw/ · annotations/ · processed/ · features/ · external/   (không commit)
├── notebooks/      01_eda · 02_results_analysis · 03_error_analysis · 04_cross_dataset
├── scripts/        prepare_data.py (TV1) · run_all_benchmarks.py (TV2)
├── src/
│   ├── data/       annotations.py · ucf_crime.py · transforms.py (TV1) · mil_dataset.py (TV3)
│   ├── models/     r2plus1d · mobilenet_gru (TV1) · i3d (TV2) · swin3d (TV3) · x3d (TV4) · heads (TV3)
│   ├── losses/     mil_loss.py (TV3)
│   ├── utils/      metrics.py (TV2) · config.py · seed.py
│   ├── train.py · train_mil.py · extract_features.py   (TV3)
│   ├── evaluate.py · benchmark.py · visualize.py       (TV2)
│   └── export_onnx.py                                  (TV4)
├── app/            video_source · inference · alerts · streamlit_app   (TV4)
├── weights/        checkpoint (không commit)
├── results/        tables/ · figures/ · logs/ · predictions/
├── reports/        report/ · slides/ · demo/
└── tests/          test_metrics.py
```
Mỗi file có ghi `[Owner: TVx]` và `TODO` ở đầu. Đã viết sẵn: định nghĩa backbone (torchvision/PyTorchVideo), ClsHead/MILHead, MIL ranking loss, metric AUC/FPR, đọc nhãn thời gian, đọc config, hàm đo latency/VRAM.

## 7. Rủi ro & phương án
| Rủi ro | Phương án |
|---|---|
| Ảnh 64×64/3fps → AUC thấp, lệch miền khi chạy video thật | Tiền xử lý trong app giống lúc train; nêu rõ hạn chế; nếu đủ thời gian thì so sánh với feature I3D trích sẵn từ video gốc |
| Báo động giả nhiều | Làm mượt + hysteresis + chọn ngưỡng theo FPR trên video Normal |
| Không đạt real-time | Backbone nhẹ (R(2+1)D), FP16/ONNX, stride lớn hơn, chạy model ở thread riêng |
| GPU hạn chế | Freeze backbone, trích feature 1 lần; train head trên Colab/Kaggle |
| Thiếu nhãn thời gian trong bản Kaggle | Tải file annotation từ trang UCF-Crime gốc (CRCV, UCF) |

## 8. Đạo đức & quyền riêng tư (nên có trong báo cáo)
- Hệ thống chỉ **hỗ trợ** con người, không tự động xử lý; luôn có người xác nhận cảnh báo.
- Rủi ro thiên lệch (bối cảnh, độ phân giải, màu sắc camera); không nhận dạng danh tính.
- Lưu trữ clip có giới hạn thời gian, chỉ lưu đoạn có sự kiện.

## Tài liệu tham khảo
- *Introduction to Deep Learning — Final Project Guidelines & Topic Catalog* (đề tài số 7, Track 1).
- Sultani et al., *Real-world Anomaly Detection in Surveillance Videos*, CVPR 2018.
- Tian et al., *RTFM*, ICCV 2021 — https://arxiv.org/abs/2101.10030
- Zhou et al., *UR-DMU*, AAAI 2023 — https://arxiv.org/abs/2302.05160
- *Auditing Frame-Level AUC in WSVAD* (2026) — https://arxiv.org/abs/2608.11985
- Mô tả bản Kaggle (mirror HF) — https://huggingface.co/datasets/hibana2077/UCF-Crime-Dataset
