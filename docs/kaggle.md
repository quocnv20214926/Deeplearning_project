# Chạy train trên Kaggle (TV3 — dùng được cho cả nhóm)

Notebook mẫu: [`notebooks/kaggle_tv3_train.ipynb`](../notebooks/kaggle_tv3_train.ipynb). Các bước dưới đây giải thích từng phần.

## 1. Đẩy code lên GitHub (trên máy)
```bash
git checkout -b feat/tv3-train-framework
git add src/train.py src/train_mil.py src/extract_features.py src/utils/config.py src/utils/model_io.py \
        src/models/swin3d.py src/models/i3d.py src/models/x3d.py tests/test_tv3_training.py \
        configs/m3_swin3d.yaml notebooks/kaggle_tv3_train.ipynb docs/ README.md TASKS.md
git commit -m "TV3: train framework, MIL, feature extraction, Swin-T backbone"
git push -u origin feat/tv3-train-framework      # rồi mở Pull Request vào master
```
Kaggle clone repo công khai qua HTTPS, nên chỉ cần push branch là notebook lấy được code.

## 2. Tạo notebook Kaggle
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → chọn `notebooks/kaggle_tv3_train.ipynb`.
2. Thanh bên phải (**Session options**):
   - **Accelerator:** `GPU T4 x2` (hoặc `GPU P100`). Code dùng 1 GPU.
   - **Internet:** `On` (để clone repo, `pip install pytorchvideo`, tải trọng số Kinetics).
   - Tài khoản cần **xác minh số điện thoại** mới bật được GPU và Internet.
3. **Add Input** → tìm `odins0n/ucf-crime-dataset` → Add. Dataset được gắn sẵn ở `/kaggle/input/...`
   (không tốn 11,8 GB tải về; notebook tự tìm thư mục chứa `Train/` và `Test/`).
4. (Tuỳ chọn) wandb: **Add-ons → Secrets** → thêm `WANDB_API_KEY`, bật quyền truy cập cho notebook.

## 3. Thứ tự chạy
| Ô | Việc | Thời gian ước tính |
|---|---|---|
| 0 | clone branch, cài pytorchvideo, đặt `CFG_OVERRIDES` | 1–2 phút |
| 1 | `prepare_data.py --steps index` (lập `frames_index.csv`), `pytest` | 5–10 phút |
| 2 | chạy thử 1 epoch trên 2 video/lớp (R(2+1)D và Swin) | 2–5 phút |
| A | trích feature R(2+1)D K400 + train MIL (**task tuần 2**) | 20–30 phút |
| B | đo tốc độ Swin, rồi fine-tune M3 Swin-T (**task tuần 3**) | vài giờ |
| C | evaluate test, feature Swin + MIL, benchmark | 30–45 phút |
| D | nén kết quả | 1 phút |

Các ô 0 → A chạy tương tác được. Ô **B (chạy thật)** nên chạy nền: **Save Version → Save & Run All (Commit)**.
Phiên commit chạy tối đa **12 giờ** kể cả khi tắt trình duyệt. Mọi thứ trong `/kaggle/working/` thành **Output**
của version đó. Quota GPU khoảng 30 giờ/tuần, nên ở chế độ tương tác hãy tắt session khi không dùng.

## 4. Theo dõi và xử lý sự cố
- Log mỗi epoch: `results/logs/swin3d_t_cls.csv` (loss, `val_clip_auc`, `val_video_auc`, `val_acc14`, phút/epoch,
  `vram_peak_gb`). Có wandb thì xem trực tiếp trên web.
- **CUDA out of memory:** thêm `--set train.batch_size=4 train.grad_accum=4` (giữ tích = 16), hoặc
  `model.grad_checkpointing=true`, hoặc `train.freeze=partial`.
- **Quá chậm** (xem dòng `clip/s` trong log): epoch dài thì đặt `--set train.samples_per_epoch=20000` để có nhiều
  điểm val/checkpoint hơn. Nếu nghẽn ở khâu đọc ảnh (GPU-Util thấp trong `nvidia-smi`), có thể đóng gói frame:
  `python scripts/prepare_data.py --steps pack` cùng `data.packed_dir=/tmp/frames` (khoảng 16 GB, mất khi hết phiên).
- **Bị ngắt / hết 12 giờ:** tạo version mới → **Add Input** → *Your Work → Notebook Output* của version cũ → copy
  `weights/swin3d_t_cls*.pt` vào `weights/` → chạy với `--resume weights/swin3d_t_cls_last.pt`.
  Feature đã trích cũng được bỏ qua tự động khi chép lại `data/features/`.
- **`pytorchvideo` lỗi import:** chỉ I3D (TV2) và X3D (TV4) cần gói này, Swin và R(2+1)D thì không.

## 5. Kết quả cần gửi cho nhóm
- `weights/swin3d_t_cls.pt`, `weights/swin3d_t_mil.pt`: lưu ở Output, hoặc tạo Kaggle Dataset riêng để TV2/TV4 Add Input.
- `results/tables/m3_swin3d_cls.json`, `results/tables/swin3d_t_mil.json`, `results/tables/r2plus1d_18_k400_mil.json`
  và `results/predictions/*.npz` → gửi TV2 để ghép bảng tổng hợp.
- Checkpoint và feature **không commit** vào git (đã có trong `.gitignore`), chỉ commit bảng JSON/CSV kết quả.
