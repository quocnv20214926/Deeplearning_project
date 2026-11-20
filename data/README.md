# Dữ liệu — [Owner: TV1]

Chỉ commit `annotations/` và `processed/splits/` (split cố định cho cả nhóm). Phần còn lại mỗi người tự tạo.
```
data/
├── raw/ucf-crime-dataset.zip            # Kaggle odins0n (64x64, 1/10 frame) — để nguyên zip, không cần giải nén
│   (hoặc raw/ucf_crime/{Train,Test}/<Class>/<Video>_<frame>.png nếu đã giải nén)
├── annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt   # nhãn thời gian test (tác giả UCF-Crime)
├── processed/
│   ├── splits/{train,val,test}.txt      # "<video> <class> <n_frames>"            ← có trong git
│   ├── splits/test_frame_labels.npz     # {video: nhãn 0/1 cho từng ảnh test}    ← có trong git
│   ├── frames_index.csv                 # mỗi ảnh một dòng: split, label, class_id, video, frame_idx, pos, path
│   ├── videos.csv                       # mỗi video một dòng: n_frames, first/last_frame, n_missing, subset
│   └── frames/{Train,Test}/<video>.npy  # uint8 (N, 64, 64, 3), ~16 GB — tuỳ chọn, đọc nhanh gấp ~2 lần PNG
├── features/<model>/<split>/<video>.npy # cho chế độ MIL (TV3)
└── external/rwf2000/                    # cross-dataset (TV2)
```

## Chuẩn bị (mỗi người chạy một lần)
```bash
kaggle datasets download -d odins0n/ucf-crime-dataset -p data/raw     # 11,8 GB
python scripts/prepare_data.py              # annotations -> index -> pack -> splits
python scripts/prepare_data.py --no-pack    # không đủ ~16 GB trống: dataset đọc PNG trực tiếp từ zip
python scripts/prepare_data.py --steps eda  # bảng + hình EDA -> results/tables/eda, results/figures/eda
```
Bước `splits` tất định (seed 42) nên tạo lại đúng file đang có trong git.
Trên **Kaggle Notebook** (dataset `odins0n/ucf-crime-dataset` gắn sẵn; output tối đa 20 GB nên không pack):
```bash
!git clone <repo> && cd Deeplearning_project && pip install -q pyyaml
!python scripts/prepare_data.py --no-pack --raw /kaggle/input/ucf-crime-dataset
```

## Kết quả chuẩn bị (giao thức `configs/base.yaml`)
| Tập | Video (Normal / bất thường) | Ảnh | Clip 16 frame | Clip bất thường |
|---|---|---|---|---|
| train | 1.449 (720 / 729) | 1.177.438 | 40.592 | 57,1% |
| val | 161 (80 / 81) | 88.907 | 4.397 | 58,9% |
| test | 290 (150 / 140) | 111.308 | 7.089 | 41,8% |

- Val = 10% video Train của **mỗi lớp** (seed 42), chia theo video nên không rò rỉ frame giữa train và val.
- Test: 290/290 video có nhãn thời gian; 8.548/111.308 ảnh (7,68%) bất thường. Ảnh thứ *k* của video ↔ frame gốc *10k*
  (đã kiểm tra: mọi video bắt đầu từ frame 0, không thiếu frame).
- Clip train/val: stride 8 (bất thường) / 16 (Normal), tối đa 64 clip/video; clip test: stride 16, phủ mọi frame.
  Nhãn clip train = nhãn video (**nhãn yếu**: chỉ ~18% frame trong video bất thường thực sự bất thường → nên dùng MIL).

## ⚠ Thứ tự 14 lớp (chỉ số đầu ra của ClsHead)
`src/data/constants.py: CLASSES` — **NormalVideos = 0**, rồi Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting,
RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism (1–13). Dataset trả `class_id` theo thứ tự này, nên
khi tính điểm bất thường `1 - softmax[normal]` và accuracy 14 lớp phải dùng đúng `CLASSES` (KHÔNG sắp xếp lại theo alphabet).

## Dùng trong code
**Train (TV3, mọi người):**
```python
from torch.utils.data import DataLoader
from src.data.ucf_crime import UCFCrimeClips, make_balanced_sampler
from src.data.transforms import build_train_transform, build_eval_transform

ds = UCFCrimeClips(cfg, "train", build_train_transform(cfg))      # -> clip (3,16,112,112), class_id, binary
dl = DataLoader(ds, batch_size=16, sampler=make_balanced_sampler(ds), num_workers=4)
val = UCFCrimeClips(cfg, "val", build_eval_transform(cfg))        # ds.items[i] = (video, start, class_id)
```
**Trích feature cho MIL (TV3):** mọi clip không chồng lấn của mỗi video, theo thứ tự thời gian, không tăng cường:
```python
ds = UCFCrimeClips(cfg, split, build_eval_transform(cfg), stride=cfg["data"]["clip_len"], max_clips=0, jitter=False)
# gom feature theo ds.items[i][0] (tên video) -> data/features/<model>/<split>/<video>.npy
```
**Đánh giá (TV2)** — thay `list_test_videos` / `load_frames_u8` trong `src/evaluate.py`:
```python
from src.data.ucf_crime import read_split, load_video, build_store
from src.data.transforms import build_eval_transform
store, tf = build_store(cfg), build_eval_transform(cfg)
labels = np.load("data/processed/splits/test_frame_labels.npz")    # labels[video] -> 0/1 từng ảnh
for video, cls, n in read_split(cfg, "test"):                      # 290 video, cls = tên lớp
    frames = load_video(cfg, "test", video, store)                 # uint8 (n, 64, 64, 3) RGB
    clip = tf(frames[pos])                                         # pos: chỉ số 16 frame -> (3,16,112,112)
```
Hoặc dùng thẳng `UCFCrimeClips(cfg, "test", tf, videos=[(video, cls, n)], stride=s)` với DataLoader;
`ds.items` cho biết vị trí bắt đầu từng clip (có thêm 1 clip căn theo cuối video).
**App (TV4):** frame OpenCV là BGR → `build_eval_transform(cfg)(frames[..., ::-1])` với `frames` (T, H, W, 3) uint8;
transform tự thu nhỏ về 64×64 rồi phóng lên 112×112 giống hệt lúc train.

## Lưu ý
- Trên Windows, khi hết VRAM thì CUDA tràn sang RAM thường thay vì báo lỗi (train chậm hàng chục lần):
  đặt `torch.cuda.set_per_process_memory_fraction(0.9)` để báo `OutOfMemoryError` ngay.
- Kiểm thử phần dữ liệu và mô hình TV1: `pytest tests/test_tv1_data.py tests/test_tv1_models.py`
  (`RUN_SLOW=1` để tải trọng số pretrained và thử train trên GPU).
